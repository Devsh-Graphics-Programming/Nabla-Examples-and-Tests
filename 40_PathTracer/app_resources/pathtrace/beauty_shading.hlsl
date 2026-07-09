#include "nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl"

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"
#include "next_event_estimator.hlsl"

[[vk::push_constant]] SBeautyPushConstants pc;

struct[raypayload] SAnyHitRetval
{
    // before sending the ray by the caller
    inline void init(const float32_t _xi, float32_t tMax = hlsl::numeric_limits<float32_t>::max)
    {
        xi   = _xi;
        rayT = tMax;
    }
    // call in AnyHit instead of AcceptHit
    inline void acceptHit(const float16_t _transparency)
    {
        // need to read the spec if an anyhit is possible that the last anyhit to run and accept a hit candidate for a ray is not the last one to
        if (rayT > spirv::RayTmaxKHR)
        {
            rayT         = spirv::RayTmaxKHR;
            transparency = _transparency;
        }
        // Note that `spirv::terminateRayKHR` is NOT the correct instruction to call (it terminates ray prematurely without considering anything else)
    }
    //

    // opacity russian roulette requires this for Discrete Probability Sampling
    float32_t xi : read(anyhit) : write(caller, anyhit);
    // need to store the t value at which the anyhit was executed, so we know whether the current closest hit comes from a confirmed anyhit
    float32_t rayT : read(caller, anyhit) : write(caller, anyhit);
    // essentially the probability of transmission
    float16_t transparency : read(caller) : write(anyhit);
    // can use additional `float16` to store BxDF mixture weights or other things so they don't need recomputing/re-fetching during shading
};

SReservoir getReservoirs(NBL_REF_ARG(LegacyBdaAccessor<SReservoir>) reservoirBuf, uint baseIndex, uint sampleIndex)
{
    uint32_t index = baseIndex + sampleIndex * gSensor.renderSize.x * gSensor.renderSize.y;
    SReservoir reservoir;
    reservoirBuf.get(index, reservoir);
    return reservoir;
}

void setReservoirs(NBL_REF_ARG(LegacyBdaAccessor<SReservoir>) reservoirBuf, uint baseIndex, uint sampleIndex, SReservoir reservoir)
{
    uint32_t index = baseIndex + sampleIndex * gSensor.renderSize.x * gSensor.renderSize.y;
    reservoirBuf.set(index, reservoir);
}

[shader("raygeneration")]
void raygen()
{
    const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);
    const SBeautyPushConstants::S16BitData unpacked16BitPC = pc.get16BitData();
    const uint32_t linearIdx = launchID.y * gSensor.renderSize.x + launchID.x;

    LegacyBdaAccessor<SReconnectionData> reconnDataPtr = LegacyBdaAccessor<SReconnectionData>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::ReconnectionDataBuf]);
    SReconnectionData rcData;
    reconnDataPtr.get(linearIdx, rcData);

    const bool adjustShadingNormal = rcData.pathLength <= 1;    // TODO: need?

    LegacyBdaAccessor<SReservoir> initialReservoirsPtr = LegacyBdaAccessor<SReservoir>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::InitialReservoirsBuf]);
    LegacyBdaAccessor<SReservoir> currentReservoirsPtr = LegacyBdaAccessor<SReservoir>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::CurrentReservoirsBuf]);

    SReservoir initialReservoir;
    initialReservoirsPtr.get(linearIdx, initialReservoir);

    if (false)  // TODO ReSTIR: check roughness greater than threshold
    {
        setReservoirs(currentReservoirsPtr, linearIdx, 0, initialReservoir);
        setReservoirs(currentReservoirsPtr, linearIdx, 1, initialReservoir);
        return;
    }

    using namespace nbl::hlsl::bxdf;
    using namespace nbl::hlsl::material_compiler3::backends::default_upt;
    using bxdf_config_t           = BxDFConfig;
    using isotropic_interaction_t = bxdf_config_t::isotropic_interaction_type;
    using light_sample_t          = bxdf_config_t::sample_type;
    using spectral_type           = bxdf_config_t::spectral_type;
    using ray_dir_info_t          = light_sample_t::ray_dir_info_type;
    using quotient_weight_type    = sampling::quotient_and_weight<spectral_type, float>;
    using value_weight_type       = sampling::value_and_weight<spectral_type, float>;
    // a little bit of persistent state
    spirv::HitObjectEXT hitObject;
    {
        // fetch random variable from memory
        const float32_t3 randVec = randgen(0u, 0u);
        // TODO: motion blur and lens DOF triplet

        // get our NDC coordinates and ray
        const float32_t2  pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(spirv::LaunchSizeKHR.xy);
        const float32_t2  NDC          = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);
        const SPrimaryRay primary      = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, NDC, float16_t2(randVec.xy));
        const SRay        ray          = primary.ray;

        // TODO: possible SER point, sorting by ray direction
        //spirv::reorderThreadWithHintEXT<uint32_t>(,);

        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval payload;
        const float                                                             tMax = pc.sensorDynamics.tMax;
        payload.init(randVec.z, tMax);
        spirv::hitObjectTraceRayEXT(hitObject, gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PATH, 0u, 0u, ray.origin, primary.tMin, ray.direction.getDirection(), tMax, payload);
        // TODO: do something with the payload's reported transparency
    }
    // TODO: Possible SER point
    const bool primaryMissed = spirv::hitObjectIsMissEXT(hitObject);
    const float32_t3 primaryRayDir = spirv::hitObjectGetWorldRayDirectionEXT(hitObject);

    SClosestHitRetval closestInfo = SClosestHitRetval::create(hitObject);

    // get previous sample
    float32_t4 previousClip = prevViewProj * closestInfo.hitPos;
    float32_t3 previousScreen = previousClip.xyz / previousClip.w;
    float32_t2 previousUV = previousScreen.xy * float32_t2(0.5f, -0.5f) + 0.5f;
    uint32_t2 previousID = hlsl::clamp(previousUV * gSensor.renderSize, hlsl::promote<uint32_t2>(0), gSensor.renderSize - hlsl::promote<uint32_t2>(1));
    uint32_t previousIdx = previousID.y * gSensor.renderSize.x + previousID.x;

    bool isPreviousValid = params.frameCount > 0 && all(previousUV > 0.f) && all(previousUV < 1.f);
    LegacyBdaAccessor<SReservoir> previousReservoirsPtr = LegacyBdaAccessor<SReservoir>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::PreviousReservoirsBuf]);

    // TODO ReSTIR: TEMPORAL REUSE HERE

    // spatial reuse
    SReservoir spatialReservoir = getReservoirs(previousReservoirsPtr, previousIdx, 0);
    spatialReservoir.vPosition = rcData.preRcHitPosition;
    spatialReservoir.vNormal = rcData.preRcNormal;

    float32_t cellSize = calculateCellSize(initialReservoir.vPosition, cameraPos, gSensor.renderSize, gSensor.restirParams);
    float32_t3 jitteredPos = rcData.preRcHitPosition + (randgen(0u, 1u) * 2.0f - 1.0f) * 0.1f * cellSize;

    int cellIdx = findCell(jitteredPos, rcData.preRcNormal, cellSize, params, checkSum);
    if (cellIdx == -1)
    {
        setReservoirs(currentReservoirsPtr, linearIdx, 1, spatialReservoir);
        return;
    }
    uint32_t cellBaseIdx, sampleCount;
    {
        bda::__ptr<uint32_t> ptr = bda::__ptr<uint32_t>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::IndexBuf]);
        BdaAccessor<uint32_t> indexPtr = BdaAccessor<uint32_t>::create(ptr);
        indexPtr.get(cellIdx, cellBaseIdx);
    }
    {
        bda::__ptr<uint32_t> ptr = bda::__ptr<uint32_t>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::CellCountersBuf]);
        BdaAccessor<uint32_t> cellCounterPtr = BdaAccessor<uint32_t>::create(ptr);
        cellCounterPtr.get(cellIdx, sampleCount);
    }

    spatialReservoir.M = hlsl::clamp(spatialReservoir.M, 0, 100);
    if (spatialReservoir.age > 100)
    {
        spatialReservoir.M = 0;
    }

    uint32_t maxSpatialIteration = 3u;  // spatialReservoir.M > 10 ? 3u : 10u;

    uint32_t increment = (sampleCount + maxSpatialIteration - 1) / maxSpatialIteration;
    uint32_t offset = round(sampleNext1D(sg) * (increment - 1));

    float32_t3 positionList[10];
    float32_t3 normalList[10];
    int MList[10];
    uint32_t nReuse = 0;
    positionList[nReuse] = rcData.preRcHitPosition;
    normalList[nReuse] = rcData.preRcNormal;
    MList[nReuse] = spatialReservoir.M;
    nReuse++;

    float32_t wSumS = spatialReservoir.M * hlsl::dot(spatialReservoir.radiance, hlsl::material_compiler3::backends::default_upt::LumaConversionCoeffs) * max(0.f, spatialReservoir.weightF);    // evalTargetPdf
    bda::__ptr<uint32_t> _csptr = bda::__ptr<uint32_t>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::CellStorageBuf]);
    BdaAccessor<uint32_t> cellStoragePtr = BdaAccessor<uint32_t>::create(_csptr);

    uint32_t reuseID = 0;
    int count = 0;
    for (uint32_t i = 0; i < sampleCount; i += increment)
    {
        count++;

        uint32_t neighborPixelIndex;
        cellStoragePtr.get(cellBaseIdx + (offset + i) % sampleCount, neighborPixelIndex);
        SReservoir neighborReservoir = getReservoirs(previousReservoirsPtr, neighborPixelIndex, (count + 1) % 2);

        if (neighborReservoir.M <= 0 || hlsl::dot(spatialReservoir.vNormal, neighborReservoir.vNormal) < normalThreshold)
            continue;

        float32_t targetPdf = hlsl::dot(neighborReservoir.radiance, hlsl::material_compiler3::backends::default_upt::LumaConversionCoeffs); // evalTargetPdf

        float32_t3 offsetB = neighborReservoir.sPosition - neighborReservoir.vPosition;
        float32_t3 offsetA = neighborReservoir.sPosition - spatialReservoir.vPosition;
            // Discard back-face.
        if (hlsl::dot(spatialReservoir.vNormal, offsetA) <= 0.f)
            targetPdf = 0.f;

        float32_t RB2 = hlsl::dot(offsetB, offsetB);
        float32_t RA2 = hlsl::dot(offsetA, offsetA);
        offsetB = hlsl::normalize(offsetB);
        offsetA = hlsl::normalize(offsetA);
        float32_t cosA = hlsl::dot(spatialReservoir.vNormal, offsetA);
        float32_t cosB = hlsl::dot(neighborReservoir.vNormal, offsetB);
        float32_t cosPhiA = -hlsl::dot(offsetA, neighborReservoir.sNormal);
        float32_t cosPhiB = -hlsl::dot(offsetB, neighborReservoir.sNormal);
        if (cosB <= 0.f || cosPhiB <= 0.f)
            continue;
        if (cosA <= 0.f || cosPhiA <= 0.f || RA2 <= 0.f || RB2 <= 0.f)
            targetPdf = 0.f;
        float32_t jacobi = hlsl::mix(clamp(RB2 * cosPhiA / (RA2 * cosPhiB), 0.f, 10.f), 0.f, RA2 * cosPhiB <= 0.f);

        targetPdf *= jacobi;

        // TODO: start at 0 or numeric_limits::min?
        const float32_t tMin = 0.f;
        const float32_t3 originMagnitude = hlsl::max(hlsl::abs(closestInfo.hitPos), hlsl::abs(spirv::hitObjectGetWorldRayOriginEXT(hitObject)));
        // TODO: should probably also take `tMax` of found hit into account
        const float32_t offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f), originMagnitude.x), hlsl::max(originMagnitude.y, originMagnitude.z)) * hlsl::exp2(-20.f);
        const float32_t3 visRayOrigin = closestInfo.hitPos + closestInfo.geometricNormal * offsetMagnitude;
        const float32_t3 visRayDir = hlsl::normalize(neighborReservoir.sPosition - visRayOrigin);

        const float32_t3 randVis = randgen(0u, 2u + i); // need this? maybe any hit can be reduced

        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval visibilityPayload;
        visibilityPayload.init(randVis.z, hlsl::numeric_limits<float32_t>::max);
        spirv::HitObjectEXT visibilityHit;
        spirv::hitObjectTraceRayEXT(visibilityHit, gTLASes[0], 0u, 0xff, ESBTO_PATH, 0u, ESBTO_PATH, visRayOrigin, tMin, visRayDir, hlsl::numeric_limits<float32_t>::max, visibilityPayload);

        bool visRayMissed = spirv::hitObjectIsMissEXT(visibilityHit);
        if (visRayMissed)
            targetPdf = 0.f;
        //targetPdf *= dot(spatialReservoir.vNormal, neighborReservoir.vNormal);
        bool updated = spatialReservoir.merge(sg, neighborReservoir, targetPdf, wSumS);
        if (updated)
            reuseID = count;

        positionList[nReuse] = neighborReservoir.vPos;
        normalList[nReuse] = neighborReservoir.vNormal;
        MList[nReuse] = neighborReservoir.M;
        nReuse++;
    }

    float z = 0;
    float chosenWeight = 0.f;
    float totalWeight = 0.f;
    for (uint32_t i = 0; i < nReuse; i++)
    {
        bool shouldTest = true;
        bool isVisible = true;
        float32_t3 dir = spatialReservoir.sPosition - positionList[i];
        if (hlsl::dot(dir, normalList[i]) < 0.f)
        {
            shouldTest = false;
            isVisible = false;
        }
        if (shouldTest)
        {
            const float32_t tMin = 0.f;
            const float32_t3 originMagnitude = hlsl::abs(positionList[i]);
            const float32_t offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f), originMagnitude.x), hlsl::max(originMagnitude.y, originMagnitude.z)) * hlsl::exp2(-20.f);
            const float32_t3 newRayOrigin = positionList[i] + normalList[i] * offsetMagnitude;
            const float32_t3 newRayDir = hlsl::normalize(spatialReservoir.sPosition - newRayOrigin);

            [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval newPayload;
            newPayload.init(randVis.z, hlsl::numeric_limits<float32_t>::max);
            spirv::HitObjectEXT newHit;
            spirv::hitObjectTraceRayEXT(newHit, gTLASes[0], 0u, 0xff, ESBTO_PATH, 0u, ESBTO_PATH, newRayOrigin, tMin, newRayDir, hlsl::numeric_limits<float32_t>::max, newPayload);
            isVisible = spirv::hitObjectIsMissEXT(newHit);
        }
        if (isVisible)
            z += MList[i];
        else if (i == 0)
            break;
    }

    float tpNewS = hlsl::dot(spatialReservoir.radiance, hlsl::material_compiler3::backends::default_upt::LumaConversionCoeffs); // evalTargetPdf
    float weight = tpNewS * z;
    float avgWeight = hlsl::mix(0.f, wSumS / weight, weight > 0.f);
    spatialReservoir.M = hlsl::clamp(spatialReservoir.M, uint16_t(0u), uint16_t(100u));
    spatialReservoir.weightF = hlsl::clamp(avgWeight, 0.f, 10.f);
    spatialReservoir.age++;

    setReservoirs(currentReservoirsPtr, linearIdx, 1, spatialReservoir);

    float32_t3 finalDir = hlsl::normalize(spatialReservoir.sPosition - spatialReservoir.vPosition);
    spectral_t finalLi = spatialReservoir.radiance * hlsl::max(0.f, spatialReservoir.weightF);

    // TODO ReSTIR: check material roughness
    const quotient_weight_type final_quo = quotient_weight_type::create(0.f, 0.f);
    {
        isotropic_interaction_t interaction = isotropic_interaction_t::create(finalDir, rcData.preRcNormal, throughput);
        typename light_sample_t::ray_dir_info_type L;
        L.direction = rcData.preRcVertexL;
        light_sample_t _sample = light_sample_t::create(L, rcData.preRcNormal);

        // TODO set up material properly
        using brdf_t = reflection::SOrenNayar<bxdf_config_t>;
        brdf_t::SCreationParams cParams;
        cParams.A = 0.f;
        const brdf_t diffuse = brdf_t::create(cParams);
        typename brdf_t::isocache_type cache;
        // TODO: cache
        final_quo = diffuse.quotientAndWeight(_sample, interaction, cache);
    }

    spectral_t color = (rcData.pathPreRadiance + rcData.pathPreThroughput * final_quo * finalLi) / params.numGIInstance;
    rwmc::CascadeAccumulator<CCascades> colorAcc = rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting, true);
    colorAcc.addSample(_static_cast<uint16_t>(0u), accum_t(color));

    gBeauty[launchID] = float32_t4(color, 1.0);
}
