#include "nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"

#include "common.hlsl"
#include "next_event_estimator.hlsl"

[[vk::push_constant]] SBeautyPushConstants pc;

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float32_t NormalCompareThreshold = 0.8f;

// Accumulation: every sample feeds BOTH outputs, a plain fp32 running mean written to the fp32
// Beauty image (gBeauty) AND the fp16 RWMC cascade splat (gRWMCCascades). Both buffers are always
// populated, so a single run yields the unbiased fp32 mean alongside the RWMC result with no build-
// time toggle. Caveat: the RWMC 16-bit per-cascade sample count wraps past 65535 spp, so for very-
// high-spp reference renders read gBeauty (fp32), the cascades are stale there.
struct CCascades
{
    using layer_type        = float16_t3;
    using sample_count_type = uint16_t;
    using weight_t          = float16_t;

    inline uint16_t getLastCascade() { return gSensor.lastCascadeIndex; }

    inline void clear()
    {
        for (uint16_t i = 0u; i <= getLastCascade(); ++i)
            gRWMCCascades[__getCoord(i)] = uint32_t2(0, 0);
    }

    inline void addSampleIntoCascadeEntry(const layer_type _sample, const uint16_t lowerCascadeIndex, const weight_t lowerCascadeLevelWeight, const weight_t higherCascadeLevelWeight, const sample_count_type sampleCount)
    {
        const weight_t reciprocalSampleCount = weight_t(1) / weight_t(sampleCount);
        uint16_t3      coord                 = __getCoord(lowerCascadeIndex);
        __splatToLayer(coord, _sample * lowerCascadeLevelWeight, sampleCount, reciprocalSampleCount);
        if (higherCascadeLevelWeight > weight_t(0))
        {
            coord.z++;
            __splatToLayer(coord, _sample * higherCascadeLevelWeight, sampleCount, reciprocalSampleCount);
        }
    }

    inline uint16_t3 __getCoord(const uint16_t cascadeIx)
    {
        uint16_t3 coord = _static_cast<uint16_t3>(spirv::LaunchIdKHR);
        coord.z         = coord.z * uint16_t(6) + cascadeIx;
        return coord;
    }

    inline void __splatToLayer(const uint16_t3 coord, const layer_type weightedSample, const sample_count_type sampleCount, const weight_t reciprocalSampleCount)
    {
        uint16_t4 data = uint16_t4(0, 0, 0, 0);
        if (sampleCount > 1)
            data = bit_cast<uint16_t4>(gRWMCCascades[coord]);
        layer_type              value          = bit_cast<layer_type>(data.xyz);
        const sample_count_type oldSampleCount = data.w;
#if NBL_RWMC_FP32_REWEIGHT
        float32_t3 v = float32_t3(value);
        v += (float32_t3(weightedSample) - v * float32_t(sampleCount - oldSampleCount)) / float32_t(sampleCount);
        value = layer_type(v);
#else
        value += (weightedSample - value * weight_t(sampleCount - oldSampleCount)) * reciprocalSampleCount;
#endif
        data                 = uint16_t4(bit_cast<uint16_t3>(value), sampleCount);
        gRWMCCascades[coord] = bit_cast<uint32_t2>(data);
    }
};

// TODO: duplicates beauty.hlsl and beauty_reservoir.hlsl, find some shared header to put
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

// TODO: duplicates beauty.hlsl and beauty_reservoir.hlsl, find some shared header to put
struct SClosestHitRetval
{
    static inline SClosestHitRetval create(NBL_REF_ARG(spirv::HitObjectEXT) hitObject)
    {
        SClosestHitRetval retval;
        {
            [[vk::ext_storage_class(spv::StorageClassHitObjectAttributeEXT)]] float32_t2 tmp;
            spirv::hitObjectGetAttributesEXT(hitObject, tmp);
            retval.barycentrics = tmp;
        }
        // Which method of barycentric interpolation is more precise? Pick your poison!
#define POSITION_RECON_METHOD 0
#if POSITION_RECON_METHOD != 0
        // compute worldspace hit position
        const float32_t3 vertices[3] = spirv::hitObjectGetIntersectionTriangleVertexPositionsEXT(hitObject);
#if POSITION_RECON_METHOD != 2
        // This way at least we stay within the triangle, and compiler can do CSE with the geometric normal calculation
        const float32_t3 modelSpacePos = vertices[0] + (vertices[1] - vertices[0]) * retval.barycentrics[0] + (vertices[2] - vertices[0]) * retval.barycentrics[1];
#else
        // This way we get less catastrophic cancellation by adding and computing the edges, but can end up outside the triangle
        const float32_t3 modelSpacePos = vertices[0] * (1.f - retval.barycentrics[0] - retval.barycentrics[1]) + vertices[1] * retval.barycentrics[0] + vertices[2] * retval.barycentrics[1];
#endif
        retval.hitPos = math::linalg::promoted_mul(hlsl::transpose(spirv::hitObjectGetObjectToWorldEXT(hitObject)), modelSpacePos);
#else
        // the way that raytracers have done this before SPV_KHR_ray_tracing_position_fetch
        retval.hitPos = spirv::hitObjectGetWorldRayOriginEXT(hitObject) + spirv::hitObjectGetWorldRayDirectionEXT(hitObject) * spirv::hitObjectGetRayTMaxEXT(hitObject);
#endif
#undef POSITION_RECON_METHOD
        retval.instancedGeometryID = spirv::hitObjectGetInstanceCustomIndexEXT(hitObject) + spirv::hitObjectGetGeometryIndexEXT(hitObject);
        retval.primitiveID         = spirv::hitObjectGetPrimitiveIndexEXT(hitObject);
        retval.geometricNormal     = reconstructGeometricNormal(hitObject);
        return retval;
    }

    float32_t3 hitPos;
    // to interpolate our vertex attributes
    float32_t2 barycentrics;
    // to get our material and geometry data back
    uint32_t instancedGeometryID;
    // to get particular Triangle's indices
    uint32_t primitiveID;
    //
    float32_t3 geometricNormal;
};

// TODO: duplicates beauty.hlsl and beauty_reservoir.hlsl, find some shared header to put
enum E_SBT_OFFSETS : uint16_t
{
    ESBTO_PATH,
    ESBTO_NEE
};

SReservoir getReservoirs(NBL_REF_ARG(LegacyBdaAccessor<SReservoir>) reservoirBuf, uint32_t baseIndex, uint32_t sampleIndex)
{
    const uint32_t framePixelCount = uint32_t(gSensor.renderSize.x) * uint32_t(gSensor.renderSize.y);
    const uint32_t index = baseIndex + sampleIndex * framePixelCount;
    SReservoir reservoir;
    if (index < framePixelCount * 2u)  // TODO: find out why I need this check, it's only crashing in nsight
        reservoirBuf.get(index, reservoir);
    return reservoir;
}

void setReservoirs(NBL_REF_ARG(LegacyBdaAccessor<SReservoir>) reservoirBuf, uint32_t baseIndex, uint32_t sampleIndex, NBL_REF_ARG(SReservoir) reservoir)
{
    const uint32_t framePixelCount = uint32_t(gSensor.renderSize.x) * uint32_t(gSensor.renderSize.y);
    const uint32_t index = baseIndex + sampleIndex * framePixelCount;
    if (index < framePixelCount * 2u)  // TODO: find out why I need this check, it's only crashing in nsight
        reservoirBuf.set(index, reservoir);
}

// Diagnostic-only NEE-proposal probe takeover
#include "nee_proposal_probe.hlsl"

// forwardNEE as a ray-tracing callable (see NBL_NEE_CALLABLE in next_event_estimator.hlsl).
[shader("callable")] 
void neeCallable(inout nbl::this_example::SNeeCallableData cd)
{
    using NEE = nbl::this_example::NextEventEstimator;

    NEE::ray_dir_info_t V;
    V.setDirection(cd.V);
    NEE::isotropic_interaction_t interaction = NEE::isotropic_interaction_t::create(V, cd.shadingNormal, cd.throughput);

    NEE::brdf_t::SCreationParams cParams;
    cParams.A                 = 0.f;
    const NEE::brdf_t diffuse = NEE::brdf_t::create(cParams);

    NEE nee                     = NEE::create();
    nee.prevDescentNeeEmitterID = cd.prevDescentNeeEmitterID;
    nee.prevDescentNeePdf       = cd.prevDescentNeePdf;

    const NEE::SForwardSample s = nee.forwardNEE(cd.hitPos, cd.shadingNormal, interaction, diffuse, cd.throughput, cd.randNEE, cd.randNEE2);

    cd.pickedDir               = s.pickedDir;
    cd.contribution            = s.contribution;
    cd.pickedEmitterID         = s.pickedEmitterID;
    cd.valid                   = s.valid ? 1u : 0u;
    cd.prevDescentNeeEmitterID = nee.prevDescentNeeEmitterID;
    cd.prevDescentNeePdf       = nee.prevDescentNeePdf;
}

[shader("callable")] 
void emissionCallable(inout nbl::this_example::SEmissionCallableData ec)
{
    using NEE             = nbl::this_example::NextEventEstimator;
    NEE nee               = NEE::create();
    nee.prevShadingHitPos = ec.prevShadingHitPos;
    nee.prevShadingNormal = ec.prevShadingNormal;
    // Negative sentinel = same-emitter cache miss: run the backward selection-pdf climb here in the
    // callable stage so its register / i-cache footprint stays out of raygen. >= 0 is the cached pdf.
    const float32_t backPdf = (ec.emitterSelectBackPdf < 0.f) ? nee.__emitterSelectBackPdf(ec.emitterIdx) : ec.emitterSelectBackPdf;
    ec.deweight             = (backPdf > 0.f) ? nee.__emissionDeweight(ec.emitterIdx, ec.currentHitPos, backPdf, ec.otherTechniqueHeuristic) : 1.f;
}

[shader("raygeneration")]
void raygen()
{
    const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);
    const SBeautyPushConstants::S16BitData unpacked16BitPC = pc.get16BitData();
    const uint32_t linearIdx = uint32_t(launchID.y) * uint32_t(gSensor.renderSize.x) + uint32_t(launchID.x);

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

    // don't advance sample?
    SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID, 0u, uint16_t(pc.sensorDynamics.keepAccumulating), pc.sensorDynamics.maxSPP);
    decltype(samplingInfo.randgen) randgen = samplingInfo.randgen;

    uint32_t sampleIndex = 0u;
    float32_t3 cameraPos;

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
        const float32_t3 randVec = randgen(0u, sampleIndex++);
        // TODO: motion blur and lens DOF triplet

        // get our NDC coordinates and ray
        const float32_t2  pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(spirv::LaunchSizeKHR.xy);
        const float32_t2  NDC          = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);
        const SPrimaryRay primary      = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, NDC, float16_t2(randVec.xy));
        const SRay        ray          = primary.ray;
        cameraPos = ray.origin;

        // TODO: possible SER point, sorting by ray direction
        //spirv::reorderThreadWithHintEXT<uint32_t>(,);

        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval payload;
        const float tMax = pc.sensorDynamics.tMax;
        payload.init(randVec.z, tMax);
        spirv::hitObjectTraceRayEXT(hitObject, gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PATH, 0u, 0u, ray.origin, primary.tMin, ray.direction.getDirection(), tMax, payload);
        // TODO: do something with the payload's reported transparency
    }
    // TODO: Possible SER point
    const bool primaryMissed = spirv::hitObjectIsMissEXT(hitObject);
    const float32_t3 primaryRayDir = spirv::hitObjectGetWorldRayDirectionEXT(hitObject);

    SClosestHitRetval closestInfo = SClosestHitRetval::create(hitObject);

    // get previous sample
    const float32_t4 previousClip = hlsl::math::linalg::promoted_mul(pc.sensorDynamics.prevViewProj, closestInfo.hitPos);
    const float32_t3 previousScreen = previousClip.xyz / previousClip.w;
    const float32_t2 previousUV = previousScreen.xy * float32_t2(0.5f, -0.5f) + 0.5f;
    const uint32_t2 previousID = uint32_t2(hlsl::clamp(previousUV * float32_t2(gSensor.renderSize), hlsl::promote<float32_t2>(0.f), float32_t2(gSensor.renderSize.x - 1u, gSensor.renderSize.y - 1u)));
    const uint32_t previousIdx = previousID.y * uint32_t(gSensor.renderSize.x) + previousID.x;

    bool isPreviousValid = gSampleCount[launchID] > 0 && hlsl::all(previousUV > hlsl::promote<float32_t2>(0.0)) && hlsl::all(previousUV < hlsl::promote<float32_t2>(1.0));
    LegacyBdaAccessor<SReservoir> previousReservoirsPtr = LegacyBdaAccessor<SReservoir>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::PreviousReservoirsBuf]);

    // TODO ReSTIR: TEMPORAL REUSE HERE
    setReservoirs(currentReservoirsPtr, linearIdx, 0, initialReservoir);

    // spatial reuse
    SReservoir spatialReservoir = getReservoirs(previousReservoirsPtr, previousIdx, 0);
    spatialReservoir.vPosition = rcData.preRcHitPosition;
    spatialReservoir.vNormal = rcData.preRcNormal;

    float32_t cellSize = calculateCellSize(initialReservoir.vPosition, cameraPos, gSensor.renderSize, gSensor.restirParams);
    float32_t3 jitteredPos = rcData.preRcHitPosition + (randgen(0u, sampleIndex++) * 2.0f - 1.0f) * 0.1f * cellSize;

    int cellIdx = findCell(jitteredPos, rcData.preRcNormal, cellSize, gSensor.restirParams, gSensor.pStorageBuffers[SensorUBOBufferAddresses::CheckSumBuf]);
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

    spatialReservoir.M = hlsl::clamp(spatialReservoir.M, uint16_t(0u), uint16_t(100u));
    if (spatialReservoir.age > 100)
    {
        spatialReservoir.M = 0;
    }

    uint32_t maxSpatialIteration = 3u;  // spatialReservoir.M > 10 ? 3u : 10u;

    uint32_t increment = (sampleCount + maxSpatialIteration - 1) / maxSpatialIteration;
    uint32_t offset = hlsl::round(randgen(0u, sampleIndex++).x * (increment - 1));

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

    uint32_t reuseID = 0u;
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < sampleCount; i += increment)
    {
        count++;

        uint32_t neighborPixelIndex;
        cellStoragePtr.get(cellBaseIdx + (offset + i) % sampleCount, neighborPixelIndex);
        SReservoir neighborReservoir = getReservoirs(previousReservoirsPtr, neighborPixelIndex, (count + 1u) % 2);

        if (neighborReservoir.M <= uint16_t(0u) || hlsl::dot(spatialReservoir.vNormal, neighborReservoir.vNormal) < NormalCompareThreshold)
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

        const float32_t3 randVis = randgen(0u, sampleIndex++); // need this? maybe any hit can be reduced

        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval visibilityPayload;
        visibilityPayload.init(randVis.z, hlsl::numeric_limits<float32_t>::max);
        spirv::HitObjectEXT visibilityHit;
        spirv::hitObjectTraceRayEXT(visibilityHit, gTLASes[0], 0u, 0xff, ESBTO_PATH, 0u, ESBTO_PATH, visRayOrigin, tMin, visRayDir, hlsl::numeric_limits<float32_t>::max, visibilityPayload);

        bool visRayMissed = spirv::hitObjectIsMissEXT(visibilityHit);
        if (visRayMissed)
            targetPdf = 0.f;
        bool updated = spatialReservoir.merge(neighborReservoir, randVis.y, targetPdf, wSumS);
        if (updated)
            reuseID = count;

        positionList[nReuse] = neighborReservoir.vPosition;
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

            const float32_t3 randVis = randgen(0u, sampleIndex++);

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
    quotient_weight_type final_quo = quotient_weight_type::create(0.f, 0.f);
    {
        typename light_sample_t::ray_dir_info_type V;
        V.direction = finalDir;
        isotropic_interaction_t interaction = isotropic_interaction_t::create(V, rcData.preRcNormal, hlsl::material_compiler3::backends::default_upt::LumaConversionCoeffs);
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

    spectral_t color = (rcData.pathPreRcRadiance + rcData.pathPreRcThroughput * final_quo.quotient() * finalLi);
    // spectral_t color = rcData.preRcNormal * hlsl::promote<spectral_t>(0.5) + hlsl::promote<spectral_t>(0.5);
    // spectral_t color = final_quo.quotient() * finalLi;
    rwmc::CascadeAccumulator<CCascades> colorAcc = rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting, true);
    colorAcc.addSample(_static_cast<uint16_t>(0u), accum_t(color));

    gBeauty[launchID] = float32_t4(color, 1.0);
}
