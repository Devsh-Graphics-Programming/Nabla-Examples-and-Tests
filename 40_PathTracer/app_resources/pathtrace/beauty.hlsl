#include "common.hlsl"

[[vk::push_constant]] SBeautyPushConstants pc;


// There's actually a huge problem with doing any throughput or accumulation modification in AnyHit shaders, they run out of order (BVH order)
// 


// Write down some thoughts about random number consumption
// - Raygen needs a triple for jitter and anyhit opacity (further triple for DoF and time/motion blur)
// - Primary Ray doesn't do backward MIS, accepts full found (part of shading) emission
//      If shading in closest hit, need to have depth (16bit) and randgen state (64bit) in payload to generate further random numbers
// - Secondary Rays do backward MIS on found emission, then:
//      1. Fetch material
//      2. generate further ray direction, modify throughput

// NEE needs to happen before throughput is multiplied with continuation BxDF ray quotient

// Because SER based on Material ID will probably greatly benefit us, the shading needs to happen in Raygen Shader.
struct SClosestHitRetval
{
    // N.B. The following 2 can only use 24bits, we could stuff other things in top 8 bit of MSB
    // TODO: shall we abuse some geometry+instance bits to determine if geometry has emission? Alternatively a clever bitfield in `uint8_t` or `uint16_t` BDA or the SBT ?
    // to get our material and geometry data back
    uint32_t instancedGeometryID;
    // to get particular Triangle's indices
    uint32_t primitiveID;
    // to interpolate our vertex attributes
    float32_t2 barycentrics;
    //
    float32_t3 hitPos;
};

//
struct[raypayload] PrimaryBeautyPayload
{
    // reinitialize for a new path
    // accumulation needs to be assigned separately (persistent between same pixel paths)
    inline void reinitPath(const float32_t rand, const float32_t sampleWeight)
    {
        reinitPath(rand,0.f); // primary visibility can't have NEE (unless doing optical lens simulation)
        throughput.clear(sampleWeight);
    }

//common stuff
    SClosestHitRetval closestRet : read(caller) : write(closesthit);
    // opacity russian roulette requires this for Discrete Probability Sampling
    float32_t xi : read(caller,anyhit) : write(caller,anyhit);
    // Put throughput before accumulation because it may need high precision and range color throughput
    // transparent (anyhit) can perform Russian Roulette to accept or reject a hit
    SThroughputs throughput : read(caller,anyhit) : write(caller,anyhit);
    // Material evaluation and stochastic opacity in anyhit shader requires we this whole struct in the payload
    SSpectralType accumulation : read(caller,anyhit) : write(caller,anyhit);
    // Has different semantics depending on the stage of the path tracer:
    // - during secondary anyhit its the previous shading normal for NEE MIS application
    // - closest hit overwrites it to pass back the geometric normal for shading in raygen
    float16_t3 normal : read(caller,anyhit) : write(caller,closesthit);
    
    // make sure we don't keep on shading
    inline void killPath()
    {
        throughput.color = promote<float32_t3>(0.f);
    }
};
struct[raypayload] GenericBeautyPayload
{
#if 0
    //
    inline void reinitVertex(const float32_t rand, const float32_t _otherTechniqueHeuristic)
    {
        xi = rand;
        otherTechniqueHeuristic = _otherTechniqueHeuristic;
    }
#endif
    // Whether to apply MIS weight if we find a NEE registered emissive
    inline bool shouldDoNEEBackwardMIS() {return otherTechniqueHeuristic>0.f;}

    // Raygen or callable sets it before shooting another ray, gets read when ray comes back with an intersection
    // MIS needs to be applied on transparent emitters, so anyhit reads but doesn't update
    // TODO up for debate whether to make this a full float, but then color throughput can be 0
    float32_t otherTechniqueHeuristic : read(caller,anyhit) : write(caller);
// common payload;
    inline void killPath() {}
};

#if 0
struct[raypayload] BeautyPayload
{


    inline void accumulate(SSpectralType contribution)
    {
        accumulation = accumulation + contribution*throughput;
    }

    //
    SClosestHitRetval<bool(SHADE_IN_CLOSEST_HIT)> closestHitReturn : read(caller) : write(closesthit);
    // TODO: options for killing specular after diffuse paths
};

enum E_SBT_OFFSETS : uint16_t
{
    ESBTO_PRIMARY,
    ESBTO_SECONDARY,
    ESBTO_NEE
};

//

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR bool HandleEnvmapInMissShader = false;

[shader("raygeneration")]
void raygen()
{
    const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);

    const SBeautyPushConstants::S16BitData unpacked16BitPC = pc.get16BitData();

    // Take n samples per frame
    // TODO: establish min/max - adaptive sampling
    uint16_t samplesThisFrame = unpacked16BitPC.maxSppPerDispatch;
    SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID,samplesThisFrame,uint16_t(pc.sensorDynamics.keepAccumulating));
    // took 64k-1 spp
    if (samplingInfo.rcpNewSampleCount==0.f)
        return;
    // weight for non RWMC contribution
    const float16_t newSamplesOverTotal = float16_t(float32_t(samplesThisFrame)*samplingInfo.rcpNewSampleCount);

    // make the accumulation compute the average of current samples
    BeautyPayload payload;
    // we reuse between paths
    payload.accumulation.clear();
    //
    [[loop]] for (uint16_t sampleIndex=samplingInfo.firstSample; sampleIndex!=samplingInfo.newSampleCount; sampleIndex++)
    {
        // trace primary ray
        float32_t3 rayOrigin,rayDir;
        {
            // fetch random variable from memory
            const float32_t3 randVec = samplingInfo.randgen(0u,sampleIndex);
            
            // get our NDC coordinates and ray
            const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f)/float32_t2(spirv::LaunchSizeKHR.xy);
            const float32_t2 NDC = float32_t2(launchID.xy)*pixelSizeNDC - promote<float32_t2>(1.f);
            const SRay ray = SRay::create(pc.sensorDynamics,pixelSizeNDC,NDC,float16_t2(randVec.xy));
            // TODO: possible SER point if doing variable spp


            {
                PrimaryBeautyPayload payload = PrimaryBeautyPayload::create(randVec.z,samplesThisFrame)
                spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PRIMARY, 0u, ESBTO_PRIMARY, ray.origin, ray.tMin, ray.direction, ray.tMax, payload);
                rayOrigin = ray.origin;
                rayDir = ray.direction;
            }
        }
        // trace further rays
        MaxContributionEstimator contribEstimator = MaxContributionEstimator::create(unpacked16BitPC.rrThroughputWeights);
        [[loop]] for (uint16_t depth=1; depth!=gSensor.lastPathDepth; depth++)
        if (contribEstimator.notCulled(payload.throughput,depth<=gSensor.lastNoRussianRouletteDepth,payload.xi))
        {
            // TODO: possible SER point right after Russian Roulette

            // two triplets for RIS between BRDF and BTDF and importance sampling within, one for NEE
            const uint16_t randDimTriplesPerDepth = 3;
            // get next random number, compensate for the single triplet ray generation used
            float32_t3 randVec = samplingInfo.randgen((depth-1)*randDimTriplesPerDepth+1,sampleIndex);

            // cast NEE rays

            // advance ray origin
            ray.origin = ray.origin+ray.direction*payload.foundT;

            // TODO: importance sample next direction
    //        ray.direction = ;

            // TODO: start at 0 or min?
            const float32_t tMin = 0.f;
            spirv::traceRayKHR(gTLASes[0],spv::RayFlagsMaskNone,0xff,ESBTO_SECONDARY,0u,ESBTO_SECONDARY,rayOrigin,tMin,rayDir,numeric_limits<float16_t>::max,payload);

            // TODO: finish NEE shadow ray
            {
                // TODO: possible SER point
            }
        }
        if (contribEstimator.notCulled(payload.throughput,true))
        {
            if (payload.shouldDoNEEBackwardMIS())
            {
                // TODO: modulate the throughputs by the MIS weight
            }
            payload.accumulate(sampleEnv(rayDir));
        }
    }

    const SSpectralType accumulation = payload.accumulation;
    // color output
//    Accumulator<gRWMCCascades> beautyAcc;
//    beautyAcc.accumulate(launchID.xy,launchID.z,float32_t3(accumulation.color),samplingInfo.rcpNewSampleCount);
    // albedo
    Accumulator<ImageAccessor_gAlbedo> albedoAcc;
    albedoAcc.accumulate(launchID.xy,launchID.z,accumulation.albedo,newSamplesOverTotal);
    // normal
    Accumulator<ImageAccessor_gNormal> normalAcc;
    normalAcc.accumulate(launchID.xy,launchID.z,correctSNorm10WhenStoringToUnorm(accumulation.normal),newSamplesOverTotal);
    // TODO: motion
    // mask
    Accumulator<ImageAccessor_gMask> maskAcc;
    maskAcc.accumulate(launchID.xy,launchID.z,vector<float16_t,1>(accumulation.transparency),newSamplesOverTotal);
}
#else
[shader("raygeneration")]
void raygen()
{
}
#endif

#if 0
template<typename Payload>
inline void commonClosestHit(NBL_REF_ARG(Payload) payload, const BuiltInTriangleIntersectionAttributes attribs)
{
    payload.closestHitReturn.instancedGeometryID = spirv::InstanceCustomIndexKHR + spirv::RayGeometryIndexKHR;
    payload.closestHitReturn.primitiveID = spirv::PrimitiveId;
    payload.closestHitReturn.barycentrics = attribs.barycentrics;

    // compute worldspace hit position
    const float32_t3 vertices[3] = spirv::HitTriangleVertexPositionsKHR;
    // Which method of barycentric interpolation is more precise? Pick your poison!
    // This way at least we stay within the triangle, and compiler can do CSE with the geometric normal calculation
    const float32_t3 modelSpacePos = vertices[0] + (vertices[1]-vertices[0]) * attribs.barycentrics[0] + (vertices[2] - vertices[0]) * attribs.barycentrics[1];
    // This way we get less catastrophic cancellation by adding and computing the edges, but can end up outside the triangle
//    const float32_t modelSpacePos = vertices[0] * (1.f-attribs.barycentrics.u-attribs.barycentrics.v) + vertices[1] * attribs.barycentrics.u + vertices[2] * attribs.barycentrics.v;
    payload.closestHitReturn.hitPos = math::linalg::promoted_mul(spirv::ObjectToWorldKHR,modelSpacePos);

#if 0 // move to raygen
    // TODO: get the material ID and UVs

    // TODO: perform shading
    {
        const float pdf = 1.f / 3.14159f;
        // consume 5 dimensions for BRDF and BTDF sampling
        payload.sampledDir = geomNormal;
        payload.throughput = payload.throughput / pdf;
        //
//        payload.otherTechniqueHeuristic = 1.f/pdf;
    }
#endif

    payload.normal = reconstructGeometricNormal();
}
#endif

//
[shader("closesthit")]
void primaryClosestHit(inout PrimaryBeautyPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
//    commonClosestHit(payload,attribs);
}
[shader("closesthit")]
void closestHit(inout GenericBeautyPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
//    commonClosestHit(payload,attribs);

#if 0 // move to raygen
    // TODO: handle emission and do NEE MIS for any emission found on current hit
    {
        // get emission stream
        // compute emission
        if (payload.shouldDoNEEBackwardMIS())
        {
            // compute NEE MIS backward weight
            // assert not inf
            // apply emissive weight
        }
        // add emissive to the contribution
    }

    // to keep path depths equal for NEE and BxDF sampling, we had to trace a this ray, but we're not going to continue
    if (lastBounce)
        payload.killPath();
    else
        commonClosestHit(payload,attribs);
#endif
}

// TODO: Anyhit transparency
// - russian roulette accept hit, means we bump throughput by rcpPdf when we chose to continue
// - reuse Z random variable coordinate from current closest hit

// TODO: do a function with MIS to do envmap lighting

//
[shader("miss")]
void primaryMiss(inout PrimaryBeautyPayload payload)
{
    // stop tracing futher rays
//    payload.killPath();
}
[shader("miss")]
void miss(inout GenericBeautyPayload payload)
{
    // stop tracing futher rays
//    payload.killPath();
}