#include "common.hlsl"

#include "nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl"

struct CCascades
{
    using layer_type = float16_t3;
    using sample_count_type = uint16_t;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t CascadeCount = 3; // TODO: refactor

    void clear(const uint32_t cascadeIx)
    {
        // NOOP and shouldn't get used
    }

};


[[vk::push_constant]] SBeautyPushConstants pc;

// There's actually a huge problem with doing any throughput or accumulation modification in AnyHit shaders, they run out of order (BVH order) and a hit behind your eventual closest hit can invoke the anyhit stage.
// 
// Most examples which multiply alpha in anyhit are super misleading, because:
// - for shadow / anyhit rays you either eventually hit an opaque (leading to a mul/replacement of transparency by 0) or you hit all opaques along the ray
// - for NEE rays you often have a finite tMax and this stops you accumulating translucency behind the emitter
// - multiplicative operations are order independent, so accumulating the visibility function can happen out of order (basis of many OIT techniques) as long as you know tMax of the closest hit
// - stochastic transparency cancels out the alpha weighting on the throughput, so there's no multiplication to perform, the throughput stays constant no matter what you do.
//   Which means it doesn't matter if you perform the test for occluded transparent geometries, you will never know, the alpha on the opaque also cancels out (shouldn't use premultiplied to shade).
// 
// However the minute you want to do stochastic RGB translucency the pdf no longer cancels out the RGB weight coefficients. While the application of `opacity/luma(opacity)` from a hit accepted as the closest,
// can be delayed until the closest hit if found, you'll start accumulating the wrong visibility from all the ignored hits. You literally have to use stochastic monochrome transparency.
// 
// Furthermore the minute you wish to add emission to the accumulation in the payload you run into Order Dependent Transparency because it requires a blend over operator.
// 
// The solutions are then as follows:
// 1. Only use Anyhit to employ stochastic transparency when the translucency weight is monochrome
// 2. Re-trace rays, find closest hit as with (1), then launch anyhit rays with known tMax - this only gets you correct RGB translucency
// 3. Use OIT techniques (A-Buffer, MLAB, WBOIT) to estimate the visibility function but without re-tracing need a robust technique which can handle "opaque transparents"
//    RGB translucency can be accumulated without sorting an A-Buffer in a O(1) pass over all intersections, also self-balancing tree and MLAB can throw out entries beyond current tMin.
//    Note that within a TLAS instances are likely to be traversed approximately in-order, and within a BLAS the primitives are too (see CWBVH8 paper with children visit order depending on ray direction signs).
//    Therefore a two tier linked list + insertion sort are a viable alternative to a self-balancing tree. To allow for emittance to be contributed by anyhit stage, it would need to be deferred to be performant,
//    the hit attributes would need to be stored alongside the translucency, so at least instance ID (possibly material ID or SBT offset), primitive ID, and the barycentrics. 
// 4. Decompose the Complex Mixture Material into a Scalar Delta Transmission plus the rest of the BxDF. The motivation is simple, for monochrome materials we have
//         DeltaTransmission*(1-alpha) + alpha*(Rest of BxDF Nodes with their Weights)
//    Where the thing getting factored is a blackbox sum of contributors, but we can reformulate any BxDF as
//         DeltaTransmission*Factor + (Rest of BxDF Nodes with their Weights)
//    Then we can simply break down the transmissive part into a monochrome part and a coloured residual, if we're unwilling to get into negative weights only option is `Transparency = min_element(Factor[0],...)`
//         DeltaTransmission * Transparency + (DeltaTransmission * (Factor-Transparency) + Rest of BxDF Nodes with their Weights)
//    We can still use stochastic transparency! Its just that whenever we accept a hit, we need pass `transparency` at the point of acceptance to the closest hit shader as to compute this
//         (DeltaTransmission * (Factor-Transparency) + Rest of BxDF Nodes with their Weights)/(1-Transparency)
//    Since Transparency can be just an approximation of the `Factor` in a monochrome form (luma) or its minimum, already computed or fetched data could be passed in payload for accepted hit
// 
// MOST IMPORTANT THING: AFTER ANYHIT ACCEPTS, ANOTHER MAY ACCEPT THATS CLOSER!
// This is very important to keep in mind when we do our Solid Angle Sampling.
// 
// Anyhit needs to pass the transparency probability to any closest hit it accepts and which then becomes the final anyhit
struct[raypayload] SAnyHitRetval
{
    // before sending the ray by the caller
    inline void init()
    {
        rayT = hlsl::numeric_limits<float32_t>::max;
    }
    // call in AnyHit instead of AcceptHit
    inline void acceptHit(const float16_t _transparency)
    {
        // need to read the spec if an anyhit is possible that the last anyhit to run and accept a hit candidate for a ray is not the last one to 
        if (rayT>spirv::RayTmaxKHR)
        {
            rayT = spirv::RayTmaxKHR;
            transparency = _transparency;
        }
        // TODO: call accept Hit intrinsic
    }
    // 
    
    // opacity russian roulette requires this for Discrete Probability Sampling
    float32_t xi : read(anyhit) : write(caller,anyhit);
    // need to store the t value at which the anyhit was executed, so we know whether the current closest hit comes from a confirmed anyhit
    float32_t rayT : read(caller,anyhit) : write(caller,anyhit);
    // essentially the probability of transmission
    float16_t transparency : read(caller) : write(anyhit);
    // can use additional `float16` to store BxDF mixture weights or other things so they don't need recomputing/re-fetching during shading
};



// Because SER based on Material ID will probably greatly benefit us, the shading needs to happen in Raygen Shader or ClosestHit executed directly by Raygen
// Lets examine what happens in the 3 options of Shading with SER Hit Objects:
// 1. Fused hitObjectTraceReorderExecuteEXT -> shading in Closest Hit
//      Miss and Closest hit still called immediately, Shading happens in both of them, only need payload to store anyhit + random number state (depth and optionally the seed)
//      but `SClosestHitRetval` gets passed to a shading function. Use NO_NULL_MISS_SHADERS definitely, and NO_NULL_CLOSEST_HIT_SHADERS if there's no blackhole materials.
// 2. hitObjectTraceRayEXT && Shading in Closest Hit with hitObjectExecuteShaderEXT
//      Only Anyhit payload needed, separate `SClosestHitRetval` payload is made in raygen and passed to the hitObjectExecuteShaderEXT, miss shader is not used.
//      Can use NO_NULL_CLOSEST_HIT and NO_NULL_MISS_SHADERS and then never invoke an invalid ClosestHit
// 3. hitObjectTraceRayEXT && Shading in Raygen
//      Only Anyhit payload needed, separate `SClosestHitRetval` is made and passed to traceRay, no closest hit shaders at all.
//      Should use NO_NULL_CLOSEST_HIT and NO_NULL_MISS_SHADERS 
struct SClosestHitRetval
{
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


// This payload will eventually not be needed with SER, and only the one below will be used
struct [raypayload] BeautyPayload
{
    inline void markAsMissed()
    {
        closestRet.geometricNormal = 45.f;
    }
    inline bool hasMissed() {return closestRet.geometricNormal[0]>1.f;}

    SClosestHitRetval closestRet : read(caller) : write(caller,closesthit);
};


enum E_SBT_OFFSETS : uint16_t
{
    ESBTO_PATH,
    ESBTO_NEE
};

// TODO: do a function with MIS to do envmap lighting

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
    const float16_t rcpSamplesThisFrame = float16_t(1)/float16_t(samplesThisFrame);

//    printf("%f %f %f",samplingInfo.rcpNewSampleCount,samplingInfo.newSampleCount,samplingInfo.firstSample);

    float16_t transparency = 0.f;
    SArbitraryOutputValues aovs;
    aovs.clear();
    [[loop]] for (uint16_t sampleIndex=samplingInfo.firstSample; sampleIndex!=samplingInfo.newSampleCount; sampleIndex++)
    {
        // For RWMC to work, every sample must be splatted individually
        accum_t color;

        const uint32_t PrimaryRayRandTripletsUsed = 2;
        // trace primary ray
        float32_t3 rayOrigin,rayDir;
        //
        bool missed;
        SClosestHitRetval closestInfo;
        {
            // fetch random variable from memory
            const float32_t3 randVec = samplingInfo.randgen(0u,sampleIndex);
            // TODO: motion blur and lens DOF triplet
            
            // get our NDC coordinates and ray
            const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f)/float32_t2(spirv::LaunchSizeKHR.xy);
            const float32_t2 NDC = float32_t2(launchID.xy)*pixelSizeNDC - promote<float32_t2>(1.f);
            const SRay ray = SRay::create(pc.sensorDynamics,pixelSizeNDC,NDC,float16_t2(randVec.xy));
            // TODO: possible SER point if doing variable spp

            // TODO: when doing anyhit opacity pass `randVec.z` into the payload
            [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] BeautyPayload payload;
            payload.markAsMissed();
            spirv::traceRayKHR(gTLASes[0],spv::RayFlagsMaskNone,0xff,ESBTO_PATH,0u,ESBTO_PATH,ray.origin,ray.tMin,ray.direction,ray.tMax,payload);

            //
            missed = payload.hasMissed();
            if (missed)
            {
                const SEnvSample _sample = sampleEnv(ray.direction);
                color = _sample.color;
                aovs = aovs + _sample.aov * rcpSamplesThisFrame;
                transparency += rcpSamplesThisFrame;
            }
            else // TODO: erase the `missed` variable and setup the struct in "wasAHit"
            {
                closestInfo = payload.closestRet;
                rayOrigin = ray.origin;
                rayDir = ray.direction;
            }
        }
        // trace further rays
        if (!missed)
        {
            //
            MaxContributionEstimator contribEstimator = MaxContributionEstimator::create(unpacked16BitPC.rrThroughputWeights);
            const uint16_t lastPathDepth = gSensor.lastPathDepth;
            //
            color = accum_t(0,0,0);
            spectral_t throughput = spectral_t(1,1,1);
            float32_t otherTechniqueHeuristic = 0.f;
            SAOVThroughputs aovThroughput;
            aovThroughput.clear(rcpSamplesThisFrame);
            // [0].xyz for BRDF Lobe sampling, then reuse [0].z for Russian Roulette, [1].xyz for BTDF Lobe sampling and [1].z for RIS lobe resampling, [2].xyz for NEE
            const uint16_t RandDimTriplesPerDepth = 3;
            [[loop]] for (uint16_t depth=1; true; depth++) // ideally peel this loop once
            {
                // TODO: get the material ID and UVs

                float32_t3 shadingNormal = closestInfo.geometricNormal;

                // TODO: possible SER point based on NEE status, and material flags

                // TODO: get AoVs from material and emission
                SAOVThroughputs nextThroughput;
                nextThroughput.clear(0.f);
                SArbitraryOutputValues aovContrib;
                aovContrib.albedo = float16_t3(1,1,1);
                aovContrib.normal = float16_t3(shadingNormal);
                // obtain full next
                printf("%d depth %f %f %f %f\n",aovThroughput.albedo[0],aovThroughput.albedo[1],aovThroughput.albedo[2],aovThroughput.transparency);
                nextThroughput = aovThroughput * nextThroughput;
                // already premultiplied by next throughput complement
                aovs = aovs + aovContrib * (aovThroughput - nextThroughput);
                aovThroughput = nextThroughput;
                
                // TODO: handle emission and do NEE MIS for any emission found on current hit
                if (false)
                {
                    // get emission stream
                    float16_t3 emission = float16_t3(0,0,0);
                    // compute emission
                    const float32_t WeightThreshold = hlsl::numeric_limits<float32_t>::min;
                    if (otherTechniqueHeuristic>WeightThreshold)
                    {
                        // compute NEE MIS backward weight on the contribution color
                        // assert not inf
                        // apply emissive weight
                    }
                    // add emissive to the contribution
                    color += emission*float16_t3(throughput);
                }

                // to keep path depths equal for NEE and BxDF sampling, we can't continue and do NEE
                if (depth==lastPathDepth)
                    break;
                
                // get next random number, compensate for the triplets ray generation used
                const uint16_t sequenceProtoDim = (depth-1)*RandDimTriplesPerDepth+PrimaryRayRandTripletsUsed;
                float32_t3 randVec = samplingInfo.randgen(sequenceProtoDim,sampleIndex);

                // perform NEE
                float32_t neeProb = 0.f;
                if (neeProb)
                {
                    if (true) // whether to perform NEE at all for this material
                    {
                        // choose regular lights or envmap

                        // TODO: SER point, top bits are NEE kind (none, regular light, envmap, then use bits of NEE random number and current position)

                        // perform the NEE sampling

                        // compute BxDF eval value, another layer of culling

                        // trace shadow rays only for contributing samples

                        // TODO: another possible SER point before casting shadow rays
                    }
                }
                
                // TODO: perform shading
                {
                    // TODO: SER point, top bits are material Flags and ID geting executed

                    const float pdf = 1.f / 3.14159f;
                    // consume additional 3 dimensions BTDF sampling and resampling
                    rayDir = shadingNormal;
                    color /= pdf;
                    throughput = throughput / pdf;
                    //
                    otherTechniqueHeuristic = 1.f/pdf;
                }

                // to keep path depths equal for NEE and BxDF sampling, we 
                if (contribEstimator.notCulled(throughput,depth<=gSensor.lastNoRussianRouletteDepth,randVec.z))
                {
                    // advance ray origin
                    rayOrigin = closestInfo.hitPos;

                    // continue the path
                    {
                        // TODO: when doing anyhit opacity pass `randVec.z` into the payload
                        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] BeautyPayload payload;
                        payload.markAsMissed();
                        // TODO: start at 0 or numeric_limits::min?
                        const float32_t tMin = 0.f;
                        spirv::traceRayKHR(gTLASes[0],spv::RayFlagsMaskNone,0xff,ESBTO_PATH,0u,ESBTO_PATH,rayOrigin,tMin,rayDir,hlsl::numeric_limits<float16_t>::max,payload);
                        if (payload.hasMissed())
                        {
                            SEnvSample _sample = sampleEnv(rayDir);
                            if (otherTechniqueHeuristic>0.f)
                            {
                                // compute NEE MIS backward weight
                                // assert not inf
                                // apply MIS to adjust _sample.color
                            }
                            color += _sample.color*throughput;
                            aovs = aovs + _sample.aov*aovThroughput;
                            transparency += aovThroughput.transparency;
                            break;
                        }
                    }
                }
            }
        }
        // color output
//        rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting).addSample(accumulation.color,samplingInfo.newSampleCount);
    }
    const bool keepAccumulating = samplingInfo.firstSample;
    // albedo
    Accumulator<ImageAccessor_gAlbedo> albedoAcc;
    albedoAcc.accumulate(launchID.xy,launchID.z,aovs.albedo,newSamplesOverTotal,keepAccumulating);
    // normal
    Accumulator<ImageAccessor_gNormal> normalAcc;
    normalAcc.accumulate(launchID.xy,launchID.z,correctSNorm10WhenStoringToUnorm(hlsl::normalize(aovs.normal)),newSamplesOverTotal,keepAccumulating);
    // TODO: motion
    // mask (TODO: do a separate pipeline for this with removed transparency calculations)
    if (gSensor.hideEnvironment)
    {
        Accumulator<ImageAccessor_gMask> maskAcc;
        vector<float16_t,1> opacity = float16_t(1)-transparency;
        maskAcc.accumulate(launchID.xy,launchID.z,opacity,newSamplesOverTotal,keepAccumulating);
    }
}


[shader("closesthit")]
void closestHit(inout BeautyPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    SClosestHitRetval closestHitReturn;
    // Which method of barycentric interpolation is more precise? Pick your poison!
#define POSITION_RECON_METHOD 0
#if POSITION_RECON_METHOD!=0
    // compute worldspace hit position
    const float32_t3 vertices[3] = spirv::HitTriangleVertexPositionsKHR;
#if POSITION_RECON_METHOD!=2
    // This way at least we stay within the triangle, and compiler can do CSE with the geometric normal calculation
    const float32_t3 modelSpacePos = vertices[0] + (vertices[1]-vertices[0]) * attribs.barycentrics[0] + (vertices[2] - vertices[0]) * attribs.barycentrics[1];
#else
    // This way we get less catastrophic cancellation by adding and computing the edges, but can end up outside the triangle
    const float32_t modelSpacePos = vertices[0] * (1.f-attribs.barycentrics.u-attribs.barycentrics.v) + vertices[1] * attribs.barycentrics.u + vertices[2] * attribs.barycentrics.v;
#endif
    closestHitReturn.hitPos = math::linalg::promoted_mul(spirv::ObjectToWorldKHR,modelSpacePos);
#else
    // the way that raytracers have done this before SPV_KHR_ray_tracing_position_fetch
    closestHitReturn.hitPos = spirv::WorldRayOriginKHR + spirv::WorldRayDirectionKHR * spirv::RayTmaxKHR;
#endif
#undef POSITION_RECON_METHOD
    closestHitReturn.barycentrics = attribs.barycentrics;
    closestHitReturn.instancedGeometryID = spirv::InstanceCustomIndexKHR + spirv::RayGeometryIndexKHR;
    closestHitReturn.primitiveID = spirv::PrimitiveId;
    closestHitReturn.geometricNormal = reconstructGeometricNormal();
    payload.closestRet = closestHitReturn;
}

// TODO: Anyhit transparency
