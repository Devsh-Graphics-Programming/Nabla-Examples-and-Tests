#include "nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl"

#include "common.hlsl"


struct CCascades
{
    using layer_type = float16_t3;
    using sample_count_type = uint16_t;
    using weight_t = float16_t;
    
    inline uint16_t getLastCascade() {return gSensor.lastCascadeIndex;}

    inline void clear()
    {
        for (uint16_t i=0u; i<=getLastCascade(); ++i)
            gRWMCCascades[__getCoord(i)] = uint32_t2(0,0);
    }

    inline void addSampleIntoCascadeEntry(const layer_type _sample, const uint16_t lowerCascadeIndex, const weight_t lowerCascadeLevelWeight, const weight_t higherCascadeLevelWeight, const sample_count_type sampleCount)
    {
        const weight_t reciprocalSampleCount = weight_t(1) / weight_t(sampleCount);
        uint16_t3 coord = __getCoord(lowerCascadeIndex);
        __splatToLayer(coord,_sample*lowerCascadeLevelWeight,sampleCount,reciprocalSampleCount);
        if (higherCascadeLevelWeight>weight_t(0))
        {
            coord.z++;
            __splatToLayer(coord,_sample*higherCascadeLevelWeight,sampleCount,reciprocalSampleCount);
        }
    }

    inline uint16_t3 __getCoord(const uint16_t cascadeIx)
    {
        uint16_t3 coord = _static_cast<uint16_t3>(spirv::LaunchIdKHR);
        coord.z = coord.z*uint16_t(6)+cascadeIx;
        return coord;
    }

    inline void __splatToLayer(const uint16_t3 coord, const layer_type weightedSample, const sample_count_type sampleCount, const weight_t reciprocalSampleCount)
    {
        uint16_t4 data = uint16_t4(0,0,0,0);
        if (sampleCount>1)
            data = bit_cast<uint16_t4>(gRWMCCascades[coord]);
        layer_type value = bit_cast<layer_type>(data.xyz);
        const sample_count_type oldSampleCount = data.w;
        value += (weightedSample - value*weight_t(sampleCount - oldSampleCount)) * reciprocalSampleCount;
        data = uint16_t4(bit_cast<uint16_t3>(value),sampleCount);
        gRWMCCascades[coord] = bit_cast<uint32_t2>(data);
    }
};

namespace nbl
{
namespace hlsl
{
namespace spirv
{
// https://github.com/microsoft/DirectXShaderCompiler/issues/6958
//[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
//[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
using HitObjectEXT = vk::SpirvOpaqueType<spv::OpTypeHitObjectEXT>;

template<typename T NBL_FUNC_REQUIRES(is_same_v<T,uint32_t>)
[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpReorderThreadWithHintEXT)]]
void reorderThreadWithHintEXT(T hint, T bits);

template<typename T NBL_FUNC_REQUIRES(is_same_v<T,uint32_t>)
[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpReorderThreadWithHitObjectEXT)]]
void reorderThreadWithHitObjectEXT([[vk::ext_reference]] HitObjectEXT hitObject, T hint, T bits);

template <typename PayloadT>
[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectTraceRayEXT)]]
void hitObjectTraceRayEXT([[vk::ext_reference]] HitObjectEXT hitObject, AccelerationStructureKHR AS, uint32_t rayFlags, uint32_t cullMask, uint32_t sbtOffset, uint32_t sbtStride, uint32_t missIndex, float32_t3 rayOrigin, float32_t rayTmin, float32_t3 rayDirection, float32_t rayTmax, [[vk::ext_reference]] PayloadT payload);

template <typename PayloadT>
[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectTraceRayEXT)]]
void hitObjectTraceRayEXT([[vk::ext_reference]] HitObjectEXT hitObject, RaytracingAccelerationStructure AS, uint32_t rayFlags, uint32_t cullMask, uint32_t sbtOffset, uint32_t sbtStride, uint32_t missIndex, float32_t3 rayOrigin, float32_t rayTmin, float32_t3 rayDirection, float32_t rayTmax, [[vk::ext_reference]] PayloadT payload);

// TODO:
// OpHitObjectTraceRayMotionEXT
// OpHitObjectRecordFromQueryEXT
// OpHitObjectRecordMissEXT
// OpHitObjectRecordMissMotionEXT 

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectRecordEmptyEXT)]]
void hitObjectRecordEmptyEXT([[vk::ext_reference]] HitObjectEXT hitObject);
 
// TODO:
// OpHitObjectExecuteShaderEXT
// OpHitObjectGetCurrentTimeEXT

template<typename HitAttributeT>
[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetAttributesEXT)]]
void hitObjectGetAttributesEXT([[vk::ext_reference]] HitObjectEXT hitObject, [[vk::ext_reference]] HitAttributeT attribs);

// OpHitObjectGetHitKindEXT

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetPrimitiveIndexEXT)]]
uint32_t hitObjectGetPrimitiveIndexEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetGeometryIndexEXT)]]
uint32_t hitObjectGetGeometryIndexEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetInstanceIdEXT)]]
uint32_t hitObjectGetInstanceIdEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetInstanceCustomIndexEXT)]]
uint32_t hitObjectGetInstanceCustomIndexEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetObjectRayOriginEXT)]]
float32_t3 hitObjectGetObjectRayOriginEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetObjectRayDirectionEXT)]]
float32_t3 hitObjectGetObjectRayDirectionEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetWorldRayDirectionEXT)]]
float32_t3 hitObjectGetWorldRayDirectionEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetWorldRayOriginEXT)]]
float32_t3 hitObjectGetWorldRayOriginEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetObjectToWorldEXT)]]
float32_t4x3 hitObjectGetObjectToWorldEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetWorldToObjectEXT)]]
float32_t4x3 hitObjectGetWorldToObjectEXT([[vk::ext_reference]] HitObjectEXT hitObject);

// the syntax to declare a function returning an Array in HLSL is insane
[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetIntersectionTriangleVertexPositionsEXT)]]
[[vk::ext_capability(spv::CapabilityRayQueryPositionFetchKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing_position_fetch")]]
float32_t3 hitObjectGetIntersectionTriangleVertexPositionsEXT([[vk::ext_reference]] HitObjectEXT hitObject)[3];

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetRayTMaxEXT)]]
float32_t hitObjectGetRayTMaxEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectGetRayTMinEXT)]]
float32_t hitObjectGetRayTMinEXT([[vk::ext_reference]] HitObjectEXT hitObject);

// TODO: OpHitObjectGetRayFlagsEXT
// TODO: OpHitObjectGetShaderBindingTableRecordIndexEXT
// TODO: OpHitObjectSetShaderBindingTableRecordIndexEXT
// TODO: OpHitObjectGetShaderRecordBufferHandleEXT

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectIsEmptyEXT)]]
bool hitObjectIsEmptyEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectIsHitEXT)]]
bool hitObjectIsHitEXT([[vk::ext_reference]] HitObjectEXT hitObject);

[[vk::ext_capability(spv::CapabilityShaderInvocationReorderEXT)]]
[[vk::ext_extension("SPV_EXT_shader_invocation_reorder")]]
[[vk::ext_instruction(spv::OpHitObjectIsMissEXT)]]
bool hitObjectIsMissEXT([[vk::ext_reference]] HitObjectEXT hitObject);
}
}
}


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
    inline void init(const float32_t _xi, float32_t tMax = hlsl::numeric_limits<float32_t>::max)
    {
        xi = _xi;
        rayT = tMax;
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

// TODO: move back to `common.hlsl`
inline float32_t3 reconstructGeometricNormal(NBL_REF_ARG(spirv::HitObjectEXT) hitObject)
{
    using namespace nbl::hlsl;

    const float32_t3 vertices[3] = spirv::hitObjectGetIntersectionTriangleVertexPositionsEXT(hitObject);

    // Do diffs in high precision, edges can be very long and dot products can easily overflow 64k max float16_t value and normalizing one extra time makes no sense
    const float32_t3 geometricNormal = hlsl::cross(vertices[1]-vertices[0],vertices[2]-vertices[0]);

    // Scales can be absolutely huge, we'd need special per-instance pre-scaled 3x3 matrices and also guarantee `geometricNormal` isn't huge
    // this would require a normalization before the matrix multiplication, making everything slower
    const float32_t4x3 w2o = spirv::hitObjectGetWorldToObjectEXT(hitObject);
    const float32_t3x3 normalMatrix = hlsl::math::linalg::truncate<3,3,3,4>(hlsl::transpose(w2o));
    // normalization also needs to be done in full floats because length squared can easily be over 64k
    return hlsl::normalize(hlsl::mul(normalMatrix,geometricNormal));
}


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
    static inline SClosestHitRetval create(NBL_REF_ARG(spirv::HitObjectEXT) hitObject)
    {
        SClosestHitRetval retval;
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
        retval.hitPos = math::linalg::promoted_mul(spirv::ObjectToWorldKHR,modelSpacePos);
    #else
        // the way that raytracers have done this before SPV_KHR_ray_tracing_position_fetch
        retval.hitPos = spirv::hitObjectGetWorldRayOriginEXT(hitObject) + spirv::hitObjectGetWorldRayDirectionEXT(hitObject) * spirv::hitObjectGetRayTMaxEXT(hitObject);
    #endif
    #undef POSITION_RECON_METHOD
        // TODO: Check this actually works
        {
            [[vk::ext_storage_class(spv::StorageClassHitObjectAttributeEXT)]] float32_t2 tmp;
            spirv::hitObjectGetAttributesEXT(hitObject,tmp);
            retval.barycentrics = tmp;
        }
        retval.instancedGeometryID = spirv::hitObjectGetInstanceCustomIndexEXT(hitObject) + spirv::hitObjectGetGeometryIndexEXT(hitObject);
        retval.primitiveID = spirv::hitObjectGetPrimitiveIndexEXT(hitObject);
        retval.geometricNormal = reconstructGeometricNormal(hitObject);
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

enum E_SBT_OFFSETS : uint16_t
{
    ESBTO_PATH,
    ESBTO_NEE
};


[[vk::push_constant]] SBeautyPushConstants pc;

// TODO: do a function with MIS to do envmap lighting


[shader("raygeneration")]
void raygen()
{
    const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);
    const SBeautyPushConstants::S16BitData unpacked16BitPC = pc.get16BitData();
    
    // Take n samples per frame
    // TODO: establish min/max - adaptive sampling
    SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID,unpacked16BitPC.maxSppPerDispatch,uint16_t(pc.sensorDynamics.keepAccumulating),_static_cast<uint16_t>(pc.sensorDynamics.maxSPP));
    // took max samples
    const uint16_t endSample = samplingInfo.newSampleCount;
    const uint16_t samplesThisFrame = endSample-samplingInfo.firstSample;
    if (samplesThisFrame==0)
        return;

    // TODO: possible SER point if doing variable spp
    //spirv::reorderThreadWithHintEXT<uint32_t>(hlsl::min<uint32_t>(samplesThisFrame,1),1);

    // weight for non RWMC contribution
    const float16_t newSamplesOverTotal = _static_cast<float16_t>(_static_cast<float32_t>(samplesThisFrame)*samplingInfo.rcpNewSampleCount);
    const float16_t rcpSamplesThisFrame = float16_t(1)/_static_cast<float16_t>(samplesThisFrame);

    float16_t transparency = 0.f;
    SArbitraryOutputValues aovs;
    aovs.clear();
    // some weird DXC and SPIR-V Tools Bug, lets try to move stuff out to temporaries and only use those
    decltype(samplingInfo.randgen) randgen = samplingInfo.randgen;
    const bool keepAccumulating = samplingInfo.firstSample;
    [[loop]] for (uint16_t sampleIndex=samplingInfo.firstSample; sampleIndex!=endSample; )
    {
        // For RWMC to work, every sample must be splatted individually
        spectral_t color;

        using namespace nbl::hlsl::bxdf;
        using namespace nbl::hlsl::material_compiler3::backends::default_upt;
        using bxdf_config_t = BxDFConfig;
        using ray_dir_info_t = bxdf_config_t::ray_dir_info_type;
        using isotropic_interaction_t = bxdf_config_t::isotropic_interaction_type;
        using light_sample_t = bxdf_config_t::sample_type;
        using quotient_pdf_type = bxdf_config_t::quotient_pdf_type;
        // a little bit of persistent state
        spirv::HitObjectEXT hitObject;
        {
            // fetch random variable from memory
            const float32_t3 randVec = randgen(0u,sampleIndex);
            // TODO: motion blur and lens DOF triplet
            
            // get our NDC coordinates and ray
            const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f)/float32_t2(spirv::LaunchSizeKHR.xy);
            const float32_t2 NDC = float32_t2(launchID.xy)*pixelSizeNDC - promote<float32_t2>(1.f);
            const SPrimaryRay primary = genPrimaryRay(pc.sensorDynamics,pixelSizeNDC,NDC,float16_t2(randVec.xy));
            const SRay ray = primary.ray;

            // TODO: possible SER point, sorting by ray direction
            //spirv::reorderThreadWithHintEXT<uint32_t>(,);

            [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval payload;
            const float tMax = pc.sensorDynamics.tMax;
            payload.init(randVec.z,tMax);
            spirv::hitObjectTraceRayEXT(hitObject,gTLASes[0],spv::RayFlagsMaskNone,0xff,ESBTO_PATH,0u,0u,ray.origin,primary.tMin,ray.direction.getDirection(),tMax,payload);
            // TODO: do something with the payload's reported transparency
        }
        // TODO: Possible SER point
        if (spirv::hitObjectIsMissEXT(hitObject))
        {
            const SEnvSample _sample = sampleEnv(spirv::hitObjectGetWorldRayDirectionEXT(hitObject));
            color = _sample.color;
            aovs = aovs + _sample.aov * rcpSamplesThisFrame;
            transparency += rcpSamplesThisFrame;
        }
        else // trace further rays
        {
            //
            MaxContributionEstimator contribEstimator = MaxContributionEstimator::create(unpacked16BitPC.rrThroughputWeights);
            const uint16_t lastPathDepth = gSensor.lastPathDepth;
            const uint16_t lastNoRussianRouletteDepth = gSensor.lastNoRussianRouletteDepth;
            //
            color = spectral_t(0,0,0);
            spectral_t throughput = spectral_t(1,1,1);
            float32_t otherTechniqueHeuristic = 0.f;
            SAOVThroughputs aovThroughput;
            aovThroughput.clear(rcpSamplesThisFrame);
            [[loop]] for (uint16_t depth=1; true; depth++) // ideally peel this loop once
            {
                // TODO: get the material ID and UVs

                // TODO: preserve spread metrics
                ray_dir_info_t V;
                V.setDirection(spirv::hitObjectGetWorldRayDirectionEXT(hitObject));

                SClosestHitRetval closestInfo = SClosestHitRetval::create(hitObject);
                const float32_t GdotV = hlsl::dot(V.getDirection(),closestInfo.geometricNormal);
                // TODO: only for twosided materials
                closestInfo.geometricNormal *= sign(GdotV);

                float32_t3 shadingNormal = closestInfo.geometricNormal;

                // TODO: possible SER point based on NEE status, and material flags

                // TODO: get AoVs from material and emission
                SAOVThroughputs nextThroughput;
                nextThroughput.clear(0.f);
                SArbitraryOutputValues aovContrib;
                aovContrib.albedo = float16_t3(1,1,1);
                aovContrib.normal = float16_t3(shadingNormal);
                // obtain full next
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
                    color += emission*throughput;
                }

                // to keep path depths equal for NEE and BxDF sampling, we can't continue and do NEE
                if (depth==lastPathDepth)
                    break;
                
                // get next random number, compensate for the triplets ray generation used
                const uint16_t sequenceProtoDim = (depth-uint16_t(1))*RandDimTriplesPerDepth+PrimaryRayRandTripletsUsed;
                float32_t3 randVec = randgen(sequenceProtoDim,sampleIndex);

                // TODO: start at 0 or numeric_limits::min?
                const float32_t tMin = 0.f;
                // should the offset be the same for NEE and Path Continuation?
                const float32_t3 originMagnitude = max(abs(closestInfo.hitPos),abs(spirv::hitObjectGetWorldRayOriginEXT(hitObject)));
                // TODO: should probably also take `tMax` of found hit into account
                const float offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f),originMagnitude.x),hlsl::max(originMagnitude.y,originMagnitude.z))*hlsl::exp2(-20.f);
                const float32_t3 newRayOrigin = closestInfo.hitPos+closestInfo.geometricNormal*offsetMagnitude;

                // perform NEE
                const float32_t neeProb = 1.f;
                if (false) // whether to perform NEE at all for this material
                {
                    float32_t3 randNEE = randgen(sequenceProtoDim+uint16_t(1),sampleIndex);
                    // choose regular lights or envmap

                    // TODO: SER point, top bits are NEE kind (none, regular light, envmap, then use bits of NEE random number and current position)

                    // perform the NEE sampling
                    float32_t3 L;//light_sample_t L;
                    float32_t pdf;
                    float32_t tMax;
                    {
                        const float32_t cosTheta = hlsl::mix(1.0f, sunConeHalfAngleCos, randNEE.x);
                        const float32_t cosTheta2 = cosTheta * cosTheta;
                        const float32_t sinTheta = hlsl::sqrt(1.0 - cosTheta2);

                        L = sunDir * cosTheta;

                        float32_t phi = 2.0 * numbers::pi<float32_t> *randNEE.y;
                        float32_t3 X, Y;
                        math::frisvad<float32_t3>(sunDir, X, Y);

                        L += (X * hlsl::cos(phi) + Y * hlsl::sin(phi)) * sinTheta;

                        pdf = 1.0 / (2.0 * numbers::pi<float32_t> * (1.0 - sunConeHalfAngleCos));
                        tMax = hlsl::numeric_limits<float16_t>::max;
                    }

                    // compute BxDF eval value, another layer of culling
#if 0
                    // trace shadow rays only for contributing samples
                    {
                       // TODO: another possible SER point before casting shadow rays
                        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval payload;
                        payload.init(tMax);
                        // TODO: change to ESBTO_NEE when ready
                        spirv::traceRayKHR(gTLASes[0],spv::RayFlagsTerminateOnFirstHitKHRMask|spv::RayFlagsSkipClosestHitShaderKHRMask,0xff,ESBTO_PATH,0u,ESBTO_PATH,newRayOrigin,tMin,L,tMax,payload);

                        if (false)
                        {
                            // apply everything
                        }
                    }
#endif
                }
                
                // TODO: perform shading
                light_sample_t bxdfSample;
                {
                    // TODO: embed a bit in the material stream whether:
                    // 1. anisotropic interaction is needed
                    // 2. whether luma contribution hint is needed
                    isotropic_interaction_t interaction = isotropic_interaction_t::create(V,shadingNormal,throughput);

                    // TODO: SER point, top bits are material Flags and ID geting executed

                    using brdf_t = reflection::SOrenNayar<bxdf_config_t>;
                    brdf_t::SCreationParams cParams;
                    cParams.A = 0.f;
                    const brdf_t diffuse = brdf_t::create(cParams);

                    //
                    bxdfSample = diffuse.generate(interaction,randVec.xy);
                    // Do I need to check `_sample.isValid()` myself before calling `forwardWeight`?
                    const quotient_pdf_type qAp = diffuse.quotient_and_pdf(bxdfSample,interaction);
                    const float forwardWeight = qAp.pdf;
                    if (forwardWeight<0.00000001f)
                        break;

                    const float32_t3 albedo = float32_t3(0.8,0.7,0.5);
                    throughput = throughput * qAp.quotient * albedo;

                    // TODO: include neeProb here
                    otherTechniqueHeuristic = 1.f/forwardWeight;
                }

                // to keep path depths equal for NEE and BxDF sampling, we 
                if (contribEstimator.notCulled(throughput,depth<=lastNoRussianRouletteDepth,randVec.z))
                {
                    // continue the path
                    {
                        const float32_t3 L = bxdfSample.getL().getDirection();
                        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval payload;
                        payload.init(randVec.z);
                        spirv::hitObjectTraceRayEXT(hitObject,gTLASes[0],spv::RayFlagsMaskNone,0xff,ESBTO_PATH,0u,0u,newRayOrigin,tMin,L,hlsl::numeric_limits<float32_t>::max,payload);
                        // TODO: do something with the payload's reported transparency
                        if (spirv::hitObjectIsMissEXT(hitObject)) // TODO: factor out into an inlineable function
                        {
                            SEnvSample _sample = sampleEnv(spirv::hitObjectGetWorldRayDirectionEXT(hitObject));
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
        // can't use pc.keepAccumulating because of variable sampling we want to do later, so just have first sample clear the RWMC
        const bool doClear = (sampleIndex++)==0;
        // color output, don't precompute `rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting)` and keep it as live state, it will spill anyway
        rwmc::CascadeAccumulator<CCascades> colorAcc = rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting,doClear);
        colorAcc.addSample(sampleIndex,accum_t(color));
    }
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
void closestHit(inout SAnyHitRetval payload, in BuiltInTriangleIntersectionAttributes attribs)
{
}

// TODO: Anyhit transparency
