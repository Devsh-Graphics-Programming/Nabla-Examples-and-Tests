#include "common.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"


using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;

[[vk::push_constant]] SBeautyPushConstants pc;


struct SSpectralType
{
    inline void clear()
    {
        transparency = float16_t(1);
        aovThroughput = float16_t3(1,1,1);
        color = normal = albedo = float16_t3(0,0,0);
        // TODO: motion
    }

    float16_t3 color;
    float16_t3 aovThroughput;
    float16_t3 albedo;
    float16_t3 normal;
    float16_t transparency;
    // TODO: motion
};

struct [raypayload] BeautyPayload
{
/*
    inline void init(const float16_t sampleWeight)
    {
        throughput = promote<float16_t3>(sampleWeight);
        accumulation.clear();
        otherTechniqueHeuristic = 0.f;
    }

    // TODO: investigate reading from scramble texture instead
    scramble_state_t dimensionScramble : read(caller,anyhit,closesthit,miss) : write(caller);
    // all shading gets done in raygen (raygen does shading via callables)
    // anyhit can mix in its own stuff during thindielectric and opaque trace
    SSpectralType accumulation : read(caller,anyhit) : write(caller,anyhit);
    // russian roulette in opacity tracing requires us to read and write
    float16_t3 throughput : read(caller,anyhit) : write(caller,anyhit);
    // world space normal
    float16_t3 normalAtOrigin : read(caller,anyhit) : write(caller,closesthit,anyhit);
    // TODO up for debate whether to make this a full float
    // Raygen or callable sets it before shooting another ray, gets read when ray comes back with an intersection
    // MIS needs to be applied on transparent emitters, so anyhit reads
    float32_t otherTechniqueHeuristic : read(caller,anyhit) : write(caller);
*/
    // TODO: options for killing specular after diffuse paths
    uint16_t depth : MAX_DEPTH_LOG2;
};

[shader("raygeneration")]
void raygen()
{
    const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);
    
    SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID,1,uint16_t(pc.sensorDynamics.keepAccumulating));
    // took 64k-1 spp
    if (samplingInfo.rcpNewSampleCount==0.f)
        return;

    SSpectralType accumulation;
    accumulation.clear();

    // TODO

//    Accumulator<gRWMCCascades> beautyAcc;
//    beautyAcc.accumulate(launchID.xy,launchID.z,float32_t3(accumulation.color),samplingInfo.rcpNewSampleCount);
    const float16_t rcpNewSampleCountF16 = float16_t(samplingInfo.rcpNewSampleCount);
    // albedo
    Accumulator<ImageAccessor_gAlbedo> albedoAcc;
    albedoAcc.accumulate(launchID.xy,launchID.z,accumulation.albedo,rcpNewSampleCountF16);
    // normal
    Accumulator<ImageAccessor_gNormal> normalAcc;
//    normalAcc.accumulate(launchID.xy,launchID.z,correctSNorm10WhenStoringToUnorm(accumulation.worldNormal),samplingInfo.rcpNewSampleCount);
    // TODO: motion
    // mask
    Accumulator<ImageAccessor_gMask> maskAcc;
    maskAcc.accumulate(launchID.xy,launchID.z,vector<float16_t,1>(accumulation.transparency),rcpNewSampleCountF16);
}

[shader("closesthit")]
void closesthit(inout BeautyPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("miss")]
void miss(inout BeautyPayload payload)
{
}