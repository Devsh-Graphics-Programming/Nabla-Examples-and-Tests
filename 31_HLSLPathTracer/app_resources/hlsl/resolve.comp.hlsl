#include <nbl/builtin/hlsl/rwmc/rwmc.hlsl>
#include "resolve_common.hlsl"
#include "rwmc_global_settings_common.hlsl"
#ifdef PERSISTENT_WORKGROUPS
#include "nbl/builtin/hlsl/math/morton.hlsl"
#endif

[[vk::push_constant]] ResolvePushConstants pc;
[[vk::image_format("rgba16f")]] [[vk::binding(0, 0)]] RWTexture2D<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(0, 1)]] RWTexture2DArray<float32_t4> cascade;

using namespace nbl;
using namespace hlsl;

NBL_CONSTEXPR uint32_t WorkgroupSize = 512;
NBL_CONSTEXPR uint32_t MAX_DEPTH_LOG2 = 4;
NBL_CONSTEXPR uint32_t MAX_SAMPLES_LOG2 = 10;

int32_t2 getCoordinates()
{
    uint32_t width, height;
    outImage.GetDimensions(width, height);
    return int32_t2(glsl::gl_GlobalInvocationID().x % width, glsl::gl_GlobalInvocationID().x / width);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
#ifdef PERSISTENT_WORKGROUPS
    uint32_t virtualThreadIndex;
    [loop]
    for (uint32_t virtualThreadBase = glsl::gl_WorkGroupID().x * WorkgroupSize; virtualThreadBase < 1920 * 1080; virtualThreadBase += glsl::gl_NumWorkGroups().x * WorkgroupSize)
    {
        virtualThreadIndex = virtualThreadBase + glsl::gl_LocalInvocationIndex().x;
        const int32_t2 coords = (int32_t2)math::Morton<uint32_t>::decode2d(virtualThreadIndex);
#else
    const int32_t2 coords = getCoordinates();
#endif

    rwmc::ReweightingParameters reweightingParameters = rwmc::computeReweightingParameters(pc.base, pc.sampleCount, pc.minReliableLuma, pc.kappa, CascadeSize);
    float32_t3 color = rwmc::reweight(reweightingParameters, cascade, coords);

    outImage[coords] = float32_t4(color, 1.0f);

#ifdef PERSISTENT_WORKGROUPS
    }
#endif
}
