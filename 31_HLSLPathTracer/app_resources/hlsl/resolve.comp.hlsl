#include <nbl/builtin/hlsl/rwmc/resolve.hlsl>
#include "resolve_common.hlsl"
#include "rwmc_global_settings_common.hlsl"
#ifdef PERSISTENT_WORKGROUPS
#include "nbl/builtin/hlsl/math/morton.hlsl"
#endif

[[vk::push_constant]] ResolvePushConstants pc;
[[vk::image_format("rgba16f")]] [[vk::binding(0)]] RWTexture2DArray<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(1)]] RWTexture2DArray<float32_t4> cascade;

using namespace nbl;
using namespace hlsl;

int32_t2 getImageExtents()
{
    uint32_t width, height, imageArraySize;
    outImage.GetDimensions(width, height, imageArraySize);
    return int32_t2(width, height);
}

[numthreads(ResolveWorkgroupSizeX, ResolveWorkgroupSizeY, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    const int32_t2 coords = int32_t2(threadID.x, threadID.y);
    const int32_t2 imageExtents = getImageExtents();
    if (coords.x >= imageExtents.x || coords.y >= imageExtents.y)
        return;

    float32_t3 color = rwmc::reweight(pc.resolveParameters, cascade, coords);

    outImage[uint3(coords.x, coords.y, 0)] = float32_t4(color, 1.0f);
}
