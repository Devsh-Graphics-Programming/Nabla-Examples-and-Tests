#include "nbl/builtin/hlsl/cpp_compat.hlsl"

[[vk::image_format("rgba16f")]] [[vk::binding(0, 0)]] RWTexture2D<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(1, 0)]] RWTexture2DArray<float32_t4> cascade;

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

float calculateLumaRec709(float32_t4 color)
{
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    const int32_t2 coords = getCoordinates();

    float r = cascade.Load(uint3(coords, 0)).r;
    float g = cascade.Load(uint3(coords, 1)).g;
    float b = cascade.Load(uint3(coords, 2)).b;
    float32_t4 color = float32_t4(r, g, b, 1.0f);
    float luma = calculateLumaRec709(color);
    float32_t4 output = float32_t4(luma, luma, luma, 1.0f);

    outImage[coords] = output;
}
