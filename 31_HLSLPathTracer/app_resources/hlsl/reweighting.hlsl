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

// this function is for testing purpose
// simply adds every cascade buffer, output shoud be nearly the same as output of default accumulator (RWMC off)
void sumCascade(in const int32_t2 coords)
{
    float32_t3 accumulation = float32_t3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 6; ++i)
    {
        float32_t4 cascadeLevel = cascade.Load(uint3(coords, i));
        accumulation += float32_t3(cascadeLevel.r, cascadeLevel.g, cascadeLevel.b);
    }

    accumulation /= 32.0f;

    float32_t4 output = float32_t4(accumulation, 1.0f);

    outImage[coords] = output;
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    // TODO: remove, ideally shader should not be called at all when we don't use RWMC
    bool useRWMC = true;
    if (!useRWMC)
        return;

    const int32_t2 coords = getCoordinates();
    sumCascade(coords);

    // zero out cascade
    for (int i = 0; i < 6; ++i)
        cascade[uint3(coords.x, coords.y, i)] = float32_t4(0.0f, 0.0f, 0.0f, 0.0f);
}
