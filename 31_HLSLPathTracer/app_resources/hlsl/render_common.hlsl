#ifndef _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_

struct SPushConstants
{
    float32_t4x4 invMVP;
    int sampleCount;
    int depth;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::combinedImageSampler]][[vk::binding(0, 2)]] Texture2D<float3> envMap;      // unused
[[vk::combinedImageSampler]][[vk::binding(0, 2)]] SamplerState envSampler;

[[vk::binding(1, 2)]] Buffer<uint3> sampleSequence;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]] Texture2D<uint2> scramblebuf; // unused
[[vk::combinedImageSampler]][[vk::binding(2, 2)]] SamplerState scrambleSampler;

[[vk::image_format("rgba16f")]][[vk::binding(0, 0)]] RWTexture2D<float32_t4> outImage;
[[vk::image_format("rgba16f")]][[vk::binding(1, 0)]] RWTexture2DArray<float32_t4> cascade;

#endif
