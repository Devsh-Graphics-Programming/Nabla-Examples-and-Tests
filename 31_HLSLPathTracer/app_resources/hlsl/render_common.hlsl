#ifndef _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_

struct SPushConstants
{
    float32_t4x4 invMVP;
    int sampleCount;
    int depth;

    uint64_t spheresAddress;
    uint64_t trianglesAddress;
    uint64_t rectanglesAddress;
    uint64_t lightsAddress;
    uint64_t bxdfsAddress;

    uint32_t sphereCount;
    uint32_t triangleCount;
    uint32_t rectangleCount;
    uint32_t lightCount;
    uint32_t bxdfCount;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::combinedImageSampler]][[vk::binding(0, 2)]] Texture2D<float3> envMap;      // unused
[[vk::combinedImageSampler]][[vk::binding(0, 2)]] SamplerState envSampler;

[[vk::binding(1, 2)]] Buffer<uint3> sampleSequence;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]] Texture2D<uint2> scramblebuf; // unused
[[vk::combinedImageSampler]][[vk::binding(2, 2)]] SamplerState scrambleSampler;

[[vk::image_format("rgba16f")]][[vk::binding(0, 0)]] RWTexture2D<float32_t4> outImage;

#endif
