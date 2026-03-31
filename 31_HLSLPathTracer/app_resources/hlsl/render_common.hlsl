#ifndef _PATHTRACER_EXAMPLE_RENDER_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_RENDER_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

using namespace nbl;
using namespace hlsl;

#define MAX_DEPTH_LOG2 4
#define MAX_SAMPLES_LOG2 10
NBL_CONSTEXPR uint32_t MaxDepthLog2 = MAX_DEPTH_LOG2;
NBL_CONSTEXPR uint32_t MaxSamplesLog2 = MAX_SAMPLES_LOG2;

struct RenderPushConstants
{
    // TODO: cut down the MVP into a compact raygen matrix to get this whole struct to less than 112 bytes!
	float32_t4x4 invMVP;
	float32_t3x4 generalPurposeLightMatrix;
    uint64_t pSampleSequence;
    // TODO: compact a bit and refactor
    uint32_t sampleCount : MAX_SAMPLES_LOG2;
    uint32_t depth : MAX_DEPTH_LOG2;
    uint32_t sequenceSampleCountLog2 : 5;
    uint32_t unused : 13;
};
#undef MAX_SAMPLES_LOG2
#undef MAX_DEPTH_LOG2

NBL_CONSTEXPR float32_t3 LightEminence = float32_t3(30.0f, 25.0f, 15.0f);
NBL_CONSTEXPR uint32_t RenderWorkgroupSizeSqrt = 8u;
NBL_CONSTEXPR uint32_t RenderWorkgroupSize = RenderWorkgroupSizeSqrt*RenderWorkgroupSizeSqrt;
NBL_CONSTEXPR uint32_t MaxDescriptorCount = 256u;
NBL_CONSTEXPR uint16_t MaxUITextureCount = 1u;

#endif
