#ifndef _PATHTRACER_EXAMPLE_RENDER_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_RENDER_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

using namespace nbl;
using namespace hlsl;

struct RenderPushConstants
{
	float32_t4x4 invMVP;
	float32_t3x4 generalPurposeLightMatrix;
    int sampleCount;
    int depth;
    uint64_t pSampleSequence;
};

NBL_CONSTEXPR float32_t3 LightEminence = float32_t3(30.0f, 25.0f, 15.0f);
NBL_CONSTEXPR uint32_t RenderWorkgroupSize = 64u;
NBL_CONSTEXPR uint32_t MAX_DEPTH_LOG2 = 4u;
NBL_CONSTEXPR uint32_t MAX_SAMPLES_LOG2 = 10u;
NBL_CONSTEXPR uint32_t MaxBufferDimensions = 3u << MAX_DEPTH_LOG2;
NBL_CONSTEXPR uint32_t MaxBufferSamples = 1u << MAX_SAMPLES_LOG2;

#endif
