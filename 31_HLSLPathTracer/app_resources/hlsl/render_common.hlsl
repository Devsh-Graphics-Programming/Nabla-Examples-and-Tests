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
    // we set using a transpose matrix
    void setLightMatrix(const float32_t4x3 matT)
    {
		lightX = matT[0];
		lightY = matT[1];
		// Z had a length and a direction, can point colinear or opposite cross product
		const float32_t3 recon = hlsl::cross(matT[0],matT[1]);
		lightZscale = hlsl::sign(hlsl::dot(recon,matT[2]))*hlsl::length(matT[2])/hlsl::length(recon);
		lightPos = matT[3];
		assert(lightMatrix()==hlsl::transpose(matT));
	}

    float32_t3x4 getLightMatrix()
    {
        float32_t4x3 retval;
        retval[0] = lightX;
        retval[1] = lightY;
        retval[2] = cross(lightX, lightY) * lightZscale;
        retval[3] = lightPos;
        return hlsl::transpose(retval);
    }

    uint64_t pSampleSequence;
	float32_t4x4 invMVP;
    float32_t3 lightX;
    float32_t3 lightY;
    float32_t lightZscale;
    float32_t3 lightPos;
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
