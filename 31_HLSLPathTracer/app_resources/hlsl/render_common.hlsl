#ifndef _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifndef __HLSL_VERSION
#include "matrix4SIMD.h"
#endif

struct RenderPushConstants
{
#ifdef __HLSL_VERSION
	float32_t4x4 invMVP;
	float32_t3x4 generalPurposeLightMatrix;
#else
	nbl::core::matrix4SIMD invMVP;
	nbl::core::matrix3x4SIMD generalPurposeLightMatrix;
#endif

	int sampleCount;
	int depth;
};

NBL_CONSTEXPR nbl::hlsl::float32_t3 LightEminence = nbl::hlsl::float32_t3(30.0f, 25.0f, 15.0f);

#endif
