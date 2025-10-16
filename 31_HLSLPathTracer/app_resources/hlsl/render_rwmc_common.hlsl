#ifndef _NBL_HLSL_PATHTRACER_RENDER_RWMC_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_RWMC_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifndef __HLSL_VERSION
#include "matrix4SIMD.h"
#endif

struct RenderRWMCPushConstants
{
#ifdef __HLSL_VERSION
	float32_t4x4 invMVP;
#else
	nbl::core::matrix4SIMD invMVP;
#endif
	int sampleCount;
	int depth;
	float start;
	float base;
	float kappa;
};

#endif
