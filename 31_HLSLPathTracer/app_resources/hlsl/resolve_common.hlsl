#ifndef _NBL_HLSL_PATHTRACER_RESOLVE_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RESOLVE_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct ResolvePushConstants
{
	uint32_t sampleCount;
	float base;
	float minReliableLuma;
	float kappa;
};

#endif
