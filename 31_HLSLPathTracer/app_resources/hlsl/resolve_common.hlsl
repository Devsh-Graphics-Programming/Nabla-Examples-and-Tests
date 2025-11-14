#ifndef _NBL_HLSL_PATHTRACER_RESOLVE_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RESOLVE_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl"

struct ResolvePushConstants
{
	uint32_t sampleCount;
	nbl::hlsl::rwmc::ResolveParameters resolveParameters;
};

NBL_CONSTEXPR uint32_t ResolveWorkgroupSizeX = 32u;
NBL_CONSTEXPR uint32_t ResolveWorkgroupSizeY = 16u;

#endif
