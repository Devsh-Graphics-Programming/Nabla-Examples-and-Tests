#ifndef _PATHTRACER_EXAMPLE_RESOLVE_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_RESOLVE_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl"

struct ResolvePushConstants
{
	nbl::hlsl::rwmc::ResolveParameters resolveParameters;
};

NBL_CONSTEXPR uint32_t ResolveWorkgroupSizeX = 32u;
NBL_CONSTEXPR uint32_t ResolveWorkgroupSizeY = 16u;
NBL_CONSTEXPR uint32_t CascadeCount = 6u;

#endif
