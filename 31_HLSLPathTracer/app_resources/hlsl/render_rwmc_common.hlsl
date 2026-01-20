#ifndef _NBL_HLSL_PATHTRACER_RENDER_RWMC_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_RWMC_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "render_common.hlsl"

struct RenderRWMCPushConstants
{
	RenderPushConstants renderPushConstants;
	int32_t packedSplattingParams;
};

#endif
