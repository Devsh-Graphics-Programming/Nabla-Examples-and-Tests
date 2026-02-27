#ifndef _PATHTRACER_EXAMPLE_RENDER_RWMC_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_RENDER_RWMC_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "render_common.hlsl"

struct RenderRWMCPushConstants
{
	RenderPushConstants renderPushConstants;
	nbl::hlsl::rwmc::SPackedSplattingParameters splattingParameters;
};

#endif
