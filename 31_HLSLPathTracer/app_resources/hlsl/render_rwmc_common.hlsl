#ifndef _PATHTRACER_EXAMPLE_RENDER_RWMC_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_RENDER_RWMC_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "rwmc_common.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "render_common.hlsl"

struct RenderRWMCPushConstants
{
	RenderPushConstants renderPushConstants;
	nbl::hlsl::rwmc::SPackedSplattingParameters splattingParameters;
};
#ifndef __HLSL_VERSION
static_assert(sizeof(RenderRWMCPushConstants)<=128,"Nabla Core Profile Guarantees only minimum of 128 bytes");
#endif

#endif
