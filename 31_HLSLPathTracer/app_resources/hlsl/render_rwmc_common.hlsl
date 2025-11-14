#ifndef _NBL_HLSL_PATHTRACER_RENDER_RWMC_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_RWMC_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "render_common.hlsl"

#ifndef __HLSL_VERSION
#include "matrix4SIMD.h"
#endif

struct RenderRWMCPushConstants
{
	RenderPushConstants renderPushConstants;
	nbl::hlsl::rwmc::SplattingParameters splattingParameters;
};

#endif
