#ifndef _NBL_THIS_EXAMPLE_RWMC_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_RWMC_HLSL_INCLUDED_


#include "renderer/shaders/common.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl"

#include <boost/preprocessor/arithmetic/sub.hpp>

namespace nbl
{
namespace this_example
{
// We do it so weirdly because https://github.com/microsoft/DirectXShaderCompiler/issues/7131
#define MAX_CASCADE_COUNT_LOG2 3

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t MaxCascadeCount = uint16_t(1u<<MAX_CASCADE_COUNT_LOG2);
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t ResolveWorkgroupSizeX = 32u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t ResolveWorkgroupSizeY = 16u;

// no uint16_t to be used because its going to be a push constant
struct SResolveConstants
{
	hlsl::rwmc::SResolveParameters resolveParameters;
	uint32_t cascadeCount;
};

}
}
#endif  // _NBL_THIS_EXAMPLE_RWMC_HLSL_INCLUDED_
