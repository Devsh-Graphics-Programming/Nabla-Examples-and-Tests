#ifndef _NBL_THIS_EXAMPLE_RWMC_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_RWMC_HLSL_INCLUDED_


#include "renderer/shaders/common.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl"

#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>

namespace nbl
{
namespace this_example
{
// We do it so weirdly because https://github.com/microsoft/DirectXShaderCompiler/issues/7131
#define MAX_CASCADE_COUNT_LOG2 3

// no uint16_t to be used because its going to be a push constant
struct SResolveConstants // TODO: move somewhere
{
	struct SProtoRWMC
	{
		hlsl::float32_t initialEmin;
		hlsl::float32_t reciprocalBase;
		hlsl::float32_t reciprocalKappa;
		hlsl::float32_t colorReliabilityFactor;
	} rwmc;
	uint32_t cascadeCount : BOOST_PP_ADD(MAX_CASCADE_COUNT_LOG2,1);
	uint32_t unused : BOOST_PP_SUB(31,MAX_CASCADE_COUNT_LOG2);
};

}
}
#endif  // _NBL_THIS_EXAMPLE_RWMC_HLSL_INCLUDED_
