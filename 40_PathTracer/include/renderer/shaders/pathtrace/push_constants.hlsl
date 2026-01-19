#ifndef _NBL_THIS_EXAMPLE_PATHTRACE_PUSH_CONSTANTS_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATHTRACE_PUSH_CONSTANTS_HLSL_INCLUDED_


#include "renderer/shaders/session.hlsl"

#include <boost/preprocessor/arithmetic/mul.hpp>


// no uint16_t to be used because its going to be a push constant
namespace nbl
{
namespace this_example
{
struct SSensorDynamics
{
	// assuming input will be ndc = [-1,1]^2 x {-1}
	hlsl::float32_t3x4 ndcToRay;
	hlsl::float32_t tMax;
	// we can adaptively sample per-pixel, but 
	uint32_t minSPP : MAX_SPP_LOG2;
	uint32_t maxSPP : MAX_SPP_LOG2;
	uint32_t unused : BOOST_PP_SUB(32,BOOST_PP_MUL(MAX_SPP_LOG2,2));
};
	
struct SPrevisPushConstants : SSensorDynamics
{
};

// We do it so weirdly because https://github.com/microsoft/DirectXShaderCompiler/issues/7131
#define MAX_SPP_PER_DISPATCH_LOG2 5
struct SBeautyPushConstants : SSensorDynamics
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxSppPerDispatchLog2 = MAX_SPP_PER_DISPATCH_LOG2;

	uint32_t maxSppPerDispatch : MAX_SPP_PER_DISPATCH_LOG2;
	uint32_t unused : 27;
};
#undef MAX_SPP_PER_DISPATCH_LOG2

struct SDebugPushConstants : SSensorDynamics
{
};

}
}
#endif  // _NBL_THIS_EXAMPLE_PATHTRACE_PUSH_CONSTANTS_HLSL_INCLUDED_
