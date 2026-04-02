#ifndef _NBL_THIS_EXAMPLE_PATHTRACE_PUSH_CONSTANTS_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATHTRACE_PUSH_CONSTANTS_HLSL_INCLUDED_


#include "renderer/shaders/session.hlsl"

#include <boost/preprocessor/arithmetic/mul.hpp>


// no uint16_t to be used because its going to be a push constant
namespace nbl
{
namespace this_example
{
	
#define MAX_SPP_LOG2 15
NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxSPPLog2 = MAX_SPP_LOG2;
// need to be able to count (represent) both 0 and Max
NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxSPP = (0x1u << MaxSPPLog2) - 1;
struct SSensorDynamics
{
	// assuming input will be ndc = [-1,1]^2 x {-1}
	hlsl::float32_t3x4 invView;
	hlsl::float32_t2x3 ndcToRay;
	hlsl::float32_t nearClip;
	hlsl::float32_t tMax;
	// we can adaptively sample per-pixel, but some bounds need to be kept
	uint32_t minSPP : MAX_SPP_LOG2;
	uint32_t maxSPP : MAX_SPP_LOG2;
	uint32_t unused : 1;
	uint32_t keepAccumulating : 1;
};
#undef MAX_SPP_LOG2
	
struct SPrevisPushConstants
{
	SSensorDynamics sensorDynamics;
};

// We do it so weirdly because https://github.com/microsoft/DirectXShaderCompiler/issues/7131
#define MAX_SPP_PER_DISPATCH_LOG2 5
struct SBeautyPushConstants
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxSppPerDispatchLog2 = MAX_SPP_PER_DISPATCH_LOG2;

	// PushConstant16bit access feature isn't ubiquitous
	struct S16BitData
	{
		// Luma conversion coefficients scaled by something proportional to the brightest light in the scene
#ifndef __HLSL_VERSION
		hlsl::
#endif
		float16_t3 rrThroughputWeights;
		// For a foveated render
		uint16_t maxSppPerDispatch;
	};


	SSensorDynamics sensorDynamics;
#ifdef __HLSL_VERSION
	uint32_t __16BitData[sizeof(S16BitData)/sizeof(uint32_t)];
	// 
	S16BitData get16BitData()
	{
		S16BitData retval;
		// TODO: implement later
		retval.rrThroughputWeights = hlsl::promote<float16_t3>(hlsl::numeric_limits<float16_t>::max); // always pass RR
		retval.maxSppPerDispatch = 3;
		return retval;
	}
#else
	S16BitData __16BitData;
#endif
};
#undef MAX_SPP_PER_DISPATCH_LOG2

struct SDebugPushConstants
{
	SSensorDynamics sensorDynamics;
	// some enum/choice of what to debug
};

}
}
#endif  // _NBL_THIS_EXAMPLE_PATHTRACE_PUSH_CONSTANTS_HLSL_INCLUDED_
