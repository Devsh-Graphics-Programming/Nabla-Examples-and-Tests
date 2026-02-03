#include "app_resources/common.hlsl"

using namespace nbl;
using namespace hlsl;

[[vk::push_constant]] PushConstantData pc;

struct device_capabilities
{
#ifdef TEST_NATIVE
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = true;
#else
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = false;
#endif
};

#ifndef OPERATION
#error "Define OPERATION!"
#endif

#ifndef NUM_LOOPS
#error "Define NUM_LOOPS!"
#endif

// NOTE added dummy output image to be able to profile with Nsight, which still doesn't support profiling headless compute shaders
[[vk::binding(2, 0)]] RWTexture2D<float32_t4> outImage; // dummy
