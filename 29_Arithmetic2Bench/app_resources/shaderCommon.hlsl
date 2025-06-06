#include "common.hlsl"

using namespace nbl;
using namespace hlsl;

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

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

#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif

#ifndef NUM_LOOPS
#error "Define NUM_LOOPS!"
#endif

// NOTE added dummy output image to be able to profile with Nsight, which still doesn't support profiling headless compute shaders
[[vk::binding(2, 0)]] RWTexture2D<float32_t4> outImage; // dummy
