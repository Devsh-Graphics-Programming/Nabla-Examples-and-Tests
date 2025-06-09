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
