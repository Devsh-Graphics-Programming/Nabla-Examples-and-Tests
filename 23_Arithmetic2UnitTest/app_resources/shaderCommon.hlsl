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
