#include "common.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

using namespace nbl;
using namespace hlsl;

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

[[vk::push_constant]] PushConstantData pc;

#ifndef OPERATION
#error "Define OPERATION!"
#endif

#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif
