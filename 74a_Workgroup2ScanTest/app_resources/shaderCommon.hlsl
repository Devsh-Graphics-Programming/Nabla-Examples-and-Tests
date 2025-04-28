#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;

// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<type_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];

// because subgroups don't match `gl_LocalInvocationIndex` snake curve addressing, we also can't load inputs that way
uint32_t globalIndex();
// since we test ITEMS_PER_WG<WorkgroupSize we need this so workgroups don't overwrite each other's outputs
bool canStore();

#ifndef OPERATION
#error "Define OPERATION!"
#endif
#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif
template<template<class> class binop>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    if (globalIndex()==0u)
        output[binop<type_t>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
        
    operation_t<typename binop<type_t>::base_t,nbl::hlsl::jit::device_capabilities> func;
    if (canStore())
        output[binop<type_t>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),func(sourceVal));
}


type_t test()
{
    const type_t sourceVal = inputValue[globalIndex()];

    subtest<bit_and>(sourceVal);
    subtest<bit_xor>(sourceVal);
    subtest<bit_or>(sourceVal);
    subtest<plus>(sourceVal);
    subtest<multiplies>(sourceVal);
    subtest<minimum>(sourceVal);
    subtest<maximum>(sourceVal);
    return sourceVal;
}

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"