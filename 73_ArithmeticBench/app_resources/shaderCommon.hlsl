#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<uint32_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];

// to get next item, move by subgroupSize
uint32_t globalFirstItemIndex(uint32_t itemIdx);
// since we test ITEMS_PER_WG<WorkgroupSize we need this so workgroups don't overwrite each other's outputs
bool canStore();

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif
//typedef decltype(inputValue[0]) type_t;
//typedef uint32_t type_t;
//typedef uint32_t4 type_t;

// #if ITEMS_PER_INVOCATION > 1
typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;
// #else
// typedef uint32_t type_t;
// #endif


#ifndef OPERATION
#error "Define OPERATION!"
#endif

#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif
template<template<class> class binop, typename T, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    // TODO static assert vector<T, N> == type_t
    //using type_t = vector<T, N>;
    using config_t = nbl::hlsl::subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;

    if (nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex()==0u)
        output[binop<T>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
        
    operation_t<params_t> func;
    type_t value = func(sourceVal);
    if (canStore())
    {
        [unroll]
        for (uint32_t i = 0; i < ITEMS_PER_INVOCATION; i++)
            output[binop<T>::BindingIndex].template Store<uint32_t>(sizeof(uint32_t) + sizeof(uint32_t) * (globalFirstItemIndex(i) + nbl::hlsl::glsl::gl_SubgroupInvocationID()), value[i]);
    }
}


type_t test()
{
    const uint32_t idx = nbl::hlsl::glsl::gl_SubgroupInvocationID();
    type_t sourceVal;
    [unroll]
    for (uint32_t i = 0; i < ITEMS_PER_INVOCATION; i++)
    {
        sourceVal[i] = inputValue[globalFirstItemIndex(i) + idx];
    }

    subtest<bit_and, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_xor, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_or, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<plus, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<multiplies, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<minimum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<maximum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    return sourceVal;
}

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
