#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;

struct PushConstantData
{
    uint64_t inputBufAddress;
    uint64_t outputAddressBufAddress;
};

[[vk::push_constant]] PushConstantData pc;

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
template<template<class> class binop, typename T, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    // TODO static assert vector<T, N> == type_t
    //using type_t = vector<T, N>;
    using config_t = nbl::hlsl::subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;

    const uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + binop<T>::BindingIndex * sizeof(uint64_t), sizeof(uint64_t));

    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, nbl::hlsl::glsl::gl_SubgroupSize());

    operation_t<params_t> func;
    type_t val = func(sourceVal);
    if (canStore())
        vk::RawBufferStore<type_t>(outputBufAddr + sizeof(uint32_t) + sizeof(type_t) * globalIndex(), val, sizeof(uint32_t));
}


type_t test()
{
    const uint32_t idx = globalIndex();
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.inputBufAddress + idx * sizeof(type_t));

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
