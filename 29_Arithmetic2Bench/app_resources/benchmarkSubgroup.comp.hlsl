#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;

uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore() {return true;}

template<template<class> class binop, typename T, uint32_t N>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    using config_t = nbl::hlsl::subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;
    type_t value = sourceVal;

    const uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + binop<T>::BindingIndex * sizeof(uint64_t), sizeof(uint64_t));

    operation_t<params_t> func;
    // [unroll]
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(value);

    if (canStore())
        vk::RawBufferStore<type_t>(outputBufAddr + sizeof(uint32_t) + sizeof(type_t) * globalIndex(), value, sizeof(uint32_t));
}

void benchmark()
{
    const uint32_t idx = globalIndex();
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.inputBufAddress + idx * sizeof(type_t));

    subbench<bit_and, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<bit_xor, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<bit_or, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<plus, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<multiplies, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<minimum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<maximum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    benchmark();
}
