#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "shaderCommon.hlsl"

uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore() {return true;}

#ifndef NUM_LOOPS
#error "Define NUM_LOOPS!"
#endif

// template<template<class> class binop, typename T, uint32_t N>
// static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
// {
//     using config_t = nbl::hlsl::subgroup::Configuration<SUBGROUP_SIZE_LOG2>;
//     using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;

//     const uint32_t storeAddr = sizeof(uint32_t) + sizeof(type_t) * globalIndex();

//     operation_t<params_t> func;
//     [unroll]
//     for (uint32_t i = 0; i < NUM_LOOPS; i++)
//     {
//         const uint32_t arrIndex = i & 7u;   // i % 8
//         output[arrIndex].template Store<type_t>(storeAddr, func(sourceVal));
//     }
// }

template<template<class> class binop, typename T, uint32_t N>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    using config_t = nbl::hlsl::subgroup::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;
    type_t value = sourceVal;

    operation_t<params_t> func;
    [unroll]
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(value);

    output[binop<T>::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * globalIndex(), value);
}

void benchmark()
{
    const uint32_t idx = globalIndex() * ITEMS_PER_INVOCATION;
    type_t sourceVal;
    [unroll]
    for (uint32_t i = 0; i < ITEMS_PER_INVOCATION; i++)
    {
        sourceVal[i] = inputValue[idx + i];
    }

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
