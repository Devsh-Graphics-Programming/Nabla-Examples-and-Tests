#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "shaderCommon.hlsl"

// NOTE added dummy output image to be able to profile with Nsight, which still doesn't support profiling headless compute shaders
[[vk::binding(2, 0)]] RWTexture2D<float32_t4> outImage; // dummy

uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore() {return true;}

#ifndef NUM_LOOPS
#error "Define NUM_LOOPS!"
#endif

template<template<class> class binop, typename T, uint32_t N>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    using config_t = nbl::hlsl::subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;
    type_t value = sourceVal;

    operation_t<params_t> func;
    // [unroll]
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(value);

    output[binop<T>::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * globalIndex(), value);
}

void benchmark()
{
    const uint32_t idx = globalIndex();
    type_t sourceVal = inputValue[idx];

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
