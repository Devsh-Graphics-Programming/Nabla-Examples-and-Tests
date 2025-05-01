#pragma shader_stage(compute)

#include "workgroupCommon.hlsl"

template<class Config, class Binop>
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(nbl::hlsl::is_same_v<dtype_t, type_t>);

    dtype_t get(const uint32_t ix)
    {
        // return inputValue[ix];
        return inputVal;
    }
    void set(const uint32_t ix, const dtype_t value)
    {
        // output[Binop::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * ix, value);
        outputVal = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }

    // to avoid multiple load/store in benchmark, also values not that important?
    dtype_t inputVal;
    dtype_t outputVal;
};

static ScratchProxy<config_t> arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    otype_t operator()()
    {
        DataProxy<config_t,Binop> dataAccessor;
        dataAccessor.inputVal = inputValue[globalIndex()];
        nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<DataProxy<config_t,Binop>, ScratchProxy<config_t> >(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
        return dataAccessor.outputVal;
    }
};


template<template<class> class binop, typename T, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    if (globalIndex()==0u)
        output[binop<T>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());

    type_t value;
    operation_t<binop<T>,nbl::hlsl::jit::device_capabilities> func;
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(); // store is done with data accessor now

    output[binop<T>::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * globalIndex(), value);
}


type_t test()
{
    const type_t sourceVal = inputValue[globalIndex()];

    subtest<bit_and, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_xor, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_or, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<plus, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<multiplies, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<minimum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<maximum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    return sourceVal;
}


uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*ITEMS_PER_WG+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore()
{
    return nbl::hlsl::workgroup::SubgroupContiguousIndex()<ITEMS_PER_WG;
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    const type_t sourceVal = test();
}