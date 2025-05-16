#pragma shader_stage(compute)

#include "workgroupCommon.hlsl"

// NOTE added dummy output image to be able to profile with Nsight, which still doesn't support profiling headless compute shaders
[[vk::binding(2, 0)]] RWTexture2D<float32_t4> outImage; // dummy

template<class Config, class Binop>
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(nbl::hlsl::is_same_v<dtype_t, type_t>);

    // we don't want to write/read storage multiple times in loop; doesn't seem optimized out in generated spirv
    template<typename AccessType>
    void get(const uint32_t ix, NBL_REF_ARG(dtype_t) value)
    {
        // value = inputValue[ix];
        value = nbl::hlsl::promote<dtype_t>(globalIndex());
    }
    template<typename AccessType>
    void set(const uint32_t ix, const dtype_t value)
    {
        // output[Binop::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * ix, value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};

static ScratchProxy arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    void operator()()
    {
        DataProxy<config_t,Binop> dataAccessor;
        nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<DataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    }
};

#ifndef NUM_LOOPS
#error "Define NUM_LOOPS!"
#endif

template<template<class> class binop, typename T, uint32_t N>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    if (globalIndex()==0u)
        output[binop<T>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());

    operation_t<binop<T>,nbl::hlsl::jit::device_capabilities> func;
    // TODO separate out store/load from DataProxy? so we don't do too many RW in benchmark
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        func(); // store is done with data accessor now
}


type_t benchmark()
{
    const type_t sourceVal = inputValue[globalIndex()];

    subbench<bit_and, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<bit_xor, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<bit_or, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<plus, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<multiplies, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<minimum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<maximum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
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
    const type_t sourceVal = benchmark();
}
