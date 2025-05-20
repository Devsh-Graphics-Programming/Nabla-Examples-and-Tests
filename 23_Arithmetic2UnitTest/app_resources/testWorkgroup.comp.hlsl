#pragma shader_stage(compute)

#include "workgroupCommon.hlsl"

template<class Config, class Binop>
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(nbl::hlsl::is_same_v<dtype_t, type_t>);

    template<typename AccessType>
    void get(const uint32_t ix, NBL_REF_ARG(dtype_t) value)
    {
        const uint32_t workgroupOffset = nbl::hlsl::glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize;
        value = vk::RawBufferLoad<dtype_t>(pc.inputBufAddress + (workgroupOffset + ix) * sizeof(dtype_t));
    }
    template<typename AccessType>
    void set(const uint32_t ix, const dtype_t value)
    {
        const uint32_t workgroupOffset = nbl::hlsl::glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize;
        uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + Binop::BindingIndex * sizeof(uint64_t));
        [unroll]
        for (uint32_t i = 0; i < Config::ItemsPerInvocation_0; i++)
            vk::RawBufferStore<uint32_t>(outputBufAddr+sizeof(uint32_t)+sizeof(dtype_t)*(workgroupOffset+ix)+i*sizeof(uint32_t), value[i]);
        // vk::RawBufferStore<dtype_t>(outputBufAddr + sizeof(uint32_t) + sizeof(dtype_t) * (workgroupOffset+ix), value, sizeof(uint32_t)); TODO why won't this work???
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


template<template<class> class binop, typename T, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + binop<T>::BindingIndex * sizeof(uint64_t));
    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, nbl::hlsl::glsl::gl_SubgroupSize());

    operation_t<binop<T>,nbl::hlsl::jit::device_capabilities> func;
    func(); // store is done with data accessor now
}


type_t test()
{
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.inputBufAddress + globalIndex() * sizeof(type_t));

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