#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/scan/arithmetic.hlsl"

static const uint32_t WORKGROUP_SIZE = 1u << WORKGROUP_SIZE_LOG2;

#include "common.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

struct PushConstantData
{
    uint64_t inputBufAddress;
    uint64_t outputAddressBufAddress;
};

[[vk::push_constant]] PushConstantData pc;

#ifndef OPERATION
#error "Define OPERATION!"
#endif

#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif

using config_t = nbl::hlsl::scan::ScanConfiguration<WORKGROUP_SIZE_LOG2, SUBGROUP_SIZE_LOG2, ITEMS_PER_INVOCATION>;

groupshared uint32_t scratch[config_t::SharedScratchElementCount];

struct ScratchProxy
{
    template<typename AccessType, typename IndexType>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
        value = scratch[ix];
    }
    template<typename AccessType, typename IndexType>
    void set(const uint32_t ix, const AccessType value)
    {
        scratch[ix] = value;
    }

    uint32_t atomicOr(const uint32_t ix, const uint32_t value)
    {
        return nbl::hlsl::glsl::atomicOr(scratch[ix],value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};

template<class Config, class Binop>
struct PreloadedDataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(nbl::hlsl::is_same_v<dtype_t, type_t>);

    NBL_CONSTEXPR_STATIC_INLINE uint32_t PreloadedDataCount = Config::VirtualWorkgroupSize / Config::WorkgroupSize;

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = preloaded[(ix-nbl::hlsl::workgroup::SubgroupContiguousIndex())>>Config::WorkgroupSizeLog2];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        preloaded[(ix-nbl::hlsl::workgroup::SubgroupContiguousIndex())>>Config::WorkgroupSizeLog2] = value;
    }

    void preload()
    {
        const uint32_t workgroupOffset = nbl::hlsl::glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize;
        [unroll]
        for (uint32_t idx = 0; idx < PreloadedDataCount; idx++)
            preloaded[idx] = vk::RawBufferLoad<dtype_t>(pc.inputBufAddress + (workgroupOffset + idx * Config::WorkgroupSize + nbl::hlsl::workgroup::SubgroupContiguousIndex()) * sizeof(dtype_t));
    }
    void unload()
    {
        const uint32_t workgroupOffset = nbl::hlsl::glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize;
        uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + Binop::BindingIndex * sizeof(uint64_t));
        [unroll]
        for (uint32_t idx = 0; idx < PreloadedDataCount; idx++)
            vk::RawBufferStore<dtype_t>(outputBufAddr + sizeof(uint32_t) + sizeof(dtype_t) * (workgroupOffset + idx * Config::WorkgroupSize + nbl::hlsl::workgroup::SubgroupContiguousIndex()), preloaded[idx], sizeof(uint32_t));
    }

    bda::__ptr<uint32_t> getScratchPtr()
    {
        return bda::__ptr<uint32_t>::create(scratchDataAddress);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }

    dtype_t preloaded[PreloadedDataCount];
    uint64_t scratchDataAddress;
};

static ScratchProxy arithmeticAccessor;


template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    // reduction returns the value of the reduction
    // scans do no return anything, but use the data accessor to do the storing directly
#if IS_REDUCTION
    void operator()()
    {
        PreloadedDataProxy<config_t,Binop> dataAccessor;
        dataAccessor.preload();
        otype_t value = nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

        [unroll]
        for (uint32_t i = 0; i < PreloadedDataProxy<config_t,Binop>::PreloadedDataCount; i++)
            dataAccessor.preloaded[i] = value;
        dataAccessor.unload();
    }
#else
    void operator()()
    {
        PreloadedDataProxy<config_t,Binop> dataAccessor;
        dataAccessor.preload();
        nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
        dataAccessor.unload();
    }
#endif
};

uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

template<class Binop>
static void subtest()
{
    uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + Binop::BindingIndex * sizeof(uint64_t));
    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, nbl::hlsl::glsl::gl_SubgroupSize());

    operation_t<Binop,nbl::hlsl::jit::device_capabilities> func;
    func();
}

void test()
{
    subtest<bit_and<uint32_t> >();
    subtest<bit_xor<uint32_t> >();
    subtest<bit_or<uint32_t> >();
    subtest<plus<uint32_t> >();
    subtest<multiplies<uint32_t> >();
    subtest<minimum<uint32_t> >();
    subtest<maximum<uint32_t> >();
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    test();
}
