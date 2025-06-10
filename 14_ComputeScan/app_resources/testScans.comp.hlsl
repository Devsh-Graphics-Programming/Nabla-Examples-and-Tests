#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/scan/arithmetic.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

static const uint32_t WORKGROUP_SIZE = 1u << WORKGROUP_SIZE_LOG2;

#include "common.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

[[vk::push_constant]] PushConstantData pc;

#ifndef OPERATION
#error "Define OPERATION!"
#endif

#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif

using namespace nbl::hlsl;
using config_t = scan::ScanConfiguration<WORKGROUP_SIZE_LOG2, SUBGROUP_SIZE_LOG2, ITEMS_PER_INVOCATION>;

groupshared uint32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

struct ScratchProxy
{
    template<typename AccessType, typename IndexType=uint16_t>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
        value = scratch[ix];
    }
    template<typename AccessType, typename IndexType=uint16_t>
    void set(const uint32_t ix, const AccessType value)
    {
        scratch[ix] = value;
    }

    uint32_t atomicOr(const uint32_t ix, const uint32_t value)
    {
        return glsl::atomicOr(scratch[ix],value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }
};

template<class Config, class Binop>
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::arith_config_t::ItemsPerInvocation_0>;

    static DataProxy<Config, Binop> create()
    {
        DataProxy<Config, Binop> retval;
        retval.workgroupOffset = glsl::gl_WorkGroupID().x * Config::arith_config_t::VirtualWorkgroupSize;
        retval.outputBufAddr = sizeof(uint32_t) + pc.pOutputBuf[Binop::BindingIndex];
        return retval;
    }

    static DataProxy<Config, Binop> create(uint32_t workGroupID)
    {
        DataProxy<Config, Binop> retval;
        retval.workgroupOffset = workGroupID * Config::arith_config_t::VirtualWorkgroupSize;
        retval.outputBufAddr = sizeof(uint32_t) + pc.pOutputBuf[Binop::BindingIndex];
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = vk::RawBufferLoad<AccessType>(pc.pInputBuf + (workgroupOffset + ix) * sizeof(AccessType));
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        vk::RawBufferStore<AccessType>(outputBufAddr + sizeof(AccessType) * (workgroupOffset+ix), value, sizeof(uint32_t));
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    uint32_t workgroupOffset;
    uint64_t outputBufAddr;
};

template<class Config, class Binop>
struct PreloadedDataProxy
{
    using dtype_t = vector<uint32_t, Config::arith_config_t::ItemsPerInvocation_0>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t PreloadedDataCount = Config::arith_config_t::VirtualWorkgroupSize / Config::WorkgroupSize;

    static PreloadedDataProxy<Config, Binop> create()
    {
        PreloadedDataProxy<Config, Binop> retval;
        retval.data = DataProxy<Config, Binop>::create();
        return retval;
    }

    static PreloadedDataProxy<Config, Binop> create(uint32_t workGroupID)
    {
        PreloadedDataProxy<Config, Binop> retval;
        retval.data = DataProxy<Config, Binop>::create(workGroupID);
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = preloaded[ix>>Config::WorkgroupSizeLog2];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        preloaded[ix>>Config::WorkgroupSizeLog2] = value;
    }

    void preload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template get<dtype_t, uint16_t>(idx * Config::WorkgroupSize + invocationIndex, preloaded[idx]);
    }
    void unload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template set<dtype_t, uint16_t>(idx * Config::WorkgroupSize + invocationIndex, preloaded[idx]);
    }

    bda::__ptr<uint32_t> getScratchPtr()
    {
        return bda::__ptr<uint32_t>::create(scratchDataAddress);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DataProxy<Config, Binop> data;
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
    void operator()()
    {
        PreloadedDataProxy<config_t,Binop> dataAccessor = PreloadedDataProxy<config_t,Binop>::create();
        dataAccessor.scratchDataAddress = pc.pScratchBuf;
        dataAccessor.preload();
#if IS_REDUCTION
        otype_t value =
#endif
        OPERATION<config_t,binop_base_t,true,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
#if IS_REDUCTION
        [unroll]
        for (uint32_t i = 0; i < PreloadedDataProxy<config_t,Binop>::PreloadedDataCount; i++)
            dataAccessor.preloaded[i] = value;
#endif
        dataAccessor.unload();
    }
};

uint32_t globalIndex()
{
    return glsl::gl_WorkGroupID().x+workgroup::SubgroupContiguousIndex();
}

template<class Binop>
static void subtest()
{
    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(pc.pOutputBuf[Binop::BindingIndex], glsl::gl_SubgroupSize());

    operation_t<Binop,jit::device_capabilities> func;
    func();
}

void test()
{
    subtest<arithmetic::bit_and<uint32_t> >();
    subtest<arithmetic::bit_xor<uint32_t> >();
    subtest<arithmetic::bit_or<uint32_t> >();
    subtest<arithmetic::plus<uint32_t> >();
    subtest<arithmetic::multiplies<uint32_t> >();
    subtest<arithmetic::minimum<uint32_t> >();
    subtest<arithmetic::maximum<uint32_t> >();
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    test();
}
