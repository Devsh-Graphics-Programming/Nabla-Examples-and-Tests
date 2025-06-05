#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

static const uint32_t WORKGROUP_SIZE = 1u << WORKGROUP_SIZE_LOG2;

#include "shaderCommon.hlsl"

using config_t = workgroup2::ArithmeticConfiguration<WORKGROUP_SIZE_LOG2, SUBGROUP_SIZE_LOG2, ITEMS_PER_INVOCATION>;

typedef vector<uint32_t, config_t::ItemsPerInvocation_0> type_t;

// final (level 1/2) scan needs to fit in one subgroup exactly
groupshared uint32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

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
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(is_same_v<dtype_t, type_t>);

    static DataProxy<Config, Binop> create()
    {
        DataProxy<Config, Binop> retval;
        retval.workgroupOffset = glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize;
        retval.outputBufAddr = sizeof(uint32_t) + vk::RawBufferLoad<uint64_t>(pc.ppOutputBuf + Binop::BindingIndex * sizeof(uint64_t));
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
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(is_same_v<dtype_t, type_t>);

    NBL_CONSTEXPR_STATIC_INLINE uint32_t PreloadedDataCount = Config::VirtualWorkgroupSize / Config::WorkgroupSize;

    static PreloadedDataProxy<Config, Binop> create()
    {
        PreloadedDataProxy<Config, Binop> retval;
        retval.data = DataProxy<Config, Binop>::create();
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
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template get<dtype_t, uint16_t>(idx * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex(), preloaded[idx]);
    }
    void unload()
    {
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template set<dtype_t, uint16_t>(idx * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex(), preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DataProxy<Config, Binop> data;
    dtype_t preloaded[PreloadedDataCount];
};

static ScratchProxy arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    // workgroup reduction returns the value of the reduction
    // workgroup scans do no return anything, but use the data accessor to do the storing directly
    void operator()()
    {
        PreloadedDataProxy<config_t,Binop> dataAccessor = PreloadedDataProxy<config_t,Binop>::create();
        dataAccessor.preload();
#if IS_REDUCTION
        otype_t value =
#endif
        OPERATION<config_t,binop_base_t,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
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
    return glsl::gl_WorkGroupID().x*ITEMS_PER_WG+workgroup::SubgroupContiguousIndex();
}

template<class Binop>
static void subtest()
{
    uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.ppOutputBuf + Binop::BindingIndex * sizeof(uint64_t));
    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, glsl::gl_SubgroupSize());

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