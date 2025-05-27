#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

static const uint32_t WORKGROUP_SIZE = 1u << WORKGROUP_SIZE_LOG2;

#include "shaderCommon.hlsl"

using config_t = nbl::hlsl::workgroup2::ArithmeticConfiguration<WORKGROUP_SIZE_LOG2, SUBGROUP_SIZE_LOG2, ITEMS_PER_INVOCATION>;

typedef vector<uint32_t, config_t::ItemsPerInvocation_0> type_t;

// final (level 1/2) scan needs to fit in one subgroup exactly
groupshared uint32_t scratch[config_t::SharedScratchElementCount];

struct ScratchProxy
{
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = scratch[ix];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
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
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(nbl::hlsl::is_same_v<dtype_t, type_t>);

    // we don't want to write/read storage multiple times in loop; doesn't seem optimized out in generated spirv
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(dtype_t) value)
    {
        // value = inputValue[ix];
        value = nbl::hlsl::promote<dtype_t>(globalIndex());
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const dtype_t value)
    {
        // output[Binop::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * ix, value);
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

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }

    dtype_t preloaded[PreloadedDataCount];
};

static ScratchProxy arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

#if IS_REDUCTION
    void operator()(PreloadedDataProxy<config_t,Binop> dataAccessor)
    {
        otype_t value = nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

        [unroll]
        for (uint32_t i = 0; i < PreloadedDataProxy<config_t,Binop>::PreloadedDataCount; i++)
            dataAccessor.preloaded[i] = value;
    }
#else
    void operator()(PreloadedDataProxy<config_t,Binop> dataAccessor)
    {
        nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    }
#endif

};

template<class Binop>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    const uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + Binop::BindingIndex * sizeof(uint64_t), sizeof(uint64_t));

    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, nbl::hlsl::glsl::gl_SubgroupSize());

    PreloadedDataProxy<config_t,Binop> dataAccessor;
    dataAccessor.preload();

    operation_t<Binop,nbl::hlsl::jit::device_capabilities> func;
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        func(dataAccessor);

    dataAccessor.unload();
}


type_t benchmark()
{
    const type_t sourceVal = vk::RawBufferLoad<type_t>(pc.inputBufAddress + globalIndex() * sizeof(type_t));

    subbench<bit_and<uint32_t> >(sourceVal);
    subbench<bit_xor<uint32_t> >(sourceVal);
    subbench<bit_or<uint32_t> >(sourceVal);
    subbench<plus<uint32_t> >(sourceVal);
    subbench<multiplies<uint32_t> >(sourceVal);
    subbench<minimum<uint32_t> >(sourceVal);
    subbench<maximum<uint32_t> >(sourceVal);
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
