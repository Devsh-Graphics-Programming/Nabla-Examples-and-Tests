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
    template<typename AccessType>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
        value = scratch[ix];
    }
    template<typename AccessType>
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
        vk::RawBufferStore<dtype_t>(outputBufAddr + sizeof(uint32_t) + sizeof(dtype_t) * (workgroupOffset+ix), value, sizeof(uint32_t));
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

    template<typename AccessType>
    void get(const uint32_t ix, NBL_REF_ARG(dtype_t) value)
    {
        value = preloaded[(ix-nbl::hlsl::workgroup::SubgroupContiguousIndex())>>Config::WorkgroupSizeLog2];
    }
    template<typename AccessType>
    void set(const uint32_t ix, const dtype_t value)
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

    // workgroup reduction returns the value of the reduction
    // workgroup scans do no return anything, but use the data accessor to do the storing directly
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


template<class Binop>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + Binop::BindingIndex * sizeof(uint64_t));
    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, nbl::hlsl::glsl::gl_SubgroupSize());

    operation_t<Binop,nbl::hlsl::jit::device_capabilities> func;
    func();
}


type_t test()
{
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.inputBufAddress + globalIndex() * sizeof(type_t));

    subtest<bit_and<uint32_t> >(sourceVal);
    subtest<bit_xor<uint32_t> >(sourceVal);
    subtest<bit_or<uint32_t> >(sourceVal);
    subtest<plus<uint32_t> >(sourceVal);
    subtest<multiplies<uint32_t> >(sourceVal);
    subtest<minimum<uint32_t> >(sourceVal);
    subtest<maximum<uint32_t> >(sourceVal);
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