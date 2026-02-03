#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

using config_t = WORKGROUP_CONFIG_T;

#include "app_resources/shaderCommon.hlsl"

typedef vector<uint32_t, config_t::ItemsPerInvocation_0> type_t;

// final (level 1/2) scan needs to fit in one subgroup exactly
groupshared uint32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

#include "nbl/examples/workgroup/DataAccessors.hlsl"
using namespace nbl::hlsl::examples::workgroup;

template<uint16_t WorkgroupSizeLog2, uint16_t VirtualWorkgroupSize, uint16_t ItemsPerInvocation>
struct RandomizedInputDataProxy
{
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t PreloadedDataCount = VirtualWorkgroupSize / WorkgroupSize;

    static RandomizedInputDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> create(uint64_t inputBuf, uint64_t outputBuf)
    {
        RandomizedInputDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> retval;
        retval.data = DataProxy<VirtualWorkgroupSize, ItemsPerInvocation>::create(inputBuf, outputBuf);
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = preloaded[ix>>WorkgroupSizeLog2];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        preloaded[ix>>WorkgroupSizeLog2] = value;
    }

    void preload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        Xoroshiro64Star xoroshiro = Xoroshiro64Star::construct(uint32_t2(invocationIndex,invocationIndex+1));
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            [unroll]
            for (uint16_t i = 0; i < ItemsPerInvocation; i++)
               preloaded[idx][i] = xoroshiro();
    }
    void unload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template set<dtype_t, uint16_t>(idx * WorkgroupSize + invocationIndex, preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DataProxy<VirtualWorkgroupSize, ItemsPerInvocation> data;
    dtype_t preloaded[PreloadedDataCount];
};

static ScratchProxy arithmeticAccessor;

using data_proxy_t = RandomizedInputDataProxy<config_t::WorkgroupSizeLog2,config_t::VirtualWorkgroupSize,config_t::ItemsPerInvocation_0>;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    void operator()(data_proxy_t dataAccessor)
    {
#if IS_REDUCTION
        otype_t value = 
#endif
        OPERATION<config_t,binop_base_t,device_capabilities>::template __call<data_proxy_t, ScratchProxy>(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
#if IS_REDUCTION
        [unroll]
        for (uint32_t i = 0; i < data_proxy_t::PreloadedDataCount; i++)
            dataAccessor.preloaded[i] = value;
#endif
    }
};

template<class Binop>
static void subbench()
{
    data_proxy_t dataAccessor = data_proxy_t::create(0, pc.pOutputBuf[Binop::BindingIndex]);
    dataAccessor.preload();

    operation_t<Binop,device_capabilities> func;
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        func(dataAccessor);

    dataAccessor.unload();
}

void benchmark()
{
    // only benchmark plus op
    subbench<arithmetic::plus<uint32_t> >();
}


[numthreads(config_t::WorkgroupSize,1,1)]
void main()
{
    benchmark();
}
