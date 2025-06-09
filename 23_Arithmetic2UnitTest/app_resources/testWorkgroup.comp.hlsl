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

#include "../../common/include/WorkgroupDataAccessors.hlsl"

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


template<class Binop>
static void subtest()
{
    if (glsl::gl_SubgroupSize()!=1u<<SUBGROUP_SIZE_LOG2)
        vk::RawBufferStore<uint32_t>(pc.pOutputBuf[Binop::BindingIndex], glsl::gl_SubgroupSize());

    operation_t<Binop,device_capabilities> func;
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