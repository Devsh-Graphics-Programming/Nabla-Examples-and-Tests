#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

using config_t = WORKGROUP_CONFIG_T;

#include "app_resources/shaderCommon.hlsl"

typedef vector<uint32_t, config_t::ItemsPerInvocation_0> type_t;

// final (level 1/2) scan needs to fit in one subgroup exactly
groupshared uint32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

#include "nbl/examples/workgroup/DataAccessors.hlsl"
using namespace nbl::hlsl::examples::workgroup;

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
        using data_proxy_t = PreloadedDataProxy<config_t::WorkgroupSizeLog2,config_t::VirtualWorkgroupSize,config_t::ItemsPerInvocation_0>;
        data_proxy_t dataAccessor = data_proxy_t::create(pc.pInputBuf, pc.pOutputBuf[Binop::BindingIndex]);
        dataAccessor.preload();
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
        dataAccessor.unload();
    }
};


template<class Binop>
static void subtest()
{
    assert(glsl::gl_SubgroupSize() == config_t::SubgroupSize)

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

[numthreads(config_t::WorkgroupSize,1,1)]
void main()
{
    test();
}