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
groupshared uint32_t scratch[config_t::SharedScratchElementCount];

#include "../../common/include/WorkgroupDataAccessors.hlsl"

static ScratchProxy arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    void operator()(PreloadedDataProxy<config_t,Binop> dataAccessor)
    {
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
    }
// #else
//     void operator()(PreloadedDataProxy<config_t,Binop> dataAccessor)
//     {
//         OPERATION<config_t,binop_base_t,device_capabilities>::template __call<PreloadedDataProxy<config_t,Binop>, ScratchProxy>(dataAccessor,arithmeticAccessor);
//         // we barrier before because we alias the accessors for Binop
//         arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
//     }
// #endif

};

template<class Binop>
static void subbench()
{
    const uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.ppOutputBuf + Binop::BindingIndex * sizeof(uint64_t), sizeof(uint64_t));

    PreloadedDataProxy<config_t,Binop> dataAccessor = PreloadedDataProxy<config_t,Binop>::create();
    dataAccessor.preload();

    operation_t<Binop,device_capabilities> func;
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        func(dataAccessor);

    dataAccessor.unload();
}

void benchmark()
{
    subbench<arithmetic::bit_and<uint32_t> >();
    subbench<arithmetic::bit_xor<uint32_t> >();
    subbench<arithmetic::bit_or<uint32_t> >();
    subbench<arithmetic::plus<uint32_t> >();
    subbench<arithmetic::multiplies<uint32_t> >();
    subbench<arithmetic::minimum<uint32_t> >();
    subbench<arithmetic::maximum<uint32_t> >();
}


uint32_t globalIndex()
{
    return glsl::gl_WorkGroupID().x*ITEMS_PER_WG+workgroup::SubgroupContiguousIndex();
}

bool canStore()
{
    return workgroup::SubgroupContiguousIndex()<ITEMS_PER_WG;
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    benchmark();
}
