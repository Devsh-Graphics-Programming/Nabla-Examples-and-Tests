#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

#include "shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup2/basic.hlsl"

typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;

uint32_t globalIndex()
{
    return glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+workgroup::SubgroupContiguousIndex();
}

template<class Binop, uint32_t N>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    using config_t = subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = subgroup2::ArithmeticParams<config_t, typename Binop::base_t, N, device_capabilities>;
    type_t value = sourceVal;

    const uint64_t outputBufAddr = pc.pOutputBuf[Binop::BindingIndex];

    operation_t<params_t> func;
    // [unroll]
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(value);

    vk::RawBufferStore<type_t>(outputBufAddr + sizeof(type_t) * globalIndex(), value, sizeof(uint32_t));
}

void benchmark()
{
    type_t sourceVal;
    Xoroshiro64Star xoroshiro = Xoroshiro64Star::construct(uint32_t2(invocationIndex,invocationIndex+1));
    [unroll]
    for (uint16_t i = 0; i < Config::ItemsPerInvocation_0; i++)
        sourceVal[i] = xoroshiro();

    subbench<arithmetic::plus<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    benchmark();
}
