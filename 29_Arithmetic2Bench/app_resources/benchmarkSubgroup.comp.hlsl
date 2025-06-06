#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

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

    const uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.ppOutputBuf + Binop::BindingIndex * sizeof(uint64_t), sizeof(uint64_t));

    operation_t<params_t> func;
    // [unroll]
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(value);

    vk::RawBufferStore<type_t>(outputBufAddr + sizeof(uint32_t) + sizeof(type_t) * globalIndex(), value, sizeof(uint32_t));
}

void benchmark()
{
    const uint32_t idx = globalIndex();
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.pInputBuf + idx * sizeof(type_t));

    subbench<arithmetic::bit_and<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<arithmetic::bit_xor<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<arithmetic::bit_or<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<arithmetic::plus<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<arithmetic::multiplies<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<arithmetic::minimum<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subbench<arithmetic::maximum<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    benchmark();
}
