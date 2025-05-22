#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;

template<class Binop, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    using config_t = nbl::hlsl::subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
    using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename Binop::base_t, N, nbl::hlsl::jit::device_capabilities>;

    const uint64_t outputBufAddr = vk::RawBufferLoad<uint64_t>(pc.outputAddressBufAddress + Binop::BindingIndex * sizeof(uint64_t), sizeof(uint64_t));

    if (globalIndex()==0u)
        vk::RawBufferStore<uint32_t>(outputBufAddr, nbl::hlsl::glsl::gl_SubgroupSize());

    operation_t<params_t> func;
    type_t val = func(sourceVal);
    if (canStore())
        vk::RawBufferStore<type_t>(outputBufAddr + sizeof(uint32_t) + sizeof(type_t) * globalIndex(), val, sizeof(uint32_t));
}

type_t test()
{
    const uint32_t idx = globalIndex();
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.inputBufAddress + idx * sizeof(type_t));

    subtest<bit_and<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_xor<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_or<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<plus<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<multiplies<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<minimum<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<maximum<uint32_t>, ITEMS_PER_INVOCATION>(sourceVal);
    return sourceVal;
}

uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore() {return true;}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    test();
}
