#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_params.hlsl"

#include "app_resources/shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup2/basic.hlsl"

template<class Binop, class device_capabilities>
using params_t = SUBGROUP_CONFIG_T;

typedef vector<uint32_t, params_t<typename arithmetic::bit_and<uint32_t>::base_t, device_capabilities>::ItemsPerInvocation> type_t;

uint32_t globalIndex()
{
    return glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+workgroup::SubgroupContiguousIndex();
}

template<class Binop>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    const uint64_t outputBufAddr = pc.pOutputBuf[Binop::BindingIndex];

    assert(glsl::gl_SubgroupSize() == params_t<typename Binop::base_t, device_capabilities>::config_t::Size)

    operation_t<params_t<typename Binop::base_t, device_capabilities> > func;
    type_t val = func(sourceVal);

    vk::RawBufferStore<type_t>(outputBufAddr + sizeof(type_t) * globalIndex(), val, sizeof(uint32_t));
}

type_t test()
{
    const uint32_t idx = globalIndex();
    type_t sourceVal = vk::RawBufferLoad<type_t>(pc.pInputBuf + idx * sizeof(type_t));

    subtest<arithmetic::bit_and<uint32_t> >(sourceVal);
    subtest<arithmetic::bit_xor<uint32_t> >(sourceVal);
    subtest<arithmetic::bit_or<uint32_t> >(sourceVal);
    subtest<arithmetic::plus<uint32_t> >(sourceVal);
    subtest<arithmetic::multiplies<uint32_t> >(sourceVal);
    subtest<arithmetic::minimum<uint32_t> >(sourceVal);
    subtest<arithmetic::maximum<uint32_t> >(sourceVal);
    return sourceVal;
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    test();
}
