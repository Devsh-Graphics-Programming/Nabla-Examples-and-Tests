#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_params.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

#include "app_resources/shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup2/basic.hlsl"

template<class Binop, class device_capabilities>
using params_t = SUBGROUP_CONFIG_T;

NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation = params_t<typename arithmetic::plus<uint32_t>::base_t, device_capabilities>::ItemsPerInvocation;

typedef vector<uint32_t, ItemsPerInvocation> type_t;

uint32_t globalIndex()
{
    return glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+workgroup::SubgroupContiguousIndex();
}

template<class Binop>
static void subbench(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    type_t value = sourceVal;

    const uint64_t outputBufAddr = pc.pOutputBuf[Binop::BindingIndex];

    operation_t<params_t<typename Binop::base_t, device_capabilities> > func;
    // [unroll]
    for (uint32_t i = 0; i < NUM_LOOPS; i++)
        value = func(value);

    vk::RawBufferStore<type_t>(outputBufAddr + sizeof(type_t) * globalIndex(), value, sizeof(uint32_t));
}

void benchmark()
{
    const uint32_t invocationIndex = globalIndex();
    type_t sourceVal;
    Xoroshiro64Star xoroshiro = Xoroshiro64Star::construct(uint32_t2(invocationIndex,invocationIndex+1));
    [unroll]
    for (uint16_t i = 0; i < ItemsPerInvocation; i++)
        sourceVal[i] = xoroshiro();

    subbench<arithmetic::plus<uint32_t> >(sourceVal);
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    benchmark();
}
