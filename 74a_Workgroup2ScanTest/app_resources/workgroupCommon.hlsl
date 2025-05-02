#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#include "common.hlsl"

static const uint32_t WORKGROUP_SIZE = 1u << WORKGROUP_SIZE_LOG2;

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

using config_t = nbl::hlsl::workgroup2::Configuration<WORKGROUP_SIZE_LOG2, SUBGROUP_SIZE_LOG2, ITEMS_PER_INVOCATION>;

typedef vector<uint32_t, config_t::ItemsPerInvocation_0> type_t;

// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<type_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];

// because subgroups don't match `gl_LocalInvocationIndex` snake curve addressing, we also can't load inputs that way
uint32_t globalIndex();
// since we test ITEMS_PER_WG<WorkgroupSize we need this so workgroups don't overwrite each other's outputs
bool canStore();

#ifndef OPERATION
#error "Define OPERATION!"
#endif
#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif

groupshared vector<uint32_t, config_t::ItemsPerInvocation_1> scratch[config_t::SubgroupSize];  // final (level 1) scan needs to fit in one subgroup exactly

template<class Config>
struct ScratchProxy
{
    using scalar_t = uint32_t;
    using stype_t = vector<uint32_t, Config::ItemsPerInvocation_1>;

    stype_t get(const uint32_t ix)
    {
        return scratch[ix];
    }
    void set(const uint32_t ix, const stype_t value)
    {
        scratch[ix] = value;
    }

    scalar_t getByComponent(const uint32_t ix)
    {
        return scratch[ix/Config::ItemsPerInvocation_1][ix&(Config::ItemsPerInvocation_1-1)];
    }
    void setByComponent(const uint32_t ix, const scalar_t value)
    {
        scratch[ix/Config::ItemsPerInvocation_1][ix&(Config::ItemsPerInvocation_1-1)] = value;
    }

    stype_t atomicOr(const uint32_t ix, const stype_t value)
    {
        return nbl::hlsl::glsl::atomicOr(scratch[ix],value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};
