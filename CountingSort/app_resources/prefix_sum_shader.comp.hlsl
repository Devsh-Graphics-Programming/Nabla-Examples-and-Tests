#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint32_t> scratch;

groupshared uint32_t prefixScratch[WorkgroupSize];

struct ScratchProxy
{
    uint32_t get(const uint32_t ix)
    {
        return prefixScratch[ix];
    }
    void set(const uint32_t ix, const uint32_t value)
    {
        prefixScratch[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

static ScratchProxy arithmeticAccessor;

groupshared uint32_t sdata[WorkgroupSize];

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();
    sdata[tid] = 0;
    uint32_t index = nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid;

    nbl::hlsl::glsl::barrier();

    if (index < pushConstants.dataElementCount)
    {
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * index);
        nbl::hlsl::glsl::atomicAdd(sdata[value - pushConstants.minimum], (uint32_t) 1);
    }

    nbl::hlsl::glsl::barrier();

    uint32_t retval = nbl::hlsl::workgroup::exclusive_scan<nbl::hlsl::plus<uint32_t>, WorkgroupSize>::template __call<ScratchProxy> (sdata[tid], arithmeticAccessor);
    arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

    nbl::hlsl::glsl::atomicAdd(scratch[tid], retval);
}