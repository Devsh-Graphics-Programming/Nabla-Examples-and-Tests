#include "common.hlsl"

#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint32_t> scratch;

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();
    uint32_t index = nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid;

    nbl::hlsl::glsl::barrier();
    
    if (index < pushConstants.dataElementCount)
    {
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * index);
        uint32_t address = nbl::hlsl::glsl::atomicAdd(scratch[value], (uint32_t) 1);
        vk::RawBufferStore(pushConstants.outputAddress + sizeof(uint32_t) * address, value);
    }
}