#include "common.hlsl"

#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint32_t> scratch;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t index = WorkgroupSize * GroupID.x + ID.x;

    if (index < pushConstants.dataElementCount)
    {
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * index);
        uint32_t address = nbl::hlsl::glsl::atomicAdd(scratch[value], (uint32_t) -1);
        vk::RawBufferStore(pushConstants.outputAddress + sizeof(uint32_t) * (address - 1), value);
    }
}