#include "common.hlsl"

#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

NBL_CONSTEXPR uint32_t BucketsPerThread = ceil((float) BucketCount / WorkgroupSize);

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint32_t> scratch;

groupshared uint32_t sdata[BucketCount];

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

    [unroll]
    for (int i = 0; i < BucketsPerThread; i++)
        sdata[BucketsPerThread * tid + i] = 0;
    uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid) * pushConstants.elementsPerWT;

    nbl::hlsl::glsl::barrier();
    
    [unroll]
    for (int i = 0; i < pushConstants.elementsPerWT; i++)
    {
        if (index + i >= pushConstants.dataElementCount)
            break;
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * (index + i));
        nbl::hlsl::glsl::atomicAdd(sdata[value - pushConstants.minimum], (uint32_t) 1);
    }

    nbl::hlsl::glsl::barrier();

    [unroll]
    for (int i = 0; i < BucketsPerThread; i++)
    {
        uint32_t count = sdata[WorkgroupSize * i + tid];
        sdata[WorkgroupSize * i + tid] = nbl::hlsl::glsl::atomicAdd(scratch[WorkgroupSize * i + tid], sdata[WorkgroupSize * i + tid]);
        [unroll]
        for (int j = 0; j < count; j++)
        {
            vk::RawBufferStore(pushConstants.outputAddress + sizeof(uint32_t) * (sdata[WorkgroupSize * i + tid] + j), WorkgroupSize * i + tid);
        }
    }
}