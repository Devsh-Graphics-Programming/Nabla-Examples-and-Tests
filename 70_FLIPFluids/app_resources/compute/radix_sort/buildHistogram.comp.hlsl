#include "sort_common.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 1)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<DATA_TYPE> inputBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> globalHistograms;
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> partitionHistogram;

groupshared uint localHistogram[NumSortBins];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_GroupThreadID, uint32_t3 groupID : SV_GroupID)
{
    uint g_id = groupID.x;
    uint l_id = threadID.x;
    uint s_id = glsl::gl_SubgroupID();
    uint ls_id = glsl::gl_SubgroupInvocationID();
    
    uint idx = s_id * glsl::gl_SubgroupSize() + ls_id;
    uint partitionIdx = g_id;
    uint partitionStart = partitionIndex * PartitionSize;

    if (partitionStart >= params.numElements)
        return;

    // clear shared histogram data
    if (idx < NumSortBins)
        localHistogram[idx] = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < NumPartitions; i++)
    {
        uint keyIdx = partitionStart + WorkgroupSize * i + idx;
        uint key = keyIdx < params.numElements ? getKey(inputBuffer[keyIdx]) : 0xffffffff;
        uint radix = bitFieldExtract(key, params.bitShift, 8);
        InterlockedAdd(localHistogram[radix], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    if (idx < NumSortBins)
    {
        partitionHistogram[NumSortBins * partitionIdx + idx] = localHistogram[idx];

        uint passIdx = params.bitShift / 8;
        InterlockedAdd(globalHistograms[NumSortBins * passIdx + idx], localHistogram[idx]);
    }
}
