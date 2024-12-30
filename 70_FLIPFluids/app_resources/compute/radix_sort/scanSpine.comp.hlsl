#include "sort_common.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 1)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<uint> globalHistograms;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> partitionHistogram;

groupshared uint reduction;
groupshared uint sScan[SUBGROUP_SIZE];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint threadID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    uint g_id = groupID.x;
    uint l_id = threadID.x;
    uint s_id = glsl::gl_SubgroupID();
    uint ls_id = glsl::gl_SubgroupInvocationID();

    uint idx = s_id * glsl::gl_SubgroupSize() + ls_id;
    uint partitionCount = (params.numElements + PartitionSize - 1) / PartitionSize;

    if (idx == 0)
        reduction = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i * WorkgroupSize < partitionCount; i++)
    {
        uint partitionIdx = i * WorkgroupSize + idx;
        uint value = partitionIdx < partitionCount ? partitionHistogram[partitionIdx * NumSortBins + g_id] : 0;
        uint prefixSum = WavePrefixSum(value) + reduction;
        uint sum = WaveActiveSum(value);

        if (WaveIsFirstLane())
            sScan[s_id] = sum;
        GroupMemoryBarrierWithGroupSync();

        if (idx < glsl::gl_NumSubgroups())
        {
            uint prefixSum = WavePrefixSum(sScan[idx]);
            uint sum = WaveActiveSum(sScan[idx]);
            sScan[idx] = prefixSum;

            if (idx == 0)
                reduction += sum;
        }
        GroupMemoryBarrierWithGroupSync();

        if (partitionIdx < partitionCount)
        {
            prefixSum += sScan[s_id];
            partitionHistogram[partitionIdx * NumSortBins + g_id] = prefixSum;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (g_id == 0)
    {
        if (idx < NumSortBins)
        {
            uint passIdx = params.bitShift / 8;
            uint value = globalHistograms[passIdx * NumSortBins + idx];
            uint prefixSum = WavePrefixSum(value);
            uint sum = WaveActiveSum(value);

            if (WaveIsFirstLane())
                sScan[s_id] = sum;
            GroupMemoryBarrierWithGroupSync();

            if (idx < NumSortBins / glsl::gl_SubgroupSize())
            {
                uint ps = WavePrefixSum(sScan[idx]);
                sScan[idx] = ps;
            }
            GroupMemoryBarrierWithGroupSync();

            prefixSum += sScan[s_id];
            globalHistograms[passIdx * NumSortBins + idx] = prefixSum;
        }
    }
}
