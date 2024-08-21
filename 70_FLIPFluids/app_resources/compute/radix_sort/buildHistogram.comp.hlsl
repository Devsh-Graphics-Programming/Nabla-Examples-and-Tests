#include "sort_common.hlsl"

[[vk::binding(0, 1)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<DATA_TYPE> inputBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> histograms;

shared DATA_TYPE[NumSortBins] histogram;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_GroupThreadID, uint32_t3 groupID : SV_GroupID)
{
    uint g_id = groupID.x;
    uint l_id = threadID.x;

    if (l_id < NumSortBins)
        histogram[l_id] = 0u;
    barrier();

    for (uint i = 0; i < params.numThreadsPerGroup; i++)
    {
        uint elementID = g_id * params.numThreadsPerGroup * WorkgroupSize + i * WorkgroupSize + l_id
        if (elementID < params.numElements)
        {
            uint bin = uint(GET_KEY(inputBuffer[elementID]) >> params.bitShift) & uint(NumSortBins - 1);
            atomicAdd(histogram[bin], 1u);
        }
    }
    barrier();

    if (l_id < NumSortBins)
        histograms[NumSortBins * g_id + l_id] = histogram[l_id];
}
