#include "sort_common.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 1)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<DATA_TYPE> inputBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> histograms;

groupshared uint histogram[NumSortBins];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_GroupThreadID, uint32_t3 groupID : SV_GroupID)
{
    uint g_id = groupID.x;
    uint l_id = threadID.x;

    if (l_id < NumSortBins)
        histogram[l_id] = 0u;
    glsl::barrier();

    for (uint i = 0; i < params.numThreadsPerGroup; i++)
    {
        uint elementID = g_id * params.numThreadsPerGroup * WorkgroupSize + i * WorkgroupSize + l_id;
        if (elementID < params.numElements)
        {
            uint bin = uint(GET_KEY(inputBuffer[elementID]) >> params.bitShift) & uint(NumSortBins - 1);
            glsl::atomicAdd(histogram[bin], 1u);
        }
    }
    glsl::barrier();

    if (l_id < NumSortBins)
        histograms[NumSortBins * g_id + l_id] = histogram[l_id];
}
