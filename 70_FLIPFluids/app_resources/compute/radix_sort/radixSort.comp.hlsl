#include "sort_common.hlsl"

using namespace nbl::hlsl;

#define GET_KEY(s) s.x

[[vk::binding(0, 1)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<DATA_TYPE> inputBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<DATA_TYPE> outputBuffer;

[[vk::binding(3, 1)]] RWStructuredBuffer<uint> histograms;

groupshared uint sums[NumSortBins / SubgroupSize];
groupshared uint globalOffsets[NumSortBins];

struct BinFlags
{
    uint flags[WorkgroupSize / 32];
};
groupshared BinFlags binFlags[NumSortBins];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint threadID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    uint g_id = groupID.x;
    uint l_id = threadID.x;
    uint s_id = glsl::gl_SubgroupID();
    uint ls_id = glsl::gl_SubgroupInvocationID();

    uint localHistogram = 0;
    uint prefixSum = 0;
    uint histogramCount = 0;

    if (l_id < NumSortBins)
    {
        uint count = 0;
        for (uint i = 0; i < params.numWorkgroups; i++)
        {
            const uint t = histograms[NumSortBins * i + l_id];
            localHistogram = (i == g_id) ? count : localHistogram;
            count += t;
        }

        histogramCount = count;
        const uint sum = glsl::subgroupAdd(histogramCount);
        prefixSum = glsl::subgroupExclusiveAdd(histogramCount);
        if (glsl::subgroupElect())
            sums[s_id] = sum;
    }
    glsl::barrier();

    if (l_id < NumSortBins)
    {
        const uint totalPrefixSums = glsl::subgroupBroadcast(glsl::subgroupExclusiveAdd(sums[ls_id]), s_id);
        const uint globalHistogram = totalPrefixSums + prefixSum;
        globalOffsets[l_id] = globalHistogram + localHistogram;
    }

    const uint flagsBin = l_id / 32;
    const uint flagsBit = 1 << (l_id % 32);

    for (uint i = 0; i < params.numThreadsPerGroup; i++)
    {
        uint elementID = g_id * params.numThreadsPerGroup * WorkgroupSize + i * WorkgroupSize + l_id;

        if (l_id < NumSortBins)
        {
            for (int j = 0; j < WorkgroupSize / 32; j++)
            {
                binFlags[l_id].flags[j] = 0u;
            }
        }
        glsl::barrier();

        DATA_TYPE e = 0;
        uint binID = 0;
        uint binOffset = 0;
        if (elementID < params.numElements)
        {
            e = inputBuffer[elementID];
            binID = uint(GET_KEY(e) >> params.bitShift) & uint(NumSortBins - 1);
            binOffset = globalOffsets[binID];
            glsl::atomicAdd(binFlags[binID].flags[flagsBin], flagsBit);
        }
        glsl::barrier();

        if (elementID < params.numElements)
        {
            uint prefix = 0;
            uint count = 0;
            for (uint j = 0; j < WorkgroupSize / 32; j++)
            {
                const uint bits = binFlags[binID].flags[j];
                const uint fullCount = countbits(bits);
                const uint partialCount = countbits(bits & (flagsBit - 1));
                prefix += (j < flagsBin) ? fullCount : 0u;
                prefix += (j == flagsBin) ? partialCount : 0u;
                count += fullCount;
            }
            outputBuffer[binOffset + prefix] = e;
            if (prefix == count - 1)
                glsl::atomicAdd(globalOffsets[binID], count);
        }
        glsl::barrier();
    }
}
