#include "sort_common.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 1)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<DATA_TYPE> inputBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<DATA_TYPE> outputBuffer;

[[vk::binding(3, 1)]] RWStructuredBuffer<uint> globalHistograms;
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> partitionHistogram;

groupshared uint localHistogram[PartitionSize];
groupshared uint histogramSums[NumSortBins];

uint bitCount(uint4 v)
{
    uint4 res = countbits(v);
    return res[0] + res[1] + res[2] + res[3];
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint threadID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    uint g_id = groupID.x;
    uint l_id = threadID.x;
    uint s_id = glsl::gl_SubgroupID();
    uint ls_id = glsl::gl_SubgroupInvocationID();

    uint numSubgroups = glsl::gl_NumSubgroups();
    uint subgroupSize = glsl::gl_SubgroupSize();
    
    uint idx = s_id * subgroupSize + ls_id;
    uint partitionIdx = g_id;
    uint partitionStart = partitionIdx * PartitionSize;

    const uint4 subgroupMask = uint4(
        (1 << ls_id) - 1,
        (1 << (ls_id - 32)) - 1,
        (1 << (ls_id - 64)) - 1,
        (1 << (ls_id - 96)) - 1
    );

    if (partitionStart >= params.numElements)
        return;

    if (idx < NumSortBins)
    {
        for (uint i = 0; i < numSubgroups; i++)
        {
            localHistogram[numSubgroups * idx + i] = 0;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // load local values
    uint keys[NumPartitions];
    uint bins[NumPartitions];
    uint offsets[NumPartitions];
    uint subgroupHist[NumPartitions];

#ifdef USE_KV_PAIRS
    uint values[NumPartitions];
#endif

    for (uint i = 0; i < NumPartitions; i++)
    {
        uint keyIdx = partitionStart + NumPartitions * subgroupSize * s_id + i * subgroupSize + ls_id;
        uint key = keyIdx < params.numElements ? getKey(inputBuffer, keyIdx) : 0xffffffff;
        keys[i] = key;

#ifdef USE_KV_PAIRS
        values[i] = keyIdx < params.numElements ? getValue(inputBuffer, keyIdx) : 0;
#endif

        uint bin = glsl::bitfieldExtract(key, params.bitShift, 8);
        bins[i] = bin;

        uint4 mask = WaveActiveBallot(true);
        [unroll]
        for (uint j = 0; j < 8; j++)
        {
            uint digit = (bin >> j) & 1;
            uint4 ballot = WaveActiveBallot(digit == 1);
            mask &= (uint4)(digit - 1) ^ ballot;
        }

        uint subgroupOffset = bitCount(subgroupMask & mask);
        uint radixCount = bitCount(mask);

        if (subgroupOffset == 0)
        {
            InterlockedAdd(localHistogram[numSubgroups * bin + s_id], radixCount);
            subgroupHist[i] = radixCount;
        }
        else
        {
            subgroupHist[i] = 0;
        }

        offsets[i] = subgroupOffset;
    }
    GroupMemoryBarrierWithGroupSync();

    // reduce, downsweep step
    for (uint i = idx; i < NumSortBins * numSubgroups; i += WorkgroupSize)
    {
        uint v = localHistogram[i];
        uint sum = WaveActiveSum(v);
        uint prefixSum = WavePrefixSum(v);
        localHistogram[i] = prefixSum;
        if (ls_id == 0)
            histogramSums[i / subgroupSize] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint reduceOffset = NumSortBins * numSubgroups / subgroupSize;
    if (idx < reduceOffset)
    {
        uint v = histogramSums[idx];
        uint sum = WaveActiveSum(v);
        uint prefixSum = WavePrefixSum(v);
        histogramSums[idx] = prefixSum;
        if (ls_id == 0)
            histogramSums[reduceOffset + idx / subgroupSize] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint reduceSize = NumSortBins * numSubgroups / subgroupSize / subgroupSize;
    if (idx < reduceSize)
    {
        uint v = histogramSums[reduceOffset + idx];
        uint prefixSum = WavePrefixSum(v);
        histogramSums[reduceOffset + idx] = prefixSum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (idx < reduceOffset)
        histogramSums[idx] += histogramSums[reduceOffset + idx / subgroupSize];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = idx; i < NumSortBins * numSubgroups; i += WorkgroupSize)
        localHistogram[i] += histogramSums[i / subgroupSize];
    GroupMemoryBarrierWithGroupSync();

    // scatter step
    for (uint i = 0; i < NumPartitions; i++)
    {
        uint bin = bins[i];
        offsets[i] += localHistogram[numSubgroups * bin + s_id];
        GroupMemoryBarrierWithGroupSync();

        if (subgroupHist[i] > 0)
            InterlockedAdd(localHistogram[numSubgroups * bin + s_id], subgroupHist[i]);
        GroupMemoryBarrierWithGroupSync();
    }

    if (idx < NumSortBins)
    {
        uint v = idx == 0 ? 0 : localHistogram[numSubgroups * idx - 1];
        uint passIdx = params.bitShift / 8;
        histogramSums[idx] = globalHistograms[NumSortBins * passIdx + idx] + partitionHistogram[NumSortBins * partitionIdx + idx] - v;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < NumPartitions; i++)
    {
        localHistogram[offsets[i]] = keys[i];
    }
    GroupMemoryBarrierWithGroupSync();

    // binning
    for (uint i = idx; i < PartitionSize; i += WorkgroupSize)
    {
        uint key = localHistogram[i];
        uint bin = glsl::bitfieldExtract(key, params.bitShift, 8);
        uint dstOffset = histogramSums[bin] + i;
        if (dstOffset < params.numElements)
            setKey(outputBuffer, dstOffset, key);

#ifdef USE_KV_PAIRS
        keys[i / WorkgroupSize] = dstOffset;
#endif
    }

#ifdef USE_KV_PAIRS
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < NumPartitions; i++)
    {
        localHistogram[offsets[i]] = values[i];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = idx; i < PartitionSize; i += WorkgroupSize)
    {
        uint value = localHistogram[i];
        setValue(outputBuffer, keys[i / WorkgroupSize], value);
    }
#endif
}
