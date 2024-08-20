#pragma kernel localRadixSort
#pragma kernel globalMerge

#include "../../common.hlsl"

struct SSortParams
{
    uint numElements;
    uint numGroups;

    uint groupOffset;
    uint bitShift;
    
    uint keyType;
    uint sortingOrder;
};

#define GET_KEY(s) s.x

[[vk::binding(1, 0)]]
cbuffer SortParams
{
    SSortParams params;
};

[[vk::binding(0, 1)]] RWStructuredBuffer<uint2> inputBuffer;
[[vk::binding(1, 1)]] RWStructuredBuffer<uint2> outputBuffer;

[[vk::binding(2, 1)]] RWStructuredBuffer<uint> firstIdxBuffer;
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> groupSumBuffer;
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> globalPrefixSumBuffer;

static const uint numElemsPerGroup = NUM_THREADS;
static const uint logNumElemsPerGroup = 7; //log2(numElemsPerGroup);
static const uint numElemsPerGroup1 = numElemsPerGroup - 1u;

static const uint nWay = 16u;
static const uint nWay1 = nWay - 1u;

static const uint sharedDataLen = numElemsPerGroup;
static const uint sharedScanLen = numElemsPerGroup;
static const uint sharedPdLen = nWay;

groupshared uint2 sharedData[sharedDataLen];
groupshared uint4 sharedScan[sharedScanLen];
groupshared uint sharedPd[sharedPdLen];

inline uint getKey4Bit(uint2 data)
{
    uint key = GET_KEY(data);
    if (params.sortingOrder == 1)
        key = ~key;

    return (key >> params.bitShift) & nWay1;
}

inline uint getValueUint16(uint4 value, uint key)
{
    return (value[key % 4u] >> (key / 4u * 8u)) & 0x000000ffu;
}
inline void setValueUint16(inout uint4 uint16_value, uint value, uint key)
{
    uint16_value[key % 4u] += value << (key / 4u * 8u);
}
inline uint4 buildSharedScanData(uint key)
{
    return (uint4)(key % 4u == uint4(0u, 1u, 2u, 3u)) << ((key / 4u) * 8u);
}


[numthreads(WorkgroupSize, 1, 1)]
void localRadixSort(uint threadID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    groupID += params.groupOffset;
    uint globalID = numElemsPerGroup * groupID + threadID;

    uint2 data = (uint2)0;
    uint key4Bit = nWay1;
    if (globalID < params.numElements)
    {
        data = inputBuffer[globalID];
        key4Bit = getKey4Bit(data);
    }

    sharedScan[threadID] = buildSharedScanData(key4Bit);
    GroupMemoryBarrierWithGroupSync();

    // build prefix sum
    [unroll(logNumElemsPerGroup)]
    for (uint offset = 1u;; offset <<= 1u)
    {
        uint4 sum = sharedScan[threadID];
        if (threadID >= offset)
        {
            sum += sharedScan[threadID - offset];
        }
        GroupMemoryBarrierWithGroupSync();
        sharedScan[threadID] = sum;
        GroupMemoryBarrierWithGroupSync();
    }

    uint4 total = sharedScan[numElemsPerGroup1];
    uint4 firstIdx = 0;
    uint runningSum = 0;

    [unroll(nWay)]
    for (uint i = 0;; i++)
    {
        setValueUint16(firstIdx, runningSum, i);
        runningSum += getValueUint16(total, i);
    }

    if (threadID < nWay)
    {
        groupSumBuffer[threadID * params.numGroups + groupID] = getValueUint16(total, threadID);
        firstIdxBuffer[threadID + nWay * groupID] = getValueUint16(firstIdx, threadID);
    }

    uint newID = getValueUint16(firstIdx, key4Bit);
    if (threadID > 0)
    {
        newID += getValueUint16(sharedScan[threadID - 1], key4Bit);
    }
    sharedData[newID] = data;
    GroupMemoryBarrierWithGroupSync();

    if (globalID < params.numElements)
    {
        outputBuffer[globalID] = sharedData[threadID];
    }
}

[numthreads(WorkgroupSize, 1, 1)]
void globalMerge(uint threadID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    groupID += params.groupOffset;
    uint globalID = numElemsPerGroup * groupID + threadID;

    if (threadID < nWay)
    {
        sharedPd[threadID] = globalPrefixSumBuffer[threadID * params.numGroups + groupID];
        sharedPd[threadID] -= firstIdxBuffer[threadID + nWay * groupID];
    }
    GroupMemoryBarrierWithGroupSync();

    if (globalID < params.numElements)
    {
        uint2 data = inputBuffer[globalID];
        uint key4Bit = getKey4Bit(data);

        uint newID = threadID + sharedPd[key4Bit];

        outputBuffer[newID] = data;
    }
}
