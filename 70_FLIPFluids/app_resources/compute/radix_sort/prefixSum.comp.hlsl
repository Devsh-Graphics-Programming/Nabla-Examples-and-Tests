#pragma kernel prefixSum
#pragma kernel addGroupSum

#include "../../common.hlsl"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define SHARED_MEMORY_ADDRESS(n) ((n) + CONFLICT_FREE_OFFSET(n))

struct SPrefixSumParams
{
    uint numElements;
    uint groupOffset;
    uint groupSumOffset;
    uint pad;
}

[[vk::binding(1, 0)]]
cbuffer PrefixSumParams
{
    SPrefixSumParams params;
};

[[vk::binding(0, 1)]] RWStructuredBuffer<uint> dataBuffer;
[[vk::binding(1, 1)]] RWStructuredBuffer<uint> groupSumBuffer;

static const uint numGroupThreads = WorkgroupSize;
static const uint numElemsPerGroup = 2u * WorkgroupSize;
static const uint numElemsPerGroup1 = numElemsPerGroup - 1u;
static const uint logNumElemsPerGroup = log2(numElemsPerGroup);
static const uint smaNumElemsPerGroup1 = SHARED_MEMORY_ADDRESS(numElemsPerGroup1);

static const uint sharedSumLen = SHARED_MEMORY_ADDRESS(numElemsPerGroup);

groupshared uint sharedSum[sharedSumLen];

[numthreads(WorkgroupSize, 1, 1)]
void prefixSum(uint threadId : SV_GroupThreadID, uint groupId : SV_GroupID)
{
    groupId += params.groupOffset;

    uint startIdx = threadId;
    uint endIdx = startIdx + numGroupThreads;

    startIdx = SHARED_MEMORY_ADDRESS(startIdx);
    endIdx = SHARED_MEMORY_ADDRESS(endIdx);

    uint globalStartIdx = threadId + numElemsPerGroup * groupId;
    uint globalEndIdx = globalStartIdx + numGroupThreads;

    sharedSum[startIdx] = globalStartIdx < params.numElements ? dataBuffer[globalStartIdx] : 0;
    sharedSum[EndIdx] = globalEndIdx < params.numElements ? dataBuffer[globalEndIdx] : 0;

    uint offset = 1u;

    [unroll(logNumElemsPerGroup)]
    for (uint du = numElemsPerGroup >> 1u;; du >>= 1u)
    {
        GroupMemoryBarrierWithGroupSync();

        if (threadId < du)
        {
            uint startIdx_u = offset * ((threadId << 1u) + 1u) - 1u;
            uint endIdx_u = offset * ((threadId << 1u) + 2u) - 1u;

            startIdx_u = SHARED_MEMORY_ADDRESS(startIdx_u);
            endIdx_u = SHARED_MEMORY_ADDRESS(endIdx_u);

            sharedSum[endIdx_u] += sharedSum[startIdx_u];
        }
        offset <<= 1u;
    }

    if (threadId == 0u)
    {
        groupSumBuffer[groupId + params.groupSumOffset] = sharedSum[smaNumElemsPerGroup1];
        sharedSum[smaNumElemsPerGroup1] = 0;
    }

    [unroll(logNumElemsPerGroup)]
    for (uint dd = 1u;; dd <<= 1u)
    {
        offset >>= 1u;

        GroupMemoryBarrierWithGroupSync();

        if (threadId < du)
        {
            uint startIdx_d = offset * ((threadId << 1u) + 1u) - 1u;
            uint endIdx_d = offset * ((threadId << 1u) + 2u) - 1u;

            startIdx_d = SHARED_MEMORY_ADDRESS(startIdx_d);
            endIdx_d = SHARED_MEMORY_ADDRESS(endIdx_d);

            uint tmp = sharedSum[startIdx_d];
            sharedSum[startIdx_d] = sharedSum[endIdx_d];
            sharedSum[endIdx_d] += tmp;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (globalStartIdx < params.numElements)
        dataBuffer[globalStartIdx] = sharedSum[startIdx];
    if (globalEndIdx < params.numElements)
        dataBuffer[globalEndIdx] = sharedSum[endIdx];
}

[numthreads(WorkgroupSize, 1, 1)]
void addGroupSum(uint threadId : SV_GroupThreadID, uint groupId : SV_GroupID)
{
    groupId += params.groupOffset;

    uint groupSum = groupSumBuffer[groupId];

    uint globalStartIdx = threadId + numElemsPerGroup * groupId;
    uint globalEndIdx = globalStartIdx + numGroupThreads;

    if (globalStartIdx < params.numElements)
        dataBuffer[globalStartIdx] += groupSum;
    if (globalEndIdx < params.numElements)
        dataBuffer[globalEndIdx] += groupSum;
}
