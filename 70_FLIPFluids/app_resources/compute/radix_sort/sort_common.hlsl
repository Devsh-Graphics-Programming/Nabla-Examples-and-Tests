#ifndef _FLIP_EXAMPLE_SORT_COMMON_HLSL
#define _FLIP_EXAMPLE_SORT_COMMON_HLSL

#define SORT_WORKGROUP_SIZE 512   // sort step
#define SUBGROUP_SIZE 32    // want it to be 32 even on amd (rdna)
#define NUM_SORT_BINS 256
#define NUM_PARTITIONS 8

#ifdef __HLSL_VERSION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

// defaults to data type unit2
// valid types: uint, uint2, int, float
#if !defined(DATA_TYPE_UINT) || !defined(DATA_TYPE_UINT2) || !defined(DATA_TYPE_INT) || !defined(DATA_TYPE_FLOAT)
#define DATA_TYPE_UINT2
#endif

#if defined(DATA_TYPE_UINT)
#define DATA_TYPE uint
#elif defined(DATA_TYPE_UINT2)
#define DATA_TYPE uint2
#elif defined(DATA_TYPE_INT)
#define DATA_TYPE int
#elif defined(DATA_TYPE_FLOAT)
#define DATA_TYPE float
#endif

#if defined(DATA_TYPE_UINT2)
#define USE_KV_PAIRS
#endif

static const uint WorkgroupSize = SORT_WORKGROUP_SIZE;
static const uint SubgroupSize = SUBGROUP_SIZE;
static const uint NumSortBins = NUM_SORT_BINS;

static const uint NumPartitions = NUM_PARTITIONS;
static const uint PartitionSize = NumPartitions * WorkgroupSize;

struct SSortParams
{
    uint numElements;
    uint bitShift;
    uint numWorkgroups;
    uint numThreadsPerGroup;
};

inline uint floatToUint(float f)
{
    uint mask = -((int) (asuint(f) >> 31)) | 0x80000000;
    return asuint(f) ^ mask;
}

inline float uintToFloat(uint u)
{
    uint mask = ((u >> 31) - 1) | 0x80000000;
    return asfloat(u ^ mask);
}

inline uint intToUint(int i)
{
    return asuint(i ^ 0x80000000);
}

inline int uintToInt(uint u)
{
    return asint(u ^ 0x80000000);
}

inline uint getKey(RWStructuredBuffer<DATA_TYPE> buffer, uint idx)
{
    uint key;
#if defined(DATA_TYPE_UINT)
    key = buffer[idx];
#elif defined(DATA_TYPE_UINT2)
    key = buffer[idx].x;
#elif defined(DATA_TYPE_INT)
    key = intToUint(buffer[idx]);
#elif defined(DATA_TYPE_FLOAT)
    key = floatToUint(buffer[idx]);
#endif

    return key;
}

inline void setKey(RWStructuredBuffer<DATA_TYPE> buffer, uint idx, uint key)
{
#if defined(DATA_TYPE_UINT)
    buffer[idx] = key;
#elif defined(DATA_TYPE_UINT2)
    buffer[idx].x = key;
#elif defined(DATA_TYPE_INT)
    buffer[idx] = uintToInt(key);
#elif defined(DATA_TYPE_FLOAT)
    buffer[idx] = uintToFloat(key);
#endif
}

#ifdef USE_KV_PAIRS
inline uint getValue(RWStructuredBuffer<DATA_TYPE> buffer, uint idx)
{
    return buffer[idx].y;
}

inline void setValue(RWStructuredBuffer<DATA_TYPE> buffer, uint idx, uint value)
{
    buffer[idx].y = value;
}
#endif

#endif
#endif