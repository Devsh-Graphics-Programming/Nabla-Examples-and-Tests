#ifndef _FLIP_EXAMPLE_SORT_COMMON_HLSL
#define _FLIP_EXAMPLE_SORT_COMMON_HLSL

#define SORT_WORKGROUP_SIZE 512   // sort step
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

inline uint getKey(DATA_TYPE data)
{
    uint key;
#if defined(DATA_TYPE_UINT)
    key = data;
#elif defined(DATA_TYPE_UINT2)
    key = data.x;
#elif defined(DATA_TYPE_INT)
    // TODO: convert int to uint
#elif defined(DATA_TYPE_FLOAT)
    // TODO: convert float to uint
#endif
}

inline uint bitFieldExtract(uint data, uint offset, uint numBits)
{
    uint mask = (1u << numBits) - 1u;
    return (data >> offset) & mask;
}

#endif
#endif