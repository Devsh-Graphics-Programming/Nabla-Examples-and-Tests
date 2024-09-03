#ifndef _FLIP_EXAMPLE_SORT_COMMON_HLSL
#define _FLIP_EXAMPLE_SORT_COMMON_HLSL

#define SORT_WORKGROUP_SIZE 256
#define NUM_SORT_BINS 256

#ifdef __HLSL_VERSION

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

#ifndef DATA_TYPE
#define DATA_TYPE uint2
#endif

#define GET_KEY(s) s.x

static const uint WorkgroupSize = SORT_WORKGROUP_SIZE;
static const uint NumSortBins = NUM_SORT_BINS;
static const uint SubgroupSize = 32;    // 32 = nv warp size, 64 = amd wavefront

struct SSortParams
{
    uint numElements;
    uint bitShift;
    uint numWorkgroups;
    uint numThreadsPerGroup;
};

#endif
#endif