// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _COUNTING_SORT_COMMON_INCLUDED_
#define _COUNTING_SORT_COMMON_INCLUDED_

struct CountingPushData
{
    uint64_t inputKeyAddress;
    uint64_t inputValueAddress;
    uint64_t histogramAddress;
    uint64_t outputKeyAddress;
    uint64_t outputValueAddress;
    uint32_t dataElementCount;
    uint32_t elementsPerWT;
    uint32_t minimum;
    uint32_t maximum;
};

using namespace nbl::hlsl;

#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"

static const uint32_t WorkgroupSize = DeviceConfigCaps::maxComputeWorkGroupInvocations;
static const uint32_t MaxBucketCount = (DeviceConfigCaps::maxComputeSharedMemorySize / sizeof(uint32_t)) / 2;
static const uint32_t BucketCount = (MaxBucketCount > 3000) ? 3000 : MaxBucketCount;

using Ptr = bda::__ptr<uint32_t>;
using PtrAccessor = BdaAccessor<uint32_t>;

groupshared uint32_t sdata[BucketCount];

struct SharedAccessor
{
    void get(const uint32_t index, NBL_REF_ARG(uint32_t) value)
    {
        value = sdata[index];
    }

    void set(const uint32_t index, const uint32_t value)
    {
        sdata[index] = value;
    }

    uint32_t atomicAdd(const uint32_t index, const uint32_t value)
    {
        return glsl::atomicAdd(sdata[index], value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

uint32_t3 glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}


#endif

#endif