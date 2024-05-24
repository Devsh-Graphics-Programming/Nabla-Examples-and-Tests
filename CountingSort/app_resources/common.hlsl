// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _COUNTING_SORT_COMMON_INCLUDED_
#define _COUNTING_SORT_COMMON_INCLUDED_

struct CountingPushData
{
    uint64_t inputKeyAddress;
    uint64_t inputValueAddress;
    uint64_t scratchAddress;
    uint64_t outputKeyAddress;
    uint64_t outputValueAddress;
    uint32_t dataElementCount;
    uint32_t elementsPerWT;
    uint32_t minimum;
    uint32_t maximum;
};

#endif