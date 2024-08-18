// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _BITONIC_SORT_COMMON_INCLUDED_
#define _BITONIC_SORT_COMMON_INCLUDED_

struct BitonicPushData
{
    uint64_t inputKeyAddress; // Address of the input key buffer
    uint64_t inputValueAddress; // Address of the input value buffer
    uint64_t outputKeyAddress; // Address of the output key buffer
    uint64_t outputValueAddress; // Address of the output value buffer
    uint32_t dataElementCount; // Total number of elements to be sorted
};

#endif
