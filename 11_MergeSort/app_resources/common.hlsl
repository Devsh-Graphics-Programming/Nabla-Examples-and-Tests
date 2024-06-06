// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _MERGE_SORT_COMMON_INCLUDED_
#define _MERGE_SORT_COMMON_INCLUDED_

struct MergeSortPushData
{
    uint64_t buffer_a_address;
    uint64_t buffer_b_address;
    uint64_t num_elements_per_array;
    uint64_t buffer_length;
};

#endif