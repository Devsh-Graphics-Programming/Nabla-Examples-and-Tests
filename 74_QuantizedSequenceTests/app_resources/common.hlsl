//// Copyright (C) 2023-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_74_QUANTIZED_SEQUENCE_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_74_QUANTIZED_SEQUENCE_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/sampling/quantized_sequence.hlsl>

using namespace nbl::hlsl;
struct QuantizedSequenceInputTestValues
{
    uint32_t3 uintVec3;

    uint32_t4 scrambleKey;
};

struct QuantizedSequenceTestValues
{
    uint32_t3 uintVec3;

    // pre decode scramble
    float32_t3 vec3_predecode;

    // post decode scramble
    uint32_t3 uintVec3_postdecode;
};

struct QuantizedSequenceTestExecutor
{
    void operator()(NBL_CONST_REF_ARG(QuantizedSequenceInputTestValues) input, NBL_REF_ARG(QuantizedSequenceTestValues) output)
    {
        {
            sampling::QuantizedSequence<uint32_t2, 3> qs = sampling::QuantizedSequence<uint32_t2, 3>::create(input.uintVec3);
            for (uint32_t i = 0; i < 3; i++)
                output.uintVec3[i] = qs.get(i);
        }
        
    }
};

#endif
