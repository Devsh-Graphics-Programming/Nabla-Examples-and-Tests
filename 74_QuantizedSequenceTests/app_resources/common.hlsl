//// Copyright (C) 2023-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_74_QUANTIZED_SEQUENCE_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_74_QUANTIZED_SEQUENCE_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/sampling/quantized_sequence.hlsl>

using namespace nbl::hlsl;
struct QuantizedSequenceInputTestValues
{
    uint32_t scalar;
    uint32_t2 uvec2;
    uint32_t3 uvec3;
    uint32_t4 uvec4;

    uint16_t scalar16;
    uint16_t2 u16vec2;
    uint16_t3 u16vec3;
    uint16_t4 u16vec4;

    float32_t3 unorm3;

    uint32_t3 scrambleKey3;
};

struct QuantizedSequenceTestValues
{
    uint32_t uintDim1;
    uint32_t2 uintDim2;
    uint32_t3 uintDim3;
    uint32_t4 uintDim4;

    uint32_t2 uintVec2_Dim2;
    uint32_t3 uintVec2_Dim3;

    uint32_t3 uintVec3_Dim3;
    uint32_t4 uintVec4_Dim4;

    uint16_t u16Dim1;
    uint16_t2 u16Dim2;
    uint16_t3 u16Dim3;
    uint16_t4 u16Dim4;

    uint16_t2 u16Vec2_Dim2;
    uint16_t4 u16Vec2_Dim4;

    uint16_t3 u16Vec3_Dim3;
    uint16_t4 u16Vec4_Dim4;

    // pre decode scramble
    float32_t3 unorm3_predecode;

    // post decode scramble
    float32_t3 unorm3_postdecode;
};

struct QuantizedSequenceTestExecutor
{
    void operator()(NBL_CONST_REF_ARG(QuantizedSequenceInputTestValues) input, NBL_REF_ARG(QuantizedSequenceTestValues) output)
    {
        // test get/set/create
        {
            sampling::QuantizedSequence<uint32_t, 1> qs = sampling::QuantizedSequence<uint32_t, 1>::create(input.scalar);
            output.uintDim1 = qs.get(0);
        }
        {
            sampling::QuantizedSequence<uint32_t, 2> qs = sampling::QuantizedSequence<uint32_t, 2>::create(input.uvec2);
            for (uint32_t i = 0; i < 2; i++)
                output.uintDim2[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint32_t, 3> qs = sampling::QuantizedSequence<uint32_t, 3>::create(input.uvec3);
            for (uint32_t i = 0; i < 3; i++)
                output.uintDim3[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint32_t, 4> qs = sampling::QuantizedSequence<uint32_t, 4>::create(input.uvec4);
            for (uint32_t i = 0; i < 4; i++)
                output.uintDim4[i] = qs.get(i);
        }

        {
            sampling::QuantizedSequence<uint32_t2, 2> qs = sampling::QuantizedSequence<uint32_t2, 2>::create(input.uvec2);
            for (uint32_t i = 0; i < 2; i++)
                output.uintVec2_Dim2[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint32_t2, 3> qs = sampling::QuantizedSequence<uint32_t2, 3>::create(input.uvec3);
            for (uint32_t i = 0; i < 3; i++)
                output.uintVec2_Dim3[i] = qs.get(i);
        }

        {
            sampling::QuantizedSequence<uint32_t3, 3> qs = sampling::QuantizedSequence<uint32_t3, 3>::create(input.uvec3);
            for (uint32_t i = 0; i < 3; i++)
                output.uintVec3_Dim3[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint32_t4, 4> qs = sampling::QuantizedSequence<uint32_t4, 4>::create(input.uvec4);
            for (uint32_t i = 0; i < 4; i++)
                output.uintVec4_Dim4[i] = qs.get(i);
        }

        // u16
        {
            sampling::QuantizedSequence<uint16_t, 1> qs = sampling::QuantizedSequence<uint16_t, 1>::create(input.scalar16);
            output.u16Dim1 = qs.get(0);
        }
        {
            sampling::QuantizedSequence<uint16_t, 2> qs = sampling::QuantizedSequence<uint16_t, 2>::create(input.u16vec2);
            for (uint32_t i = 0; i < 2; i++)
                output.u16Dim2[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint16_t, 3> qs = sampling::QuantizedSequence<uint16_t, 3>::create(input.u16vec3);
            for (uint32_t i = 0; i < 3; i++)
                output.u16Dim3[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint16_t, 4> qs = sampling::QuantizedSequence<uint16_t, 4>::create(input.u16vec4);
            for (uint32_t i = 0; i < 4; i++)
                output.u16Dim4[i] = qs.get(i);
        }

        {
            sampling::QuantizedSequence<uint16_t2, 2> qs = sampling::QuantizedSequence<uint16_t2, 2>::create(input.u16vec2);
            for (uint32_t i = 0; i < 2; i++)
                output.u16Vec2_Dim2[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint16_t2, 4> qs = sampling::QuantizedSequence<uint16_t2, 4>::create(input.u16vec4);
            for (uint32_t i = 0; i < 4; i++)
                output.u16Vec2_Dim4[i] = qs.get(i);
        }

        {
            sampling::QuantizedSequence<uint16_t3, 3> qs = sampling::QuantizedSequence<uint16_t3, 3>::create(input.u16vec3);
            for (uint32_t i = 0; i < 3; i++)
                output.u16Vec3_Dim3[i] = qs.get(i);
        }
        {
            sampling::QuantizedSequence<uint16_t4, 4> qs = sampling::QuantizedSequence<uint16_t4, 4>::create(input.u16vec4);
            for (uint32_t i = 0; i < 4; i++)
                output.u16Vec4_Dim4[i] = qs.get(i);
        }

        // test encode/decode
        {
            sampling::QuantizedSequence<uint32_t2, 3> qs = sampling::QuantizedSequence<uint32_t2, 3>::template encode<float, true>(input.unorm3);
            output.unorm3_predecode = qs.template decode<float>(input.scrambleKey3);
        }
        {
            sampling::QuantizedSequence<uint32_t2, 3> qs = sampling::QuantizedSequence<uint32_t2, 3>::template encode<float, false>(input.unorm3);
            sampling::QuantizedSequence<uint32_t2, 3> key = sampling::QuantizedSequence<uint32_t2, 3>::create(input.scrambleKey3);
            output.unorm3_postdecode = qs.template decode<float>(key);
        }
    }
};

#endif
