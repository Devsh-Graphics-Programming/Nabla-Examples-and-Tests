//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/emulated_float64_t_test/common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<TestValues> testValuesOutput;

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    const emulated::emulated_float64_t a = emulated::emulated_float64_t::create(20.0f);
    const emulated::emulated_float64_t b = emulated::emulated_float64_t::create(10.0f);

    // "constructors"
    testValuesOutput[0].intCreateVal = emulated::emulated_float64_t::create(24).data;
    testValuesOutput[0].uintCreateVal = emulated::emulated_float64_t::create(24u).data;
    testValuesOutput[0].uint64CreateVal = emulated::emulated_float64_t::create(24ull).data;
    testValuesOutput[0].floatCreateVal = emulated::emulated_float64_t::create(1.2f).data;
    testValuesOutput[0].doubleCreateVal = emulated::emulated_float64_t::create(1.2).data;
    emulated::emulated_float64_t::create(min16int(2));

    // arithmetic operators
    testValuesOutput[0].additionVal = (a+b).data;
    testValuesOutput[0].substractionVal = (a-b).data;
    testValuesOutput[0].multiplicationVal = (a*b).data;
    testValuesOutput[0].divisionVal = (a/b).data;

    // relational operators
    testValuesOutput[0].lessOrEqualVal = (a<=b);
    testValuesOutput[0].greaterOrEqualVal = (a>=b);
    testValuesOutput[0].equalVal = (a==b);
    testValuesOutput[0].notEqualVal = (a!=b);
    testValuesOutput[0].lessVal = (a<b);
    testValuesOutput[0].greaterVal = (a>b);

    // conversion operators
    testValuesOutput[0].convertionToBoolVal = bool(a);
    testValuesOutput[0].convertionToIntVal = int(a);
    testValuesOutput[0].convertionToUint32Val = uint32_t(a);
    testValuesOutput[0].convertionToUint64Val = uint64_t(a);
    testValuesOutput[0].convertionToFloatVal = float(a);
    testValuesOutput[0].convertionToDoubleVal = float64_t(a);
    //testValuesOutput[0].convertionToHalfVal = half(a);
}
