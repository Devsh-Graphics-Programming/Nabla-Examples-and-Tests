//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/emulated_float64_t_test/common.hlsl"
#include <nbl/builtin/hlsl/bit.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<TestValues<false, true> > testValuesOutput;

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    //const nbl::hlsl::emulated_float64_t<false, true> a = nbl::hlsl::emulated_float64_t<false, true>::create(float64_t(20.0));
    //const nbl::hlsl::emulated_float64_t<false, true> b = nbl::hlsl::emulated_float64_t<false, true>::create(float64_t(10.0));

    //// "constructors"
    //testValuesOutput[0].int32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(24).data;
    //testValuesOutput[0].int64CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(24).data;
    //testValuesOutput[0].uint32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(24u).data;
    //testValuesOutput[0].uint64CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(24ull).data;
    //testValuesOutput[0].float32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(1.2f).data;
    //testValuesOutput[0].float64CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(1.2).data;
    ////nbl::hlsl::emulated_float64_t::create(min16int(2));

    //// arithmetic operators
    //testValuesOutput[0].additionVal = (a+b).data;
    //testValuesOutput[0].substractionVal = (a-b).data;
    //testValuesOutput[0].multiplicationVal = (a*b).data;
    //testValuesOutput[0].divisionVal = (a/b).data;

    //// relational operators
    //testValuesOutput[0].lessOrEqualVal = int(a<=b);
    //testValuesOutput[0].greaterOrEqualVal = int(a>=b);
    //testValuesOutput[0].equalVal = int(a==b);
    //testValuesOutput[0].notEqualVal = int(a!=b);
    //testValuesOutput[0].lessVal = int(a<b);
    //testValuesOutput[0].greaterVal = int(a>b);

    const nbl::hlsl::emulated_float64_t<false, true> a = nbl::hlsl::emulated_float64_t<false, true>::create(float64_t(20.0));
    const nbl::hlsl::emulated_float64_t<false, true> b = nbl::hlsl::emulated_float64_t<false, true>::create(float64_t(10.0));

    // "constructors"
    testValuesOutput[0].int32CreateVal = 2;
}
