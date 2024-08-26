//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/emulated_float64_t_test/common.hlsl"
#include <nbl/builtin/hlsl/bit.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<TestValues<false, true> > testValuesOutput;

[[vk::push_constant]]
PushConstants pc;

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    const nbl::hlsl::emulated_float64_t<false, true> a = nbl::hlsl::bit_cast<emulated_float64_t<false, true> >(pc.a);
    const nbl::hlsl::emulated_float64_t<false, true> b = nbl::hlsl::bit_cast<emulated_float64_t<false, true> >(pc.b);

    // "constructors"
    testValuesOutput[0].int32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.int32).data;
    testValuesOutput[0].int64CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.int64).data;
    testValuesOutput[0].uint32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.uint32).data;
    testValuesOutput[0].uint64CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.uint64).data;
    testValuesOutput[0].float32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.float32).data;
    //testValuesOutput[0].float64CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.float64).data;
    //nbl::hlsl::emulated_float64_t::create(min16int(2));

    portable_float64_t<> asdf = nbl::hlsl::emulated_float64_t<false, true>::create(pc.constrTestVals.float32);
    float asdf2 = nbl::hlsl::_static_cast<float>(asdf);
    testValuesOutput[0].float32CreateVal = nbl::hlsl::emulated_float64_t<false, true>::create(asdf2).data;

    emulated_vector_t2<emulated_float64_t<false, true> > vec;
    vec.x = emulated_float64_t<false, true>::create(1.5f);
    vec.y = emulated_float64_t<false, true>::create(1.25f);

    float2 asdf3 = nbl::hlsl::_static_cast<float2>(vec);

    testValuesOutput[0].int32CreateVal = asdf3;
    testValuesOutput[0].int64CreateVal = asdf3;

    // arithmetic operators
    testValuesOutput[0].additionVal = (a+b).data;
    testValuesOutput[0].substractionVal = (a-b).data;
    testValuesOutput[0].multiplicationVal = (a*b).data;
    testValuesOutput[0].divisionVal = (a/b).data;

    // relational operators
    testValuesOutput[0].lessOrEqualVal = int(a<=b);
    testValuesOutput[0].greaterOrEqualVal = int(a>=b);
    testValuesOutput[0].equalVal = int(a==b);
    testValuesOutput[0].notEqualVal = int(a!=b);
    testValuesOutput[0].lessVal = int(a<b);
    testValuesOutput[0].greaterVal = int(a>b);
}
