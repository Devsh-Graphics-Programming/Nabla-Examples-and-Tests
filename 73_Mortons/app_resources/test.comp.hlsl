//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include "testCommon.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<InputTestValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<TestValues> outputTestValues;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    if (invocationID.x == 0)
        fillTestValues(inputTestValues[0], outputTestValues[0]);
}
