//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include "testCommon.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<InputTestValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<TestValues> outputTestValues;

[numthreads(WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    const uint invID = invocationID.x;
    if (invID >= TEST_COUNT)
        return;

    TestExecutor executor;
    executor(inputTestValues[invID], outputTestValues[invID]);
}
