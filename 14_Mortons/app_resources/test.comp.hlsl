//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include "testCommon.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<InputTestValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<TestValues> outputTestValues;

[numthreads(1, 1, 1)]
[shader("compute")]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    uint32_t testID = glsl::gl_GlobalInvocationID().x;
    fillTestValues(inputTestValues[testID], outputTestValues[testID]);
}
