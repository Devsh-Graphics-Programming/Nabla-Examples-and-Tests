//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "testCommon2.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<InputTestValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<TestValues> outputTestValues;

[numthreads(256, 1, 1)]
[shader("compute")]
void main()
{
    const uint invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
    TestExecutor2 executor;
    executor(inputTestValues[invID], outputTestValues[invID]);
}
 
