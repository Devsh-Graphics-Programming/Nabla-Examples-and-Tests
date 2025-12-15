//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "common.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<IntrinsicsIntputTestValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<IntrinsicsTestValues> outputTestValues;

[numthreads(WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main()
{
    const uint invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
    IntrinsicsTestExecutor executor;
    executor(inputTestValues[invID], outputTestValues[invID]);
}