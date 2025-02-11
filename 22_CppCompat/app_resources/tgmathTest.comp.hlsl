//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<TgmathIntputTestValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<TgmathTestValues> outputTestValues;

[numthreads(256, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    if(invocationID.x == 0)
        outputTestValues[0].fillTestValues(inputTestValues[0]);
}
