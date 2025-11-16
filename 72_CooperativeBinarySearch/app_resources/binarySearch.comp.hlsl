// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma wave shader_stage(compute)

#include "common.h"
using namespace nbl::hlsl;

[[vk::push_constant]] ConstantBuffer<PushConstants> Constants;
[[vk::binding(0)]] StructuredBuffer<uint> Histogram;
[[vk::binding(1)]] RWStructuredBuffer<uint> Output;

static const uint32_t GroupsharedSize = 256;

[numthreads(256, 1, 1)]
void main(const uint3 thread : SV_DispatchThreadID, const uint3 groupThread : SV_GroupThreadID, const uint3 group : SV_GroupID)
{

}