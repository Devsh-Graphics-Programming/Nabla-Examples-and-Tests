// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] AutoexposurePushData pushData;

[numthreads(DeviceSubgroupSize, DeviceSubgroupSize, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
}