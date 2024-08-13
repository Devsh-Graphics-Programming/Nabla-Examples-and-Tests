// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/luma_meter.hlsl"
#include "app_resources/common.hlsl"

// shared accross frag & compute - binding 0 set 3
[[vk::combinedImageSampler]] [[vk::binding(0, 3)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 3)]] SamplerState samplerState;

[[vk::push_constant]] AutoexposurePushData pushData;

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(DeviceSubgroupSize, DeviceSubgroupSize, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
}