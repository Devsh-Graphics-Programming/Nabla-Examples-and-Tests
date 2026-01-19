// Copyright (C) 2024-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "renderer/shaders/present/push_constants.hlsl"
// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace ext::FullScreenTriangle;

[[vk::binding(0)]] Texture2DArray images[DefaultResolvePushConstants::ImageCount];
[[vk::binding(1)]] SamplerState samplerState;

[[vk::push_constant]] DefaultResolvePushConstants pc;

[shader("pixel")]
float32_t4 present_default(SVertexAttributes vxAttr) : SV_Target0
{
    return float32_t4(images[pc.imageIndex].SampleLevel(samplerState,float32_t3(vxAttr.uv,0.f),0.f).rgb,1.0f);
}
