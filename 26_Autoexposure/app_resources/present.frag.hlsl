// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma wave shader_stage(fragment)

#include "app_resources/common.hlsl"

// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::combinedImageSampler]] [[vk::binding(0, 3)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 3)]] SamplerState samplerState;

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
    return texture.SampleLevel(samplerState, vxAttr.uv, 0.f);
}