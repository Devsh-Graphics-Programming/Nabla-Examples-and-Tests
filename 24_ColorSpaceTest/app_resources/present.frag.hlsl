// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(fragment)

// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

[[vk::combinedImageSampler]][[vk::binding(0,3)]] Texture2DArray texture;
[[vk::combinedImageSampler]][[vk::binding(0,3)]] SamplerState samplerState;

[[vk::location(0)]] float32_t4 main(nbl::hlsl::ext::FullScreenTriangle::SVertexAttributes vxAttr)
{
    pixelColor = texture.SampleLevel(samplerState,float32_t3(vxAttr.uv,0.f),0.0);
}