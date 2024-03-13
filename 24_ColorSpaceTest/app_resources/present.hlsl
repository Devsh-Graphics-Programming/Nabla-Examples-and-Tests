// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// vertex shader is provided by the fullScreenTriangle extension
#pragma wave shader_stage(fragment)

[[vk::combinedImageSampler]][[vk::binding(0,3)]] Texture2DArray texture;
[[vk::combinedImageSampler]][[vk::binding(0,3)]] SamplerState samplerState;

// layout(location = 0) in vec2 TexCoord;
// layout(location = 0) out vec4 pixelColor;

void main()
{
//    pixelColor = textureLod(tex0,TexCoord,0.0);
}