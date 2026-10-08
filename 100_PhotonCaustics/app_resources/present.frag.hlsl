// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(fragment)

#include "app_resources/common.hlsl"

#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D    hdrTexture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState hdrSampler;
        
[[vk::push_constant]] SPresentPushConstants pc;

float32_t3 reinhard(float32_t3 x)
{
    return x / ((float32_t3)(1.0f) + x);
}

float32_t3 acesFilm(float32_t3 x)
{
    const float32_t a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
    const float32_t3 hdr = hdrTexture.Sample(hdrSampler, vxAttr.uv).rgb * pc.exposure;

    float32_t3 mapped;
    if (pc.tonemapOperator == 1)
        mapped = reinhard(hdr);
    else if (pc.tonemapOperator == 2)
        mapped = acesFilm(hdr);
    else
        mapped = hdr;

    return float32_t4(mapped, 1.0f);
}