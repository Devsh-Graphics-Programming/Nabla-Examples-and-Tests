// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma wave shader_stage(fragment)

// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::binding(0, 3)]] Texture2D<float32_t2> warpMap;
[[vk::combinedImageSampler]][[vk::binding(1, 3)]] Texture2D<float32_t4> envMap;
[[vk::combinedImageSampler]][[vk::binding(1, 3)]] SamplerState envMapSampler;

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
    uint width;
    uint height;
    warpMap.GetDimensions(width, height);
    float32_t2 uv = warpMap.Load(uint32_t3(width * vxAttr.uv.x, height * vxAttr.uv.y, 0));
    float32_t4 color = envMap.Sample(envMapSampler, uv);

    return float32_t4(color.xyz, 1.0);
}
