// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma wave shader_stage(fragment)

// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl::ext::FullScreenTriangle;

#include "push_constants.hlsl"
    

[[vk::combinedImageSampler]][[vk::binding(0,3)]] Texture2DArray texture;
[[vk::combinedImageSampler]][[vk::binding(0,3)]] SamplerState samplerState;


[[vk::push_constant]] push_constants_t pc;

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
    const float32_t2 repeatCoord = vxAttr.uv*float32_t2(pc.grid);
    const int32_t layer = int32_t(repeatCoord.y)*pc.grid.x+int32_t(repeatCoord.x);
    return texture.Sample(samplerState,float32_t3(repeatCoord,layer));
}