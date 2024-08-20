// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma wave shader_stage(fragment)

#include "nbl/builtin/hlsl/colorspace/EOTF.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/colorspace/OETF.hlsl"
#include "nbl/builtin/hlsl/tonemapper/operators.hlsl"
#include "app_resources/common.hlsl"

// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

// binding 0 set 1
[[vk::combinedImageSampler]] [[vk::binding(0, 1)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 1)]] SamplerState samplerState;

[[vk::push_constant]] AutoexposurePushData pushData;

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
    float32_t3 color = colorspace::oetf::sRGB(texture.Sample(samplerState, vxAttr.uv).rgb);
    float32_t3 CIEColor = mul(colorspace::sRGBtoXYZ, color);

    tonemapper::Reinhard<float32_t> reinhard = tonemapper::Reinhard<float32_t>::create(pushData.EV);

    float32_t3 tonemappedColor = mul(colorspace::decode::XYZtoscRGB, reinhard(CIEColor));

    return float32_t4(colorspace::eotf::sRGB(tonemappedColor), 1.0);
}
