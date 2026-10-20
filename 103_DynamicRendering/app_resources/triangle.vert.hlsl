// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

using namespace nbl::hlsl;

struct PSInput
{
    float32_t4 position : SV_Position;
    float32_t3 color : COLOR0;
};

[shader("vertex")]
PSInput VSMain(uint32_t vertexID : SV_VertexID)
{
    const float32_t2 positions[3] = {
        float32_t2( 0.0f,-0.5f),
        float32_t2( 0.5f, 0.5f),
        float32_t2(-0.5f, 0.5f)
    };
    const float32_t3 colors[3] = {
        float32_t3(1.f, 0.f, 0.f),
        float32_t3(0.f, 1.f, 0.f),
        float32_t3(0.f, 0.f, 1.f)
    };

    PSInput output;
    output.position = float32_t4(positions[vertexID], 0.f, 1.f);
    output.color = colors[vertexID];
    return output;
}
