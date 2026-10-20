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

[shader("pixel")]
float32_t4 PSMain(PSInput input) : SV_Target
{
    return float32_t4(input.color, 1.f);
}
