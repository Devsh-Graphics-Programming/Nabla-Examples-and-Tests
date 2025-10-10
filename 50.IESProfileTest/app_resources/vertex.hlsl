// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "VSInput.hlsl"
#include "PSInput.hlsl"

[shader("vertex")]
PSInput VSMain(VSInput input)
{
    PSInput output;
    output.position = float4(input.position, 1.f);

    return output;
}