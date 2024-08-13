// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _AUTOEXPOSURE_COMMON_INCLUDED_
#define _AUTOEXPOSURE_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct AutoexposurePushData
{
    float meteringWindowScaleX, meteringWindowScaleY;
    float meteringWindowOffsetX, meteringWindowOffsetY;
    float lumaMin, lumaMax;
    uint32_t sampleCountX, sampleCountY;
    uint32_t viewportSizeX, viewportSizeY;
    uint64_t lumaMeterBDA;
};

#endif