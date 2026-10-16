//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_UTILS_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_UTILS_HLSL_INCLUDED_
#include <nbl/builtin/hlsl/bit.hlsl>
#include <nbl/builtin/hlsl/random/pcg.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

// unused
uint32_t packSilhouette(const uint32_t s[7])
{
    uint32_t packed = 0;
    uint32_t size = s[0] & 0x7; // 3 bits for size

    // Pack vertices LSB-first (vertex1 in lowest 3 bits above size)
    for (uint32_t i = 1; i <= 6; ++i)
    {
        uint32_t v = s[i];
        if (v < 0)
            v = 0;                            // replace unused vertices with 0
        packed |= (v & 0x7) << (3 * (i - 1)); // vertex i-1 shifted by 3*(i-1)
    }

    // Put size in the MSB (bits 29-31 for a 32-bit uint32_t, leaving 29 bits for vertices)
    packed |= (size & 0x7) << 29;

    return packed;
}

#endif // _SOLID_ANGLE_VIS_EXAMPLE_UTILS_HLSL_INCLUDED_
