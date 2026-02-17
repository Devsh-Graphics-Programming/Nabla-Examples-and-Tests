//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_UTILS_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_UTILS_HLSL_INCLUDED_
#include <nbl/builtin/hlsl/random/pcg.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

// TODO: implemented somewhere else?
// Bit rotation helpers
uint32_t rotl(uint32_t value, uint32_t bits, uint32_t width)
{
    // mask for the width
    uint32_t mask = (width == 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
    value &= mask;

    // Map bits==width -> 0
    bits &= -(bits < width);

    return ((value << bits) | (value >> (width - bits))) & mask;
}

uint32_t rotr(uint32_t value, uint32_t bits, uint32_t width)
{
    uint32_t mask = ((1u << width) - 1u);
    value &= mask;

    // Map bits==width -> 0
    bits &= -(bits < width);

    return ((value >> bits) | (value << (width - bits))) & mask;
}

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

float32_t2 hammersleySample(uint32_t i, uint32_t numSamples)
{
    return float32_t2(
        float32_t(i) / float32_t(numSamples),
        float32_t(reversebits(i)) / 4294967295.0f);
}

float32_t2 nextRandomUnorm2(inout nbl::hlsl::Xoroshiro64StarStar rnd)
{
    return float32_t2(
        float32_t(rnd()) * 2.3283064365386963e-10,
        float32_t(rnd()) * 2.3283064365386963e-10);
}

#endif // _SOLID_ANGLE_VIS_EXAMPLE_UTILS_HLSL_INCLUDED_
