#ifndef _UTILS_HLSL_
#define _UTILS_HLSL_

// TODO: implemented somewhere else?
// Bit rotation helpers
uint32_t rotl(uint32_t value, uint32_t bits, uint32_t width)
{
    bits = bits % width;
    uint32_t mask = (1u << width) - 1u;
    value &= mask;
    return ((value << bits) | (value >> (width - bits))) & mask;
}

uint32_t rotr(uint32_t value, uint32_t bits, uint32_t width)
{
    bits = bits % width;
    uint32_t mask = (1u << width) - 1u;
    value &= mask;
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

#endif // _UTILS_HLSL_
