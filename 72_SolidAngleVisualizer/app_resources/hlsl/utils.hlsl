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


#endif // _UTILS_HLSL_
