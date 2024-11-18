#ifndef _FLIP_EXAMPLE_KERNEL_HLSL
#define _FLIP_EXAMPLE_KERNEL_HLSL

#ifdef __HLSL_VERSION

float getWeight(float3 pPos, float3 cPos, float invSpacing)
{
    float3 dist = abs((pPos - cPos) * invSpacing);
    float3 weight = saturate(1.0f - dist);
    return weight.x * weight.y * weight.z;
}

#endif
#endif