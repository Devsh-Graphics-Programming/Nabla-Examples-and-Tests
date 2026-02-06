#ifndef _THIS_EXAMPLE_FALSE_COLOR_HLSL_INCLUDED_
#define _THIS_EXAMPLE_FALSE_COLOR_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/tgmath.hlsl"

namespace nbl
{
namespace hlsl
{
namespace this_example
{
namespace ies
{

NBL_CONSTEXPR_STATIC_INLINE uint32_t FalseColorStopCount = 6u;

inline float32_t falseColorStop(uint32_t idx)
{
    switch (idx)
    {
        case 0u: return 0.0f;
        case 1u: return 0.15f;
        case 2u: return 0.35f;
        case 3u: return 0.55f;
        case 4u: return 0.75f;
        default: return 1.0f;
    }
}

inline float32_t3 falseColor(float32_t v)
{
    v = nbl::hlsl::clamp(v, float32_t(0.0f), float32_t(1.0f));
    v = nbl::hlsl::pow(v, float32_t(0.8f));

    const float32_t3 c0 = float32_t3(0.0f, 0.0f, 0.0f);
    const float32_t3 c1 = float32_t3(0.0f, 0.0f, 0.35f);
    const float32_t3 c2 = float32_t3(0.10f, 0.20f, 0.90f);
    const float32_t3 c3 = float32_t3(0.70f, 0.05f, 0.80f);
    const float32_t3 c4 = float32_t3(1.00f, 0.30f, 1.00f);
    const float32_t3 c5 = float32_t3(1.00f, 1.00f, 1.00f);

    if (v < 0.15f)
    {
        const float32_t t = v / 0.15f;
        return c0 + (c1 - c0) * t;
    }
    else if (v < 0.35f)
    {
        const float32_t t = (v - 0.15f) / (0.35f - 0.15f);
        return c1 + (c2 - c1) * t;
    }
    else if (v < 0.55f)
    {
        const float32_t t = (v - 0.35f) / (0.55f - 0.35f);
        return c2 + (c3 - c2) * t;
    }
    else if (v < 0.75f)
    {
        const float32_t t = (v - 0.55f) / (0.75f - 0.55f);
        return c3 + (c4 - c3) * t;
    }
    else
    {
        const float32_t t = (v - 0.75f) / (1.0f - 0.75f);
        return c4 + (c5 - c4) * t;
    }
}

}
}
}
}

#endif
