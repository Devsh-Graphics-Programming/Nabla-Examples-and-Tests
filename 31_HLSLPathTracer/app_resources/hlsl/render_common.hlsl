#ifndef _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACER_RENDER_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifndef __HLSL_VERSION
#include "matrix4SIMD.h"
#endif

struct RenderPushConstants
{
#ifdef __HLSL_VERSION
    float32_t4x4 invMVP;
#else
    nbl::hlsl::float32_t4x4 invMVP;
#endif
    int sampleCount;
    int depth;
    uint64_t pSampleSequence;
};

struct QuantizedSequence
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Bits = 21u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t XZMask = (0x1u << Bits) - 1u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t YMask = (0x1u << (32u-Bits)) - 1u;

    uint32_t getX() { return data[0] & XZMask; }
    uint32_t getY()
    {
        uint32_t y = data[0] >> Bits;
        y |= (data[1] >> Bits) << (32u-Bits);
        return y;
    }
    uint32_t getZ() { return data[1] & XZMask; }

    void setX(uint32_t x)
    {
        data[0] &= ~XZMask;
        data[0] |= x & XZMask;
    }
    void setY(uint32_t y)
    {
        data[0] &= XZMask;
        data[1] &= XZMask;
        data[0] |= (y & YMask) << Bits;
        data[1] |= (y >> (32u-Bits) & YMask) << Bits;
    }
    void setZ(uint32_t z)
    {
        data[1] &= ~XZMask;
        data[1] |= z & XZMask;
    }

    uint32_t data[2];
};

NBL_CONSTEXPR nbl::hlsl::float32_t3 LightEminence = nbl::hlsl::float32_t3(30.0f, 25.0f, 15.0f);
NBL_CONSTEXPR uint32_t RenderWorkgroupSize = 64u;
NBL_CONSTEXPR uint32_t MAX_DEPTH_LOG2 = 4u;
NBL_CONSTEXPR uint32_t MAX_SAMPLES_LOG2 = 10u;
NBL_CONSTEXPR uint32_t MaxBufferDimensions = 3u << MAX_DEPTH_LOG2;
NBL_CONSTEXPR uint32_t MaxBufferSamples = 1u << MAX_SAMPLES_LOG2;

#endif
