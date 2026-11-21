#include "nbl/builtin/hlsl/cpp_compat/basic.h"

template<typename Caps>
struct DummyCaps
{
#if defined(NBL_TEST_HAS_I)
    static const uint32_t u16 = Caps::i::u16;
    static const uint32_t u32 = Caps::i::u32;
#endif
#if defined(NBL_TEST_HAS_M)
    static const uint32_t md = Caps::m::md;
    static const bool en = Caps::m::en;
#endif
#if defined(NBL_TEST_HAS_F)
    static const float fmin = Caps::f::min;
#endif
#if defined(NBL_TEST_HAS_D)
    static const double dmax = Caps::d::max;
#endif
#if defined(NBL_TEST_HAS_LIMITS)
    static const uint32_t lim = Caps::maxImageDimension2D;
#endif
#if defined(NBL_TEST_HAS_FEATURES)
    static const bool feat = Caps::shaderCullDistance;
#endif
#if defined(NBL_TEST_HAS_T)
    static const uint32_t sel = Caps::t::sel;
#endif
};

template<typename Caps>
uint32_t dummyValue()
{
    uint32_t v = 0u;
#if defined(NBL_TEST_HAS_I)
    v += DummyCaps<Caps>::u16 + DummyCaps<Caps>::u32;
#endif
#if defined(NBL_TEST_HAS_M)
    v += DummyCaps<Caps>::md;
    v += DummyCaps<Caps>::en ? 1u : 0u;
#endif
#if defined(NBL_TEST_HAS_FEATURES)
    if (DummyCaps<Caps>::feat)
    {
#if defined(NBL_TEST_HAS_LIMITS)
        v += DummyCaps<Caps>::lim;
#endif
    }
#elif defined(NBL_TEST_HAS_LIMITS)
    v += DummyCaps<Caps>::lim;
#endif
#if defined(NBL_TEST_HAS_F)
    float f = DummyCaps<Caps>::fmin;
    v += uint32_t(f);
#endif
#if defined(NBL_TEST_HAS_D)
    double d = DummyCaps<Caps>::dmax;
    v += uint32_t(d);
#endif
#if defined(NBL_TEST_HAS_T)
    v += DummyCaps<Caps>::sel;
#endif
    return v;
}

[numthreads(1, 1, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint32_t v = dummyValue<DeviceConfigCaps>();
    if (tid.x == 0u)
    {
        uint32_t sink = v;
        if (sink == 0u)
            return;
    }
}
