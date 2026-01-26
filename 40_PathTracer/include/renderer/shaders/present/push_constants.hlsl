#ifndef _NBL_THIS_EXAMPLE_PRESENT_PUSH_CONSTANTS_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PRESENT_PUSH_CONSTANTS_HLSL_INCLUDED_


#include "renderer/shaders/resolve/rwmc.hlsl"


// no uint16_t to be used because its going to be a push constant
namespace nbl
{
namespace this_example
{
using namespace nbl::hlsl;
	
struct SDefaultResolvePushConstants
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ImageCount = 16;

    struct Regular
    {
        // if more than 1.f
        float32_t2 scale;
        // to visualize what will get cropped out
        float32_t2 _min,_max;
    };
    struct Cubemap
    {
        // theoretically we only need inverse of product of 3x3 view with very sparse 4x4
        float32_t4x4 invProjView;
    };
#ifndef __HLSL_VERSION
    union
    {
        Regular regular;
        Cubemap cubemap;
    };
#else
    // note how this is a conversion to a copy, and not handing out of a reference
    // Ergo, its not a true "union"
    inline Regular regular()
    {
        Regular retval;
        retval.scale = __union.invProjView[0].xy;
        retval.crop = __union.invProjView[0].zw;
        retval.limit = __union.invProjView[1].xy;
        return retval;
    }
    inline Cubemap cubemap() {return __union;}

    Cubemap __union;
#endif
    // 3 extra bits for cube layer
    uint32_t isCubemap : 1;
    uint32_t layer : MAX_CASCADE_COUNT_LOG2;
    uint32_t imageIndex : BOOST_PP_SUB(31,MAX_CASCADE_COUNT_LOG2);
};

}
}
#endif  // _NBL_THIS_EXAMPLE_PRESENT_PUSH_CONSTANTS_HLSL_INCLUDED_
