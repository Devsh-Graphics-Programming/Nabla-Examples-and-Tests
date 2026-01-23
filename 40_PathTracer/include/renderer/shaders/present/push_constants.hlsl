#ifndef _NBL_THIS_EXAMPLE_PRESENT_PUSH_CONSTANTS_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PRESENT_PUSH_CONSTANTS_HLSL_INCLUDED_


#include "renderer/shaders/resolve/rwmc.hlsl"


// no uint16_t to be used because its going to be a push constant
namespace nbl
{
namespace this_example
{
	
struct DefaultResolvePushConstants
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ImageCount = 16;

    struct Regular
    {
        float32_t2 cropOffset;
        float32_t2 scale;
    };
    struct Cubemap
    {
        // TODO
    };
    union
    {
        Regular regular;
        Cubemap cubemap;
    };
    // 3 extra bits for cube layer
    uint32_t isCubemap : 1;
    uint32_t layer : BOOST_PP_ADD(MAX_CASCADE_COUNT_LOG2,3);
    uint32_t imageIndex : BOOST_PP_SUB(28,MAX_CASCADE_COUNT_LOG2);
};

}
}
#endif  // _NBL_THIS_EXAMPLE_PRESENT_PUSH_CONSTANTS_HLSL_INCLUDED_
