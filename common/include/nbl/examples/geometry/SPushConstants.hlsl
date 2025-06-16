#ifndef _NBL_EXAMPLES_S_PUSH_CONSTANTS_HLSL_
#define _NBL_EXAMPLES_S_PUSH_CONSTANTS_HLSL_


#include "nbl/examples/common/SBasicViewParameters.hlsl"


namespace nbl
{
namespace hlsl
{
namespace examples
{
namespace geometry_creator_scene
{

struct SPushConstants
{
	SBasicViewParameters basic;
	uint32_t positionView : 11;
	uint32_t normalView : 10;
	uint32_t uvView : 11;
};

}
}
}
}
#endif

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/