#ifndef _NBL_EXAMPLES_S_PUSH_CONSTANTS_HLSL_
#define _NBL_EXAMPLES_S_PUSH_CONSTANTS_HLSL_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl
{
namespace hlsl
{
namespace examples
{
namespace geometry_creator_scene
{

struct SInstanceMatrices
{
	float32_t4x4 worldViewProj;
	float32_t3x3 normal;
};

struct SPushConstants
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t DescriptorCount = (0x1<<16)-1;

	SInstanceMatrices matrices;
	uint32_t positionView : 16;
	uint32_t normalView : 16;
};

}
}
}
}
#endif

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/