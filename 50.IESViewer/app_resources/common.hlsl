#ifndef _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#define QUANT_ERROR_ADMISSIBLE 1/1024
#define WORKGROUP_SIZE 256u
#define WORKGROUP_DIMENSION 16u
#define MAX_IES_IMAGES 6969

namespace nbl
{
namespace hlsl
{
namespace this_example
{
namespace ies
{

struct SInstanceMatrices
{
    float32_t4x4 worldViewProj;
    float32_t3x3 normal;
};

struct CdcPC
{
    uint64_t hAnglesBDA;
    uint64_t vAnglesBDA;
    uint64_t dataBDA;
    uint32_t mode : 8;
    uint32_t symmetry : 8;
    uint32_t texIx : 16;
	uint32_t hAnglesCount;
    uint32_t vAnglesCount;
    float32_t maxIValue;
    float32_t zAngleDegreeRotation;

	uint32_t pad;
};

struct SpherePC
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t DescriptorCount = (0x1<<16)-1;
	SInstanceMatrices matrices;
    uint32_t positionView : 16;
    uint32_t normalView : 16;
	float32_t radius;
	uint16_t texIx;
};

struct PushConstants
{
	CdcPC cdc;
	SpherePC sphere;
};
	
}		
}
}
}
#endif // _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
