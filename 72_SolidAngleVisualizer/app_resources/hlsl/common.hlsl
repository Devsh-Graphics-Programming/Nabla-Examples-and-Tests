#ifndef _SOLID_ANGLE_VIS_COMMON_HLSL_
#define _SOLID_ANGLE_VIS_COMMON_HLSL_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"



struct PushConstants
{
	nbl::hlsl::float32_t3x4 modelMatrix;
	nbl::hlsl::float32_t4 viewport;
};


#endif // _SOLID_ANGLE_VIS_COMMON_HLSL_
