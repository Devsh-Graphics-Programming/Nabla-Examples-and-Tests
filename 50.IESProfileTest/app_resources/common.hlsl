#ifndef _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// -> TODO: use NBL_CONTEXPR or something
#ifndef UINT16_MAX
#define UINT16_MAX 65535u // would be cool if we have this define somewhere or GLSL do
#endif // UINT16_MAX
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795f // would be cool if we have this define somewhere or GLSL do
#endif // M_PI

#define M_HALF_PI M_PI/2.0f // would be cool if we have this define somewhere or GLSL do
#define QUANT_ERROR_ADMISSIBLE 1/1024

#define WORKGROUP_SIZE 256u
#define WORKGROUP_DIMENSION 16u
// <- + wipe whatever we already have

// TODO: since NSC prebuilds into SPIRV - maybe could make it a CMake option with a default val
#define MAX_IES_IMAGES 4u * 6969

using namespace nbl::hlsl;

struct PushConstants
{
	uint64_t hAnglesBDA;
	uint64_t vAnglesBDA;
	uint64_t dataBDA;
	float64_t maxIValue;

	uint32_t hAnglesCount;
	uint32_t vAnglesCount;
	uint32_t dataCount;

	uint32_t mode;
	uint32_t texIx;
	float32_t zAngleDegreeRotation;

	uint32_t dummy;

	#ifdef __HLSL_VERSION
	float64_t getHorizontalAngle(uint32_t i) { return vk::RawBufferLoad<float64_t>(hAnglesBDA + sizeof(float64_t) * i, sizeof(float64_t)); }
	float64_t getVerticalAngle(uint32_t i) { return vk::RawBufferLoad<float64_t>(vAnglesBDA + sizeof(float64_t) * i, sizeof(float64_t)); }
	float64_t getData(uint32_t i) { return vk::RawBufferLoad<float64_t>(dataBDA + sizeof(float64_t) * i, sizeof(float64_t)); }
	#endif // __HLSL_VERSION
};

#endif // _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
