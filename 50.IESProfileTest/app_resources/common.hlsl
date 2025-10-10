#ifndef _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// -> TODO: use NBL_CONTEXPR or something
#ifndef UINT16_MAX
#define UINT16_MAX 65535u // would be cool if we have this define somewhere or GLSL do
#endif
#define M_PI 3.1415926535897932384626433832795f // would be cool if we have this define somewhere or GLSL do
#define M_HALF_PI M_PI/2.0f // would be cool if we have this define somewhere or GLSL do
#define QUANT_ERROR_ADMISSIBLE 1/1024

#define WORKGROUP_SIZE 256u
#define WORKGROUP_DIMENSION 16u
// <- + wipe whatever we already have

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
	float32_t zAngleDegreeRotation;

	uint32_t mode;

	#ifdef __HLSL_VERSION
	float32_t getHorizontalAngle(uint32_t ix) { return vk::RawBufferLoad<float32_t>(hAnglesBDA + sizeof(float32_t) * ix); }
	float32_t getVerticalAngle(uint32_t ix) { return vk::RawBufferLoad<float32_t>(vAnglesBDA + sizeof(float32_t) * ix); }
	float32_t getData(uint32_t ix) { return vk::RawBufferLoad<float32_t>(dataBDA + sizeof(float32_t) * ix); }
	#endif // __HLSL_VERSION
};

#endif // _THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
