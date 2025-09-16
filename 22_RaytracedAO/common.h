#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_


#define RAYCOUNT_N_BUFFERING_LOG2 2
#define RAYCOUNT_N_BUFFERING (0x1<<RAYCOUNT_N_BUFFERING_LOG2)
#define RAYCOUNT_SHIFT (32-RAYCOUNT_N_BUFFERING_LOG2)

#define MAX_TRIANGLES_IN_BATCH 16384

// need to bump to 2 in case of NEE + MIS, 3 in case of Path Guiding
#define SAMPLING_STRATEGY_COUNT 2

#define WORKGROUP_SIZE 256
#if WORKGROUP_SIZE!=256
#error "Hardcoded 16 should be NBL_SQRT(WORKGROUP_SIZE)"
#endif
#define WORKGROUP_DIM 16

#define MAX_SAMPLERS_COMPUTE 16


#ifdef __cplusplus
	#define uint uint32_t
	struct uvec2
	{
		uint x,y;
	};
	struct vec2
	{
		float x,y;
	};
	struct vec3
	{
		float x,y,z;
	};
	#define vec4 nbl::core::vectorSIMDf
	#define mat4 nbl::core::matrix4SIMD
	#define mat4x3 nbl::core::matrix3x4SIMD
#else
// for some reason Mitsuba uses Left-Handed coordinate system with Y-up for Envmaps
vec3 worldSpaceToMitsubaEnvmap(in vec3 worldSpace)
{
	return vec3(-worldSpace.z,worldSpace.xy);
}
vec3 mitsubaEnvmapToWorldSpace(in vec3 mitsubaEnvmaSpace)
{
	return vec3(mitsubaEnvmaSpace.yz,-mitsubaEnvmaSpace.x);
}
#endif


#endif
