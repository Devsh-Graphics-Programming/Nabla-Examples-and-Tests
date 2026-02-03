#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_


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
