#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_

#ifndef UINT16_MAX
#define UINT16_MAX 65535u // would be cool if we have this define somewhere or GLSL do
#endif
#define M_PI 3.1415926535897932384626433832795f // would be cool if we have this define somewhere or GLSL do
#define M_HALF_PI M_PI/2.0f // would be cool if we have this define somewhere or GLSL do
#define QUANT_ERROR_ADMISSIBLE 1/1024

#define WORKGROUP_SIZE 256u
#define WORKGROUP_DIMENSION 16u

#endif // _COMMON_INCLUDED_
