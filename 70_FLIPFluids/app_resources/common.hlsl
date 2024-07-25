#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 256;

struct Particle
{
    float id;
    float pad0[3];
    float32_t4 position;
    float32_t4 velocity;
}

#ifdef __HLSL_VERSION
struct SMVPParams
{
	float4x4 MVP;
	float3x4 MV;
	float3x3 normalMat;
};
#endif
