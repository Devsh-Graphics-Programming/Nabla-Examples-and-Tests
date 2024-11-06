#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

#ifdef __HLSL_VERSION
struct VertexData
{
    float3 position;
    float4 color;
    float2 uv;
    float3 normal;
};

struct SCameraParameters //! matches CPU version size & alignment (160, 4)
{
    float3 camPos;

	float4x4 MVP;
	float3x4 V;
    float3x4 P;
};
#endif

#endif  // RQG_COMMON_HLSL
