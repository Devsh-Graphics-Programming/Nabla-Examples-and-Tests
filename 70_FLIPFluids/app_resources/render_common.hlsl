#ifndef _FLIP_EXAMPLE_RENDER_COMMON_HLSL
#define _FLIP_EXAMPLE_RENDER_COMMON_HLSL

#ifdef __HLSL_VERSION
struct VertexInfo
{
    float4 position;
	float4 vsSpherePos;
    
    float radius;
    float pad;

    float4 color;
    float2 uv;
};

struct PSInput
{
	float4 position : SV_Position;
	float2 uv : TEXCOORD0;
	nointerpolation float3 vsSpherePos : TEXCOORD1;
    nointerpolation float radius : TEXCOORD2;
    nointerpolation float4 color : TEXCOORD3;
};

struct SParticleRenderParams
{
    float radius;
    float zNear;
    float zFar;
    float pad;
};
#endif

#endif