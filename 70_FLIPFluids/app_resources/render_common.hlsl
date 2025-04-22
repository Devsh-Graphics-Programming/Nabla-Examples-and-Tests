#ifndef _FLIP_EXAMPLE_RENDER_COMMON_HLSL
#define _FLIP_EXAMPLE_RENDER_COMMON_HLSL
#include "nbl/builtin/hlsl/bda/struct_declare.hlsl"

struct SParticleRenderParams
{
    float radius;
    float zNear;
    float zFar;
};

// TODO: This struct shouldn't exist if there's no "vertex generation" shader
struct VertexInfo;
// TODO: don't use 4D vectors for 3D quantities
NBL_HLSL_DEFINE_STRUCT((VertexInfo),
    ((position, float32_t4))
    ((vsSpherePos, float32_t4))
    ((radius, float32_t))
    ((color, float32_t4))
    ((uv, float32_t2))
);

#ifdef __HLSL_VERSION
struct PSInput
{
	float4 position : SV_Position;
	float2 uv : TEXCOORD0;
	nointerpolation float3 vsSpherePos : TEXCOORD1;
    nointerpolation float radius : TEXCOORD2;
    nointerpolation float4 color : TEXCOORD3; // TODO: unless you plan on using transparency, don't use RGBA, use RGB (float3) instead
};
#endif

#endif