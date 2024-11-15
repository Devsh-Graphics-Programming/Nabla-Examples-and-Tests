#ifndef _CAD_EXAMPLE_GEOTEXTURE_COMMON_HLSL_INCLUDED_
#define _CAD_EXAMPLE_GEOTEXTURE_COMMON_HLSL_INCLUDED_

#include "../globals.hlsl"

// Handle multiple geo textures, separate set, array of texture? index allocator? or multiple sets?
NBL_CONSTEXPR uint32_t MaxGeoTextures = 256; 

// GeoTexture Oriented Bounding Box
struct GeoTextureOBB
{
    float64_t2 topLeft; // 2 * 8 = 16 bytes
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (32)
};

#ifdef __HLSL_VERSION
struct PSInput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : COLOR0;
};

// Push Constant 
[[vk::push_constant]] GeoTextureOBB geoTextureOBB;

// Set 0 - Scene Data and Globals, buffer bindings don't change the buffers only get updated
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);

// Set 1 - Window dependant data which has higher update frequency due to multiple windows and resize need image recreation and descriptor writes
[[vk::binding(0, 1)]] Texture2D<float4> geoTexture : register(t0);
[[vk::binding(1, 1)]] SamplerState geoTextureSampler : register(s0);
#endif

#endif