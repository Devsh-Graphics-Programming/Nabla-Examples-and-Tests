struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color : COLOR;
    [[vk::location(2)]] float2 uv : TEXCOORD;
    [[vk::location(3)]] float3 normal : NORMAL;
};

#include "template/gc.vertex.hlsl"
