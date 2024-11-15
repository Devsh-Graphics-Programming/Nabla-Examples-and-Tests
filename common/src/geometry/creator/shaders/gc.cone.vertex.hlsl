struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color : COLOR;
    [[vk::location(2)]] float3 normal : NORMAL;
};

#include "template/gc.vertex.hlsl"
