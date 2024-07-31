#pragma shader_stage(vertex)

#include "common.hlsl"

struct PSInput
{
	float4 position : SV_Position;
	float4 color : COLOR0;
};

// set 1, binding 0
[[vk::binding(0, 1)]]
cbuffer CameraData
{
    SMVPParams params;
};

PSInput main(uint vertexID : SV_VertexID)
{
    PSInput output;

    const float4 position[3] = {
		float4(0.5,-0.5,0,1),
		float4(0,0.5,0,1),
		float4(-0.5,-0.5,0,1),
	};

    const float4 colors[3] = {
		float4(1,0,0,1),
		float4(0,1,0,1),
		float4(0,0,1,1),
	};

    output.position = float4(position[vertexID % 3]);
    output.color = float4(colors[vertexID % 3]);

    return output;
}