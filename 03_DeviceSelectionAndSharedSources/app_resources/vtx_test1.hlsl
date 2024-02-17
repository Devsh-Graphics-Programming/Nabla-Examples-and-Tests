#pragma shader_stage(vertex)

#include "common_test1.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<uint> uints : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float> floats : register(t1);

PSInput main(uint vertexID : SV_VertexID)
{
    PSInput outV;
    outV.data1[vertexID].x = 1;
    outV.data2[vertexID].x = 1.0f;
    return outV;
}