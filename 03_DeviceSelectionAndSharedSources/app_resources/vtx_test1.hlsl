#pragma shader_stage(vertex)

#include "common_test1.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<uint> uints;
[[vk::binding(1, 0)]] StructuredBuffer<float> floats;
[[vk::binding(2,1)]] ByteAddressBuffer inputs[2];

PSInput main(uint vertexID : SV_VertexID)
{
    PSInput outV;
    outV.data1[vertexID].x = uints[0];
    outV.data2[vertexID].x = floats[0];
    outputBuff.Store<uint32_t>(0, uints[1]);

    return outV;
}