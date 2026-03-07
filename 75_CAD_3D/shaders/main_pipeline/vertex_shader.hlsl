#pragma shader_stage(vertex)

#include "common.hlsl"

[shader("vertex")]
PSInput vtxMain(uint vertexID : SV_VertexID)
{
    PSInput outV;
    TriangleMeshVertex vtx = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * vertexID, 8u);

    outV.position.x = _static_cast<float>(vtx.pos.x);
    outV.position.y = _static_cast<float>(vtx.pos.y);
    outV.position.z = _static_cast<float>(vtx.pos.z);
    outV.position.w = 1.0f;

    return outV;
}
