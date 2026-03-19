#pragma shader_stage(vertex)

#include "common.hlsl"

[shader("vertex")]
PSInput vtxMain(uint vertexID : SV_VertexID)
{
    PSInput outV;
    TriangleMeshVertex vtx = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * vertexID, 8u);

    // calculate object space normal, for now we can treat it as the world space normal
    {
        const uint32_t currentVertexWithinTriangleIndex = vertexID % 3;
        const uint32_t firstVertexOfCurrentTriangleIndex = vertexID - currentVertexWithinTriangleIndex;

        TriangleMeshVertex triangleVertices[3];
        triangleVertices[0] = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * firstVertexOfCurrentTriangleIndex, 8u);
        triangleVertices[1] = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * (firstVertexOfCurrentTriangleIndex + 1), 8u);
        triangleVertices[2] = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * (firstVertexOfCurrentTriangleIndex + 2), 8u);

        // TODO: calculate on pfloat64_t
        float32_t3 vertex0 = _static_cast<float32_t3>(triangleVertices[0].pos);
        float32_t3 vertex1 = _static_cast<float32_t3>(triangleVertices[1].pos);
        float32_t3 vertex2 = _static_cast<float32_t3>(triangleVertices[2].pos);

        float32_t3 triangleEdge0 = vertex1 - vertex0;
        float32_t3 triangleEdge1 = vertex2 - vertex0;

        outV.setNormal((normalize(cross(triangleEdge1, triangleEdge0)) + 1.0f) * 0.5f);
    }

    pfloat64_t4 pos;
    pos.x = vtx.pos.x;
    pos.y = vtx.pos.y;
    pos.z = vtx.pos.z;
    pos.w = _static_cast<pfloat64_t>(1.0f);


    outV.setHeight(_static_cast<float>(pos.y));

    //pos = mul(pc.viewProjectionMatrix, pos);
    // TODO: use pc.viewProjectionMatrix and multiply it with pfloat64_t4 pos instead fix portable_matrix with portable_float multiplication
    float4x4 viewProjMatrix;
    for (int i = 0; i < 4; ++i)
    {
        viewProjMatrix[i][0] = _static_cast<float>(pc.viewProjectionMatrix[i].x);
        viewProjMatrix[i][1] = _static_cast<float>(pc.viewProjectionMatrix[i].y);
        viewProjMatrix[i][2] = _static_cast<float>(pc.viewProjectionMatrix[i].z);
        viewProjMatrix[i][3] = _static_cast<float>(pc.viewProjectionMatrix[i].w);
    }

    outV.setScreenSpaceVertexAttribs(_static_cast<float4>(pos).xyz);

    /*if (vertexID == 0)
    {
        printf("%f, %f, %f, %f", a[0][0], a[0][1], a[0][2], a[0][3]);
        printf("%f, %f, %f, %f", a[1][0], a[1][1], a[1][2], a[1][3]);
        printf("%f, %f, %f, %f", a[2][0], a[2][1], a[2][2], a[2][3]);
        printf("%f, %f, %f, %f", a[3][0], a[3][1], a[3][2], a[3][3]);
    }*/

    outV.position = mul(viewProjMatrix, _static_cast<float4>(pos));

    return outV;
}
