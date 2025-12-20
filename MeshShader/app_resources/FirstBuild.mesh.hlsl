//https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#primitive-attributes

struct SInterpolants{
    float4 ndc : SV_Position;
};
struct Primo {
    uint vertexID : SV_PrimitiveID;
};

[numthreads(WORKGROUP_SIZE,1,1)]
[outputtopology("point")]

[shader("mesh")]
void main(
    in uint3 ID : SV_DispatchThreadID,
    out vertices SInterpolants verts[WORKGROUP_SIZE],
    out indices uint prims[WORKGROUP_SIZE]
)
{
    verts[ID.x].ndc = float32_t4(ID.x, 0.0, 0.0, 1.0);
    prims[ID.x] = ID.x;
    SetMeshOutputCounts(WORKGROUP_SIZE, WORKGROUP_SIZE);
}