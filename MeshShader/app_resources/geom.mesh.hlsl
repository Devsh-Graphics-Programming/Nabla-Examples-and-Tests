//https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#primitive-attributes

#include "task_mesh_common.hlsl"

//            (binding, set)
[[vk::binding(0, 0)]] StructuredBuffer<float3> MeshVertexBuffer;

struct VertexOut {
    float32_t4 ndc : SV_Position;
    float32_t3 meta : COLOR1;
};

[numthreads(WORKGROUP_SIZE,1,1)]

[outputtopology("triangle")]
[shader("mesh")]
void main(
    in uint3 id : SV_DispatchThreadID,
    in uint3 groupThreadID : SV_GroupThreadID,
    out vertices VertexOut verts[WORKGROUP_SIZE],
    out indices uint3 prims[WORKGROUP_SIZE]
)
{

    // i havent benchmarked this personally, but my understandign is that AMD devices prefer mesh shaders to be "by primitive"
    // and that nvidia devices prefer mesh shaders to be "by vertex".
    // ideally, i'd benchmark both and setup branches so that each device can specialize the shader basedo n what it likes 
    //(theres a property in VkMeshProperties that would indicate this)
    if (id.x < pc.vertCount) {
        const float32_t3 position = MeshVertexBuffer[id.x];

        // verts[id.x].ndc = mul(float32_t4(position, 1.0), worldViewProj);
        verts[id.x].ndc = mul(pc.mvp, float32_t4(position, 1.0));

        verts[id.x].meta = position;
    }

    // im just assuming its a triangle list right now. wont work if its not
    if (id.x < pc.vertCount / 3) {

        prims[id.x] = uint3(
                        id.x * 3, 
                        id.x * 3 + 1, 
                        id.x * 3 + 2
                    );
    }

    

    SetMeshOutputCounts(pc.vertCount, pc.vertCount / 3);
}