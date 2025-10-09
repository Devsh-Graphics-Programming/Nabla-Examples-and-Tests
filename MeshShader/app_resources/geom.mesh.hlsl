//https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#primitive-attributes

#include "task_mesh_common.hlsl"


//utb is short for "uniform texel buffer", or its a storage buffer with vec4s
[[vk::binding(0)]] StructuredBuffer<float32_t4> utbs[PushDescCount];
//none of the objects use the index buffer

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
    in payload TaskToMeshPayload taskToMeshPayload,
    out vertices VertexOut verts[WORKGROUP_SIZE],
    out indices uint3 prims[WORKGROUP_SIZE]
)
{   
    MeshData meshDataCopy = meshData[taskToMeshPayload.objectType[groupThreadID.x]];

    //if the ratio isnt 1 object to 1 transform, the payload can be used to pass in a transform index
    //or if it isnt 1 task shader launching every mesh shader, the payload will need to handle
    const float32_t4x4 worldViewProj = pc.viewProj * transform[groupThreadID.x];


    if(id.x < meshDataCopy.vertCount){
        const float32_t3 position = utbs[meshDataCopy.positionView][id.x].xyz;

        verts[id.x].ndc = mul(float32_t4(position, 1.0), worldViewProj);


        if (meshDataCopy.normalView < PushDescCount) { // && meshDataCopy.objType != CONE_OBJECT - just going to set cone_object normalView to pushdesccount
            verts[id.x].meta = utbs[meshDataCopy.normalView][id.x].xyz;
        }
        else {
        //i could reconstruct the normal right here in the mesh shader for the cone
            //verts[id.x].meta = mul(inverse(transpose(pc.matrices.normal)),position);
            verts[id.x].meta = float32_t3(0.0, 0.0, 0.0); //id like to check if cones even have a normal first
        }
    }

    if(id.x < meshDataCopy.primCount){
        if(meshDataCopy.objType == T_FAN_OBJECT_TYPE){
            uint3 prim = uint3(0, id.x + 1, id.x + 2);
            if(prim.y >= meshDataCopy.vertCount){
                //not adding
            }
            if(prim.z >= meshDataCopy.vertCount){
                prim.z = 1;
            }
            prims[id.x] = prim;
        }
        else{
            uint3 prim = uint3(id.x, id.x + 1, id.x + 2);
            bool lessThan = (prim.x < meshDataCopy.vertCount) && (prim.y < meshDataCopy.vertCount) && (prim.z < meshDataCopy.vertCount);
            if(lessThan){
                prims[id.x] = prim;
            }
        }
    }


    SetMeshOutputCounts(meshDataCopy.vertCount, meshDataCopy.primCount);
}