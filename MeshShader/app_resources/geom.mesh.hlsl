//https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#primitive-attributes

#include "task_mesh_common.hlsl"


//utb is short for "uniform texel buffer", could also be considered a storage buffer with vec4s
//the gpu probably does something different with the data between a utb and a storage buffer but idk
[[vk::binding(0)]] Buffer<float32_t4> utbs[PushDescCount];

//binding 1, set 0, the mesh data is in binding 0
[[vk::binding(1, 0)]] Buffer<uint> indices[];

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

        // verts[id.x].ndc = mul(float32_t4(position, 1.0), worldViewProj);
        verts[id.x].ndc = mul(worldViewProj, float32_t4(position, 1.0));


        if (meshDataCopy.normalView < PushDescCount) { // && meshDataCopy.objType != CONE_OBJECT - just going to set cone_object normalView to pushdesccount
            verts[id.x].meta = utbs[meshDataCopy.normalView][id.x].xyz;
        }
        else {
        //i could reconstruct the normal right here in the mesh shader for the cone
            //verts[id.x].meta = mul(inverse(transpose(pc.matrices.normal)),position);
            verts[id.x].meta = float32_t3(0.0, 0.0, 0.0); //id like to check if cones even have a normal first
        }
    }

    //uint outputVertexCount = meshDataCopy.vertCount; //not necessary right now
    uint outputPrimitiveCount = meshDataCopy.primCount;
    //we're assuming primCount == vertCount, but most of the time a 
    //index buffer will exist, so i'll leave the branch in the EXAMPLE
    if(id.x < meshDataCopy.primCount){ 
        //a fan has 0 at the center, then around a circle it'll have 1 at 12o'clock, (relatively speaking)
        //numbers increment as you go clockwise (again, relatively speaking)
        //so if id.x + 2 is greater than prim count, it wraps back around to 1
        if(meshDataCopy.objType == T_FAN_OBJECT_TYPE){
            uint3 prim = uint3(0, id.x + 1, id.x + 2);
            if(prim.y >= meshDataCopy.vertCount){
                //not adding
            }
            else if (prim.z >= meshDataCopy.vertCount) {
                prim.z = 1;
                prims[id.x] = prim;
                printf("adding prim - {%u:%u:%u}", prims[id.x].x, prims[id.x].y, prims[id.x].z);
            }
            else {
                prims[id.x] = prim;
                printf("adding prim - {%u:%u:%u}", prims[id.x].x, prims[id.x].y, prims[id.x].z);
            }
        }
        /* probably incorrect
        else if(triangle strip){
            uint3 prim = uint3(id.x, id.x + 1, id.x + 2);
            bool lessThan = (prim.x < meshDataCopy.vertCount) && (prim.y < meshDataCopy.vertCount) && (prim.z < meshDataCopy.vertCount);
            if (lessThan) {
                prims[id.x] = prim;
                printf("adding prim [triangle strip type]- {%u:%u:%u} : {%u:%u:%u}", prims[id.x].x, prims[id.x].y, prims[id.x].z, prim.x, prim.y, prim.z);
            }
        }
        */
        else { // triangle list.
            outputPrimitiveCount = meshDataCopy.vertCount / 3;
            if (id.x < (meshDataCopy.primCount / 3)) { //probably incorrect for a indexed triangle list idk
                prims[id.x].x = id.x * 3;
                prims[id.x].y = id.x * 3 + 1;
                prims[id.x].z = id.x * 3 + 2;
                printf("adding prim [triangle strip type]- {%u:%u:%u}", prims[id.x].x, prims[id.x].y, prims[id.x].z);
            }
        }
    }

    SetMeshOutputCounts(meshDataCopy.vertCount, outputPrimitiveCount);
}