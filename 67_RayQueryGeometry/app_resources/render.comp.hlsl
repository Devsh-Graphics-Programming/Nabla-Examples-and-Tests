#include "common.hlsl"

struct SPushConstants
{
    bool useIndex;
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1, 0)]]
cbuffer CameraData
{
    SCameraParameters params;
};

[[vk::binding(2, 0)]] RWTexture2D<float4> outImage;

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    uint2 coords = threadID.xy;
    uint2 resolution;
    outImage.GetDimensions(resolution.x, resolution.y);
    uint2 texCoords = uint2(float(coords.x) / resolution.x, 1.0 - float(coords.y) / resolution.y);

    if (any(coords >= resolution))
        return;

    float4 color = float4(0, 0, 0, 1);
    
    float4 NDC = float4(texCoords * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    float3 targetPos;
    {
        float4 tmp = mul(params.invMVP, NDC);
        targetPos = tmp.xyz / tmp.w;
        NDC.z = 1.0;
    }

    RayDesc ray;
    ray.TMin = 0.01;
    ray.TMax = 1000.0;
    ray.Origin = params.camPos;
    ray.Direction = normalize(targetPos - params.camPos);

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(topLevelAS, 0, 0xFF, ray);

    while (query.Proceed()) {}

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        const int primID = query.CommittedPrimitiveIndex();

        uint idxOffset = primID * 3;
        VertexData v0, v1, v2;

        if (pc.useIndex)
        {
            uint i0 = vk::RawBufferLoad<uint32_t>(pc.indexBufferAddress + idxOffset * sizeof(uint32_t) + 0);
            uint i1 = vk::RawBufferLoad<uint32_t>(pc.indexBufferAddress + idxOffset * sizeof(uint32_t) + 1);
            uint i2 = vk::RawBufferLoad<uint32_t>(pc.indexBufferAddress + idxOffset * sizeof(uint32_t) + 2);

            v0 = vk::RawBufferLoad<VertexData>(pc.vertexBufferAddress + i0 * sizeof(VertexData));
            v1 = vk::RawBufferLoad<VertexData>(pc.vertexBufferAddress + i1 * sizeof(VertexData));
            v2 = vk::RawBufferLoad<VertexData>(pc.vertexBufferAddress + i2 * sizeof(VertexData));
        }
        else
        {
            v0 = vk::RawBufferLoad<VertexData>(pc.vertexBufferAddress + idxOffset * sizeof(VertexData) + 0);
            v1 = vk::RawBufferLoad<VertexData>(pc.vertexBufferAddress + idxOffset * sizeof(VertexData) + 1);
            v2 = vk::RawBufferLoad<VertexData>(pc.vertexBufferAddress + idxOffset * sizeof(VertexData) + 2);
        }
        

        float3 barycentrics = float3(0.0, query.CommittedTriangleBarycentrics());
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

        float3 normalInterp = barycentrics.x * v0.normal + barycentrics.y * v1.normal + barycentrics.z * v2.normal;
        color = float4(normalInterp, 1.0);
    }

    outImage[coords] = color;
}
