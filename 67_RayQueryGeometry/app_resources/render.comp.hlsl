#include "common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

struct SPushConstants
{
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

float3 getNormalsFromMask(uint32_t n)   // also probably need template specialization for each geometry XD
{
    // this still doesn't feel right
    // int8_t3 --> SSCALED (converts to float) --> unpackUnorm (read as uint --> float / 255)
    float4 v = nbl::hlsl::spirv::unpackUnorm4x8(n) * 255.0;
    return v.xyz;
}

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    uint2 coords = threadID.xy;
    uint2 resolution;
    outImage.GetDimensions(resolution.x, resolution.y);
    float2 texCoords = float2(float(coords.x) / resolution.x, 1.0 - float(coords.y) / resolution.y);

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

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(topLevelAS, 0, 0xFF, ray);

    while (query.Proceed()) {}

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        const int primID = query.CommittedPrimitiveIndex();

        uint idxOffset = primID * 3;

        // TODO: put into struct per geometry type
        const uint vertexStride = 24;
        const uint byteOffset = 18;
        const uint indexType = 0;   // value from obj.indexType

        uint i0, i1, i2;

        switch (indexType)
        {
            case 0: // EIT_16BIT
            {
                i0 = uint32_t(vk::RawBufferLoad<uint16_t>(pc.indexBufferAddress + (idxOffset + 0) * sizeof(uint16_t)));
                i1 = uint32_t(vk::RawBufferLoad<uint16_t>(pc.indexBufferAddress + (idxOffset + 1) * sizeof(uint16_t)));
                i2 = uint32_t(vk::RawBufferLoad<uint16_t>(pc.indexBufferAddress + (idxOffset + 2) * sizeof(uint16_t)));
            }
            break;
            case 1: // EIT_32BIT
            {
                i0 = vk::RawBufferLoad<uint32_t>(pc.indexBufferAddress + (idxOffset + 0) * sizeof(uint32_t));
                i1 = vk::RawBufferLoad<uint32_t>(pc.indexBufferAddress + (idxOffset + 1) * sizeof(uint32_t));
                i2 = vk::RawBufferLoad<uint32_t>(pc.indexBufferAddress + (idxOffset + 2) * sizeof(uint32_t));
            }
            break;
            default:    // EIT_NONE
            {
                i0 = idxOffset;
                i1 = idxOffset + 1;
                i2 = idxOffset + 2;
            }
        }

        uint32_t v0 = vk::RawBufferLoad<uint32_t>(pc.vertexBufferAddress + i0 * vertexStride + byteOffset);
        uint32_t v1 = vk::RawBufferLoad<uint32_t>(pc.vertexBufferAddress + i1 * vertexStride + byteOffset);
        uint32_t v2 = vk::RawBufferLoad<uint32_t>(pc.vertexBufferAddress + i2 * vertexStride + byteOffset);

        float3 n0 = getNormalsFromMask(v0) * 0.5 + 0.5;
        float3 n1 = getNormalsFromMask(v1) * 0.5 + 0.5;
        float3 n2 = getNormalsFromMask(v2) * 0.5 + 0.5;

        float3 barycentrics = float3(0.0, query.CommittedTriangleBarycentrics());
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

        float3 normalInterp = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
        //color = float4(normalInterp, 1.0);
        color = float4(0.8, 0, 0.2, 1.0);
    }

    outImage[coords] = color;
}
