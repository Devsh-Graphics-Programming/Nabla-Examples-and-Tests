#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

float32_t3 fetchVertexNormal(int instID, int primID, STriangleGeomInfo geom, float2 bary)
{
    uint idxOffset = primID * 3;
    
    const uint indexType = geom.indexType;
    const uint vertexStride = geom.vertexStride;
    
    const uint32_t objType = geom.objType;
    const uint64_t indexBufferAddress = geom.indexBufferAddress;
    
    uint i0, i1, i2;
    switch (indexType)
    {
        case 0: // EIT_16BIT
        {
            i0 = uint32_t(vk::RawBufferLoad < uint16_t > (indexBufferAddress + (idxOffset + 0) * sizeof(uint16_t), 2u));
            i1 = uint32_t(vk::RawBufferLoad < uint16_t > (indexBufferAddress + (idxOffset + 1) * sizeof(uint16_t), 2u));
            i2 = uint32_t(vk::RawBufferLoad < uint16_t > (indexBufferAddress + (idxOffset + 2) * sizeof(uint16_t), 2u));
        }
        break;
        case 1: // EIT_32BIT
        {
            i0 = vk::RawBufferLoad < uint32_t > (indexBufferAddress + (idxOffset + 0) * sizeof(uint32_t));
            i1 = vk::RawBufferLoad < uint32_t > (indexBufferAddress + (idxOffset + 1) * sizeof(uint32_t));
            i2 = vk::RawBufferLoad < uint32_t > (indexBufferAddress + (idxOffset + 2) * sizeof(uint32_t));
        }
        break;
        default: // EIT_NONE
        {
            i0 = idxOffset;
            i1 = idxOffset + 1;
            i2 = idxOffset + 2;
        }
    }

    const uint64_t normalBufferAddress = geom.normalBufferAddress;

    float3 n0, n1, n2;
    switch (objType)
    {
        case OT_CUBE:
        case OT_SPHERE:
        case OT_RECTANGLE:
        case OT_CYLINDER:
        //case OT_ARROW:
        case OT_CONE:
        {
            // TODO: document why the alignment is 2 here and nowhere else? isnt the `vertexStride` aligned to more than 2 anyway?
            uint32_t v0 = vk::RawBufferLoad<uint32_t>(normalBufferAddress + i0 * 4);
            uint32_t v1 = vk::RawBufferLoad<uint32_t>(normalBufferAddress + i1 * 4);
            uint32_t v2 = vk::RawBufferLoad<uint32_t>(normalBufferAddress + i2 * 4);

            n0 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v0).xyz);
            n1 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v1).xyz);
            n2 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v2).xyz);
        }
        break;
        case OT_ICOSPHERE:
        default:
        {
            n0 = normalize(vk::RawBufferLoad<float3>(normalBufferAddress + i0 * 12));
            n1 = normalize(vk::RawBufferLoad<float3>(normalBufferAddress + i1 * 12));
            n2 = normalize(vk::RawBufferLoad<float3>(normalBufferAddress + i2 * 12));
        }
    }

    float3 barycentrics = float3(0.0, bary);
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;
    return normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);
}

[shader("closesthit")]
void main(inout PrimaryPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const int primID = PrimitiveIndex();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const float32_t3 vertexNormal = fetchVertexNormal(instID, primID, geom, attribs.barycentrics);
    const float32_t3 worldNormal = normalize(mul(vertexNormal, WorldToObject3x4()).xyz);

    payload.materialId = MaterialId::createTriangle(instID);

    payload.worldNormal = worldNormal;
    payload.rayDistance = RayTCurrent();

}