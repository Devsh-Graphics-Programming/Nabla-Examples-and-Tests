#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

float3 unpackNormals3x10(uint32_t v)
{
    // host side changes float32_t3 to EF_A2B10G10R10_SNORM_PACK32
    // follows unpacking scheme from https://github.com/KhronosGroup/SPIRV-Cross/blob/main/reference/shaders-hlsl/frag/unorm-snorm-packing.frag
    int signedValue = int(v);
    int3 pn = int3(signedValue << 22, signedValue << 12, signedValue << 2) >> 22;
    return clamp(float3(pn) / 511.0, -1.0, 1.0);
}

struct VertexData
{
    float32_t3 position;
    float32_t3 normal;
};

VertexData fetchVertexData(int instID, int primID, STriangleGeomInfo geom, float2 bary)
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

    const uint64_t vertexBufferAddress = geom.vertexBufferAddress;
    float32_t3 p0 = vk::RawBufferLoad < float32_t3 > (vertexBufferAddress + i0 * vertexStride);
    float32_t3 p1 = vk::RawBufferLoad < float32_t3 > (vertexBufferAddress + i1 * vertexStride);
    float32_t3 p2 = vk::RawBufferLoad < float32_t3 > (vertexBufferAddress + i2 * vertexStride);

    const uint64_t normalVertexBufferAddress = vertexBufferAddress + s_offsetsToNormalBytes[objType];
    float3 n0, n1, n2;
    switch (objType)
    {
        case OT_CUBE:
        {
                uint32_t v0 = vk::RawBufferLoad < uint32_t > (normalVertexBufferAddress + i0 * vertexStride, 2u);
                uint32_t v1 = vk::RawBufferLoad < uint32_t > (normalVertexBufferAddress + i1 * vertexStride, 2u);
                uint32_t v2 = vk::RawBufferLoad < uint32_t > (normalVertexBufferAddress + i2 * vertexStride, 2u);

                n0 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v0).xyz);
                n1 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v1).xyz);
                n2 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v2).xyz);
            }
            break;
        case OT_SPHERE:
        case OT_CYLINDER:
        case OT_ARROW:
        case OT_CONE:
        {
                uint32_t v0 = vk::RawBufferLoad < uint32_t > (normalVertexBufferAddress + i0 * vertexStride);
                uint32_t v1 = vk::RawBufferLoad < uint32_t > (normalVertexBufferAddress + i1 * vertexStride);
                uint32_t v2 = vk::RawBufferLoad < uint32_t > (normalVertexBufferAddress + i2 * vertexStride);

                n0 = normalize(unpackNormals3x10(v0));
                n1 = normalize(unpackNormals3x10(v1));
                n2 = normalize(unpackNormals3x10(v2));
            }
            break;
        case OT_RECTANGLE:
        case OT_DISK:
        case OT_ICOSPHERE:
        default:
        {
                n0 = normalize(vk::RawBufferLoad <
                float3 > (normalVertexBufferAddress + i0 * vertexStride));
                n1 = normalize(vk::RawBufferLoad <
                float3 > (normalVertexBufferAddress + i1 * vertexStride));
                n2 = normalize(vk::RawBufferLoad <
                float3 > (normalVertexBufferAddress + i2 * vertexStride));
            }
    }

    float3 barycentrics = float3(0.0, bary);
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    VertexData data;
    data.position = barycentrics.x * p0 + barycentrics.y * p1 + barycentrics.z * p2;
    data.normal = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);
    return data;
}

[shader("closesthit")]
void main(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const int primID = PrimitiveIndex();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const VertexData vertexData = fetchVertexData(instID, primID, geom, attribs.barycentrics);
    const float32_t3 worldNormal = normalize(mul(vertexData.normal, WorldToObject3x4()).xyz);

    payload.material = geom.material;
    payload.worldNormal = worldNormal;
    payload.rayDistance = RayTCurrent();

}