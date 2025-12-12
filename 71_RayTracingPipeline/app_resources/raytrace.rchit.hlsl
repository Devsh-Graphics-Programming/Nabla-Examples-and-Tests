#include "common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

float3 calculateNormals(int primID, STriangleGeomInfo geom, float2 bary)
{
    const uint indexType = geom.indexType;
    const uint normalType = geom.normalType;

    const uint64_t vertexBufferAddress = geom.vertexBufferAddress;
    const uint64_t indexBufferAddress = geom.indexBufferAddress;
    const uint64_t normalBufferAddress = geom.normalBufferAddress;

    uint32_t3 indices;
    if (indexBufferAddress == 0)
    {
        indices[0] = primID * 3;
        indices[1] = indices[0] + 1;
        indices[2] = indices[0] + 2;
    }
    else {
        switch (indexType)
        {
            case 0: // EIT_16BIT
                indices = uint32_t3((nbl::hlsl::bda::__ptr<uint16_t3>::create(indexBufferAddress)+primID).deref().load());
                break;
            case 1: // EIT_32BIT
                indices = uint32_t3((nbl::hlsl::bda::__ptr<uint32_t3>::create(indexBufferAddress)+primID).deref().load());
                break;
        }
    }

    if (normalBufferAddress == 0)
    {
        float3 v0 = (nbl::hlsl::bda::__ptr<float3>::create(vertexBufferAddress) + indices[0]).deref().load();
        float3 v1 = (nbl::hlsl::bda::__ptr<float3>::create(vertexBufferAddress) + indices[1]).deref().load();
        float3 v2 = (nbl::hlsl::bda::__ptr<float3>::create(vertexBufferAddress) + indices[2]).deref().load();

        return normalize(cross(v2 - v0, v1 - v0));
    }

    float3 n0, n1, n2;
    switch (normalType)
    {
        case NT_R8G8B8A8_SNORM:
        {
            uint32_t v0 = (nbl::hlsl::bda::__ptr<uint32_t>::create(normalBufferAddress) + indices[0]).deref().load();
            uint32_t v1 = (nbl::hlsl::bda::__ptr<uint32_t>::create(normalBufferAddress) + indices[1]).deref().load();
            uint32_t v2 = (nbl::hlsl::bda::__ptr<uint32_t>::create(normalBufferAddress) + indices[2]).deref().load();

            n0 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v0).xyz);
            n1 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v1).xyz);
            n2 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v2).xyz);
        }
        break;
        case NT_R32G32B32_SFLOAT:
        {
            float3 v0 = (nbl::hlsl::bda::__ptr<float3>::create(normalBufferAddress) + indices[0]).deref().load();
            float3 v1 = (nbl::hlsl::bda::__ptr<float3>::create(normalBufferAddress) + indices[1]).deref().load();
            float3 v2 = (nbl::hlsl::bda::__ptr<float3>::create(normalBufferAddress) + indices[2]).deref().load();

            n0 = normalize(v0);
            n1 = normalize(v1);
            n2 = normalize(v2);
        }
        break;
    }

    float3 barycentrics = float3(0.0, bary);
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;        

    return barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
}


[shader("closesthit")]
void main(inout PrimaryPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int primID = spirv::PrimitiveId;
    const int instanceCustomIndex = spirv::InstanceCustomIndexKHR;
    const int geometryIndex = spirv::RayGeometryIndexKHR;
    const static uint64_t STriangleGeomInfoAlignment = nbl::hlsl::alignment_of_v<STriangleGeomInfo>;
    const STriangleGeomInfo geom = vk::BufferPointer<STriangleGeomInfo, STriangleGeomInfoAlignment>(pc.triangleGeomInfoBuffer + (instanceCustomIndex + geometryIndex) * sizeof(STriangleGeomInfo)).Get();
    const float32_t3 vertexNormal = calculateNormals(primID, geom, attribs.barycentrics);
    const float32_t3 worldNormal = normalize(mul(vertexNormal, transpose(spirv::WorldToObjectKHR)).xyz);

    payload.materialId = MaterialId::createTriangle(instanceCustomIndex);

    payload.worldNormal = worldNormal;
    payload.rayDistance = spirv::RayTmaxKHR;

}