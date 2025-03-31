#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;


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