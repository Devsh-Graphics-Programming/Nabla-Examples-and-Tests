#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[shader("closesthit")]
void main(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const int primID = PrimitiveIndex();
    float32_t3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    SProceduralGeomInfo sphere = vk::RawBufferLoad < SProceduralGeomInfo > (pc.proceduralGeomInfoBuffer + primID * sizeof(SProceduralGeomInfo));

    // Computing the normal at hit position
    float32_t3 worldNormal = normalize(worldPosition - sphere.center);

    payload.material = sphere.material;
    payload.worldNormal = worldNormal;
    payload.rayDistance = RayTCurrent();

}