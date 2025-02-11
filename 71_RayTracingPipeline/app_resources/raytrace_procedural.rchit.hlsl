#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[shader("closesthit")]
void main(inout HitPayload payload, in ProceduralHitAttribute attrib)
{
    const float32_t3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const float32_t3 worldNormal = normalize(worldPosition - attrib.center);

    payload.material = attrib.material;
    payload.worldNormal = worldNormal;
    payload.rayDistance = RayTCurrent();

}