#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("closesthit")]
void main(inout PrimaryPayload payload, in ProceduralHitAttribute attrib)
{
    const float32_t3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const float32_t3 worldNormal = normalize(worldPosition - attrib.center);

    payload.material = attrib.material;
    payload.worldNormal = worldNormal;
    payload.rayDistance = RayTCurrent();

}