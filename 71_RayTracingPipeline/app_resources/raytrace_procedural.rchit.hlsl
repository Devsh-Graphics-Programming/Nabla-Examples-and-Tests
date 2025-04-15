#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("closesthit")]
void main(inout PrimaryPayload payload, in ProceduralHitAttribute attrib)
{
    const float32_t3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const float32_t3 worldNormal = normalize(worldPosition - attrib.center);

    payload.materialId = MaterialId::createProcedural(PrimitiveIndex()); // we use negative value to indicate that this is procedural

    payload.worldNormal = worldNormal;
    payload.rayDistance = RayTCurrent();

}