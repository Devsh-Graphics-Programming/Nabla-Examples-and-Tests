#include "common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[shader("closesthit")]
void main(inout PrimaryPayload payload, in ProceduralHitAttribute attrib)
{
    const float32_t3 worldPosition = spirv::WorldRayOriginKHR + spirv::WorldRayDirectionKHR * spirv::RayTmaxKHR;
    const float32_t3 worldNormal = normalize(worldPosition - attrib.center);

    payload.materialId = MaterialId::createProcedural(spirv::PrimitiveId); // we use negative value to indicate that this is procedural

    payload.worldNormal = worldNormal;
    payload.rayDistance = spirv::RayTmaxKHR;

}