#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout OcclusionPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const Material material = nbl::hlsl::_static_cast<Material>(geom.material);
    
    payload.attenuation = (1.f-material.alpha) * payload.attenuation;
    // arbitrary constant
//    if (payload.attenuation < 1.f/1024.f)
//        TerminateRay();
    IgnoreHit();
}
