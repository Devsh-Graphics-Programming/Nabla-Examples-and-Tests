#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const Material material = unpackMaterial(geom.material);
    
    if (material.dissolve == 0.0 || material.dissolve < payload.dissolveThreshold)
    {
        IgnoreHit();
    }
}
