#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout OcclusionPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const Material material = unpackMaterial(geom.material);
    
    if (material.isTransparent())
    {
        payload.attenuation = material.alpha * payload.attenuation;
        IgnoreHit();
    }
    else
    {
        payload.attenuation = 0;
        AcceptHitAndEndSearch();
    }

}
