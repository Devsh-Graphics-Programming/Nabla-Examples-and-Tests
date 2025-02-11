#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const Material material = unpackMaterial(geom.material);
    
    if (material.illum != 4)
    {
        payload.attenuation = 0;
        AcceptHitAndEndSearch();
    }
    else
    {
        payload.attenuation = (1 - material.dissolve) * payload.attenuation;
        IgnoreHit();
    }

}
