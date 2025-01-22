#include "common.hlsl"
#include "random.hlsl"

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

#if defined(USE_COLOR_PAYLOAD)
using AnyHitPayload = ColorPayload;
#elif defined(USE_SHADOW_PAYLOAD)
using AnyHitPayload = ShadowPayload;
#endif

[shader("anyhit")]
void main(inout AnyHitPayload p, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const SGeomInfo geom = vk::RawBufferLoad < SGeomInfo > (pc.geometryInfoBuffer + instID * sizeof(SGeomInfo));
    
    if (geom.material.illum != 4)
        return;

    if (geom.material.dissolve == 0.0)
        IgnoreHit();
    else if (rnd(p.seed) > geom.material.dissolve)
        IgnoreHit();
}
