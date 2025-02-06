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
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const Material material = unpackMaterial(geom.material);
    
    if (material.illum != 4)
        return;

    uint32_t seed = p.seed;
    if (material.dissolve == 0.0)
        IgnoreHit();
    else if (rnd(seed) > material.dissolve)
        IgnoreHit();
}
