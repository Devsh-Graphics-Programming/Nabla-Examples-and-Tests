#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[shader("closesthit")]
void main(inout ColorPayload p, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const int primID = PrimitiveIndex();
    float32_t3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    SProceduralGeomInfo sphere = vk::RawBufferLoad < SProceduralGeomInfo > (pc.proceduralGeomInfoBuffer + primID * sizeof(SProceduralGeomInfo));

    // Computing the normal at hit position
    float32_t3 worldNormal = normalize(worldPosition - sphere.center);

    RayLight cLight;
    cLight.inHitPosition = worldPosition;
    CallShader(pc.light.type, cLight);

    // Material of the object
    Material mat = sphere.material;

    // Diffuse
    float3 diffuse = computeDiffuse(sphere.material, cLight.outLightDir, worldNormal);
    float3 specular = float3(0, 0, 0);
    float attenuation = 1;

    // Tracing shadow ray only if the light is visible from the surface
    if (dot(worldNormal, cLight.outLightDir) > 0)
    {
        RayDesc rayDesc;
        rayDesc.Origin = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        rayDesc.Direction = cLight.outLightDir;
        rayDesc.TMin = 0.01;
        rayDesc.TMax = cLight.outLightDistance;

        uint flags =
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE |
            RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

        ShadowPayload shadowPayload;
        shadowPayload.isShadowed = true;
        shadowPayload.seed = p.seed;
        TraceRay(topLevelAS, flags, 0xFF, ERT_OCCLUSION, 0, EMT_PRIMARY, rayDesc, shadowPayload);

        bool isShadowed = shadowPayload.isShadowed;
        if (isShadowed)
        {
            attenuation = 0.3;
        }
        else
        {
            specular = computeSpecular(sphere.material, WorldRayDirection(), cLight.outLightDir, worldNormal);
        }
    }

    p.hitValue = (cLight.outIntensity * attenuation * (diffuse + specular));
}