#include "common.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

static const int32_t s_sampleCount = 10;
static const float32_t3 s_clearColor = float32_t3(0.3, 0.3, 0.8);

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1, 0)]] RWTexture2D<float32_t4> colorImage;

float32_t nextRandomUnorm(inout nbl::hlsl::Xoroshiro64StarStar rnd)
{
    return float32_t(rnd()) / float32_t(0xFFFFFFFF);
}

[shader("raygeneration")]
void main()
{
    const uint32_t3 launchID = DispatchRaysIndex();
    const uint32_t3 launchSize = DispatchRaysDimensions();
    const uint32_t2 coords = launchID.xy;

    const uint32_t seed1 = nbl::hlsl::random::Pcg::create(pc.frameCounter)();
    const uint32_t seed2 = nbl::hlsl::random::Pcg::create(launchID.y * launchSize.x + launchID.x)();
    nbl::hlsl::Xoroshiro64StarStar rnd = nbl::hlsl::Xoroshiro64StarStar::construct(uint32_t2(seed1, seed2));

    float32_t3 hitValues = float32_t3(0, 0, 0);
    for (uint32_t sample_i = 0; sample_i < s_sampleCount; sample_i++)
    {
        const float32_t r1 = nextRandomUnorm(rnd);
        const float32_t r2 = nextRandomUnorm(rnd);
        const float32_t2 subpixelJitter = pc.frameCounter == 0 ? float32_t2(0.5f, 0.5f) : float32_t2(r1, r2);

        const float32_t2 pixelCenter = float32_t2(coords) + subpixelJitter;
        const float32_t2 inUV = pixelCenter / float32_t2(launchSize.xy);

        const float32_t2 d = inUV * 2.0 - 1.0;
        const float32_t4 tmp = mul(pc.invMVP, float32_t4(d.x, d.y, 1, 1));
        const float32_t3 targetPos = tmp.xyz / tmp.w;

        const float32_t3 camDirection = normalize(targetPos - pc.camPos);

        RayDesc rayDesc;
        rayDesc.Origin = pc.camPos;
        rayDesc.Direction = camDirection;
        rayDesc.TMin = 0.01;
        rayDesc.TMax = 10000.0;
        
        PrimaryPayload payload;
        payload.pcg = PrimaryPayload::generator_t::create(rnd());
        TraceRay(topLevelAS, RAY_FLAG_NONE, 0xff, ERT_PRIMARY, 0, EMT_PRIMARY, rayDesc, payload);

        const float32_t rayDistance = payload.rayDistance;
        if (rayDistance < 0)
        {
            hitValues += s_clearColor;
            continue;
        }

        const float32_t3 worldPosition = pc.camPos + (camDirection * rayDistance);

        // make sure to call with least live state
        RayLight cLight;
        cLight.inHitPosition = worldPosition;
        CallShader(pc.light.type, cLight);

        const float32_t3 worldNormal = payload.worldNormal;

        Material material;
        MaterialId materialId = payload.materialId;
        // we use negative index to indicate that this is a procedural geometry
        if (materialId.isHitProceduralGeom())
        {
            const MaterialPacked materialPacked = vk::RawBufferLoad<MaterialPacked>(pc.proceduralGeomInfoBuffer + materialId.getMaterialIndex() * sizeof(SProceduralGeomInfo));
            material = nbl::hlsl::_static_cast<Material>(materialPacked);
        }
        else
        {
            const MaterialPacked materialPacked = vk::RawBufferLoad<MaterialPacked>(pc.triangleGeomInfoBuffer + materialId.getMaterialIndex() * sizeof(STriangleGeomInfo));
            material = nbl::hlsl::_static_cast<Material>(materialPacked);
        }

        float32_t attenuation = 1;

        if (dot(worldNormal, cLight.outLightDir) > 0)
        {
            RayDesc rayDesc;
            rayDesc.Origin = worldPosition;
            rayDesc.Direction = cLight.outLightDir;
            rayDesc.TMin = 0.01;
            rayDesc.TMax = cLight.outLightDistance;

            uint32_t shadowRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
            OcclusionPayload occlusionPayload;
            occlusionPayload.attenuation = 1;
            TraceRay(topLevelAS, shadowRayFlags, 0xFF, ERT_OCCLUSION, 0, EMT_OCCLUSION, rayDesc, occlusionPayload);

            attenuation = occlusionPayload.attenuation;
            if (occlusionPayload.attenuation > 0.0001)
            {
                const float32_t3 diffuse = computeDiffuse(material, cLight.outLightDir, worldNormal);
                const float32_t3 specular = computeSpecular(material, camDirection, cLight.outLightDir, worldNormal);
                hitValues += (cLight.outIntensity * attenuation * (diffuse + specular));
            }
        }
        hitValues += material.ambient;
    }

    const float32_t3 hitValue = hitValues / s_sampleCount;

    if (pc.frameCounter > 0)
    {
        float32_t a = 1.0f / float32_t(pc.frameCounter + 1);
        float32_t3 oldColor = colorImage[coords].xyz;
        colorImage[coords] = float32_t4(lerp(oldColor, hitValue, a), 1.0f);
    }
    else
    {
        colorImage[coords] = float32_t4(hitValue, 1.0f);
    }
}
