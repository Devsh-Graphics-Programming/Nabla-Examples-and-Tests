#include "common.hlsl"
#include "random.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

static const int32_t s_sampleCount = 10;

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1, 0)]] RWTexture2D<float32_t4> colorImage;

float32_t3 reinhardTonemap(float32_t3 v)
{
    return v / (1.0f + v);
}

[shader("raygeneration")]
void main()
{
    uint32_t3 launchID = DispatchRaysIndex();
    uint32_t3 launchSize = DispatchRaysDimensions();
    uint32_t2 coords = launchID.xy;
    uint32_t seed = tea(launchID.y * launchSize.x + launchID.x, pc.frameCounter);

    float32_t3 hitValues = float32_t3(0, 0, 0);
    for (uint32_t sample_i = 0; sample_i < s_sampleCount; sample_i++)
    {
        const float32_t r1 = rnd(seed);
        const float32_t r2 = rnd(seed);
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
        rayDesc.TMin = 0.001;
        rayDesc.TMax = 10000.0;
        
        HitPayload payload;
        payload.seed = seed;
        TraceRay(topLevelAS, RAY_FLAG_NONE, 0xff, ERT_PRIMARY, 0, EMT_PRIMARY, rayDesc, payload);

        if (payload.rayDistance < 0)
        {
            hitValues += float32_t3(0.3, 0.3, 0.3);
            continue;
        }

        const float32_t3 worldPosition = pc.camPos + (camDirection * payload.rayDistance);
        const float32_t3 worldNormal = payload.worldNormal;
        const Material material = unpackMaterial(payload.material);
        RayLight cLight;
        cLight.inHitPosition = worldPosition;
        CallShader(pc.light.type, cLight);

        const float32_t3 diffuse = computeDiffuse(material, cLight.outLightDir, worldNormal);
        float32_t3 specular = float32_t3(0, 0, 0);
        float32_t attenuation = 0;

        if (dot(worldNormal, cLight.outLightDir) > 0)
        {
            RayDesc rayDesc;
            rayDesc.Origin = worldPosition;
            rayDesc.Direction = cLight.outLightDir;
            rayDesc.TMin = 0.01;
            rayDesc.TMax = cLight.outLightDistance;

            ShadowPayload shadowPayload;
            shadowPayload.attenuation = -1; // negative attenuation indicate occlusion happening. will be multiplied by -1 in miss shader.
            TraceRay(topLevelAS, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, ERT_OCCLUSION, 0, EMT_OCCLUSION, rayDesc, shadowPayload);

            if (shadowPayload.attenuation > 0)
            {
                specular = computeSpecular(material, camDirection, cLight.outLightDir, worldNormal);
                attenuation = shadowPayload.attenuation;
            }
        }
        hitValues += (cLight.outIntensity * attenuation * (diffuse + specular));
    }

    float32_t3 hitValue = hitValues / s_sampleCount;

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
