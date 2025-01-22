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

        float32_t3 direction = normalize(targetPos - pc.camPos);

        RayDesc rayDesc;
        rayDesc.Origin = pc.camPos;
        rayDesc.Direction = direction;
        rayDesc.TMin = 0.01;
        rayDesc.TMax = 1000.0;
        
        ColorPayload payload;
        payload.seed = seed;
        TraceRay(topLevelAS, RAY_FLAG_NONE, 0xff, 0, 0, 0, rayDesc, payload);

        hitValues += payload.hitValue;
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
