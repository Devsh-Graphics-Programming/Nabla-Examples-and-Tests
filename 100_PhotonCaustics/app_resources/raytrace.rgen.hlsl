// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
//--------------------------------------------------------------------------
#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

#include "nbl/builtin/hlsl/random/pcg.hlsl" // for random seed
#include "nbl/builtin/hlsl/math/functions.hlsl" // why not some math
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl" // diffuse ray random direction thingy
//--------------------------------------------------------------------------
// Lets get some naming straight for debug views etc.
// L = light
// E = eye
// D = diffuse bounce
// S = specular bounce (metal or glass)
//--------------------------------------------------------------------------
// Resources
using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure sceneTLAS;
[[vk::binding(1, 0)]] RWTexture2D<float32_t4> hdrImage;
[[vk::binding(2, 0)]] RWTexture2D<float32_t4> accumulationImage;

[[vk::push_constant]] SPushConstants pc;
//--------------------------------------------------------------------------
[shader("raygeneration")]
void main()
{
    const uint32_t3 launchID   = spirv::LaunchIdKHR;
    const uint32_t3 launchSize = spirv::LaunchSizeKHR;
    const int32_t2  pixel      = int32_t2(launchID.xy);

    // Seed varies per pixel AND per frame
    const uint32_t pixelIndex = uint32_t(pixel.x) + uint32_t(pixel.y) * launchSize.x;
    const uint32_t frameHash  = random::PCG32::construct(pc.accumulatedFrames)();
    random::PCG32  rng        = random::PCG32::construct(pixelIndex + frameHash);

    // Multiple spp radiance accumulator
    float32_t3 pixelRadiance = (float32_t3)0.0f;

    for (uint32_t sampleIndex = 0; sampleIndex < NUM_SAMPLES_PER_PIXEL; ++sampleIndex)
    {
        // Jitter inside the pixel for each sample ==> FREE AA!
        const float32_t2 jitter = float32_t2(rnd(rng), rnd(rng)); // see already in [0...1] uniform range
        const float32_t2 uv     = (float32_t2(pixel) + jitter) / float32_t2(launchSize.xy);
        const float32_t2 ndc    = uv * 2.0f - 1.0f;

        // get the camera ray target in World space
        const float32_t4 tmp    = mul(pc.invMVP, float32_t4(ndc.x, ndc.y, 1.0f, 1.0f));
        const float32_t3 target = tmp.xyz / tmp.w;

        float32_t3 origin    = pc.camPos;
        float32_t3 direction = normalize(target - pc.camPos);

        // radiance   = light gathered back to the eye along this path
        // throughput = surviving fraction of it, attenuated at every bounce
        float32_t3 radiance   = (float32_t3)0.0f;
        float32_t3 throughput = (float32_t3)1.0f;

        for (uint32_t bounce = 0; bounce < NUM_MAX_BOUNCES; ++bounce)
        {
            [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
            RayPayload payload;
            payload.missed = false;

            spirv::traceRayKHR(sceneTLAS,
                                spv::RayFlagsOpaqueKHRMask,
                                0xFF, 0, 0, 0,
                                origin, RAY_TMIN, direction, RAY_TMAX, payload);

            // Light is found by walking into it, wven miss shader can contribute light which is why we test for misses after adding radiance
            radiance += throughput * payload.emission;

            if (payload.missed)
                break;

            // The whole Lambertian BRDF, after the pi and cosine cancel
            throughput *= payload.albedo; // whats left after absorption of light

            // A fully absorbed path can never contribute again, so kill it
            if (max(throughput.r, max(throughput.g, throughput.b)) <= 0.0f)
                break;

            // add offset to adjust going in and out of surfaces, get new origin
            origin = payload.position + payload.normal * RAY_ORIGIN_OFFSET;
                
            if (payload.metallic > 0.5f)
            {
                const float32_t3 reflected = reflect(direction, payload.normal);
                // if it goes inside, ignore it
                if (dot(reflected, payload.normal) <= 0.0f)
                    break;
                direction = reflected;
            }
            else if (payload.transmission > 0.5f)
            {
                // Determine the IOR on each side of the surface.
                const float32_t etaIncident    = payload.frontFace ? 1.0f : payload.ior;
                const float32_t etaTransmitted = payload.frontFace ? payload.ior : 1.0f;

                const float32_t etaRatio = etaIncident / etaTransmitted;

                // Cosine of the angle between the incident ray and the surface normal.
                const float32_t cosTheta = min(dot(-direction, payload.normal), 1.0f);

                const float32_t3 refracted = refract(direction, payload.normal, etaRatio);

                // total internal reflection, damn snell
                const bool tir = dot(refracted, refracted) == 0.0f;

                // prob of being reflected, if hits our seeded run, we reflect or we refract
                const float32_t reflectProb = tir ? 1.0f : fresnelDielectric(cosTheta, etaIncident, etaTransmitted);

                if (rnd(rng) < reflectProb)
                {
                    direction = reflect(direction, payload.normal);
                    origin    = payload.position + payload.normal * RAY_ORIGIN_OFFSET;
                }
                else
                {

                    // The refracted ray crosses to the opposite side of the surface,
                    // so offset the origin in the opposite direction.
                    direction = refracted;
                    origin    = payload.position - payload.normal * RAY_ORIGIN_OFFSET;
                }
            }
            else
            {
                // diffuse ==> so randomly walk somewhere
                direction = cosineSampleHemisphere(payload.normal, rng);
            }
        }
        pixelRadiance += radiance;
    }

    pixelRadiance /= float32_t(NUM_SAMPLES_PER_PIXEL);

    float32_t3 accumulated = pixelRadiance;
    if (pc.accumulatedFrames > 0)
        accumulated += accumulationImage[pixel].rgb;
    accumulationImage[pixel] = float32_t4(accumulated, 1.0f);

    const float32_t3 averagedAccumulation = accumulated / float32_t(pc.accumulatedFrames + 1);
    hdrImage[pixel] = float32_t4(averagedAccumulation, 1.0f);
}