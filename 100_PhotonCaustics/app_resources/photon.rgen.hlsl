// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
//--------------------------------------------------------------------------
#include "app_resources/common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl" // buffer device address, directly load into shader via dma instruction
//--------------------------------------------------------------------------
[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure sceneTLAS;

//--------------------------------------------------------------------------
void storePhotonAtomic(float32_t3 position, float32_t3 arrivedFrom, float32_t3 power)
{
    // Claim a slot with an atomic bump
    BdaAccessor<uint32_t> counter = BdaAccessor<uint32_t>::create(bda::__ptr<uint32_t>::create(pc.photonCounterBuffer));
    const uint32_t slot = counter.atomicAdd(0ull, 1u);

    // if we are full bail out
    if (slot >= pc.photonCount)
        return;

    SPhoton ph;
    ph.position  = position;
    ph.direction = arrivedFrom;
    ph.power     = power;

    const static uint64_t PhotonAlign = nbl::hlsl::alignment_of_v<SPhoton>;
    vk::BufferPointer<SPhoton, PhotonAlign>(pc.photonBuffer + slot * sizeof(SPhoton)).Get() = ph;
}
//--------------------------------------------------------------------------
[shader("raygeneration")]
void main()
{
    const uint32_t photonIx = spirv::LaunchIdKHR.x;
    const uint32_t lightCount = pc.lightCount;
    if (lightCount == 0)
        return;

    // hash seed on accumm frames + photon idx shouldn't be same as the seed for ray tracer
    const uint32_t s1 = random::PCG32::construct(pc.accumulatedFrames * 0x9E3779B9u)();
    const uint32_t s2 = random::PCG32::construct(photonIx)();
    random::PCG32 rng = random::PCG32::construct(s1 + s2);

    // Pick a light index randomly
    const uint32_t        lIdx       = min(uint32_t(rnd(rng) * lightCount), lightCount - 1);
    const static uint64_t LightAlign = nbl::hlsl::alignment_of_v<SLight>;
    const SLight          light      = vk::BufferPointer <SLight, LightAlign>(pc.lightBuffer + lIdx * sizeof(SLight)).Get();

    // pick a point on the light triangle and the initial random normal
    const float32_t u = rnd(rng), v = rnd(rng);
    // The sqrt is what keeps the density even, without it samples bunch at v0
    const float32_t su = sqrt(u);
    const float32_t3 start = light.v0 * (1.0f - su)
                           + light.v1 * (su * (1.0f - v))
                           + light.v2 * (su * v);
    const float32_t3 lightN = normalize(cross(light.v1 - light.v0, light.v2 - light.v0));

    // set initial direction and light flux
    float32_t3 origin    = start + lightN * RAY_ORIGIN_OFFSET;
    float32_t3 direction = cosineSampleHemisphere(lightN, rng);
    float32_t3 power     = light.emission
                            * (numbers::pi<float32_t> * light.area * float32_t(lightCount))
                            * pc.photonScale;

    uint32_t specularBounces = 0; // this is a caustic only map!

    for (uint32_t bounce = 0; bounce < NUM_MAX_BOUNCES; ++bounce)
    {
        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
        RayPayload payload;
        payload.missed = false;

        spirv::traceRayKHR(sceneTLAS, spv::RayFlagsOpaqueKHRMask, 0xFF, 0, 0, 0,
                           origin, RAY_TMIN, direction, RAY_TMAX, payload);
        if (payload.missed)
            break;

        const bool isMetal   = payload.metallic     > 0.5f;
        const bool isGlass   = payload.transmission > 0.5f;
        const bool isDiffuse = !isMetal && !isGlass;

        if (isDiffuse)
        {
            // Caustic-only map ==> store ONLY if the photon already bounced offsomething specular
            if (specularBounces > 0)
                storePhotonAtomic(payload.position, -direction, power);
            break;
        }

        origin = payload.position + payload.normal * RAY_ORIGIN_OFFSET;

        if (isMetal)
        {
            const float32_t3 reflected = reflect(direction, payload.normal);
            if (dot(reflected, payload.normal) <= 0.0f)
                break;
            direction = reflected;
            power *= payload.albedo;
            specularBounces++;
        } else {
            // Determine the IOR on each side of the surface.
            const float32_t etaIncident = payload.frontFace ? 1.0f : payload.ior;
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
                direction = refracted;
                origin    = payload.position - payload.normal * RAY_ORIGIN_OFFSET;
            }
            power *= payload.albedo;
            specularBounces++;
        }

        // same as thorughput if it's too less bail out this ray
        if (max(power.r, max(power.g, power.b)) <= 0.0f)
            break;
    }
}