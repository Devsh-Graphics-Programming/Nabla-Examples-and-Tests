// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _PHOTON_CAUSTICS_COMMON_HLSL_
#define _PHOTON_CAUSTICS_COMMON_HLSL_
//-----------------------------------------------------------------------------
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/basic.h"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#endif
//-----------------------------------------------------------------------------
// Defines/Constants
NBL_CONSTEXPR uint32_t NUM_MAX_BOUNCES       = 8;
NBL_CONSTEXPR uint32_t NUM_SAMPLES_PER_PIXEL = 16;
NBL_CONSTEXPR float    RAY_ORIGIN_OFFSET     = 1.0e-4f;
NBL_CONSTEXPR float    RAY_TMIN              = 1.0e-4f;
NBL_CONSTEXPR float    RAY_TMAX              = 10000.0f;
NBL_CONSTEXPR uint32_t DEBUG_PHOTONS_BIT     = 0x00000001u;
NBL_CONSTEXPR uint32_t DEBUG_PHOTON_MAP_DISABLE_SPECULAR_CONCENTRATION = 0x00000002u;
//-----------------------------------------------------------------------------
// Compat types
using namespace nbl::hlsl;

struct SPresentPushConstants
{
    float32_t exposure;
    uint32_t tonemapOperator;
};

struct SPushConstants
{
    float32_t4x4 invMVP; // inverse(projection * view), reconstructs the eye ray
    float32_t3 camPos;
    uint32_t accumulatedFrames;

    uint64_t geomInfoBuffer;
    uint64_t lightBuffer;
    uint64_t photonBuffer;

    uint32_t lightCount;
    uint32_t photonCount;
    float32_t gatherRadius;
    float32_t photonScale;

    uint32_t debugFlags;
};

// Sooo... we use custom geometry and scene data for the RayTracing pipeline which means we need our own DS for scene/vertex/mats
struct Material
{
    float32_t3 albedo;
    float32_t3 emission;
    float32_t  metallic;
    float32_t  roughness;
    float32_t  ior;
    float32_t  transmission;
};

struct SGeomInfo
{
    Material material;
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;   // 0 => implicit triangle list (primID*3)
    uint64_t normalBufferAddress;  // 0 => flat face normal from positions
    uint32_t indexType;            // 0 = uint16, 1 = uint32
    uint32_t normalType;           // 0 = R32G32B32_SFLOAT, 1 = R8G8B8A8_SNORM
};
#ifdef __HLSL_VERSION
NBL_REGISTER_OBJ_TYPE(SGeomInfo, 8)
#endif

struct SVertex
{
    float32_t3 position;
    float32_t3 normal;
};

// An emissive triangle. The eye pass finds lights by bumping into them, but
// photons have to START somewhere, so the light geometry needs an explicit
// list. Area is precomputed so the shader never redoes a cross product.
struct SLight
{
    float32_t3 v0, v1, v2;
    float32_t3 emission;
    float32_t area;
};
#ifdef __HLSL_VERSION
NBL_REGISTER_OBJ_TYPE(SLight, 4)
#endif

struct SPhoton
{
    float32_t3 position;
    float32_t3 direction;
    float32_t3 power;
};
#ifdef __HLSL_VERSION
NBL_REGISTER_OBJ_TYPE(SPhoton, 4)
#endif

// uniform grid over the scene, photons are binned into it at emission time
NBL_CONSTEXPR uint32_t GRID_DIM   = 16;
NBL_CONSTEXPR uint32_t GRID_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;

// uint64 first so they stay 8 byte aligned under scalar layout
struct SPhotonMapHeader
{
    uint64_t cellCountsAddr;
    uint64_t cellPhotonsAddr;
    float32_t3 photonMapCenter;
    float32_t photonMapRadius;
    float32_t3 gridMin;
    float32_t pad0;
    float32_t3 gridInvCellSize;
    uint32_t photonCounter;
};
#ifdef __HLSL_VERSION
NBL_REGISTER_OBJ_TYPE(SPhotonMapHeader, 8)
#endif

NBL_CONSTEXPR uint64_t PHOTON_COUNTER_OFFSET = 60;
NBL_CONSTEXPR uint64_t PHOTON_ARRAY_OFFSET = 64;

#ifdef __HLSL_VERSION
int32_t3 photonGridCoord(float32_t3 p, float32_t3 gridMin, float32_t3 gridInvCellSize)
{
    return clamp(int32_t3((p - gridMin) * gridInvCellSize), (int32_t3)0, (int32_t3)(GRID_DIM - 1));
}

uint32_t photonGridFlatten(int32_t3 c)
{
    return uint32_t((c.z * GRID_DIM + c.y) * GRID_DIM + c.x);
}
#endif

//--------------------------------------------------------------------------
// Common Functions
#ifdef __HLSL_VERSION
// Material eval
// normalize [0, 1] because we need this to add a random jitter UV offset in that range
// TODO: do we have UINT32_MAX in HLSL?
float32_t rnd(inout random::PCG32 rng)
{
    return float32_t(rng()) * (1.0f / 4294967296.0f);
}

// get normal aligned ranom direction for diffuse reflections
float32_t3 cosineSampleHemisphere(float32_t3 n, inout random::PCG32 rng)
{
    // sample in a canonical frame where the normal is +Z, then rotate to world
    const float32_t3 local = sampling::ProjectedHemisphere < float32_t > ::
    __generate(float32_t2(rnd(rng), rnd(rng)));

    // adjust to world normal
    float32_t3 tangent, bitangent;
    math::frisvad < float32_t3 > (n, tangent, bitangent);
    return normalize(tangent * local.x + bitangent * local.y + n * local.z);
}

// Fresnel Schlick approximation for Dielectrics:
// F = F0 + (1 - F0) * (1 - cosTheta)^5
//
// F0 = ((etaIncident - etaTransmitted) /
//       (etaIncident + etaTransmitted))^2
float32_t fresnelDielectric(float32_t cosTheta, float32_t etaI, float32_t etaT)
{
    const float32_t etaRatio = etaI / etaT;
    const float32_t f0Ratio = (1.0f - etaRatio) / (1.0f + etaRatio);
    const float32_t f0 = f0Ratio * f0Ratio;

    const float32_t m = 1.0f - clamp(cosTheta, 0.0f, 1.0f);
    const float32_t m2 = m * m;

    return f0 + (1.0f - f0) * (m2 * m2 * m);
}
#endif
//-----------------------------------------------------------------------------

#ifdef __HLSL_VERSION
struct [raypayload] RayPayload
{
    float32_t3 position     : read(caller) : write(closesthit);
    float32_t3 normal       : read(caller) : write(closesthit);
    float32_t3 albedo       : read(caller) : write(closesthit,miss);
    float32_t3 emission     : read(caller) : write(closesthit,miss);
    float32_t  metallic     : read(caller) : write(closesthit);
    float32_t  roughness    : read(caller) : write(closesthit);
    float32_t  ior          : read(caller) : write(closesthit);
    float32_t  transmission : read(caller) : write(closesthit);
    bool       frontFace    : read(caller) : write(closesthit);
    bool       missed       : read(caller) : write(closesthit,miss);
};
#endif

#endif
