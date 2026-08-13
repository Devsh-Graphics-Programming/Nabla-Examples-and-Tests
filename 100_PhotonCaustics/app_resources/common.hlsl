// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _PHOTON_CAUSTICS_COMMON_HLSL_
#define _PHOTON_CAUSTICS_COMMON_HLSL_
//-----------------------------------------------------------------------------
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/basic.h"
//-----------------------------------------------------------------------------
// Defines/Constants
NBL_CONSTEXPR uint32_t NUM_MAX_BOUNCES       = 8;
NBL_CONSTEXPR uint32_t NUM_SAMPLES_PER_PIXEL = 16;
NBL_CONSTEXPR float    RAY_ORIGIN_OFFSET     = 1.0e-4f;
NBL_CONSTEXPR float    RAY_TMIN              = 1.0e-4f;
NBL_CONSTEXPR float    RAY_TMAX              = 10000.0f;
//-----------------------------------------------------------------------------
// Compat types

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

enum NormalType : uint32_t
{
    NT_R8G8B8A8_SNORM,
    NT_R32G32B32_SFLOAT,
};
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
