// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
//--------------------------------------------------------------------------
#include "app_resources/common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
//--------------------------------------------------------------------------
using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;
//--------------------------------------------------------------------------
// Vertex/normal fetch adapted from 71_RayTracingPipeline
float32_t3 calculateNormal(uint32_t primID, SGeomInfo geom, float32_t2 bary)
{
    uint32_t3 indices;
    if (geom.indexBufferAddress == 0)
    {
        indices[0] = primID * 3;
        indices[1] = indices[0] + 1;
        indices[2] = indices[0] + 2;
    }
    else
    {
        if (geom.indexType == 0) // 16-bit
            indices = uint32_t3((bda::__ptr<uint16_t3>::create(geom.indexBufferAddress) + primID).deref().load());
        else // 32-bit
            indices = uint32_t3((bda::__ptr<uint32_t3>::create(geom.indexBufferAddress) + primID).deref().load());
    }

    if (geom.normalBufferAddress == 0)
    {
        // No normal buffer: flat face normal from the positions themselves.
        const float32_t3 v0 = (bda::__ptr<float32_t3>::create(geom.vertexBufferAddress) + indices[0]).deref().load();
        const float32_t3 v1 = (bda::__ptr<float32_t3>::create(geom.vertexBufferAddress) + indices[1]).deref().load();
        const float32_t3 v2 = (bda::__ptr<float32_t3>::create(geom.vertexBufferAddress) + indices[2]).deref().load();
        return normalize(cross(v2 - v0, v1 - v0));
    }

    float32_t3 n0, n1, n2;
    if (geom.normalType == 1) // R8G8B8A8_SNORM, packed
    {
        const uint32_t p0 = (bda::__ptr<uint32_t>::create(geom.normalBufferAddress) + indices[0]).deref().load();
        const uint32_t p1 = (bda::__ptr<uint32_t>::create(geom.normalBufferAddress) + indices[1]).deref().load();
        const uint32_t p2 = (bda::__ptr<uint32_t>::create(geom.normalBufferAddress) + indices[2]).deref().load();
        n0 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(p0).xyz);
        n1 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(p1).xyz);
        n2 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(p2).xyz);
    }
    else // R32G32B32_SFLOAT
    {
        n0 = normalize((bda::__ptr<float32_t3>::create(geom.normalBufferAddress) + indices[0]).deref().load());
        n1 = normalize((bda::__ptr<float32_t3>::create(geom.normalBufferAddress) + indices[1]).deref().load());
        n2 = normalize((bda::__ptr<float32_t3>::create(geom.normalBufferAddress) + indices[2]).deref().load());
    }

    // The intersection reports only two barycentrics; the third belongs to
    // vertex 0, so it must come FIRST for bary[i] to weight vertex i.
    const float32_t3 baryFull = float32_t3(1.0f - bary.x - bary.y, bary.x, bary.y);
    return baryFull.x * n0 + baryFull.y * n1 + baryFull.z * n2;
}

//--------------------------------------------------------------------------
[shader("closesthit")]
void main(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const uint32_t primID      = spirv::PrimitiveId;
    const uint32_t instanceIdx = spirv::InstanceCustomIndexKHR;

    const static uint64_t GeomAlign = nbl::hlsl::alignment_of_v<SGeomInfo>;
    const SGeomInfo geom = vk::BufferPointer<SGeomInfo, GeomAlign>(pc.geomInfoBuffer + instanceIdx * sizeof(SGeomInfo)).Get();

    // Interpolate in object space, then object -> world
    float32_t3 normal = calculateNormal(primID, geom, attribs.barycentrics);
    normal = normalize(mul(normal, transpose(spirv::WorldToObjectKHR)).xyz);

    // Make the shading normal oppose the incoming ray, couldnt find GLSL frontFace...
    const bool frontFace = dot(WorldRayDirection(), normal) < 0.0f;
    if (!frontFace)
        normal = -normal;

    payload.position     = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.normal       = normal;
    payload.albedo       = geom.material.albedo;
    payload.emission     = geom.material.emission;
    payload.metallic     = geom.material.metallic;
    payload.roughness    = geom.material.roughness;
    payload.ior          = geom.material.ior;
    payload.transmission = geom.material.transmission;
    payload.frontFace    = frontFace;
    payload.missed       = false;
}