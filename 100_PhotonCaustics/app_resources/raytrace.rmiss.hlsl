// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
//
// Port of VK_PathTracing/shaders/miss.rmiss. Matches your current shader
// exactly: sky sampling is disabled (black miss), so the only light in the
// scene is whatever emissive geometry you place. Reporting the miss through
// 'emission' means raygen needs no special case -- the same
// 'radiance += throughput * emission' line handles both this and any
// emitter the ray actually hits.
// RT stage is inferred from the [shader("miss")] attribute below.
#include "app_resources/common.hlsl"

using namespace nbl::hlsl;

[shader("miss")]
void main(inout RayPayload payload)
{
    // TODO: https://www.shadertoy.com/view/wslyWs
    payload.emission = (float32_t3) 0.0f; // could sample env HDR here... using gl_WorldRayDirectionEXT as direction;
    payload.missed   = true;
}