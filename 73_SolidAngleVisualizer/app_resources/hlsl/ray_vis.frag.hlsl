//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include "utils.hlsl"

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

// Visualizes a ray as an arrow from origin in NDC space
//  Returns color (rgb), intensity (a), and depth (in extra component)
struct ArrowResult
{
    float32_t4 color : SV_Target0;
    float32_t depth : SV_Depth;
};

[[vk::push_constant]] struct PushConstantRayVis pc;

#if VISUALIZE_SAMPLES
#include "drawing.hlsl"

// Ray-AABB intersection in world space
// Returns the distance to the nearest intersection point, or -1 if no hit
float32_t rayAABBIntersection(float32_t3 rayOrigin, float32_t3 rayDir, float32_t3 aabbMin, float32_t3 aabbMax)
{
    float32_t3 invDir = 1.0f / rayDir;
    float32_t3 t0 = (aabbMin - rayOrigin) * invDir;
    float32_t3 t1 = (aabbMax - rayOrigin) * invDir;

    float32_t3 tmin = min(t0, t1);
    float32_t3 tmax = max(t0, t1);

    float32_t tNear = max(max(tmin.x, tmin.y), tmin.z);
    float32_t tFar = min(min(tmax.x, tmax.y), tmax.z);

    // Check if ray intersects AABB
    if (tNear > tFar || tFar < 0.0)
        return -1.0;

    // Return the nearest positive intersection
    return tNear >= 0.0 ? tNear : tFar;
}

// Project 3D point to NDC space
float32_t2 projectToNDC(float32_t3 worldPos, float32_t4x4 viewProj, float32_t aspect)
{
    float32_t4 clipPos = mul(viewProj, float32_t4(worldPos, 1.0));
    clipPos /= clipPos.w;

    // Apply aspect ratio correction
    clipPos.x *= aspect;

    return clipPos.xy;
}

ArrowResult visualizeRayAsArrow(float32_t3 rayOrigin, float32_t4 directionAndPdf, float32_t arrowLength, float32_t2 ndcPos, float32_t aspect)
{
    ArrowResult result;
    result.color = float32_t4(0, 0, 0, 0);
    result.depth = 0.0; // Far plane in reversed-Z

    float32_t3 rayDir = normalize(directionAndPdf.xyz);
    float32_t pdf = directionAndPdf.w;

    // Define the 3D line segment
    float32_t3 worldStart = rayOrigin;
    float32_t3 worldEnd = rayOrigin + rayDir * arrowLength;

    // Transform to view space (camera space) for clipping
    float32_t4x4 viewMatrix = pc.viewProjMatrix; // If you have view matrix separately, use that
    // For now, we'll work in clip space and check w values

    float32_t4 clipStart = mul(pc.viewProjMatrix, float32_t4(worldStart, 1.0));
    float32_t4 clipEnd = mul(pc.viewProjMatrix, float32_t4(worldEnd, 1.0));

    // Clip against near plane (w = 0 plane in clip space)
    // If both points are behind camera, reject
    if (clipStart.w <= 0.001 && clipEnd.w <= 0.001)
        return result;

    // If line crosses the near plane, clip it
    float32_t t0 = 0.0;
    float32_t t1 = 1.0;

    if (clipStart.w <= 0.001)
    {
        // Start is behind camera, clip to near plane
        float32_t t = (0.001 - clipStart.w) / (clipEnd.w - clipStart.w);
        t0 = saturate(t);
        clipStart = lerp(clipStart, clipEnd, t0);
        worldStart = lerp(worldStart, worldEnd, t0);
    }

    if (clipEnd.w <= 0.001)
    {
        // End is behind camera, clip to near plane
        float32_t t = (0.001 - clipStart.w) / (clipEnd.w - clipStart.w);
        t1 = saturate(t);
        clipEnd = lerp(clipStart, clipEnd, t1);
        worldEnd = lerp(worldStart, worldEnd, t1);
    }

    // Now check if the clipped segment is valid
    if (t0 >= t1)
        return result;

    // Perspective divide to NDC
    float32_t2 ndcStart = clipStart.xy / clipStart.w;
    float32_t2 ndcEnd = clipEnd.xy / clipEnd.w;

    // Apply aspect ratio correction
    ndcStart.x *= aspect;
    ndcEnd.x *= aspect;

    // Calculate arrow direction in NDC
    float32_t2 arrowVec = ndcEnd - ndcStart;
    float32_t arrowNDCLength = length(arrowVec);

    // Skip if arrow is too small on screen
    if (arrowNDCLength < 0.005)
        return result;

    // Calculate perpendicular distance to line segment in NDC space
    float32_t2 toPixel = ndcPos - ndcStart;
    float32_t t_ndc = saturate(dot(toPixel, arrowVec) / dot(arrowVec, arrowVec));

    // Draw line shaft
    float32_t lineThickness = 0.002;
    float32_t lineIntensity = lineSegment(ndcPos, ndcStart, ndcEnd, lineThickness);

    // Calculate perspective-correct depth
    if (lineIntensity > 0.0)
    {
        // Interpolate in clip space
        float32_t4 clipPos = lerp(clipStart, clipEnd, t_ndc);

        // Compute NDC depth for reversed-Z
        float32_t depthNDC = clipPos.z / clipPos.w;
        result.depth = 1.0f - depthNDC;

        // Clip against valid depth range
        if (result.depth < 0.0 || result.depth > 1.0)
        {
            lineIntensity = 0.0;
        }
    }

    // Modulate by PDF
    float32_t pdfIntensity = saturate(pdf * 0.5);
    float32_t3 finalColor = float32_t3(pdfIntensity, pdfIntensity, pdfIntensity);

    result.color = float32_t4(finalColor, lineIntensity);
    return result;
}

// Returns both tMin (entry) and tMax (exit) for ray-AABB intersection
struct AABBIntersection
{
    float32_t tMin; // Distance to front face (entry point)
    float32_t tMax; // Distance to back face (exit point)
    bool hit;       // Whether ray intersects the AABB at all
};

AABBIntersection rayAABBIntersectionFull(float32_t3 origin, float32_t3 dir, float32_t3 boxMin, float32_t3 boxMax)
{
    AABBIntersection result;
    result.hit = false;
    result.tMin = 0.0f;
    result.tMax = 0.0f;

    float32_t3 invDir = 1.0f / dir;
    float32_t3 t0 = (boxMin - origin) * invDir;
    float32_t3 t1 = (boxMax - origin) * invDir;

    float32_t3 tmin = min(t0, t1);
    float32_t3 tmax = max(t0, t1);

    result.tMin = max(max(tmin.x, tmin.y), tmin.z);
    result.tMax = min(min(tmax.x, tmax.y), tmax.z);

    // Ray intersects if tMax >= tMin and tMax > 0
    result.hit = (result.tMax >= result.tMin) && (result.tMax > 0.0f);

    // If we're inside the box, tMin will be negative
    // In that case, we want to use tMax (exit point)
    if (result.tMin < 0.0f)
        result.tMin = 0.0f;

    return result;
}
#endif // VISUALIZE_SAMPLES

// [shader("pixel")]
[[vk::location(0)]] ArrowResult main(SVertexAttributes vx)
{
    ArrowResult output;
#if VISUALIZE_SAMPLES
    output.color = float32_t4(0.0, 0.0, 0.0, 0.0);
    output.depth = 0.0;       // Far plane in reversed-Z (near=0, far=1)
    float32_t maxDepth = 0.0; // Track closest depth (minimum in reversed-Z)
    float32_t aaWidth = length(float32_t2(ddx(vx.uv.x), ddy(vx.uv.y)));

    // Convert to NDC space with aspect ratio correction
    float32_t2 ndcPos = vx.uv * 2.0f - 1.0f;
    float32_t aspect = pc.viewport.z / pc.viewport.w;
    ndcPos.x *= aspect;

    for (uint32_t v = 0; v < DebugDataBuffer[0].clippedSilhouetteVertexCount; v++)
    {
        float32_t4 clipPos = mul(pc.viewProjMatrix, float32_t4(DebugDataBuffer[0].clippedSilhouetteVertices[v], 1.0));
        float32_t3 ndcPosVertex = clipPos.xyz / clipPos.w;
        if (ndcPosVertex.z < maxDepth)
            continue;

        float32_t4 intensity = drawCorner(ndcPosVertex, ndcPos, aaWidth, 0.03, 0.0, colorLUT[DebugDataBuffer[0].clippedSilhouetteVerticesIndices[v]]);

        // Update depth only where we drew something
        if (any(intensity.rgb > 0.0))
        {
            output.color.rgb += intensity.rgb;
            maxDepth = max(maxDepth, 1.0f - ndcPosVertex.z);
        }
    }

    uint32_t sampleCount = DebugDataBuffer[0].sampleCount;

    for (uint32_t i = 0; i < sampleCount; i++)
    {
        float32_t3 rayOrigin = float32_t3(0, 0, 0);
        float32_t4 directionAndPdf = DebugDataBuffer[0].rayData[i];
        float32_t3 rayDir = normalize(directionAndPdf.xyz);

        // Define cube bounds in local space
        float32_t3 cubeLocalMin = float32_t3(-0.5, -0.5, -0.5);
        float32_t3 cubeLocalMax = float32_t3(0.5, 0.5, 0.5);

        // Transform ray to local space of the cube (using precomputed inverse)
        float32_t3 localRayOrigin = mul(pc.invModelMatrix, float32_t4(rayOrigin, 1.0)).xyz;
        float32_t3 localRayDir = normalize(mul(pc.invModelMatrix, float32_t4(rayDir, 0.0)).xyz);

        // Get both entry and exit distances
        AABBIntersection intersection = rayAABBIntersectionFull(localRayOrigin, localRayDir, cubeLocalMin, cubeLocalMax);

        float32_t arrowLength;
        float32_t3 arrowColor;

        if (intersection.hit)
        {
            // Use tMax (exit point at back face) instead of tMin (entry point at front face)
            float32_t3 localExitPoint = localRayOrigin + localRayDir * intersection.tMax;
            float32_t3 worldExitPoint = mul(pc.modelMatrix, float32_t4(localExitPoint, 1.0)).xyz;
            arrowLength = length(worldExitPoint - rayOrigin);
            arrowColor = float32_t3(0.0, 1.0, 0.0); // Green for valid samples
        }
        else
        {
            // Ray doesn't intersect - THIS SHOULD NEVER HAPPEN with correct sampling!
            float32_t3 cubeCenter = mul(pc.modelMatrix, float32_t4(0, 0, 0, 1)).xyz;
            arrowLength = length(cubeCenter - rayOrigin) + 2.0;
            arrowColor = float32_t3(1.0, 0.0, 0.0); // Red for BROKEN samples
        }

        ArrowResult arrow = visualizeRayAsArrow(rayOrigin, directionAndPdf, arrowLength, ndcPos, aspect);

        // Only update depth if arrow was actually drawn
        if (arrow.color.a > 0.0)
        {
            maxDepth = max(maxDepth, arrow.depth);
        }

        // Modulate arrow color by its alpha (only add where arrow is visible)
        output.color.rgb += arrowColor * arrow.color.a;
        output.color.a = max(output.color.a, arrow.color.a);
    }

    // Clamp to prevent overflow
    output.color = saturate(output.color);
    output.color.a = 1.0;

    // Write the closest depth (minimum in reversed-Z)
    output.depth = maxDepth;

#endif
    return output;
}
