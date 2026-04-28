//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include "debug_vis.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include "utils.hlsl"

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::push_constant]] struct PushConstantRayVis pc;

#include "drawing.hlsl"

struct RayVisOutput
{
    float32_t4 color : SV_Target0;
    float32_t depth : SV_Depth;
};

// [shader("pixel")]
[[vk::location(0)]] RayVisOutput main(SVertexAttributes vx)
{
    RayVisOutput output;
    output.color = float32_t4(0.0, 0.0, 0.0, 0.0);
    output.depth = 0.0;       // Far plane in reversed-Z (near=0, far=1)
    float32_t maxDepth = 0.0; // Track closest depth (minimum in reversed-Z)
    float32_t aaWidth = length(float32_t2(ddx(vx.uv.x), ddy(vx.uv.y)));

    // Convert to NDC space with aspect ratio correction
    float32_t2 ndcPos = vx.uv * 2.0f - 1.0f;
    float32_t aspect = pc.viewport.z / pc.viewport.w;
    ndcPos.x *= aspect;
    VisContext::begin(ndcPos, float32_t3(0, 0, 0), aaWidth);

    // Draw vertices in 3D
    for (uint32_t v = 0; v < DebugDataBuffer[0].silhouette.clippedVertexCount; v++)
    {
        float32_t4 clipPos = mul(pc.viewProjMatrix, float32_t4(DebugDataBuffer[0].silhouette.clippedVertices[v], 1.0));
        float32_t3 ndcPosVertex = clipPos.xyz / clipPos.w;
        ndcPosVertex.x *= aspect;
        if (ndcPosVertex.z < maxDepth)
            continue;

        float32_t4 intensity = SphereDrawer::drawDot(ndcPosVertex, 0.03, 0.0, colorLUT[DebugDataBuffer[0].silhouette.clippedVertexIndices[v]]);

        // Update depth only where we drew something
        if (intensity.a > 0.0)
        {
            VisContext::add(intensity);
            maxDepth = max(maxDepth, 1.0f - ndcPosVertex.z);
        }
    }

    // Draw sample rays
    for (uint32_t i = 0; i < DebugDataBuffer[0].sampling.sampleCount; i++)
    {
        float32_t3 rayOrigin = float32_t3(0, 0, 0);
        float32_t4 directionAndPdf = DebugDataBuffer[0].sampling.rayData[i];
        float32_t3 rayDir = normalize(directionAndPdf.xyz);

        shapes::OBBView<float32_t> obb = shapes::OBBView<float32_t>::create(pc.modelMatrix);
        shapes::OBBView<float32_t>::Intersection intersection = obb.rayIntersection(rayOrigin, rayDir);

        float32_t arrowLength;
        float32_t3 arrowColor;

        if (intersection.hit)
        {
            // Use tMax (exit point at back face)
            float32_t3 worldExitPoint = rayOrigin + rayDir * intersection.tMax;
            arrowLength = intersection.tMax;
            arrowColor = float32_t3(0.0, 1.0, 0.0); // Green for valid samples
        }
        else
        {
            // Ray doesn't intersect
            float32_t3 cubeCenter = obb.getCenter();
            arrowLength = length(cubeCenter - rayOrigin) + 2.0; // make it a little taller
            arrowColor = float32_t3(1.0, 0.0, 0.0); // Red for BROKEN samples
        }

        SphereDrawer::ArrowResult arrow = SphereDrawer::visualizeRayAsArrow(rayOrigin, directionAndPdf, arrowLength, ndcPos, aspect, pc.viewProjMatrix);

        // Only update depth if arrow was actually drawn
        if (arrow.color.a > 0.0)
        {
            maxDepth = max(maxDepth, arrow.depth);
        }

        // Modulate arrow color by its alpha (only add where arrow is visible)
        VisContext::add(float32_t4(arrowColor * arrow.color.a, 0.0));
        output.color.a = max(output.color.a, arrow.color.a);
    }

    // Clamp to prevent overflow
    output.color.rgb += VisContext::flush().rgb;
    output.color = saturate(output.color);
    output.color.a = 1.0;

    // Write the closest depth (minimum in reversed-Z)
    output.depth = maxDepth;

    return output;
}
