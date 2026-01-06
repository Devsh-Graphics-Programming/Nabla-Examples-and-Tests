#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include "utils.hlsl"

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::push_constant]] struct PushConstantRayVis pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<ResultData> DebugDataBuffer;
#define VISUALIZE_SAMPLES 1
#include "Drawing.hlsl"

// Ray-AABB intersection in world space
// Returns the distance to the nearest intersection point, or -1 if no hit
float32_t rayAABBIntersection(float32_t3 rayOrigin, float32_t3 rayDir, float32_t3 aabbMin, float32_t3 aabbMax)
{
    float32_t3 invDir = 1.0 / rayDir;
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

// Visualizes a ray as an arrow from origin in NDC space
//  Returns color (rgb), intensity (a), and depth (in extra component)
struct ArrowResult
{
    float32_t4 color : SV_Target0;
    float32_t depth : SV_Depth;
};

ArrowResult visualizeRayAsArrow(float32_t3 rayOrigin, float32_t4 directionAndPdf, float32_t arrowLength, float32_t2 ndcPos, float32_t aspect)
{
    ArrowResult result;
    result.color = float32_t4(0, 0, 0, 0);
    result.depth = 0.0;

    float32_t3 rayDir = normalize(directionAndPdf.xyz);
    float32_t pdf = directionAndPdf.w;

    float32_t3 rayEnd = rayOrigin + rayDir * arrowLength;

    // Project start and end points to NDC space
    float32_t2 ndcStart = projectToNDC(rayOrigin, pc.viewProjMatrix, aspect);
    float32_t2 ndcEnd = projectToNDC(rayEnd, pc.viewProjMatrix, aspect);

    // Get clip space positions
    float32_t4 clipStart = mul(pc.viewProjMatrix, float32_t4(rayOrigin, 1.0));
    float32_t4 clipEnd = mul(pc.viewProjMatrix, float32_t4(rayEnd, 1.0));

    // Calculate arrow properties in NDC space
    float32_t arrowNDCLength = length(ndcEnd - ndcStart);

    // Skip if arrow is too small on screen (in NDC units)
    if (arrowNDCLength < 0.01)
        return result;

    // Calculate the parametric position along the arrow shaft IN NDC
    float32_t2 pa = ndcPos - ndcStart;
    float32_t2 ba = ndcEnd - ndcStart;
    float32_t t_ndc = saturate(dot(pa, ba) / dot(ba, ba));

    // Draw line shaft
    float32_t lineThickness = 0.002;
    float32_t lineIntensity = lineSegment(ndcPos, ndcStart, ndcEnd, lineThickness);

    // Calculate depth at this pixel's position along the arrow
    if (lineIntensity > 0.0)
    {
        // Interpolate in CLIP space for perspective-correct depth
        float32_t4 clipPos = lerp(clipStart, clipEnd, t_ndc);
        float32_t depthNDC = clipPos.z / clipPos.w;

        // Convert to reversed depth [0,1] -> [1,0]
        result.depth = 1.0 - depthNDC;

        // Clip against depth range (like hardware would)
        // In reversed depth: near=1.0, far=0.0
        if (result.depth < 0.0 || result.depth > 1.0)
        {
            lineIntensity = 0.0; // Outside depth range, clip it
        }
    }

    // Modulate by PDF
    float32_t pdfIntensity = saturate(pdf * 0.5);

    float32_t3 finalColor = pdfIntensity;

    result.color = float32_t4(finalColor, lineIntensity);
    return result;
}

// Transform a point by inverse of model matrix (world to local space)
float32_t3 worldToLocal(float32_t3 worldPos, float32_t3x4 modelMatrix)
{
    // Manually construct 4x4 from 3x4
    float32_t4x4 model4x4 = float32_t4x4(
        modelMatrix[0],
        modelMatrix[1],
        modelMatrix[2],
        float32_t4(0.0, 0.0, 0.0, 1.0));
    float32_t4x4 invModel = inverse(model4x4);
    return mul(invModel, float32_t4(worldPos, 1.0)).xyz;
}

// Transform a direction by inverse of model matrix (no translation)
float32_t3 worldToLocalDir(float32_t3 worldDir, float32_t3x4 modelMatrix)
{
    // Manually construct 4x4 from 3x4
    float32_t4x4 model4x4 = float32_t4x4(
        modelMatrix[0],
        modelMatrix[1],
        modelMatrix[2],
        float32_t4(0.0, 0.0, 0.0, 1.0));
    float32_t4x4 invModel = inverse(model4x4);
    return mul(invModel, float32_t4(worldDir, 0.0)).xyz;
}
[[vk::location(0)]] ArrowResult main(SVertexAttributes vx)
{
    ArrowResult output;
    output.color = float32_t4(0.0, 0.0, 0.0, 0.0);
    output.depth = 0.0;       // Default to far plane in reversed depth
    float32_t maxDepth = 0.0; // Track the closest depth (maximum in reversed depth)

    // Convert to NDC space with aspect ratio correction
    float32_t2 ndcPos = vx.uv * 2.0f - 1.0f;
    float32_t aspect = pc.viewport.z / pc.viewport.w;
    ndcPos.x *= aspect;

    // Draw clipped silhouett vertices using drawCorners()
    for (uint32_t v = 0; v < DebugDataBuffer[0].clippedSilhouetteVertexCount; v++)
    {
        float32_t4 clipPos = mul(pc.viewProjMatrix, float32_t4(DebugDataBuffer[0].clippedSilhouetteVertices[v], 1.0));
        float32_t3 ndcPosVertex = clipPos.xyz / clipPos.w; // Perspective divide to get NDC

        float32_t4 intensity = drawCorner(ndcPosVertex, ndcPos, 0.005, 0.01, 0.01, float32_t3(1.0, 0.0, 0.0));

        output.color += intensity;
        output.depth = intensity > 0.0 ? 1.0 : output.depth; // Update depth
        maxDepth = max(maxDepth, output.depth);
    }

    int sampleCount = DebugDataBuffer[0].sampleCount;

    for (int i = 0; i < sampleCount; i++)
    {
        float32_t3 rayOrigin = float32_t3(0, 0, 0);
        float32_t4 directionAndPdf = DebugDataBuffer[0].rayData[i];
        float32_t3 rayDir = normalize(directionAndPdf.xyz);

        // Define cube bounds in local space (unit cube from -0.5 to 0.5, adjust as needed)
        float32_t3 cubeLocalMin = float32_t3(-0.5, -0.5, -0.5);
        float32_t3 cubeLocalMax = float32_t3(0.5, 0.5, 0.5);

        // Transform ray to local space of the cube
        float32_t3 localRayOrigin = worldToLocal(rayOrigin, pc.modelMatrix);
        float32_t3 localRayDir = normalize(worldToLocalDir(rayDir, pc.modelMatrix));

        // Perform intersection test in local space
        float32_t hitDistance = rayAABBIntersection(localRayOrigin, localRayDir, cubeLocalMin, cubeLocalMax);

        float32_t arrowLength;
        if (hitDistance > 0.0)
        {
            // Calculate world space hit distance
            // We need to account for the scaling in the model matrix
            float32_t3 localHitPoint = localRayOrigin + localRayDir * hitDistance;
            float32_t3 worldHitPoint = mul(pc.modelMatrix, float32_t4(localHitPoint, 1.0)).xyz;
            arrowLength = length(worldHitPoint - rayOrigin);
        }
        else
        {
            // No intersection, use fallback (e.g., fixed length or distance to cube center)
            float32_t3 cubeCenter = mul(pc.modelMatrix, float32_t4(0, 0, 0, 1)).xyz;
            arrowLength = length(cubeCenter - rayOrigin) + 2.0;
        }

        ArrowResult arrow = visualizeRayAsArrow(rayOrigin, directionAndPdf, arrowLength, ndcPos, aspect);
        maxDepth = max(maxDepth, arrow.depth);

        // Additive blending
        output.color.rgb += hitDistance > 0.0 ? arrow.color.rgb : float32_t3(1.0, 0.0, 0.0);
        output.color.a = max(output.color.a, arrow.color.a);
    }

    // Clamp to prevent overflow
    output.color = saturate(output.color);
    output.color.a = 1.0;

    // Write the closest depth (maximum in reversed depth)
    // ONLY write depth if we actually drew something
    output.depth = output.color.a > 0.0 ? maxDepth : 0.0;

    return output;
}