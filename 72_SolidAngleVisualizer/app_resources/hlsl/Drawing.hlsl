#ifndef _DEBUG_HLSL_
#define _DEBUG_HLSL_
#include "common.hlsl"

float2 sphereToCircle(float3 spherePoint)
{
    if (spherePoint.z >= 0.0f)
    {
        return spherePoint.xy * CIRCLE_RADIUS;
    }
    else
    {
        float r2 = (1.0f - spherePoint.z) / (1.0f + spherePoint.z);
        float uv2Plus1 = r2 + 1.0f;
        return (spherePoint.xy * uv2Plus1 / 2.0f) * CIRCLE_RADIUS;
    }
}

float4 drawGreatCircleArc(float3 fragPos, float3 points[2], int visibility, float aaWidth)
{
    if (visibility == 0) return float4(0,0,0,0);
    
    float3 v0 = normalize(points[0]);
    float3 v1 = normalize(points[1]);
    float3 p = normalize(fragPos);
    
    float3 arcNormal = normalize(cross(v0, v1));
    float dist = abs(dot(p, arcNormal));
    
    float dotMid = dot(v0, v1);
    bool onArc = (dot(p, v0) >= dotMid) && (dot(p, v1) >= dotMid);
    
    if (!onArc) return float4(0,0,0,0);
    
    float avgDepth = (length(points[0]) + length(points[1])) * 0.5f;
    float depthScale = 3.0f / avgDepth;
    
    float baseWidth = (visibility == 1) ? 0.01f : 0.005f;
    float width = min(baseWidth * depthScale, 0.02f);
    
    float alpha = 1.0f - smoothstep(width - aaWidth, width + aaWidth, dist);
    
    float4 edgeColor = (visibility == 1) ? 
        float4(0.0f, 0.5f, 1.0f, 1.0f) :
        float4(1.0f, 0.0f, 0.0f, 1.0f);
    
    float intensity = (visibility == 1) ? 1.0f : 0.5f;
    return edgeColor * alpha * intensity;
}

float4 drawHiddenEdges(float3 spherePos, uint32_t silEdgeMask, float aaWidth)
{
    float4 color = float4(0,0,0,0);
    float3 hiddenEdgeColor = float3(0.1, 0.1, 0.1);
    
    for (int i = 0; i < 12; i++)
    {
        if ((silEdgeMask & (1u << i)) == 0)
        {
            int2 edge = allEdges[i];
            float3 edgePoints[2] = { corners[edge.x], corners[edge.y] };
            float4 edgeContribution = drawGreatCircleArc(spherePos, edgePoints, 1, aaWidth);
            color += float4(hiddenEdgeColor * edgeContribution.a, edgeContribution.a);
        }
    }
    return color;
}

float4 drawCorners(float3 spherePos, float2 p, float aaWidth)
{
    float4 color = float4(0,0,0,0);
    for (int i = 0; i < 8; i++)
    {
        float3 corner3D = normalize(corners[i]);
        float2 cornerPos = sphereToCircle(corner3D);
        float dist = length(p - cornerPos);
        float dotSize = 0.02f;
        float dotAlpha = 1.0f - smoothstep(dotSize - aaWidth, dotSize + aaWidth, dist);
        if (dotAlpha > 0.0f)
        {
            float3 dotColor = colorLUT[i];
            color += float4(dotColor * dotAlpha, dotAlpha);
        }
    }
    return color;
}

float4 drawRing(float2 p, float aaWidth)
{
    float positionLength = length(p);
    float ringWidth = 0.002f;
    float ringDistance = abs(positionLength - CIRCLE_RADIUS);
    float ringAlpha = 1.0f - smoothstep(ringWidth - aaWidth, ringWidth + aaWidth, ringDistance);
    return ringAlpha * float4(1, 1, 1, 1);
}

// Check if a face on the hemisphere is visible from camera at origin
bool isFaceVisible(float3 faceCenter, float3 faceNormal)
{
    float3 viewVec = normalize(-faceCenter); // Vector from camera to face
    return dot(faceNormal, viewVec) > 0.0f;
}

int getEdgeVisibility(int edgeIdx)
{
    int2 faces = edgeToFaces[edgeIdx];

    // Transform normals to world space
    float3x3 rotMatrix = (float3x3)pc.modelMatrix;
    float3 n_world_f1 = mul(rotMatrix, localNormals[faces.x]);
    float3 n_world_f2 = mul(rotMatrix, localNormals[faces.y]);

    bool visible1 = isFaceVisible(faceCenters[faces.x], n_world_f1);
    bool visible2 = isFaceVisible(faceCenters[faces.y], n_world_f2);

    // Silhouette: exactly one face visible
    if (visible1 != visible2) return 1;

    // Inner edge: both faces visible
    if (visible1 && visible2) return 2;

    // Hidden edge: both faces hidden
    return 0;
}

#if DEBUG_DATA
uint32_t computeGroundTruthEdgeMask()
{
    uint32_t mask = 0u;
    NBL_UNROLL
    for (int j = 0; j < 12; j++)
    {
        // getEdgeVisibility returns 1 for a silhouette edge based on 3D geometry
        if (getEdgeVisibility(j) == 1)
        {
            mask |= (1u << j);
        }
    }
    return mask;
}

void validateEdgeVisibility(uint32_t sil, int vertexCount, uint32_t generatedSilMask)
{
    uint32_t mismatchAccumulator = 0;

    // The Ground Truth now represents the full 3D silhouette, clipped or not.
    uint32_t groundTruthMask = computeGroundTruthEdgeMask();

    // The comparison checks if the generated mask perfectly matches the full 3D ground truth.
    uint32_t mismatchMask = groundTruthMask ^ generatedSilMask;

    if (mismatchMask != 0)
    {
        NBL_UNROLL
        for (int j = 0; j < 12; j++)
        {
            if ((mismatchMask >> j) & 1u)
            {
                int2 edge = allEdges[j];
                // Accumulate vertex indices where error occurred
                mismatchAccumulator |= (1u << edge.x) | (1u << edge.y);
            }
        }
    }
    
    // Simple Write (assuming all fragments calculate the same result)
    InterlockedOr(DebugDataBuffer[0].edgeVisibilityMismatch, mismatchAccumulator);
}
#endif


#endif // _DEBUG_HLSL_
