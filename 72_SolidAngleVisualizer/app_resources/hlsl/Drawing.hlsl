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

float drawGreatCircleArc(float3 fragPos, float3 points[2], float aaWidth, float width = 0.01f)
{
    float3 v0 = normalize(points[0]);
    float3 v1 = normalize(points[1]);
    float3 p = normalize(fragPos);

    float3 arcNormal = normalize(cross(v0, v1));
    float dist = abs(dot(p, arcNormal));

    float dotMid = dot(v0, v1);
    bool onArc = (dot(p, v0) >= dotMid) && (dot(p, v1) >= dotMid);

    if (!onArc)
        return 0.0f;

    float avgDepth = (length(points[0]) + length(points[1])) * 0.5f;
    float depthScale = 3.0f / avgDepth;

    width = min(width * depthScale, 0.02f);
    float alpha = 1.0f - smoothstep(width - aaWidth, width + aaWidth, dist);

    return alpha;
}

float4 drawHiddenEdges(float3 spherePos, uint32_t silEdgeMask, float aaWidth)
{
    float4 color = 0;
    float3 hiddenEdgeColor = float3(0.1, 0.1, 0.1);

    NBL_UNROLL
    for (int i = 0; i < 12; i++)
    {
        // skip silhouette edges
        if (silEdgeMask & (1u << i))
            continue;

        int2 edge = allEdges[i];

        float3 v0 = normalize(getVertex(edge.x));
        float3 v1 = normalize(getVertex(edge.y));

        bool neg0 = v0.z < 0.0f;
        bool neg1 = v1.z < 0.0f;

        // fully hidden
        if (neg0 && neg1)
            continue;

        float3 p0 = v0;
        float3 p1 = v1;

        // clip if needed
        if (neg0 ^ neg1)
        {
            float t = v0.z / (v0.z - v1.z);
            float3 clip = normalize(lerp(v0, v1, t));

            p0 = neg0 ? clip : v0;
            p1 = neg1 ? clip : v1;
        }

        float3 pts[2] = {p0, p1};
        float4 c = drawGreatCircleArc(spherePos, pts, aaWidth, 0.005f);
        color += float4(hiddenEdgeColor * c.a, c.a);
    }

    return color;
}

float4 drawCorners(float3 spherePos, float2 p, float aaWidth)
{
    float4 color = 0;

    float dotSize = 0.02f;
    float innerDotSize = dotSize * 0.5f;

    for (int i = 0; i < 8; i++)
    {
        float3 corner3D = normalize(getVertex(i));
        float2 cornerPos = sphereToCircle(corner3D);

        float dist = length(p - cornerPos);

        // outer dot
        float outerAlpha = 1.0f - smoothstep(dotSize - aaWidth,
                                             dotSize + aaWidth,
                                             dist);

        if (outerAlpha <= 0.0f)
            continue;

        float3 dotColor = colorLUT[i];
        color += float4(dotColor * outerAlpha, outerAlpha);

        // -------------------------------------------------
        // inner black dot for hidden corners
        // -------------------------------------------------
        if (corner3D.z < 0.0f)
        {
            float innerAlpha = 1.0f - smoothstep(innerDotSize - aaWidth,
                                                 innerDotSize + aaWidth,
                                                 dist);

            // ensure it stays inside the outer dot
            innerAlpha *= outerAlpha;

            float3 innerColor = float3(0.0, 0.0, 0.0);
            color -= float4(innerAlpha.xxx, 0.0f);
        }
    }

    return color;
}

float4 drawRing(float2 p, float aaWidth)
{
    float positionLength = length(p);
    float ringWidth = 0.003f;
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
    if (visible1 != visible2)
        return 1;

    // Inner edge: both faces visible
    if (visible1 && visible2)
        return 2;

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
