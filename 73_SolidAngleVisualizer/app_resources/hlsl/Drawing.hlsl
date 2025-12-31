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

float drawCross2D(float2 fragPos, float2 center, float size, float thickness)
{
    float2 p = abs(fragPos - center);

    // Check if point is inside the cross (horizontal or vertical bar)
    bool inHorizontal = (p.x <= size && p.y <= thickness);
    bool inVertical = (p.y <= size && p.x <= thickness);

    return (inHorizontal || inVertical) ? 1.0f : 0.0f;
}

float4 drawHiddenEdges(float3 spherePos, uint32_t silEdgeMask, float aaWidth)
{
    float4 color = 0;
    float3 hiddenEdgeColor = float3(0.1, 0.1, 0.1);

    NBL_UNROLL
    for (int32_t i = 0; i < 12; i++)
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

float4 drawCorners(float2 p, float aaWidth)
{
    float4 color = 0;

    float dotSize = 0.02f;
    float innerDotSize = dotSize * 0.5f;

    for (int32_t i = 0; i < 8; i++)
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

float4 drawClippedSilhouetteVertices(float2 p, ClippedSilhouette silhouette, float aaWidth)
{
    float4 color = 0;
    float dotSize = 0.03f;

    for (uint i = 0; i < silhouette.count; i++)
    {
        float3 corner3D = normalize(silhouette.vertices[i]);
        float2 cornerPos = sphereToCircle(corner3D);

        float dist = length(p - cornerPos);

        // Smooth circle for the vertex
        float alpha = 1.0f - smoothstep(dotSize * 0.8f, dotSize, dist);

        if (alpha > 0.0f)
        {
            // Color gradient: Red (index 0) to Cyan (last index)
            // This helps verify the CCW winding order visually
            float t = float(i) / float(max(1u, silhouette.count - 1));
            float3 vertexColor = lerp(float3(1, 0, 0), float3(0, 1, 1), t);

            color += float4(vertexColor * alpha, alpha);
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

// Returns the number of visible faces and populates the faceIndices array
uint getVisibleFaces(int3 region, out uint faceIndices[3])
{
    uint count = 0;

    // Check X axis
    if (region.x == 0)
        faceIndices[count++] = 3; // X+
    else if (region.x == 2)
        faceIndices[count++] = 2; // X-

    // Check Y axis
    if (region.y == 0)
        faceIndices[count++] = 5; // Y+
    else if (region.y == 2)
        faceIndices[count++] = 4; // Y-

    // Check Z axis
    if (region.z == 0)
        faceIndices[count++] = 1; // Z+
    else if (region.z == 2)
        faceIndices[count++] = 0; // Z-

    return count;
}

float4 drawVisibleFaceOverlay(float3 spherePos, int3 region, float aaWidth)
{
    uint faceIndices[3];
    uint count = getVisibleFaces(region, faceIndices);
    float4 color = 0;

    for (uint i = 0; i < count; i++)
    {
        uint fIdx = faceIndices[i];
        float3 n = localNormals[fIdx];

        // Transform normal to world space (using the same logic as your corners)
        float3 worldNormal = -normalize(mul((float3x3)pc.modelMatrix, n));
        worldNormal.z = -worldNormal.z; // Invert Z for correct orientation

        // Very basic visualization: highlight if the sphere position
        // is generally pointing towards that face's normal
        float alignment = dot(spherePos, worldNormal);
        if (alignment > 0.95f)
        {
            // Use different colors for different face indices
            color += float4(colorLUT[fIdx % 24], 0.5f);
        }
    }
    return color;
}

// Check if a face on the hemisphere is visible from camera at origin
bool isFaceVisible(float3 faceCenter, float3 faceNormal)
{
    float3 viewVec = normalize(-faceCenter); // Vector from camera to face
    return dot(faceNormal, viewVec) > 0.0f;
}

float4 drawFaces(float3 spherePos, float aaWidth)
{
    float4 color = 0.0f;
    float3 p = normalize(spherePos);

    float3x3 rotMatrix = (float3x3)pc.modelMatrix;

    // Check each of the 6 faces
    for (int32_t faceIdx = 0; faceIdx < 6; faceIdx++)
    {
        float3 n_world = mul(rotMatrix, localNormals[faceIdx]);

        // Check if face is visible
        if (!isFaceVisible(faceCenters[faceIdx], n_world))
            continue;

        // Get the 4 corners of this face
        float3 faceVerts[4];
        for (int32_t i = 0; i < 4; i++)
        {
            int32_t cornerIdx = faceToCorners[faceIdx][i];
            faceVerts[i] = normalize(getVertex(cornerIdx));
        }

        // Compute face center for winding
        float3 faceCenter = float3(0, 0, 0);
        for (int32_t i = 0; i < 4; i++)
            faceCenter += faceVerts[i];
        faceCenter = normalize(faceCenter);

        // Check if point is inside this face
        bool isInside = true;
        float minDist = 1e10;

        for (int32_t i = 0; i < 4; i++)
        {
            float3 v0 = faceVerts[i];
            float3 v1 = faceVerts[(i + 1) % 4];

            // Skip edges behind camera
            if (v0.z < 0.0f && v1.z < 0.0f)
            {
                isInside = false;
                break;
            }

            // Great circle normal
            float3 edgeNormal = normalize(cross(v0, v1));

            // Ensure normal points inward
            if (dot(edgeNormal, faceCenter) < 0.0f)
                edgeNormal = -edgeNormal;

            float d = dot(p, edgeNormal);

            if (d < -1e-6f)
            {
                isInside = false;
                break;
            }

            minDist = min(minDist, abs(d));
        }

        if (isInside)
        {
            float alpha = smoothstep(0.0f, aaWidth * 2.0f, minDist);

            // Use colorLUT based on face index (0-5)
            float3 faceColor = colorLUT[faceIdx];

            float shading = saturate(p.z * 0.8f + 0.2f);
            color += float4(faceColor * shading * alpha, alpha);
        }
    }

    return color;
}

int32_t getEdgeVisibility(int32_t edgeIdx)
{

    // Adjacency of edges to faces
    // Corrected Adjacency of edges to faces
    static const int2 edgeToFaces[12] = {
        // Edge Index:  | allEdges[i]  | Shared Faces:

        /* 0 (0-1) */ {4, 0}, // Y- (4) and Z- (0)
        /* 1 (2-3) */ {5, 0}, // Y+ (5) and Z- (0)
        /* 2 (4-5) */ {4, 1}, // Y- (4) and Z+ (1)
        /* 3 (6-7) */ {5, 1}, // Y+ (5) and Z+ (1)

        /* 4 (0-2) */ {2, 0}, // X- (2) and Z- (0)
        /* 5 (1-3) */ {3, 0}, // X+ (3) and Z- (0)
        /* 6 (4-6) */ {2, 1}, // X- (2) and Z+ (1)
        /* 7 (5-7) */ {3, 1}, // X+ (3) and Z+ (1)

        /* 8 (0-4) */ {2, 4},  // X- (2) and Y- (4)
        /* 9 (1-5) */ {3, 4},  // X+ (3) and Y- (4)
        /* 10 (2-6) */ {2, 5}, // X- (2) and Y+ (5)
        /* 11 (3-7) */ {3, 5}  // X+ (3) and Y+ (5)
    };

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
    for (int32_t j = 0; j < 12; j++)
    {
        // getEdgeVisibility returns 1 for a silhouette edge based on 3D geometry
        if (getEdgeVisibility(j) == 1)
        {
            mask |= (1u << j);
        }
    }
    return mask;
}

void validateEdgeVisibility(uint32_t sil, int32_t vertexCount, uint32_t generatedSilMask)
{
    uint32_t mismatchAccumulator = 0;

    // The Ground Truth now represents the full 3D silhouette, clipped or not.
    uint32_t groundTruthMask = computeGroundTruthEdgeMask();

    // The comparison checks if the generated mask perfectly matches the full 3D ground truth.
    uint32_t mismatchMask = groundTruthMask ^ generatedSilMask;

    if (mismatchMask != 0)
    {
        NBL_UNROLL
        for (int32_t j = 0; j < 12; j++)
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
