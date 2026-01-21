#ifndef _DEBUG_HLSL_
#define _DEBUG_HLSL_

#include "common.hlsl"
#include "gpu_common.hlsl"

// Check if a face on the hemisphere is visible from camera at origin
bool isFaceVisible(float32_t3 faceCenter, float32_t3 faceNormal)
{
    float32_t3 viewVec = normalize(-faceCenter); // Vector from camera to face
    return dot(faceNormal, viewVec) > 0.0f;
}

// doesn't change Z coordinate
float32_t3 sphereToCircle(float32_t3 spherePoint)
{
    if (spherePoint.z >= 0.0f)
    {
        return float32_t3(spherePoint.xy * CIRCLE_RADIUS, spherePoint.z);
    }
    else
    {
        float32_t r2 = (1.0f - spherePoint.z) / (1.0f + spherePoint.z);
        float32_t uv2Plus1 = r2 + 1.0f;
        return float32_t3((spherePoint.xy * uv2Plus1 / 2.0f) * CIRCLE_RADIUS, spherePoint.z);
    }
}

#if VISUALIZE_SAMPLES

float32_t drawGreatCircleArc(float32_t3 fragPos, float32_t3 points[2], float32_t aaWidth, float32_t width = 0.01f)
{
    float32_t3 v0 = normalize(points[0]);
    float32_t3 v1 = normalize(points[1]);
    float32_t3 ndc = normalize(fragPos);

    float32_t3 arcNormal = normalize(cross(v0, v1));
    float32_t dist = abs(dot(ndc, arcNormal));

    float32_t dotMid = dot(v0, v1);
    bool onArc = (dot(ndc, v0) >= dotMid) && (dot(ndc, v1) >= dotMid);

    if (!onArc)
        return 0.0f;

    float32_t avgDepth = (length(points[0]) + length(points[1])) * 0.5f;
    float32_t depthScale = 3.0f / avgDepth;

    width = min(width * depthScale, 0.02f);
    float32_t alpha = 1.0f - smoothstep(width - aaWidth, width + aaWidth, dist);

    return alpha;
}

float32_t drawCross2D(float32_t2 fragPos, float32_t2 center, float32_t size, float32_t thickness)
{
    float32_t2 ndc = abs(fragPos - center);

    // Check if point is inside the cross (horizontal or vertical bar)
    bool inHorizontal = (ndc.x <= size && ndc.y <= thickness);
    bool inVertical = (ndc.y <= size && ndc.x <= thickness);

    return (inHorizontal || inVertical) ? 1.0f : 0.0f;
}

float32_t4 drawHiddenEdges(float32_t3x4 modelMatrix, float32_t3 spherePos, uint32_t silEdgeMask, float32_t aaWidth)
{
    float32_t4 color = 0;
    float32_t3 hiddenEdgeColor = float32_t3(0.1, 0.1, 0.1);

    NBL_UNROLL
    for (uint32_t i = 0; i < 12; i++)
    {
        // skip silhouette edges
        if (silEdgeMask & (1u << i))
            continue;

        uint32_t2 edge = allEdges[i];

        float32_t3 v0 = normalize(getVertex(modelMatrix, edge.x));
        float32_t3 v1 = normalize(getVertex(modelMatrix, edge.y));

        bool neg0 = v0.z < 0.0f;
        bool neg1 = v1.z < 0.0f;

        // fully hidden
        if (neg0 && neg1)
            continue;

        float32_t3 p0 = v0;
        float32_t3 p1 = v1;

        // clip if needed
        if (neg0 ^ neg1)
        {
            float32_t t = v0.z / (v0.z - v1.z);
            float32_t3 clip = normalize(lerp(v0, v1, t));

            p0 = neg0 ? clip : v0;
            p1 = neg1 ? clip : v1;
        }

        float32_t3 pts[2] = {p0, p1};
        float32_t c = drawGreatCircleArc(spherePos, pts, aaWidth, 0.003f);
        color += float32_t4(hiddenEdgeColor * c, c);
    }

    return color;
}

float32_t4 drawCorner(float32_t3 cornerNDCPos, float32_t2 ndc, float32_t aaWidth, float32_t dotSize, float32_t innerDotSize, float32_t3 dotColor)
{
    float32_t4 color = float32_t4(0, 0, 0, 0);
    float32_t dist = length(ndc - cornerNDCPos.xy);

    // outer dot
    float32_t outerAlpha = 1.0f - smoothstep(dotSize - aaWidth,
                                             dotSize + aaWidth,
                                             dist);

    if (outerAlpha <= 0.0f)
        return color;

    color += float32_t4(dotColor * outerAlpha, outerAlpha);

    // -------------------------------------------------
    // inner black dot for hidden corners
    // -------------------------------------------------
    if (cornerNDCPos.z < 0.0f && innerDotSize > 0.0)
    {
        float32_t innerAlpha = 1.0f - smoothstep(innerDotSize - aaWidth,
                                                 innerDotSize + aaWidth,
                                                 dist);

        // ensure it stays inside the outer dot
        innerAlpha *= outerAlpha;

        color -= float32_t4(innerAlpha.xxx, 0.0f);
    }

    return color;
}

// Draw a line segment in NDC space
float32_t lineSegment(float32_t2 ndc, float32_t2 a, float32_t2 b, float32_t thickness)
{
    float32_t2 pa = ndc - a;
    float32_t2 ba = b - a;
    float32_t h = saturate(dot(pa, ba) / dot(ba, ba));
    float32_t dist = length(pa - ba * h);
    return smoothstep(thickness, thickness * 0.5, dist);
}

// Draw an arrow head (triangle) in NDC space
float32_t arrowHead(float32_t2 ndc, float32_t2 tip, float32_t2 direction, float32_t size)
{
    // Create perpendicular vector
    float32_t2 perp = float32_t2(-direction.y, direction.x);

    // Three points of the arrow head triangle
    float32_t2 p1 = tip;
    float32_t2 p2 = tip - direction * size + perp * size * 0.5;
    float32_t2 p3 = tip - direction * size - perp * size * 0.5;

    // Check if point is inside triangle using barycentric coordinates
    float32_t2 v0 = p3 - p1;
    float32_t2 v1 = p2 - p1;
    float32_t2 v2 = ndc - p1;

    float32_t dot00 = dot(v0, v0);
    float32_t dot01 = dot(v0, v1);
    float32_t dot02 = dot(v0, v2);
    float32_t dot11 = dot(v1, v1);
    float32_t dot12 = dot(v1, v2);

    float32_t invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    float32_t u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float32_t v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    bool inside = (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0);

    // Add some antialiasing
    float32_t minDist = min(min(
                                length(ndc - p1),
                                length(ndc - p2)),
                            length(ndc - p3));

    return inside ? 1.0 : smoothstep(0.02, 0.0, minDist);
}

// Helper to draw an edge with proper color mapping
float32_t4 drawEdge(uint32_t originalEdgeIdx, float32_t3 pts[2], float32_t3 spherePos, float32_t aaWidth, float32_t width = 0.003f)
{
    float32_t4 edgeContribution = drawGreatCircleArc(spherePos, pts, aaWidth, width);
    return float32_t4(colorLUT[originalEdgeIdx] * edgeContribution.a, edgeContribution.a);
};

float32_t4 drawCorners(float32_t3x4 modelMatrix, float32_t2 ndc, float32_t aaWidth, float32_t dotSize)
{
    float32_t4 color = float32_t4(0, 0, 0, 0);

    float32_t innerDotSize = dotSize * 0.5f;

    for (uint32_t i = 0; i < 8; i++)
    {
        float32_t3 cornerCirclePos = sphereToCircle(normalize(getVertex(modelMatrix, i)));
        color += drawCorner(cornerCirclePos, ndc, aaWidth, dotSize, 0.0, colorLUT[i]);
    }

    return color;
}

float32_t4 drawClippedSilhouetteVertices(float32_t2 ndc, ClippedSilhouette silhouette, float32_t aaWidth)
{
    float32_t4 color = 0;
    float32_t dotSize = 0.03f;

    for (uint i = 0; i < silhouette.count; i++)
    {
        float32_t3 cornerCirclePos = sphereToCircle(normalize(silhouette.vertices[i]));
        float32_t dist = length(ndc - cornerCirclePos.xy);

        // Smooth circle for the vertex
        float32_t alpha = 1.0f - smoothstep(dotSize * 0.8f, dotSize, dist);

        if (alpha > 0.0f)
        {
            // Color gradient: Red (index 0) to Cyan (last index)
            // This helps verify the CCW winding order visually
            float32_t t = float32_t(i) / float32_t(max(1u, silhouette.count - 1));
            float32_t3 vertexColor = lerp(float32_t3(1, 0, 0), float32_t3(0, 1, 1), t);

            color += float32_t4(vertexColor * alpha, alpha);
        }
    }
    return color;
}

float32_t4 drawRing(float32_t2 ndc, float32_t aaWidth)
{
    float32_t positionLength = length(ndc);
    float32_t ringWidth = 0.003f;
    float32_t ringDistance = abs(positionLength - CIRCLE_RADIUS);
    float32_t ringAlpha = 1.0f - smoothstep(ringWidth - aaWidth, ringWidth + aaWidth, ringDistance);
    return ringAlpha * float32_t4(1, 1, 1, 1);
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

float32_t4 drawVisibleFaceOverlay(float32_t3x4 modelMatrix, float32_t3 spherePos, int3 region, float32_t aaWidth)
{
    uint faceIndices[3];
    uint count = getVisibleFaces(region, faceIndices);

    float32_t4 color = 0;

    for (uint i = 0; i < count; i++)
    {
        uint fIdx = faceIndices[i];
        float32_t3 n = localNormals[fIdx];

        // Transform normal to world space (using the same logic as your corners)
        float32_t3 worldNormal = -normalize(mul((float3x3)modelMatrix, n));
        worldNormal.z = -worldNormal.z; // Invert Z for correct orientation

        // Very basic visualization: highlight if the sphere position
        // is generally pointing towards that face's normal
        float32_t alignment = dot(spherePos, worldNormal);
        if (alignment > 0.95f)
        {
            // Use different colors for different face indices
            color += float32_t4(colorLUT[fIdx % 24], 0.5f);
        }
    }
    return color;
}

float32_t4 drawFaces(float32_t3x4 modelMatrix, float32_t3 spherePos, float32_t aaWidth)
{
    float32_t4 color = 0.0f;
    float32_t3 ndc = normalize(spherePos);

    float3x3 rotMatrix = (float3x3)modelMatrix;

    // Check each of the 6 faces
    for (uint32_t faceIdx = 0; faceIdx < 6; faceIdx++)
    {
        float32_t3 n_world = mul(rotMatrix, localNormals[faceIdx]);

        // Check if face is visible
        if (!isFaceVisible(faceCenters[faceIdx], n_world))
            continue;

        // Get the 4 corners of this face
        float32_t3 faceVerts[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            uint32_t cornerIdx = faceToCorners[faceIdx][i];
            faceVerts[i] = normalize(getVertex(modelMatrix, cornerIdx));
        }

        // Compute face center for winding
        float32_t3 faceCenter = float32_t3(0, 0, 0);
        for (uint32_t i = 0; i < 4; i++)
            faceCenter += faceVerts[i];
        faceCenter = normalize(faceCenter);

        // Check if point is inside this face
        bool isInside = true;
        float32_t minDist = 1e10;

        for (uint32_t i = 0; i < 4; i++)
        {
            float32_t3 v0 = faceVerts[i];
            float32_t3 v1 = faceVerts[(i + 1) % 4];

            // Skip edges behind camera
            if (v0.z < 0.0f && v1.z < 0.0f)
            {
                isInside = false;
                break;
            }

            // Great circle normal
            float32_t3 edgeNormal = normalize(cross(v0, v1));

            // Ensure normal points inward
            if (dot(edgeNormal, faceCenter) < 0.0f)
                edgeNormal = -edgeNormal;

            float32_t d = dot(ndc, edgeNormal);

            if (d < -1e-6f)
            {
                isInside = false;
                break;
            }

            minDist = min(minDist, abs(d));
        }

        if (isInside)
        {
            float32_t alpha = smoothstep(0.0f, aaWidth * 2.0f, minDist);

            // Use colorLUT based on face index (0-5)
            float32_t3 faceColor = colorLUT[faceIdx];

            float32_t shading = saturate(ndc.z * 0.8f + 0.2f);
            color += float32_t4(faceColor * shading * alpha, alpha);
        }
    }

    return color;
}

#endif // VISUALIZE_SAMPLES

#if DEBUG_DATA

uint32_t getEdgeVisibility(float32_t3x4 modelMatrix, uint32_t edgeIdx)
{

    // Adjacency of edges to faces
    // Corrected Adjacency of edges to faces
    static const uint32_t2 edgeToFaces[12] = {
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

    uint32_t2 faces = edgeToFaces[edgeIdx];

    // Transform normals to world space
    float3x3 rotMatrix = (float3x3)modelMatrix;
    float32_t3 n_world_f1 = mul(rotMatrix, localNormals[faces.x]);
    float32_t3 n_world_f2 = mul(rotMatrix, localNormals[faces.y]);

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

uint32_t computeGroundTruthEdgeMask(float32_t3x4 modelMatrix)
{
    uint32_t mask = 0u;
    NBL_UNROLL
    for (uint32_t j = 0; j < 12; j++)
    {
        // getEdgeVisibility returns 1 for a silhouette edge based on 3D geometry
        if (getEdgeVisibility(modelMatrix, j) == 1)
        {
            mask |= (1u << j);
        }
    }
    return mask;
}

void validateEdgeVisibility(float32_t3x4 modelMatrix, uint32_t sil, uint32_t vertexCount, uint32_t generatedSilMask)
{
    uint32_t mismatchAccumulator = 0;

    // The Ground Truth now represents the full 3D silhouette, clipped or not.
    uint32_t groundTruthMask = computeGroundTruthEdgeMask(modelMatrix);

    // The comparison checks if the generated mask perfectly matches the full 3D ground truth.
    uint32_t mismatchMask = groundTruthMask ^ generatedSilMask;

    if (mismatchMask != 0)
    {
        NBL_UNROLL
        for (uint32_t j = 0; j < 12; j++)
        {
            if ((mismatchMask >> j) & 1u)
            {
                uint32_t2 edge = allEdges[j];
                // Accumulate vertex indices where error occurred
                mismatchAccumulator |= (1u << edge.x) | (1u << edge.y);
            }
        }
    }

    // Simple Write (assuming all fragments calculate the same result)
    InterlockedOr(DebugDataBuffer[0].edgeVisibilityMismatch, mismatchAccumulator);
}
#endif // DEBUG_DATA

#endif // _DEBUG_HLSL_
