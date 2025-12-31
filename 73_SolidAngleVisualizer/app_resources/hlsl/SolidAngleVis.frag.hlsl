#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include "utils.hlsl"

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::push_constant]] struct PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<ResultData> DebugDataBuffer;

static const float CIRCLE_RADIUS = 0.5f;

// --- Geometry Utils ---
struct ClippedSilhouette
{
    float32_t3 vertices[7];
    uint32_t count;
};

static const float32_t3 constCorners[8] = {
    float32_t3(-1, -1, -1), float32_t3(1, -1, -1), float32_t3(-1, 1, -1), float32_t3(1, 1, -1),
    float32_t3(-1, -1, 1), float32_t3(1, -1, 1), float32_t3(-1, 1, 1), float32_t3(1, 1, 1)};

static const int32_t2 allEdges[12] = {
    {0, 1},
    {2, 3},
    {4, 5},
    {6, 7}, // X axis
    {0, 2},
    {1, 3},
    {4, 6},
    {5, 7}, // Y axis
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7}, // Z axis
};

// Maps face index (0-5) to its 4 corner indices in CCW order
static const uint32_t faceToCorners[6][4] = {
    {0, 2, 3, 1}, // Face 0: Z-
    {4, 5, 7, 6}, // Face 1: Z+
    {0, 4, 6, 2}, // Face 2: X-
    {1, 3, 7, 5}, // Face 3: X+
    {0, 1, 5, 4}, // Face 4: Y-
    {2, 6, 7, 3}  // Face 5: Y+
};

static float32_t3 corners[8];
static float32_t3 faceCenters[6] = {
    float32_t3(0, 0, 0), float32_t3(0, 0, 0), float32_t3(0, 0, 0),
    float32_t3(0, 0, 0), float32_t3(0, 0, 0), float32_t3(0, 0, 0)};

static const float32_t3 localNormals[6] = {
    float32_t3(0, 0, -1), // Face 0 (Z-)
    float32_t3(0, 0, 1),  // Face 1 (Z+)
    float32_t3(-1, 0, 0), // Face 2 (X-)
    float32_t3(1, 0, 0),  // Face 3 (X+)
    float32_t3(0, -1, 0), // Face 4 (Y-)
    float32_t3(0, 1, 0)   // Face 5 (Y+)
};

// TODO: unused, remove later
// Vertices are ordered CCW relative to the camera view.
static const int32_t silhouettes[27][7] = {
    {6, 1, 3, 2, 6, 4, 5},   // 0: Black
    {6, 2, 6, 4, 5, 7, 3},   // 1: White
    {6, 0, 4, 5, 7, 3, 2},   // 2: Gray
    {6, 1, 3, 7, 6, 4, 5},   // 3: Red
    {4, 4, 5, 7, 6, -1, -1}, // 4: Green
    {6, 0, 4, 5, 7, 6, 2},   // 5: Blue
    {6, 0, 1, 3, 7, 6, 4},   // 6: Yellow
    {6, 0, 1, 5, 7, 6, 4},   // 7: Magenta
    {6, 0, 1, 5, 7, 6, 2},   // 8: Cyan
    {6, 1, 3, 2, 6, 7, 5},   // 9: Orange
    {4, 2, 6, 7, 3, -1, -1}, // 10: Light Orange
    {6, 0, 4, 6, 7, 3, 2},   // 11: Dark Orange
    {4, 1, 3, 7, 5, -1, -1}, // 12: Pink
    {6, 0, 4, 6, 7, 3, 2},   // 13: Light Pink
    {4, 0, 4, 6, 2, -1, -1}, // 14: Deep Rose
    {6, 0, 1, 3, 7, 5, 4},   // 15: Purple
    {4, 0, 1, 5, 4, -1, -1}, // 16: Light Purple
    {6, 0, 1, 5, 4, 6, 2},   // 17: Indigo
    {6, 0, 2, 6, 7, 5, 1},   // 18: Dark Green
    {6, 0, 2, 6, 7, 3, 1},   // 19: Lime
    {6, 0, 4, 6, 7, 3, 1},   // 20: Forest Green
    {6, 0, 2, 3, 7, 5, 1},   // 21: Navy
    {4, 0, 2, 3, 1, -1, -1}, // 22: Sky Blue
    {6, 0, 4, 6, 2, 3, 1},   // 23: Teal
    {6, 0, 2, 3, 7, 5, 4},   // 24: Brown
    {6, 0, 2, 3, 1, 5, 4},   // 25: Tan/Beige
    {6, 1, 5, 4, 6, 2, 3}    // 26: Dark Brown
};

// Binary packed silhouettes
static const uint32_t binSilhouettes[27] = {
    0b11000000000000101100110010011001,
    0b11000000000000011111101100110010,
    0b11000000000000010011111101100000,
    0b11000000000000101100110111011001,
    0b10000000000000000000110111101100,
    0b11000000000000010110111101100000,
    0b11000000000000100110111011001000,
    0b11000000000000100110111101001000,
    0b11000000000000010110111101001000,
    0b11000000000000101111110010011001,
    0b10000000000000000000011111110010,
    0b11000000000000010011111110100000,
    0b10000000000000000000101111011001,
    0b11000000000000010011111110100000,
    0b10000000000000000000010110100000,
    0b11000000000000100101111011001000,
    0b10000000000000000000100101001000,
    0b11000000000000010110100101001000,
    0b11000000000000001101111110010000,
    0b11000000000000001011111110010000,
    0b11000000000000001011111110100000,
    0b11000000000000001101111011010000,
    0b10000000000000000000001011010000,
    0b11000000000000001011010110100000,
    0b11000000000000100101111011010000,
    0b11000000000000100101001011010000,
    0b11000000000000011010110100101001,
};

int32_t getSilhouetteVertex(uint32_t packedSil, int32_t index)
{
    return (packedSil >> (3 * index)) & 0x7;
}

// Get silhouette size
int32_t getSilhouetteSize(uint32_t sil)
{
    return (sil >> 29) & 0x7;
}

// Check if vertex has negative z
bool getVertexZNeg(int32_t vertexIdx)
{
#if FAST
    float32_t3 localPos = float32_t3(
        (vertexIdx & 1) ? 1.0f : -1.0f,
        (vertexIdx & 2) ? 1.0f : -1.0f,
        (vertexIdx & 4) ? 1.0f : -1.0f);

    float transformedZ = dot(pc.modelMatrix[2].xyz, localPos) + pc.modelMatrix[2].w;
    return transformedZ < 0.0f;
#else
    return corners[vertexIdx].z < 0.0f;
#endif
}

// Get world position of cube vertex
float32_t3 getVertex(int32_t vertexIdx)
{
#if FAST
    // Reconstruct local cube corner from index bits
    float sx = (vertexIdx & 1) ? 1.0f : -1.0f;
    float sy = (vertexIdx & 2) ? 1.0f : -1.0f;
    float sz = (vertexIdx & 4) ? 1.0f : -1.0f;

    float32_t4x3 model = transpose(pc.modelMatrix);

    // Transform to world
    // Full position, not just Z like getVertexZNeg
    return model[0].xyz * sx +
           model[1].xyz * sy +
           model[2].xyz * sz +
           model[3].xyz;
    // return mul(pc.modelMatrix, float32_t4(sx, sy, sz, 1.0f));
#else
    return corners[vertexIdx];
#endif
}

#include "Drawing.hlsl"
#include "Sampling.hlsl"

void setDebugData(uint32_t sil, int32_t3 region, int32_t configIndex)
{
#if DEBUG_DATA
    DebugDataBuffer[0].region = uint32_t3(region);
    DebugDataBuffer[0].silhouetteIndex = uint32_t(configIndex);
    DebugDataBuffer[0].silhouetteVertexCount = uint32_t(getSilhouetteSize(sil));
    for (int32_t i = 0; i < 6; i++)
    {
        DebugDataBuffer[0].vertices[i] = uint32_t(getSilhouetteVertex(sil, i));
    }
    DebugDataBuffer[0].silhouette = sil;
#endif
}

float32_t2 toCircleSpace(float32_t2 uv)
{
    float32_t2 p = uv * 2.0f - 1.0f;
    float aspect = pc.viewport.z / pc.viewport.w;
    p.x *= aspect;
    return p;
}

uint32_t packSilhouette(const int32_t s[7])
{
    uint32_t packed = 0;
    int32_t size = s[0] & 0x7; // 3 bits for size

    // Pack vertices LSB-first (vertex1 in lowest 3 bits above size)
    for (int32_t i = 1; i <= 6; ++i)
    {
        int32_t v = s[i];
        if (v < 0)
            v = 0;                            // replace unused vertices with 0
        packed |= (v & 0x7) << (3 * (i - 1)); // vertex i-1 shifted by 3*(i-1)
    }

    // Put size in the MSB (bits 29-31 for a 32-bit uint32_t, leaving 29 bits for vertices)
    packed |= (size & 0x7) << 29;

    return packed;
}

void computeCubeGeo()
{
    for (int32_t i = 0; i < 8; i++)
        corners[i] = mul(pc.modelMatrix, float32_t4(constCorners[i], 1.0f)).xyz;

    for (int32_t f = 0; f < 6; f++)
    {
        faceCenters[f] = float32_t3(0, 0, 0);
        for (int32_t v = 0; v < 4; v++)
            faceCenters[f] += corners[faceToCorners[f][v]];
        faceCenters[f] /= 4.0f;
    }
}

// Helper to draw an edge with proper color mapping
float32_t4 drawEdge(int32_t originalEdgeIdx, float32_t3 pts[2], float32_t3 spherePos, float aaWidth, float width = 0.01f)
{
    float32_t4 edgeContribution = drawGreatCircleArc(spherePos, pts, aaWidth, width);
    return float32_t4(colorLUT[originalEdgeIdx] * edgeContribution.a, edgeContribution.a);
};

float32_t4 computeSilhouette(uint32_t vertexCount, uint32_t sil, float32_t3 spherePos, float aaWidth, out ClippedSilhouette silhouette)
{
    float32_t4 color = float32_t4(0, 0, 0, 0);
    silhouette.count = 0;

    // Build clip mask (z < 0)
    int32_t clipMask = 0u;
    NBL_UNROLL
    for (int32_t i = 0; i < 4; i++)
        clipMask |= (getVertexZNeg(getSilhouetteVertex(sil, i)) ? 1u : 0u) << i;

    if (vertexCount == 6)
    {
        NBL_UNROLL
        for (int32_t i = 4; i < 6; i++)
            clipMask |= (getVertexZNeg(getSilhouetteVertex(sil, i)) ? 1u : 0u) << i;
    }

    int32_t clipCount = countbits(clipMask);

#if 0
    // Early exit if fully clipped
    if (clipCount == vertexCount)
        return color;

    // No clipping needed - fast path
    if (clipCount == 0)
    {
        for (int32_t i = 0; i < vertexCount; i++)
        {
            int32_t i0 = i;
            int32_t i1 = (i + 1) % vertexCount;

            float32_t3 v0 = getVertex(getSilhouetteVertex(sil, i0));
            float32_t3 v1 = getVertex(getSilhouetteVertex(sil, i1));
            float32_t3 pts[2] = {v0, v1};

            color += drawEdge(i1, pts, spherePos, aaWidth);
        }
        return color;
    }
#endif

    // Rotate clip mask so positives come first
    uint32_t invertedMask = ~clipMask & ((1u << vertexCount) - 1u);
    bool wrapAround = ((clipMask & 1u) != 0u) &&
                      ((clipMask & (1u << (vertexCount - 1))) != 0u);
    int32_t rotateAmount = wrapAround
                               ? firstbitlow(invertedMask)   // -> First POSITIVE
                               : firstbithigh(clipMask) + 1; // -> First vertex AFTER last negative

    uint32_t rotatedClipMask = rotr(clipMask, rotateAmount, vertexCount);
    uint32_t rotatedSil = rotr(sil, rotateAmount * 3, vertexCount * 3);

    int32_t positiveCount = vertexCount - clipCount;

    // ALWAYS compute both clip points
    int32_t lastPosIdx = positiveCount - 1;
    int32_t firstNegIdx = positiveCount;
    float32_t3 vLastPos = getVertex(getSilhouetteVertex(rotatedSil, lastPosIdx));
    float32_t3 vFirstNeg = getVertex(getSilhouetteVertex(rotatedSil, firstNegIdx));
    float t = vLastPos.z / (vLastPos.z - vFirstNeg.z);
    float32_t3 clipA = lerp(vLastPos, vFirstNeg, t);

    float32_t3 vLastNeg = getVertex(getSilhouetteVertex(rotatedSil, vertexCount - 1));
    float32_t3 vFirstPos = getVertex(getSilhouetteVertex(rotatedSil, 0));
    t = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
    float32_t3 clipB = lerp(vLastNeg, vFirstPos, t);

    // Draw positive edges
    NBL_UNROLL
    for (int32_t i = 0; i < positiveCount; i++)
    {
        float32_t3 v0 = getVertex(getSilhouetteVertex(rotatedSil, i));

        // ONLY use clipA if we are at the end of the positive run AND there's a clip
        bool isLastPositive = (i == positiveCount - 1);
        bool useClipA = (clipCount > 0) && isLastPositive;

        // If not using clipA, wrap around to the next vertex
        float32_t3 v1 = useClipA ? clipA : getVertex(getSilhouetteVertex(rotatedSil, (i + 1) % vertexCount));

        float32_t3 pts[2] = {v0, v1};
        color += drawEdge((i + 1) % vertexCount, pts, spherePos, aaWidth);

        silhouette.vertices[silhouette.count++] = v0;
    }

    if (clipCount > 0 && clipCount < vertexCount)
    {
        // NP edge
        float32_t3 vFirst = getVertex(getSilhouetteVertex(rotatedSil, 0));
        float32_t3 npPts[2] = {clipB, vFirst};
        color += drawEdge(0, npPts, spherePos, aaWidth);

        // Horizon arc
        float32_t3 arcPts[2] = {clipA, clipB};
        color += drawEdge(23, arcPts, spherePos, aaWidth, 0.6f);

        silhouette.vertices[silhouette.count++] = clipA;
        silhouette.vertices[silhouette.count++] = clipB;
    }

#if DEBUG_DATA
    DebugDataBuffer[0].clipMask = clipMask;
    DebugDataBuffer[0].clipCount = clipCount;
    DebugDataBuffer[0].rotatedClipMask = rotatedClipMask;
    DebugDataBuffer[0].rotateAmount = rotateAmount;
    DebugDataBuffer[0].positiveVertCount = positiveCount;
    DebugDataBuffer[0].wrapAround = (uint32_t)wrapAround;
    DebugDataBuffer[0].rotatedSil = rotatedSil;

#endif
    return color;
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
    float32_t4 color = float32_t4(0, 0, 0, 0);
    for (int32_t i = 0; i < 1; i++)
    {
        float aaWidth = length(float32_t2(ddx(vx.uv.x), ddy(vx.uv.y)));
        float32_t2 p = toCircleSpace(vx.uv);

        float32_t2 normalized = p / CIRCLE_RADIUS;
        float r2 = dot(normalized, normalized);

        float32_t3 spherePos;
        if (r2 <= 1.0f)
        {
            spherePos = float32_t3(normalized.x, normalized.y, sqrt(1.0f - r2));
        }
        else
        {
            float uv2Plus1 = r2 + 1.0f;
            spherePos = float32_t3(normalized.x * 2.0f, normalized.y * 2.0f, 1.0f - r2) / uv2Plus1;
        }
        spherePos = normalize(spherePos);

        computeCubeGeo();

        float32_t4x3 columnModel = transpose(pc.modelMatrix);
        float32_t3 obbCenter = columnModel[3].xyz;
        float32_t3x3 upper3x3 = (float32_t3x3)columnModel;
        float32_t3 rcpSqScales = rcp(float32_t3(
            dot(upper3x3[0], upper3x3[0]),
            dot(upper3x3[1], upper3x3[1]),
            dot(upper3x3[2], upper3x3[2])));
        float32_t3 normalizedProj = mul(upper3x3, obbCenter) * rcpSqScales;

        int32_t3 region = int32_t3(
            normalizedProj.x < -1.0f ? 0 : (normalizedProj.x > 1.0f ? 2 : 1),
            normalizedProj.y < -1.0f ? 0 : (normalizedProj.y > 1.0f ? 2 : 1),
            normalizedProj.z < -1.0f ? 0 : (normalizedProj.z > 1.0f ? 2 : 1));

        int32_t configIndex = region.x + region.y * 3 + region.z * 9;

        // uint32_t sil = packSilhouette(silhouettes[configIndex]);
        uint32_t sil = binSilhouettes[configIndex];

        int32_t vertexCount = getSilhouetteSize(sil);

        uint32_t silEdgeMask = 0; // TODO: take from 'fast' computeSilhouette()
#if DEBUG_DATA
        {
            for (int32_t i = 0; i < vertexCount; i++)
            {
                int32_t vIdx = i % vertexCount;
                int32_t v1Idx = (i + 1) % vertexCount;

                int32_t v0Corner = getSilhouetteVertex(sil, vIdx);
                int32_t v1Corner = getSilhouetteVertex(sil, v1Idx);
                // Mark edge as part of silhouette
                for (int32_t e = 0; e < 12; e++)
                {
                    int32_t2 edge = allEdges[e];
                    if ((edge.x == v0Corner && edge.y == v1Corner) ||
                        (edge.x == v1Corner && edge.y == v0Corner))
                    {
                        silEdgeMask |= (1u << e);
                    }
                }
            }
            validateEdgeVisibility(sil, vertexCount, silEdgeMask);
        }
#endif

        uint32_t positiveCount = 0;

        ClippedSilhouette silhouette;
        color += computeSilhouette(vertexCount, sil, spherePos, aaWidth, silhouette);
        // Draw clipped silhouette vertices
        // color += drawClippedSilhouetteVertices(p, silhouette, aaWidth);

        SamplingData samplingData = buildSamplingDataFromSilhouette(silhouette, pc.samplingMode);

        uint32_t faceIndices[3];
        uint32_t visibleFaceCount = getVisibleFaces(region, faceIndices);

        // For debugging: Draw a small indicator of which faces are found
        // color += drawVisibleFaceOverlay(spherePos, region, aaWidth);

        // color += drawFaces(spherePos, aaWidth);

        // Draw samples on sphere
        color += visualizeSamples(vx.uv, spherePos, silhouette, pc.samplingMode, samplingData, 64);

        // Or draw 2D sample space (in a separate viewport)
        // color += visualizePrimarySampleSpace(vx.uv, pc.samplingMode, 64, aaWidth);

        setDebugData(sil, region, configIndex);
        // color += drawHiddenEdges(spherePos, silEdgeMask, aaWidth);
        color += drawCorners(p, aaWidth);
        color += drawRing(p, aaWidth);

        if (all(vx.uv >= float32_t2(0.49f, 0.49f)) && all(vx.uv <= float32_t2(0.51f, 0.51f)))
        {
            return float32_t4(colorLUT[configIndex], 1.0f);
        }
    }

    return color;
}