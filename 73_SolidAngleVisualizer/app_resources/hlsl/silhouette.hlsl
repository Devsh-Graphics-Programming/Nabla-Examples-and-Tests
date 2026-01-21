#ifndef _SILHOUETTE_HLSL_
#define _SILHOUETTE_HLSL_

#include "gpu_common.hlsl"
#include "utils.hlsl"

// Special index values for clip points
static const uint32_t CLIP_POINT_A = 23; // Clip point between last positive and first negative
static const uint32_t CLIP_POINT_B = 24; // Clip point between last negative and first positive

// Compute region and configuration index from model matrix
uint32_t computeRegionAndConfig(float32_t3x4 modelMatrix, out uint32_t3 region, out uint32_t configIndex, out uint32_t vertexCount)
{
    float32_t4x3 columnModel = transpose(modelMatrix);
    float32_t3 obbCenter = columnModel[3].xyz;
    float32_t3x3 upper3x3 = (float32_t3x3)columnModel;

    float32_t3 rcpSqScales = rcp(float32_t3(
        dot(upper3x3[0], upper3x3[0]),
        dot(upper3x3[1], upper3x3[1]),
        dot(upper3x3[2], upper3x3[2])));

    float32_t3 normalizedProj = mul(upper3x3, obbCenter) * rcpSqScales;

    region = uint32_t3(
        normalizedProj.x < -0.5f ? 0 : (normalizedProj.x > 0.5f ? 2 : 1),
        normalizedProj.y < -0.5f ? 0 : (normalizedProj.y > 0.5f ? 2 : 1),
        normalizedProj.z < -0.5f ? 0 : (normalizedProj.z > 0.5f ? 2 : 1));

    configIndex = region.x + region.y * 3u + region.z * 9u;

    // uint32_t sil = packSilhouette(silhouettes[configIndex]);
    uint32_t sil = binSilhouettes[configIndex];
    vertexCount = getSilhouetteSize(sil);

    return sil;
}

#if VISUALIZE_SAMPLES
float32_t4
#else
void
#endif
computeSilhouette(float32_t3x4 modelMatrix, uint32_t vertexCount, uint32_t sil
#if VISUALIZE_SAMPLES
                  ,
                  float32_t3 spherePos, float32_t aaWidth
#endif
                  ,
                  NBL_REF_ARG(ClippedSilhouette) silhouette)
{
#if VISUALIZE_SAMPLES
    float32_t4 color = float32_t4(0, 0, 0, 0);
#endif

    silhouette.count = 0;

    // Build clip mask (z < 0)
    uint32_t clipMask = 0u;
    NBL_UNROLL
    for (uint32_t i = 0; i < 4; i++)
        clipMask |= (getVertexZNeg(modelMatrix, getSilhouetteVertex(sil, i)) ? 1u : 0u) << i;

    if (vertexCount == 6)
    {
        NBL_UNROLL
        for (uint32_t i = 4; i < 6; i++)
            clipMask |= (getVertexZNeg(modelMatrix, getSilhouetteVertex(sil, i)) ? 1u : 0u) << i;
    }

    uint32_t clipCount = countbits(clipMask);

#if 0
    // Early exit if fully clipped
    if (clipCount == vertexCount)
        return color;

    // No clipping needed - fast path
    if (clipCount == 0)
    {
        for (uint32_t i = 0; i < vertexCount; i++)
        {
            uint32_t i0 = i;
            uint32_t i1 = (i + 1) % vertexCount;
            float32_t3 v0 = getVertex(modelMatrix, getSilhouetteVertex(sil, i0));
            silhouette.vertices[silhouette.count] = v0;
            silhouette.indices[silhouette.count++] = i0;  // Original index (no rotation)

#if VISUALIZE_SAMPLES
            float32_t3 v1 = getVertex(modelMatrix, getSilhouetteVertex(sil, i1));
            float32_t3 pts[2] = {v0, v1};
            color += drawEdge(i1, pts, spherePos, aaWidth);
#endif
        }
        return color;
    }
#endif

    // Rotate clip mask so positives come first
    uint32_t invertedMask = ~clipMask & ((1u << vertexCount) - 1u);
    bool wrapAround = ((clipMask & 1u) != 0u) && ((clipMask & (1u << (vertexCount - 1))) != 0u);
    uint32_t rotateAmount = wrapAround
                                ? firstbitlow(invertedMask)   // -> First POSITIVE
                                : firstbithigh(clipMask) + 1; // -> First vertex AFTER last negative

    uint32_t rotatedClipMask = rotr(clipMask, rotateAmount, vertexCount);
    uint32_t rotatedSil = rotr(sil, rotateAmount * 3, vertexCount * 3);
    uint32_t positiveCount = vertexCount - clipCount;

    // ALWAYS compute both clip points
    uint32_t lastPosIdx = positiveCount - 1;
    uint32_t firstNegIdx = positiveCount;

    float32_t3 vLastPos = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, lastPosIdx));
    float32_t3 vFirstNeg = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, firstNegIdx));
    float32_t t = vLastPos.z / (vLastPos.z - vFirstNeg.z);
    float32_t3 clipA = lerp(vLastPos, vFirstNeg, t);

    float32_t3 vLastNeg = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, vertexCount - 1));
    float32_t3 vFirstPos = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, 0));
    t = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
    float32_t3 clipB = lerp(vLastNeg, vFirstPos, t);

    NBL_UNROLL
    for (uint32_t i = 0; i < positiveCount; i++)
    {
        // Get raw vertex
        float32_t3 v0 = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, i));
        bool isLastPositive = (i == positiveCount - 1);
        bool useClipA = (clipCount > 0) && isLastPositive;

        // Compute original index before rotation
        uint32_t originalIndex = (i + rotateAmount) % vertexCount;

#if VISUALIZE_SAMPLES
        float32_t3 v1 = useClipA ? clipA : getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, (i + 1) % vertexCount));
        float32_t3 pts[2] = {normalize(v0), normalize(v1)};
        color += drawEdge((i + 1) % vertexCount, pts, spherePos, aaWidth);
#endif

#if DEBUG_DATA
        DebugDataBuffer[0].clippedSilhouetteVertices[silhouette.count] = v0;
        DebugDataBuffer[0].clippedSilhouetteVerticesIndices[silhouette.count] = originalIndex;
#endif
        silhouette.vertices[silhouette.count++] = normalize(v0);
    }

    if (clipCount > 0 && clipCount < vertexCount)
    {
        float32_t3 vFirst = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, 0));

#if VISUALIZE_SAMPLES
        float32_t3 npPts[2] = {normalize(clipB), normalize(vFirst)};
        color += drawEdge(0, npPts, spherePos, aaWidth);

        float32_t3 arcPts[2] = {normalize(clipA), normalize(clipB)};
        color += drawEdge(23, arcPts, spherePos, aaWidth, 0.6f);
#endif

#if DEBUG_DATA
        DebugDataBuffer[0].clippedSilhouetteVertices[silhouette.count] = clipA;
        DebugDataBuffer[0].clippedSilhouetteVerticesIndices[silhouette.count] = CLIP_POINT_A;
#endif
        silhouette.vertices[silhouette.count++] = normalize(clipA);

#if DEBUG_DATA
        DebugDataBuffer[0].clippedSilhouetteVertices[silhouette.count] = clipB;
        DebugDataBuffer[0].clippedSilhouetteVerticesIndices[silhouette.count] = CLIP_POINT_B;
#endif
        silhouette.vertices[silhouette.count++] = normalize(clipB);
    }

#if DEBUG_DATA
    DebugDataBuffer[0].clippedSilhouetteVertexCount = silhouette.count;
    DebugDataBuffer[0].clipMask = clipMask;
    DebugDataBuffer[0].clipCount = clipCount;
    DebugDataBuffer[0].rotatedClipMask = rotatedClipMask;
    DebugDataBuffer[0].rotateAmount = rotateAmount;
    DebugDataBuffer[0].positiveVertCount = positiveCount;
    DebugDataBuffer[0].wrapAround = (uint32_t)wrapAround;
    DebugDataBuffer[0].rotatedSil = rotatedSil;
#endif

#if VISUALIZE_SAMPLES
    return color;
#endif
}

#endif // _SILHOUETTE_HLSL_
