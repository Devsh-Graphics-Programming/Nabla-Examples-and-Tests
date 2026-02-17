//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_

#include "gpu_common.hlsl"

struct ClippedSilhouette
{
    float32_t3 vertices[MAX_SILHOUETTE_VERTICES]; // Max 7 vertices after clipping, unnormalized
    uint32_t count;

    void normalize()
    {
        vertices[0] = nbl::hlsl::normalize(vertices[0]);
        vertices[1] = nbl::hlsl::normalize(vertices[1]);
        vertices[2] = nbl::hlsl::normalize(vertices[2]);
        if (count > 3)
        {
            vertices[3] = nbl::hlsl::normalize(vertices[3]);
            if (count > 4)
            {
                vertices[4] = nbl::hlsl::normalize(vertices[4]);
                if (count > 5)
                {
                    vertices[5] = nbl::hlsl::normalize(vertices[5]);
                    if (count > 6)
                    {
                        vertices[6] = nbl::hlsl::normalize(vertices[6]);
                    }
                }
            }
        }
    }

    // Compute the silhouette centroid (average direction)
    float32_t3 getCenter()
    {
        float32_t3 sum = float32_t3(0, 0, 0);

        NBL_UNROLL
        for (uint32_t i = 0; i < MAX_SILHOUETTE_VERTICES; i++)
        {
            if (i < count)
                sum += vertices[i];
        }

        return nbl::hlsl::normalize(sum);
    }

    static uint32_t computeRegionAndConfig(float32_t3x4 modelMatrix, out uint32_t3 region, out uint32_t configIndex, out uint32_t vertexCount)
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

        uint32_t sil = binSilhouettes[configIndex];
        vertexCount = getSilhouetteSize(sil);

        return sil;
    }

    void compute(float32_t3x4 modelMatrix, uint32_t vertexCount, uint32_t sil)
    {
        count = 0;

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

        // Invert clip mask to find first positive vertex
        uint32_t invertedMask = ~clipMask & ((1u << vertexCount) - 1u);

        // Check if wrap-around is needed (first and last bits negative)
        bool wrapAround = ((clipMask & 1u) != 0u) && ((clipMask & (1u << (vertexCount - 1))) != 0u);

        // Compute rotation amount
        uint32_t rotateAmount = wrapAround
                                    ? firstbitlow(invertedMask)   // first positive
                                    : firstbithigh(clipMask) + 1; // first vertex after last negative

        // Rotate masks
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
            float32_t3 v0 = getVertex(modelMatrix, getSilhouetteVertex(rotatedSil, i));

#if DEBUG_DATA
            uint32_t originalIndex = (i + rotateAmount) % vertexCount;
            DebugDataBuffer[0].clippedSilhouetteVertices[count] = v0;
            DebugDataBuffer[0].clippedSilhouetteVerticesIndices[count] = originalIndex;
#endif
            vertices[count++] = v0;
        }

        if (clipCount > 0 && clipCount < vertexCount)
        {
#if DEBUG_DATA
            DebugDataBuffer[0].clippedSilhouetteVertices[count] = clipA;
            DebugDataBuffer[0].clippedSilhouetteVerticesIndices[count] = CLIP_POINT_A;
#endif
            vertices[count++] = clipA;

#if DEBUG_DATA
            DebugDataBuffer[0].clippedSilhouetteVertices[count] = clipB;
            DebugDataBuffer[0].clippedSilhouetteVerticesIndices[count] = CLIP_POINT_B;
#endif
            vertices[count++] = clipB;
        }

#if DEBUG_DATA
        DebugDataBuffer[0].clippedSilhouetteVertexCount = count;
        DebugDataBuffer[0].clipMask = clipMask;
        DebugDataBuffer[0].clipCount = clipCount;
        DebugDataBuffer[0].rotatedClipMask = rotatedClipMask;
        DebugDataBuffer[0].rotateAmount = rotateAmount;
        DebugDataBuffer[0].positiveVertCount = positiveCount;
        DebugDataBuffer[0].wrapAround = (uint32_t)wrapAround;
        DebugDataBuffer[0].rotatedSil = rotatedSil;
#endif
    }
};

struct SilEdgeNormals
{
    float16_t3 edgeNormals[MAX_SILHOUETTE_VERTICES]; // 10.5 floats instead of 21
    uint32_t count;

    // Better not use and calculate it while creating the sampler
    static SilEdgeNormals create(NBL_CONST_REF_ARG(ClippedSilhouette) sil)
    {
        SilEdgeNormals result = (SilEdgeNormals)0;
        result.count = sil.count;

        float32_t3 v0 = sil.vertices[0];
        float32_t3 v1 = sil.vertices[1];
        float32_t3 v2 = sil.vertices[2];

        result.edgeNormals[0] = float16_t3(cross(v0, v1));
        result.edgeNormals[1] = float16_t3(cross(v1, v2));

        if (sil.count > 3)
        {
            float32_t3 v3 = sil.vertices[3];
            result.edgeNormals[2] = float16_t3(cross(v2, v3));

            if (sil.count > 4)
            {
                float32_t3 v4 = sil.vertices[4];
                result.edgeNormals[3] = float16_t3(cross(v3, v4));

                if (sil.count > 5)
                {
                    float32_t3 v5 = sil.vertices[5];
                    result.edgeNormals[4] = float16_t3(cross(v4, v5));

                    if (sil.count > 6)
                    {
                        float32_t3 v6 = sil.vertices[6];
                        result.edgeNormals[5] = float16_t3(cross(v5, v6));
                        result.edgeNormals[6] = float16_t3(cross(v6, v0));
                    }
                    else
                    {
                        result.edgeNormals[5] = float16_t3(cross(v5, v0));
                    }
                }
                else
                {
                    result.edgeNormals[4] = float16_t3(cross(v4, v0));
                }
            }
            else
            {
                result.edgeNormals[3] = float16_t3(cross(v3, v0));
            }
        }
        else
        {
            result.edgeNormals[2] = float16_t3(cross(v2, v0));
        }

        return result;
    }

    bool isInside(float32_t3 dir)
    {
        float16_t3 d = float16_t3(dir);
        half maxDot = dot(d, edgeNormals[0]);
        maxDot = max(maxDot, dot(d, edgeNormals[1]));
        maxDot = max(maxDot, dot(d, edgeNormals[2]));
        maxDot = max(maxDot, dot(d, edgeNormals[3]));
        maxDot = max(maxDot, dot(d, edgeNormals[4]));
        maxDot = max(maxDot, dot(d, edgeNormals[5]));
        maxDot = max(maxDot, dot(d, edgeNormals[6]));
        return maxDot <= float16_t(0.0f);
    }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_
