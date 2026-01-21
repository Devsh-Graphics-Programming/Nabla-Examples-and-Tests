#ifndef _PARALLELOGRAM_SAMPLING_HLSL_
#define _PARALLELOGRAM_SAMPLING_HLSL_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>

#define MAX_SILHOUETTE_VERTICES 7
#define MAX_CURVE_APEXES 2
#define GET_PROJ_VERT(i) vertices[i].xy *CIRCLE_RADIUS

// ============================================================================
// Core structures
// ============================================================================

struct Parallelogram
{
    float16_t2 corner;
    float16_t2 axisDir;
    float16_t width;
    float16_t height;
};

struct PrecomputedSilhouette
{
    float16_t3 edgeNormals[MAX_SILHOUETTE_VERTICES]; // 10.5 floats instead of 21
    uint32_t count;
};

struct ParallelogramSilhouette
{
    Parallelogram para;
    PrecomputedSilhouette silhouette;
};

// ============================================================================
// Silhouette helpers
// ============================================================================

PrecomputedSilhouette precomputeSilhouette(NBL_CONST_REF_ARG(ClippedSilhouette) sil)
{
    PrecomputedSilhouette result;
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
                    result.edgeNormals[6] = float16_t3(0.0f, 0.0f, 0.0f);
                }
            }
            else
            {
                result.edgeNormals[4] = float16_t3(cross(v4, v0));
                result.edgeNormals[5] = float16_t3(0.0f, 0.0f, 0.0f);
                result.edgeNormals[6] = float16_t3(0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            result.edgeNormals[3] = float16_t3(cross(v3, v0));
            result.edgeNormals[4] = float16_t3(0.0f, 0.0f, 0.0f);
            result.edgeNormals[5] = float16_t3(0.0f, 0.0f, 0.0f);
            result.edgeNormals[6] = float16_t3(0.0f, 0.0f, 0.0f);
        }
    }
    else
    {
        result.edgeNormals[2] = float16_t3(cross(v2, v0));
        result.edgeNormals[3] = float16_t3(0.0f, 0.0f, 0.0f);
        result.edgeNormals[4] = float16_t3(0.0f, 0.0f, 0.0f);
        result.edgeNormals[5] = float16_t3(0.0f, 0.0f, 0.0f);
        result.edgeNormals[6] = float16_t3(0.0f, 0.0f, 0.0f);
    }

    return result;
}

bool isInsideSilhouetteFast(float32_t3 dir, NBL_CONST_REF_ARG(PrecomputedSilhouette) sil)
{
    float16_t3 d = float16_t3(dir);
    half maxDot = dot(d, sil.edgeNormals[0]);
    maxDot = max(maxDot, dot(d, sil.edgeNormals[1]));
    maxDot = max(maxDot, dot(d, sil.edgeNormals[2]));
    maxDot = max(maxDot, dot(d, sil.edgeNormals[3]));
    maxDot = max(maxDot, dot(d, sil.edgeNormals[4]));
    maxDot = max(maxDot, dot(d, sil.edgeNormals[5]));
    maxDot = max(maxDot, dot(d, sil.edgeNormals[6]));
    return maxDot <= half(0.0f);
}
float32_t3 circleToSphere(float32_t2 circlePoint)
{
    float32_t2 xy = circlePoint / CIRCLE_RADIUS;
    float32_t xy_len_sq = dot(xy, xy);

    // if (xy_len_sq >= 1.0f)
    //     return float32_t3(0, 0, 0);

    return float32_t3(xy, sqrt(1.0f - xy_len_sq));
}

bool isEdgeConvex(float32_t3 S, float32_t3 E)
{
    return nbl::hlsl::cross2D(S.xy, E.xy) < -1e-6f;
}

// ============================================================================
// Curve evaluation helpers
// ============================================================================

// Evaluate curve point at t using rsqrt
float32_t2 evalCurvePoint(float32_t3 S, float32_t3 E, float32_t t)
{
    float32_t3 v = S + t * (E - S);
    float32_t invLen = rsqrt(dot(v, v));
    return v.xy * (invLen * CIRCLE_RADIUS);
}

// Evaluate tangent at arbitrary t
float32_t2 evalCurveTangent(float32_t3 S, float32_t3 E, float32_t t)
{
    float32_t3 v = S + t * (E - S);
    float32_t vLenSq = dot(v, v);

    if (vLenSq < 1e-12f)
        return normalize(E.xy - S.xy);

    float32_t3 p = v * rsqrt(vLenSq);
    float32_t3 vPrime = E - S;
    float32_t2 tangent2D = (vPrime - p * dot(p, vPrime)).xy;

    float32_t len = length(tangent2D);
    return (len > 1e-7f) ? tangent2D / len : normalize(E.xy - S.xy);
}

// Get both endpoint tangents efficiently (shares SdotE computation)
void getProjectedTangents(float32_t3 S, float32_t3 E, out float32_t2 t0, out float32_t2 t1)
{
    float32_t SdotE = dot(S, E);

    float32_t2 tangent0_2D = (E - S * SdotE).xy;
    float32_t2 tangent1_2D = (E * SdotE - S).xy;

    float32_t len0Sq = dot(tangent0_2D, tangent0_2D);
    float32_t len1Sq = dot(tangent1_2D, tangent1_2D);

    const float32_t eps = 1e-14f;

    if (len0Sq > eps && len1Sq > eps)
    {
        t0 = tangent0_2D * rsqrt(len0Sq);
        t1 = tangent1_2D * rsqrt(len1Sq);
        return;
    }

    // Rare fallback path
    float32_t2 diff = E.xy - S.xy;
    float32_t diffLenSq = dot(diff, diff);
    float32_t2 fallback = diffLenSq > eps ? diff * rsqrt(diffLenSq) : float32_t2(1.0f, 0.0f);

    t0 = len0Sq > eps ? tangent0_2D * rsqrt(len0Sq) : fallback;
    t1 = len1Sq > eps ? tangent1_2D * rsqrt(len1Sq) : fallback;
}

// Compute apex with clamping to prevent apex explosion
void computeApexClamped(float32_t2 p0, float32_t2 p1, float32_t2 t0, float32_t2 t1, out float32_t2 apex)
{
    float32_t denom = t0.x * t1.y - t0.y * t1.x;
    float32_t2 center = (p0 + p1) * 0.5f;

    if (abs(denom) < 1e-6f)
    {
        apex = center;
        return;
    }

    float32_t2 dp = p1 - p0;
    float32_t s = (dp.x * t1.y - dp.y * t1.x) / denom;
    apex = p0 + s * t0;

    float32_t2 toApex = apex - center;
    float32_t distSq = dot(toApex, toApex);
    float32_t maxDistSq = CIRCLE_RADIUS * CIRCLE_RADIUS * 4.0f;

    if (distSq > maxDistSq)
    {
        apex = center + toApex * (CIRCLE_RADIUS * 2.0f * rsqrt(distSq));
    }
}

void testPoint(inout float32_t minAlong, inout float32_t maxAlong, inout float32_t minPerp, inout float32_t maxPerp, float32_t2 pt, float32_t2 axisDir, float32_t2 perpDir)
{
    float32_t projAlong = dot(pt, axisDir);
    float32_t projPerp = dot(pt, perpDir);

    minAlong = min(minAlong, projAlong);
    maxAlong = max(maxAlong, projAlong);
    minPerp = min(minPerp, projPerp);
    maxPerp = max(maxPerp, projPerp);
}

template <uint32_t I>
void testEdgeForAxisFast(inout float32_t minAlong, inout float32_t maxAlong, inout float32_t minPerp, inout float32_t maxPerp,
                         uint32_t count, uint32_t n3Mask, float32_t2 axisDir, float32_t2 perpDir,
                         const float32_t3 vertices[MAX_SILHOUETTE_VERTICES])
{
    const uint32_t nextIdx = (I + 1 < count) ? I + 1 : 0;

    testPoint(minAlong, maxAlong, minPerp, maxPerp, GET_PROJ_VERT(I), axisDir, perpDir);

    if (n3Mask & (1u << I))
    {
        float32_t2 midPoint = evalCurvePoint(vertices[I], vertices[nextIdx], 0.5f);
        testPoint(minAlong, maxAlong, minPerp, maxPerp, midPoint, axisDir, perpDir);
    }
}

float32_t computeBoundingBoxAreaForAxisFast(NBL_CONST_REF_ARG(float32_t3) vertices[MAX_SILHOUETTE_VERTICES], uint32_t n3Mask, uint32_t count, float32_t2 axisDir)
{
    float32_t2 perpDir = float32_t2(-axisDir.y, axisDir.x);

    float32_t minAlong = 1e10f;
    float32_t maxAlong = -1e10f;
    float32_t minPerp = 1e10f;
    float32_t maxPerp = -1e10f;

    testEdgeForAxisFast<0>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
    testEdgeForAxisFast<1>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
    testEdgeForAxisFast<2>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
    if (count > 3)
    {
        testEdgeForAxisFast<3>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
        if (count > 4)
        {
            testEdgeForAxisFast<4>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
            if (count > 5)
            {
                testEdgeForAxisFast<5>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
                if (count > 6)
                {
                    testEdgeForAxisFast<6>(minAlong, maxAlong, minPerp, maxPerp, count, n3Mask, axisDir, perpDir, vertices);
                }
            }
        }
    }

    return (maxAlong - minAlong) * (maxPerp - minPerp);
}

void tryCaliperDir(inout float32_t bestArea, inout float32_t2 bestDir, const float32_t2 dir, const float32_t3 vertices[MAX_SILHOUETTE_VERTICES], uint32_t n3Mask, uint32_t count)
{
    float32_t area = computeBoundingBoxAreaForAxisFast(vertices, n3Mask, count, dir);

    if (area < bestArea)
    {
        bestArea = area;
        bestDir = dir;
    }
}

template <uint32_t I>
inline void processEdge(inout float32_t bestArea, inout float32_t2 bestDir, inout uint32_t convexMask, inout uint32_t n3Mask, uint32_t count, const float32_t3 vertices[MAX_SILHOUETTE_VERTICES])
{
    const uint32_t nextIdx = (I + 1 < count) ? I + 1 : 0;
    float32_t3 S = vertices[I];
    float32_t3 E = vertices[nextIdx];

    float32_t2 t0, t1;
    getProjectedTangents(S, E, t0, t1);

    tryCaliperDir(bestArea, bestDir, t0, vertices, n3Mask, count);

    if (isEdgeConvex(S, E))
    {
        convexMask |= (1u << I);
        tryCaliperDir(bestArea, bestDir, t1, vertices, n3Mask, count);

        if (dot(t0, t1) < 0.5f)
        {
            n3Mask |= (1u << I);
            float32_t2 tangentAtMid = evalCurveTangent(S, E, 0.5f);
            tryCaliperDir(bestArea, bestDir, tangentAtMid, vertices, n3Mask, count);
        }
    }
}

template <uint32_t I>
inline void testEdgeForAxisAccurate(inout float32_t minAlong, inout float32_t maxAlong, inout float32_t minPerp, inout float32_t maxPerp, uint32_t count, uint32_t convexMask, uint32_t n3Mask,
                                    float32_t2 axisDir, float32_t2 perpDir, const float32_t3 vertices[MAX_SILHOUETTE_VERTICES])
{
    const uint32_t nextIdx = (I + 1 < count) ? I + 1 : 0;
    float32_t2 projectedVertex = vertices[I].xy * CIRCLE_RADIUS;

    testPoint(minAlong, maxAlong, minPerp, maxPerp, projectedVertex, axisDir, perpDir);

    bool isN3 = (n3Mask & (1u << I)) != 0;
    bool isConvex = (convexMask & (1u << I)) != 0;

    if (!isN3 && !isConvex)
        return;

    float32_t3 S = vertices[I];
    float32_t3 E = vertices[nextIdx];
    float32_t2 midPoint = evalCurvePoint(S, E, 0.5f);

    if (isN3)
    {
        testPoint(minAlong, maxAlong, minPerp, maxPerp, midPoint, axisDir, perpDir);
    }

    if (isConvex)
    {
        float32_t2 t0, endTangent;
        getProjectedTangents(S, E, t0, endTangent);

        if (dot(t0, perpDir) > 0.0f)
        {
            float32_t2 apex0;
            if (isN3)
            {
                float32_t2 tangentAtMid = evalCurveTangent(S, E, 0.5f);
                computeApexClamped(projectedVertex, midPoint, t0, tangentAtMid, apex0);
                testPoint(minAlong, maxAlong, minPerp, maxPerp, apex0, axisDir, perpDir);

                if (dot(tangentAtMid, perpDir) > 0.0f)
                {
                    float32_t2 apex1;
                    computeApexClamped(midPoint, E.xy * CIRCLE_RADIUS, tangentAtMid, endTangent, apex1);
                    testPoint(minAlong, maxAlong, minPerp, maxPerp, apex1, axisDir, perpDir);
                }
            }
            else
            {
                computeApexClamped(projectedVertex, E.xy * CIRCLE_RADIUS, t0, endTangent, apex0);
                testPoint(minAlong, maxAlong, minPerp, maxPerp, apex0, axisDir, perpDir);
            }
        }
    }
}

Parallelogram buildParallelogramForAxisAccurate(const float32_t3 vertices[MAX_SILHOUETTE_VERTICES], uint32_t convexMask, uint32_t n3Mask, uint32_t count, float32_t2 axisDir)
{
    float32_t2 perpDir = float32_t2(-axisDir.y, axisDir.x);

    float32_t minAlong = 1e10f;
    float32_t maxAlong = -1e10f;
    float32_t minPerp = 1e10f;
    float32_t maxPerp = -1e10f;

    testEdgeForAxisAccurate<0>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
    testEdgeForAxisAccurate<1>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
    testEdgeForAxisAccurate<2>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
    if (count > 3)
    {
        testEdgeForAxisAccurate<3>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
        if (count > 4)
        {
            testEdgeForAxisAccurate<4>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
            if (count > 5)
            {
                testEdgeForAxisAccurate<5>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
                if (count > 6)
                {
                    testEdgeForAxisAccurate<6>(minAlong, maxAlong, minPerp, maxPerp, count, convexMask, n3Mask, axisDir, perpDir, vertices);
                }
            }
        }
    }

    Parallelogram result;
    result.width = float16_t(maxAlong - minAlong);
    result.height = float16_t(maxPerp - minPerp);
    result.axisDir = float16_t2(axisDir);
    result.corner = float16_t2(minAlong * axisDir + minPerp * float16_t2(-axisDir.y, axisDir.x));

    return result;
}

Parallelogram findMinimumBoundingBoxCurved(const float32_t3 vertices[MAX_SILHOUETTE_VERTICES], uint32_t count
#if VISUALIZE_SAMPLES
                                           ,
                                           float32_t2 ndc, float32_t3 spherePos, float32_t aaWidth,
                                           inout float32_t4 color
#endif
)
{
    uint32_t convexMask = 0;
    uint32_t n3Mask = 0;
    float32_t bestArea = 1e10f;
    float32_t2 bestDir = float32_t2(1.0f, 0.0f);

    processEdge<0>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
    processEdge<1>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
    processEdge<2>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
    if (count > 3)
    {
        processEdge<3>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
        if (count > 4)
        {
            processEdge<4>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
            if (count > 5)
            {
                processEdge<5>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
                if (count > 6)
                {
                    processEdge<6>(bestArea, bestDir, convexMask, n3Mask, count, vertices);
                }
            }
        }
    }

    tryCaliperDir(bestArea, bestDir, float32_t2(1.0f, 0.0f), vertices, n3Mask, count);
    tryCaliperDir(bestArea, bestDir, float32_t2(0.0f, 1.0f), vertices, n3Mask, count);

    Parallelogram best = buildParallelogramForAxisAccurate(vertices, convexMask, n3Mask, count, bestDir);

#if VISUALIZE_SAMPLES
    for (uint32_t i = 0; i < count; i++)
    {
        if (convexMask & (1u << i))
        {
            uint32_t nextIdx = (i + 1) % count;
            float32_t2 p0 = vertices[i].xy * CIRCLE_RADIUS;
            float32_t2 p1 = vertices[nextIdx].xy * CIRCLE_RADIUS;

            float32_t2 t0, endTangent;
            getProjectedTangents(vertices[i], vertices[nextIdx], t0, endTangent);

            if (n3Mask & (1u << i))
            {
                float32_t2 tangentAtMid = evalCurveTangent(vertices[i], vertices[nextIdx], 0.5f);
                float32_t2 midPoint = evalCurvePoint(vertices[i], vertices[nextIdx], 0.5f);

                float32_t2 apex0, apex1;
                computeApexClamped(p0, midPoint, t0, tangentAtMid, apex0);
                computeApexClamped(midPoint, p1, tangentAtMid, endTangent, apex1);

                color += drawCorner(float32_t3(apex0, 0.0f), ndc, aaWidth, 0.03, 0.0f, float32_t3(1, 0, 1));
                color += drawCorner(float32_t3(midPoint, 0.0f), ndc, aaWidth, 0.02, 0.0f, float32_t3(0, 1, 0));
                color += drawCorner(float32_t3(apex1, 0.0f), ndc, aaWidth, 0.03, 0.0f, float32_t3(1, 0.5, 0));
            }
            else
            {
                float32_t2 apex;
                computeApexClamped(p0, p1, t0, endTangent, apex);
                color += drawCorner(float32_t3(apex, 0.0f), ndc, aaWidth, 0.03, 0.0f, float32_t3(1, 0, 1));
            }
        }
    }
#endif

    return best;
}
// ============================================================================
// Main entry points
// ============================================================================

ParallelogramSilhouette buildParallelogram(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette
#if VISUALIZE_SAMPLES
                                           ,
                                           float32_t2 ndc, float32_t3 spherePos, float32_t aaWidth,
                                           inout float32_t4 color
#endif
)
{
    ParallelogramSilhouette result;

    // if (silhouette.count < 3)
    // {
    //     result.para.corner = float32_t2(0, 0);
    //     result.para.edge0 = float32_t2(1, 0);
    //     result.para.edge1 = float32_t2(0, 1);
    //     result.para.area = 1.0f;
    //     return result;
    // }

    result.para = findMinimumBoundingBoxCurved(silhouette.vertices, silhouette.count
#if VISUALIZE_SAMPLES
                                               ,
                                               ndc, spherePos, aaWidth, color
#endif
    );

#if DEBUG_DATA
    DebugDataBuffer[0].parallelogramArea = result.para.width * result.para.height;
#endif
    result.silhouette = precomputeSilhouette(silhouette);

    return result;
}

float32_t3 sampleFromParallelogram(NBL_CONST_REF_ARG(ParallelogramSilhouette) paraSilhouette, float32_t2 xi, out float32_t pdf, out bool valid)
{
    float16_t2 axisDir = paraSilhouette.para.axisDir;
    float16_t2 perpDir = float16_t2(-axisDir.y, axisDir.x);

    float16_t2 circleXY = paraSilhouette.para.corner +
                          float16_t(xi.x) * paraSilhouette.para.width * axisDir +
                          float16_t(xi.y) * paraSilhouette.para.height * perpDir;

    float32_t3 direction = circleToSphere(circleXY);

    valid = (direction.z > 0.0f) && isInsideSilhouetteFast(direction, paraSilhouette.silhouette);
    pdf = valid ? (1.0f / (paraSilhouette.para.width * paraSilhouette.para.height)) : 0.0f;

    return direction;
}

#endif // _PARALLELOGRAM_SAMPLING_HLSL_
