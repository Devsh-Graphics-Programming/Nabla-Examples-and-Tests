//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_PARALLELOGRAM_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_PARALLELOGRAM_SAMPLING_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include "silhouette.hlsl"
#include "drawing.hlsl"

#define MAX_CURVE_APEXES 2
#define GET_PROJ_VERT(i) silhouette.vertices[i].xy *CIRCLE_RADIUS

// ============================================================================
// Minimum bounding rectangle on projected sphere
// ============================================================================
struct Parallelogram
{
    float16_t2 corner;
    float16_t2 axisDir;
    float16_t width;
    float16_t height;

    // ========================================================================
    // Projection helpers
    // ========================================================================

    static float32_t3 circleToSphere(float32_t2 circlePoint)
    {
        float32_t2 xy = circlePoint / CIRCLE_RADIUS;
        float32_t xy_len_sq = dot(xy, xy);
        return float32_t3(xy, sqrt(1.0f - xy_len_sq));
    }

    // ========================================================================
    // Curve evaluation helpers
    // ========================================================================

    static float32_t2 evalCurvePoint(float32_t3 S, float32_t3 E, float32_t t)
    {
        float32_t3 v = S + t * (E - S);
        float32_t invLen = rsqrt(dot(v, v));
        return v.xy * (invLen * CIRCLE_RADIUS);
    }

    static float32_t2 evalCurveTangent(float32_t3 S, float32_t3 E, float32_t t)
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

    // Get both endpoint tangents (shares SdotE computation)
    static void getProjectedTangents(float32_t3 S, float32_t3 E, out float32_t2 t0, out float32_t2 t1)
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
    static void computeApexClamped(float32_t2 p0, float32_t2 p1, float32_t2 t0, float32_t2 t1, out float32_t2 apex)
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

    // ========================================================================
    // Bounding box computation (rotating calipers)
    //
    // testEdgeForAxis<I, Accurate> and computeBoundsForAxis<Accurate> are
    // templated on a bool to select between two precision levels:
    //
    // Accurate=false (used by tryCaliperDir, O(N^2) total calls):
    //   Tests vertices + edge midpoints only. Cheap (just dot products) and
    //   sufficient for *ranking* candidate axes, even though it may
    //   underestimate the true extent of convex edges.
    //
    // Accurate=true (used by buildForAxis, called once):
    //   Also computes tangent-line apex intersections for convex edges to
    //   find the true extremum. Great circle arcs that project as convex
    //   curves can bulge beyond their endpoints; the apex (tangent
    //   evaluation + line intersection + clamping) captures this but is
    //   ~4x more expensive per edge.
    //
    // The fast path gives the same relative ranking of axes (the
    // approximation error is consistent across candidates), so the
    // cheapest axis found by Fast is also the cheapest under Accurate.
    // ========================================================================

    static void testPoint(inout float32_t minAlong, inout float32_t maxAlong, inout float32_t minPerp, inout float32_t maxPerp, float32_t2 pt, float32_t2 dir, float32_t2 perpDir)
    {
        float32_t projAlong = dot(pt, dir);
        float32_t projPerp = dot(pt, perpDir);

        minAlong = min(minAlong, projAlong);
        maxAlong = max(maxAlong, projAlong);
        minPerp = min(minPerp, projPerp);
        maxPerp = max(maxPerp, projPerp);
    }

    // Accurate=false (Fast): tests vertex + midpoint only. Used O(N^2) times for axis ranking.
    // Accurate=true:         also computes tangent-line apex for convex edges. Used once for final rect.
    template <uint32_t I, bool Accurate = false>
    static void testEdgeForAxis(inout float32_t minAlong, inout float32_t maxAlong, inout float32_t minPerp, inout float32_t maxPerp, const ClippedSilhouette silhouette, uint32_t convexMask, uint32_t n3Mask, float32_t2 dir, float32_t2 perpDir)
    {
        const uint32_t nextIdx = (I + 1 < silhouette.count) ? I + 1 : 0;
        const float32_t2 projectedVertex = GET_PROJ_VERT(I);

        testPoint(minAlong, maxAlong, minPerp, maxPerp, projectedVertex, dir, perpDir);

        bool isN3 = (n3Mask & (1u << I)) != 0;

        if (Accurate)
        {
            bool isConvex = (convexMask & (1u << I)) != 0;

            if (!isN3 && !isConvex)
                return;

            float32_t3 S = silhouette.vertices[I];
            float32_t3 E = silhouette.vertices[nextIdx];
            float32_t2 midPoint = evalCurvePoint(S, E, 0.5f);

            if (isN3)
            {
                testPoint(minAlong, maxAlong, minPerp, maxPerp, midPoint, dir, perpDir);
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
                        testPoint(minAlong, maxAlong, minPerp, maxPerp, apex0, dir, perpDir);

                        if (dot(tangentAtMid, perpDir) > 0.0f)
                        {
                            float32_t2 apex1;
                            computeApexClamped(midPoint, E.xy * CIRCLE_RADIUS, tangentAtMid, endTangent, apex1);
                            testPoint(minAlong, maxAlong, minPerp, maxPerp, apex1, dir, perpDir);
                        }
                    }
                    else
                    {
                        computeApexClamped(projectedVertex, E.xy * CIRCLE_RADIUS, t0, endTangent, apex0);
                        testPoint(minAlong, maxAlong, minPerp, maxPerp, apex0, dir, perpDir);
                    }
                }
            }
        }
        else
        {
            if (isN3)
            {
                float32_t2 midPoint = evalCurvePoint(silhouette.vertices[I], silhouette.vertices[nextIdx], 0.5f);
                testPoint(minAlong, maxAlong, minPerp, maxPerp, midPoint, dir, perpDir);
            }
        }
    }

    // Unrolled bounding box computation for a given axis direction.
    // Accurate=false: fast path for axis ranking during candidate selection.
    // Accurate=true:  tight bounds with apex computation for the final rectangle.
    template <bool Accurate = false>
    static void computeBoundsForAxis(inout float32_t minAlong, inout float32_t maxAlong, inout float32_t minPerp, inout float32_t maxPerp, const ClippedSilhouette silhouette, uint32_t convexMask, uint32_t n3Mask, float32_t2 dir, float32_t2 perpDir)
    {
        testEdgeForAxis<0, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
        testEdgeForAxis<1, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
        testEdgeForAxis<2, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
        if (silhouette.count > 3)
        {
            testEdgeForAxis<3, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
            if (silhouette.count > 4)
            {
                testEdgeForAxis<4, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
                if (silhouette.count > 5)
                {
                    testEdgeForAxis<5, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
                    if (silhouette.count > 6)
                    {
                        testEdgeForAxis<6, Accurate>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);
                    }
                }
            }
        }
    }

    static void tryCaliperDir(inout float32_t bestArea, inout float32_t2 bestDir, const float32_t2 dir, const ClippedSilhouette silhouette, uint32_t n3Mask)
    {
        float32_t2 perpDir = float32_t2(-dir.y, dir.x);

        float32_t minAlong = 1e10f;
        float32_t maxAlong = -1e10f;
        float32_t minPerp = 1e10f;
        float32_t maxPerp = -1e10f;

        computeBoundsForAxis<false>(minAlong, maxAlong, minPerp, maxPerp, silhouette, 0, n3Mask, dir, perpDir);

        float32_t area = (maxAlong - minAlong) * (maxPerp - minPerp);
        if (area < bestArea)
        {
            bestArea = area;
            bestDir = dir;
        }
    }

    template <uint32_t I>
    static void processEdge(inout float32_t bestArea, inout float32_t2 bestDir, inout uint32_t convexMask, inout uint32_t n3Mask, const ClippedSilhouette silhouette, inout SilEdgeNormals precompSil)
    {
        const uint32_t nextIdx = (I + 1 < silhouette.count) ? I + 1 : 0;
        float32_t3 S = silhouette.vertices[I];
        float32_t3 E = silhouette.vertices[nextIdx];
        precompSil.edgeNormals[I] = float16_t3(cross(S, E));

        float32_t2 t0, t1;
        getProjectedTangents(S, E, t0, t1);

        tryCaliperDir(bestArea, bestDir, t0, silhouette, n3Mask);

        if (nbl::hlsl::cross2D(S.xy, E.xy) < -1e-6f)
        {
            convexMask |= (1u << I);
            tryCaliperDir(bestArea, bestDir, t1, silhouette, n3Mask);

            if (dot(t0, t1) < 0.5f)
            {
                n3Mask |= (1u << I);
                float32_t2 tangentAtMid = evalCurveTangent(S, E, 0.5f);
                tryCaliperDir(bestArea, bestDir, tangentAtMid, silhouette, n3Mask);
            }
        }
    }

    // ========================================================================
    // Factory methods
    // ========================================================================

    static Parallelogram buildForAxis(const ClippedSilhouette silhouette, uint32_t convexMask, uint32_t n3Mask, float32_t2 dir)
    {
        float32_t2 perpDir = float32_t2(-dir.y, dir.x);

        float32_t minAlong = 1e10f;
        float32_t maxAlong = -1e10f;
        float32_t minPerp = 1e10f;
        float32_t maxPerp = -1e10f;

        computeBoundsForAxis<true>(minAlong, maxAlong, minPerp, maxPerp, silhouette, convexMask, n3Mask, dir, perpDir);

        Parallelogram result;
        result.width = float16_t(maxAlong - minAlong);
        result.height = float16_t(maxPerp - minPerp);
        result.axisDir = float16_t2(dir);
        result.corner = float16_t2(minAlong * dir + minPerp * float16_t2(-dir.y, dir.x));

        return result;
    }

    // Silhouette vertices must be normalized before calling create()
    static Parallelogram create(const ClippedSilhouette silhouette, out SilEdgeNormals precompSil
#if VISUALIZE_SAMPLES
                                ,
                                float32_t2 ndc, float32_t3 spherePos, float32_t aaWidth,
                                inout float32_t4 color
#endif
    )
    {
        precompSil = (SilEdgeNormals)0;
        precompSil.count = silhouette.count;

        uint32_t convexMask = 0;
        uint32_t n3Mask = 0;
        float32_t bestArea = 1e10f;
        float32_t2 bestDir = float32_t2(1.0f, 0.0f);

        processEdge<0>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
        processEdge<1>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
        processEdge<2>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
        if (silhouette.count > 3)
        {
            processEdge<3>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
            if (silhouette.count > 4)
            {
                processEdge<4>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
                if (silhouette.count > 5)
                {
                    processEdge<5>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
                    if (silhouette.count > 6)
                    {
                        processEdge<6>(bestArea, bestDir, convexMask, n3Mask, silhouette, precompSil);
                    }
                }
            }
        }

        tryCaliperDir(bestArea, bestDir, float32_t2(1.0f, 0.0f), silhouette, n3Mask);
        tryCaliperDir(bestArea, bestDir, float32_t2(0.0f, 1.0f), silhouette, n3Mask);

        Parallelogram best = buildForAxis(silhouette, convexMask, n3Mask, bestDir);

#if VISUALIZE_SAMPLES
        for (uint32_t i = 0; i < silhouette.count; i++)
        {
            if (convexMask & (1u << i))
            {
                uint32_t nextIdx = (i + 1) % silhouette.count;
                float32_t2 p0 = GET_PROJ_VERT(i);
                float32_t2 p1 = GET_PROJ_VERT(nextIdx);

                float32_t2 t0, endTangent;
                getProjectedTangents(silhouette.vertices[i], silhouette.vertices[nextIdx], t0, endTangent);

                if (n3Mask & (1u << i))
                {
                    float32_t2 tangentAtMid = evalCurveTangent(silhouette.vertices[i], silhouette.vertices[nextIdx], 0.5f);
                    float32_t2 midPoint = evalCurvePoint(silhouette.vertices[i], silhouette.vertices[nextIdx], 0.5f);

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
#if DEBUG_DATA
        DebugDataBuffer[0].parallelogramArea = best.width * best.height;
#endif

        return best;
    }

    float32_t3 sample(NBL_CONST_REF_ARG(SilEdgeNormals) silhouette, float32_t2 xi, out float32_t pdf, out bool valid)
    {
        float16_t2 perpDir = float16_t2(-axisDir.y, axisDir.x);

        float16_t2 circleXY = corner +
                              float16_t(xi.x) * width * axisDir +
                              float16_t(xi.y) * height * perpDir;

        float32_t3 direction = circleToSphere(circleXY);

        valid = direction.z > 0.0f && silhouette.isInside(direction);
        // PDF in solid angle measure: the rectangle is in circle-space (scaled by CIRCLE_RADIUS),
        // and the orthographic projection Jacobian is dA_circle/dω = CIRCLE_RADIUS^2 * z
        pdf = valid ? (CIRCLE_RADIUS * CIRCLE_RADIUS * direction.z / (float32_t(width) * float32_t(height))) : 0.0f;

        return direction;
    }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_PARALLELOGRAM_SAMPLING_HLSL_INCLUDED_
