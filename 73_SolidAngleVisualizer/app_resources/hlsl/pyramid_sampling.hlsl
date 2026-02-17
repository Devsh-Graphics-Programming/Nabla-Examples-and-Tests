//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_

#include "gpu_common.hlsl"

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>

#include "silhouette.hlsl"
#include "drawing.hlsl"

// ============================================================================
// Spherical Rectangle Bound via Rotating Calipers
//
// Bounds the silhouette with a spherical rectangle (intersection of two
// orthogonal lunes). Each lune is defined by two great circles (planes
// through the origin). The rectangle is parameterized for downstream
// samplers (Urena, bilinear, biquadratic) in pyramid_sampling/*.hlsl.
//
// Algorithm:
// 1. Rotating Calipers: Find the edge that minimizes the lune-width proxy
//    dot(cross(A, B), C) = sin(edge_len) * sin(angular_dist)
//    No per-edge normalization needed, scalar triple product suffices.
//
// 2. Build orthonormal frame from the minimum-width edge:
//    - axis1 = normalize(cross(A, B)), pole of the primary lune
//    - axis2, axis3 complete the frame via edge-based candidate search
//      (tryPrimaryFrameCandidate), oriented toward silhouette center
//
// 3. Project vertices onto the frame as (x/z, y/z)
//    to find the bounding rectangle extents (rectR0, rectExtents)
//
// 4. Fallback: if the primary frame leaves vertices near the z=0 plane,
//    fix axis3 = camera forward (0,0,1) and search axis1/axis2 via
//    tryFallbackFrameCandidate
//
// Key property: If all vertices are inside a great circle half-space,
// then all edges (geodesic arcs) are also inside. No edge extremum
// checking needed (unlike parallelogram_sampling which works in
// projected 2D space where arcs can bulge beyond vertices).
// ============================================================================
// Spherical rectangle bound: stores the orthonormal frame and gnomonic
// projection extents. Consumed by UrenaSampler, BilinearSampler, BiquadraticSampler.
struct SphericalPyramid
{
	// Orthonormal frame for the bounding region
	float32_t3 axis1; // Primary axis (from minimum-width edge's great circle normal)
	float32_t3 axis2; // Secondary axis (perpendicular to axis1)
	float32_t3 axis3; // Forward axis, toward silhouette (primary) or camera forward (fallback)

	// SphericalRectangle parameters (in the local frame where axis3 is Z)
	float32_t3 rectR0;		// Corner position in local frame
	float32_t2 rectExtents; // Width (along axis1) and height (along axis2)
	float32_t solidAngle;	// Solid angle of the bounding region (steradians)

	// ========================================================================
	// Rotating Calipers - Minimum Width Edge Finding (Scalar Triple Product)
	// ========================================================================

	// Simplified metric: dot(cross(A, B), C) = sin(edge_len) * sin(angular_dist)
	// This is a lune-area proxy, no per-edge normalization needed for comparison.
	// Per-vertex cost: one dot product with precomputed edge normal.
	// Per-edge cost: one cross product (replaces addition + rsqrt).
	//
	// Triangular column-major traversal (rotating calipers pattern):
	//   Vertex V_j checks against edges 0..j-2.
	//   V2 -> edge 0;  V3 -> edges 0,1;  V4 -> edges 0,1,2;  etc.
	//   Total checks: (N-2)(N-1)/2 instead of N(N-2).
	//
	// Endpoints: dot(cross(A,B), A) = dot(cross(A,B), B) = 0, never affect max.
	static void findMinimumWidthEdge(const ClippedSilhouette silhouette, out uint32_t bestEdge, out float32_t3 bestV0, out float32_t3 bestV1, out float32_t bestWidth, out SilEdgeNormals precompSil)
	{
		precompSil = (SilEdgeNormals)0;
		precompSil.count = silhouette.count;

		// Edge normals: cross(v[i], v[i+1]), inward-facing for CCW-from-origin winding
		float32_t3 en0 = cross(silhouette.vertices[0], silhouette.vertices[1]);
		precompSil.edgeNormals[0] = float16_t3(en0);
		float32_t3 en1 = cross(silhouette.vertices[1], silhouette.vertices[2]);
		precompSil.edgeNormals[1] = float16_t3(en1);

		// Per-edge max(dot(en_i, v_j)), positive = inside, maximum = widest vertex
		float32_t maxDot0 = dot(silhouette.vertices[2], en0); // V2 vs edge 0

		float32_t maxDot1 = 1e10f;
		float32_t maxDot2 = 1e10f;
		float32_t maxDot3 = 1e10f;
		float32_t maxDot4 = 1e10f;

		if (silhouette.count > 3)
		{
			float32_t3 en2 = cross(silhouette.vertices[2], silhouette.vertices[3]);
			precompSil.edgeNormals[2] = float16_t3(en2);

			// V3 vs edges 0, 1
			float32_t3 v3 = silhouette.vertices[3];
			maxDot0 = max(maxDot0, dot(v3, en0));
			maxDot1 = dot(v3, en1);

			if (silhouette.count > 4)
			{
				float32_t3 en3 = cross(silhouette.vertices[3], silhouette.vertices[4]);
				precompSil.edgeNormals[3] = float16_t3(en3);

				// V4 vs edges 0, 1, 2
				float32_t3 v4 = silhouette.vertices[4];
				maxDot0 = max(maxDot0, dot(v4, en0));
				maxDot1 = max(maxDot1, dot(v4, en1));
				maxDot2 = dot(v4, en2);

				if (silhouette.count > 5)
				{
					float32_t3 en4 = cross(silhouette.vertices[4], silhouette.vertices[5]);
					precompSil.edgeNormals[4] = float16_t3(en4);

					// V5 vs edges 0, 1, 2, 3
					float32_t3 v5 = silhouette.vertices[5];
					maxDot0 = max(maxDot0, dot(v5, en0));
					maxDot1 = max(maxDot1, dot(v5, en1));
					maxDot2 = max(maxDot2, dot(v5, en2));
					maxDot3 = dot(v5, en3);

					if (silhouette.count > 6)
					{
						// V6 vs edges 0, 1, 2, 3, 4
						float32_t3 v6 = silhouette.vertices[6];
						maxDot0 = max(maxDot0, dot(v6, en0));
						maxDot1 = max(maxDot1, dot(v6, en1));
						maxDot2 = max(maxDot2, dot(v6, en2));
						maxDot3 = max(maxDot3, dot(v6, en3));
						maxDot4 = dot(v6, en4);
					}
				}
			}
		}

		// Best edge: minimum maxDot, no per-edge normalization needed.
		// Relative epsilon prevents tie-breaking flicker when two edges have
		// nearly identical widths — the current winner is "sticky" unless a
		// new edge is meaningfully better (0.1% narrower).
		const float32_t EDGE_SELECT_EPS = 1e-3f;

		bestWidth = maxDot0;
		bestEdge = 0;
		bestV0 = silhouette.vertices[0];
		bestV1 = silhouette.vertices[1];

		if (silhouette.count > 3)
		{
			bool better = maxDot1 < bestWidth * (1.0f - EDGE_SELECT_EPS);
			bestWidth = better ? maxDot1 : bestWidth;
			bestEdge = better ? 1 : bestEdge;
			bestV0 = better ? silhouette.vertices[1] : bestV0;
			bestV1 = better ? silhouette.vertices[2] : bestV1;

			if (silhouette.count > 4)
			{
				better = maxDot2 < bestWidth * (1.0f - EDGE_SELECT_EPS);
				bestWidth = better ? maxDot2 : bestWidth;
				bestEdge = better ? 2 : bestEdge;
				bestV0 = better ? silhouette.vertices[2] : bestV0;
				bestV1 = better ? silhouette.vertices[3] : bestV1;

				if (silhouette.count > 5)
				{
					better = maxDot3 < bestWidth * (1.0f - EDGE_SELECT_EPS);
					bestWidth = better ? maxDot3 : bestWidth;
					bestEdge = better ? 3 : bestEdge;
					bestV0 = better ? silhouette.vertices[3] : bestV0;
					bestV1 = better ? silhouette.vertices[4] : bestV1;

					if (silhouette.count > 6)
					{
						better = maxDot4 < bestWidth * (1.0f - EDGE_SELECT_EPS);
						bestWidth = better ? maxDot4 : bestWidth;
						bestEdge = better ? 4 : bestEdge;
						bestV0 = better ? silhouette.vertices[4] : bestV0;
						bestV1 = better ? silhouette.vertices[5] : bestV1;
					}
				}
			}
		}

		// Check the last 2 edges missed by the triangular traversal:
		// Edge count-2: vertices[count-2] -> vertices[count-1], check V0..V[count-3]
		// Edge count-1: vertices[count-1] -> vertices[0],       check V1..V[count-2]
		// Explicit per-count unrolling avoids the generic loop with runtime index comparisons.
		{
			// Penultimate edge: vertices[count-2] -> vertices[count-1]
			const uint32_t penIdx = silhouette.count - 2;
			float32_t3 enPen = cross(silhouette.vertices[penIdx], silhouette.vertices[penIdx + 1]);
			precompSil.edgeNormals[penIdx] = float16_t3(enPen);
			float32_t maxDotPen = dot(silhouette.vertices[0], enPen);
			if (silhouette.count > 3)
			{
				maxDotPen = max(maxDotPen, dot(silhouette.vertices[1], enPen));
				if (silhouette.count > 4)
				{
					maxDotPen = max(maxDotPen, dot(silhouette.vertices[2], enPen));
					if (silhouette.count > 5)
					{
						maxDotPen = max(maxDotPen, dot(silhouette.vertices[3], enPen));
						if (silhouette.count > 6)
						{
							maxDotPen = max(maxDotPen, dot(silhouette.vertices[4], enPen));
						}
					}
				}
			}

			bool betterPen = maxDotPen < bestWidth * (1.0f - EDGE_SELECT_EPS);
			bestWidth = betterPen ? maxDotPen : bestWidth;
			bestEdge = betterPen ? penIdx : bestEdge;
			bestV0 = betterPen ? silhouette.vertices[penIdx] : bestV0;
			bestV1 = betterPen ? silhouette.vertices[penIdx + 1] : bestV1;

			// Last edge: vertices[count-1] -> vertices[0] (wrap-around)
			const uint32_t lastIdx = silhouette.count - 1;
			float32_t3 enLast = cross(silhouette.vertices[lastIdx], silhouette.vertices[0]);
			precompSil.edgeNormals[lastIdx] = float16_t3(enLast);
			float32_t maxDotLast = dot(silhouette.vertices[1], enLast);
			if (silhouette.count > 3)
			{
				maxDotLast = max(maxDotLast, dot(silhouette.vertices[2], enLast));
				if (silhouette.count > 4)
				{
					maxDotLast = max(maxDotLast, dot(silhouette.vertices[3], enLast));
					if (silhouette.count > 5)
					{
						maxDotLast = max(maxDotLast, dot(silhouette.vertices[4], enLast));
						if (silhouette.count > 6)
						{
							maxDotLast = max(maxDotLast, dot(silhouette.vertices[5], enLast));
						}
					}
				}
			}

			bool betterLast = maxDotLast < bestWidth * (1.0f - EDGE_SELECT_EPS);
			bestWidth = betterLast ? maxDotLast : bestWidth;
			bestEdge = betterLast ? lastIdx : bestEdge;
			bestV0 = betterLast ? silhouette.vertices[lastIdx] : bestV0;
			bestV1 = betterLast ? silhouette.vertices[0] : bestV1;
		}
	}

	// ========================================================================
	// Template-Unrolled Projection Helpers
	// ========================================================================

	// Project a single vertex onto candidate axes, updating bounds and minZ in one fused pass
	template <uint32_t I>
	static void projectAndBound(const float32_t3 vertices[MAX_SILHOUETTE_VERTICES], float32_t3 projAxis1, float32_t3 projAxis2, float32_t3 projAxis3, NBL_REF_ARG(float32_t4) bound, NBL_REF_ARG(float32_t) minZ)
	{
		float32_t3 v = vertices[I];
		float32_t x = dot(v, projAxis1);
		float32_t y = dot(v, projAxis2);
		float32_t z = dot(v, projAxis3);
		minZ = min(minZ, z);
		float32_t rcpZ = rcp(z);
		float32_t projX = x * rcpZ;
		float32_t projY = y * rcpZ;
		bound.x = min(bound.x, projX);
		bound.y = min(bound.y, projY);
		bound.z = max(bound.z, projX);
		bound.w = max(bound.w, projY);
	}

	// Project all silhouette vertices (template-unrolled, fused bounds + minZ)
	static void projectAllVertices(const ClippedSilhouette silhouette, float32_t3 projAxis1, float32_t3 projAxis2, float32_t3 projAxis3, NBL_REF_ARG(float32_t4) bound, NBL_REF_ARG(float32_t) minZ)
	{
		bound = float32_t4(1e10f, 1e10f, -1e10f, -1e10f);
		minZ = 1e10f;
		projectAndBound<0>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
		projectAndBound<1>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
		projectAndBound<2>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
		if (silhouette.count > 3)
		{
			projectAndBound<3>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
			if (silhouette.count > 4)
			{
				projectAndBound<4>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
				if (silhouette.count > 5)
				{
					projectAndBound<5>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
					if (silhouette.count > 6)
					{
						projectAndBound<6>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound, minZ);
					}
				}
			}
		}
	}

	// ========================================================================
	// Template-Unrolled Frame Candidate Selection
	// ========================================================================

	// Try an edge as frame candidate for the primary path (axis1 fixed, find best axis2/axis3)
	template <uint32_t I, bool CheckCount = false>
	static void tryPrimaryFrameCandidate(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, float32_t3 fixedAxis1, float32_t3 axis3Ref,
										 NBL_REF_ARG(float32_t) bestArea, NBL_REF_ARG(float32_t3) bestAxis2,
										 NBL_REF_ARG(float32_t3) bestAxis3, NBL_REF_ARG(bool) found,
										 NBL_REF_ARG(float32_t) bestMinZ, NBL_REF_ARG(float32_t4) bestBound)
	{
		const uint32_t j = CheckCount ? ((I + 1 < silhouette.count) ? I + 1 : 0) : I + 1;
		float32_t3 edge = silhouette.vertices[j] - silhouette.vertices[I];

		// Candidate axis2: perpendicular to edge, in plane perpendicular to axis1
		float32_t3 axis2Cand = cross(fixedAxis1, edge);
		float32_t lenSq = dot(axis2Cand, axis2Cand);
		if (lenSq < 1e-14f)
			return;
		axis2Cand *= rsqrt(lenSq);

		// Candidate axis3: completes the frame
		float32_t3 axis3Cand = cross(fixedAxis1, axis2Cand);

		// Ensure axis3 points toward center (same hemisphere as reference)
		if (dot(axis3Cand, axis3Ref) < 0.0f)
		{
			axis2Cand = -axis2Cand;
			axis3Cand = -axis3Cand;
		}

		// Fused: check all vertices have positive z AND compute bounding rect in one pass
		float32_t4 bound;
		float32_t minZ;
		projectAllVertices(silhouette, fixedAxis1, axis2Cand, axis3Cand, bound, minZ);

		// Skip if any vertex would have z <= 0
		if (minZ <= 1e-6f)
			return;

		float32_t rectArea = (bound.z - bound.x) * (bound.w - bound.y);
		if (rectArea < bestArea)
		{
			bestArea = rectArea;
			bestAxis2 = axis2Cand;
			bestAxis3 = axis3Cand;
			bestMinZ = minZ;
			bestBound = bound;
			found = true;
		}
	}

	// Try an edge as frame candidate for the fallback path (axis3 fixed, find best axis1/axis2)
	template <uint32_t I, bool CheckCount = false>
	static void tryFallbackFrameCandidate(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, float32_t3 fixedAxis3, NBL_REF_ARG(float32_t) bestArea, NBL_REF_ARG(float32_t3) bestAxis1, NBL_REF_ARG(float32_t3) bestAxis2, NBL_REF_ARG(uint32_t) bestEdge, NBL_REF_ARG(float32_t4) bestBound)
	{
		const uint32_t j = CheckCount ? ((I + 1 < silhouette.count) ? I + 1 : 0) : I + 1;
		float32_t3 edge = silhouette.vertices[j] - silhouette.vertices[I];

		float32_t3 edgeInPlane = edge - fixedAxis3 * dot(edge, fixedAxis3);
		float32_t lenSq = dot(edgeInPlane, edgeInPlane);
		if (lenSq < 1e-14f)
			return;

		float32_t3 axis1Cand = edgeInPlane * rsqrt(lenSq);
		float32_t3 axis2Cand = cross(fixedAxis3, axis1Cand);

		float32_t4 bound;
		float32_t minZ;
		projectAllVertices(silhouette, axis1Cand, axis2Cand, fixedAxis3, bound, minZ);

		float32_t rectArea = (bound.z - bound.x) * (bound.w - bound.y);
		if (rectArea < bestArea)
		{
			bestArea = rectArea;
			bestAxis1 = axis1Cand;
			bestAxis2 = axis2Cand;
			bestBound = bound;
			bestEdge = I;
		}
	}

	// ========================================================================
	// Visualization
	// ========================================================================

#if VISUALIZE_SAMPLES
	float32_t4 visualize(float32_t3 spherePos, float32_t2 ndc, float32_t aaWidth)
	{
		float32_t4 color = float32_t4(0, 0, 0, 0);

		// Colors for visualization
		float32_t3 boundColor1 = float32_t3(1.0f, 0.5f, 0.5f); // Light red for axis1 bounds
		float32_t3 boundColor2 = float32_t3(0.5f, 0.5f, 1.0f); // Light blue for axis2 bounds
		float32_t3 centerColor = float32_t3(1.0f, 1.0f, 0.0f); // Yellow for center

		float32_t x0 = rectR0.x;
		float32_t x1 = rectR0.x + rectExtents.x;
		float32_t y0 = rectR0.y;
		float32_t y1 = rectR0.y + rectExtents.y;
		float32_t z = rectR0.z;

		// Great circle normals for the 4 edges (in local frame, then transform to world)
		float32_t3 bottomNormalLocal = normalize(float32_t3(0, -z, y0));
		float32_t3 topNormalLocal = normalize(float32_t3(0, z, -y1));
		float32_t3 leftNormalLocal = normalize(float32_t3(-z, 0, x0));
		float32_t3 rightNormalLocal = normalize(float32_t3(z, 0, -x1));

		// Transform to world space
		float32_t3 bottomNormal = bottomNormalLocal.x * axis1 + bottomNormalLocal.y * axis2 + bottomNormalLocal.z * axis3;
		float32_t3 topNormal = topNormalLocal.x * axis1 + topNormalLocal.y * axis2 + topNormalLocal.z * axis3;
		float32_t3 leftNormal = leftNormalLocal.x * axis1 + leftNormalLocal.y * axis2 + leftNormalLocal.z * axis3;
		float32_t3 rightNormal = rightNormalLocal.x * axis1 + rightNormalLocal.y * axis2 + rightNormalLocal.z * axis3;

		// Draw the 4 bounding great circles
		color += drawGreatCircleHalf(bottomNormal, spherePos, axis3, aaWidth, boundColor2, 0.004f);
		color += drawGreatCircleHalf(topNormal, spherePos, axis3, aaWidth, boundColor2, 0.004f);
		color += drawGreatCircleHalf(leftNormal, spherePos, axis3, aaWidth, boundColor1, 0.004f);
		color += drawGreatCircleHalf(rightNormal, spherePos, axis3, aaWidth, boundColor1, 0.004f);

		// Draw center point (center of the rectangle projected onto sphere)
		float32_t centerX = (x0 + x1) * 0.5f;
		float32_t centerY = (y0 + y1) * 0.5f;
		float32_t3 centerLocal = normalize(float32_t3(centerX, centerY, z));
		float32_t3 centerWorld = centerLocal.x * axis1 - centerLocal.y * axis2 + centerLocal.z * axis3;

		float32_t3 centerCircle = sphereToCircle(centerWorld);
		color += drawCorner(centerCircle, ndc, aaWidth, 0.025f, 0.0f, centerColor);

		color += drawCorner(axis1, ndc, aaWidth, 0.025f, 0.0f, float32_t3(1.0f, 0.0f, 0.0f));
		color += drawCorner(axis2, ndc, aaWidth, 0.025f, 0.0f, float32_t3(0.0f, 1.0f, 0.0f));
		color += drawCorner(axis3, ndc, aaWidth, 0.025f, 0.0f, float32_t3(0.0f, 0.0f, 1.0f));

		return color;
	}
#endif // VISUALIZE_SAMPLES

	// ========================================================================
	// Factory
	// ========================================================================

	static SphericalPyramid create(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, NBL_REF_ARG(SilEdgeNormals) silEdgeNormals
#if VISUALIZE_SAMPLES
								   ,
								   float32_t2 ndc, float32_t3 spherePos, float32_t aaWidth, inout float32_t4 color
#endif
	)
	{
		SphericalPyramid self;

		// Step 1: Find minimum-width edge using rotating calipers with lune metric
		uint32_t bestEdge;
		float32_t3 bestV0, bestV1;
		float32_t minWidth;
		findMinimumWidthEdge(silhouette, bestEdge, bestV0, bestV1, minWidth, silEdgeNormals);

		// Step 2: Build orthonormal frame from best edge
		// axis1 = perpendicular to the best edge's great circle (primary caliper direction)
		self.axis1 = normalize(cross(bestV0, bestV1));

		// Compute centroid for reference direction
		float32_t3 center = silhouette.getCenter();
		float32_t3 centerInPlane = center - self.axis1 * dot(center, self.axis1);
		float32_t3 axis3Ref = normalize(centerInPlane);

		// Step 2b: Try each edge-aligned rotation around axis1 to find the axis2/axis3
		// orientation that keeps all vertices in the positive half-space with minimum
		// bounding rectangle area
		float32_t bestRectArea = 1e20f;
		float32_t3 bestAxis2 = cross(axis3Ref, self.axis1);
		float32_t3 bestAxis3 = axis3Ref;
		bool foundValidFrame = false;
		float32_t bestMinZ = 0.0f;
		float32_t4 bounds = float32_t4(-0.1f, -0.1f, 0.1f, 0.1f);

		tryPrimaryFrameCandidate<0>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
		tryPrimaryFrameCandidate<1>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
		tryPrimaryFrameCandidate<2>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
		if (silhouette.count > 3)
		{
			tryPrimaryFrameCandidate<3, true>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
			if (silhouette.count > 4)
			{
				tryPrimaryFrameCandidate<4, true>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
				if (silhouette.count > 5)
				{
					tryPrimaryFrameCandidate<5, true>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
					if (silhouette.count > 6)
					{
						tryPrimaryFrameCandidate<6, true>(silhouette, self.axis1, axis3Ref, bestRectArea, bestAxis2, bestAxis3, foundValidFrame, bestMinZ, bounds);
					}
				}
			}
		}

		self.axis2 = bestAxis2;
		self.axis3 = bestAxis3;

		// Fallback: if the primary path failed (no valid frame found, or axis3 leaves
		// vertices too close to the z=0 singularity), fix axis3 = camera forward and
		// search for the best axis1/axis2 rotation around it.
		if (!foundValidFrame || bestMinZ < 0.15f)
		{
			// Use camera forward as axis3 (all silhouette vertices have z > 0 by construction)
			self.axis3 = float32_t3(0.0f, 0.0f, 1.0f);

			// Find optimal axis1/axis2 rotation around axis3 by trying each edge
			float32_t bestFallbackArea = 1e20f;
			// axis3 = (0,0,1), so cross((0,0,1), (1,0,0)) = (0,1,0), cross((0,0,1), (0,1,0)) = (-1,0,0)
			self.axis1 = float32_t3(0.0f, 1.0f, 0.0f);
			self.axis2 = float32_t3(-1.0f, 0.0f, 0.0f);

			tryFallbackFrameCandidate<0>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
			tryFallbackFrameCandidate<1>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
			tryFallbackFrameCandidate<2>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
			if (silhouette.count > 3)
			{
				tryFallbackFrameCandidate<3, true>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
				if (silhouette.count > 4)
				{
					tryFallbackFrameCandidate<4, true>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
					if (silhouette.count > 5)
					{
						tryFallbackFrameCandidate<5, true>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
						if (silhouette.count > 6)
						{
							tryFallbackFrameCandidate<6, true>(silhouette, self.axis3, bestFallbackArea, self.axis1, self.axis2, bestEdge, bounds);
						}
					}
				}
			}
		}

		// Degenerate bounds check (single computation, after primary/fallback decision)
		if (bounds.x >= bounds.z || bounds.y >= bounds.w)
			bounds = float32_t4(-0.1f, -0.1f, 0.1f, 0.1f);

		self.rectR0 = float32_t3(bounds.xy, 1.0f);
		self.rectExtents = float32_t2(bounds.zw - bounds.xy);

#if VISUALIZE_SAMPLES
		color += drawCorner(center, ndc, aaWidth, 0.05f, 0.0f, float32_t3(1.0f, 0.0f, 1.0f));
		color += visualizeBestCaliperEdge(silhouette.vertices, bestEdge, silhouette.count, spherePos, aaWidth);
		color += self.visualize(spherePos, ndc, aaWidth);
#endif

#if DEBUG_DATA
		DebugDataBuffer[0].pyramidAxis1 = self.axis1;
		DebugDataBuffer[0].pyramidAxis2 = self.axis2;
		DebugDataBuffer[0].pyramidCenter = center;
		DebugDataBuffer[0].pyramidHalfWidth1 = (atan(bounds.z) - atan(bounds.x)) * 0.5f;
		DebugDataBuffer[0].pyramidHalfWidth2 = (atan(bounds.w) - atan(bounds.y)) * 0.5f;
		DebugDataBuffer[0].pyramidSolidAngle = self.solidAngle;
		DebugDataBuffer[0].pyramidBestEdge = bestEdge;
		DebugDataBuffer[0].pyramidMin1 = bounds.x;
		DebugDataBuffer[0].pyramidMin2 = bounds.y;
		DebugDataBuffer[0].pyramidMax1 = bounds.z;
		DebugDataBuffer[0].pyramidMax2 = bounds.w;
#endif

		return self;
	}
};

#include "pyramid_sampling/urena.hlsl"
#include "pyramid_sampling/bilinear.hlsl"
#include "pyramid_sampling/biquadratic.hlsl"

#endif // _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
