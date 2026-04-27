//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_

#include "common.hlsl"

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>

#include "silhouette.hlsl"
#include "drawing.hlsl"

// ============================================================================
// Spherical Pyramid: gnomonic bounding rectangle for silhouette sampling.
//
// Algorithm (SphericalPyramid::create):
// 1. Adaptive axis3: blend silhouette centroid toward (0,0,1) to keep
//    all vertices in the positive gnomonic half-space. Branchless.
// 2. Rotating calipers: try each edge projected perpendicular to axis3,
//    keep the axis1/axis2 rotation with minimum gnomonic bounding area.
//    Edge normals are fused into this pass (cross products from the same
//    vertex loads).
// 3. Sign-stabilize axis1 against a world-space reference.
//
// axis3 is not stored, reconstructed as cross(axis1, axis2).
// rectR0 is float2 (z is always 1.0 in gnomonic space).
// ============================================================================
struct SphericalPyramid
{
   float32_t3 axis1; // edge-aligned, perpendicular to axis3
   float32_t3 axis2; // = cross(axis3, axis1); axis3 reconstructed via getAxis3()
   float32_t2 rectR0; // gnomonic bounding rect corner (z=1 implicit)
   float32_t2 rectExtents;

   float32_t3 getAxis3() NBL_CONST_MEMBER_FUNC { return cross(axis1, axis2); }

   // ========================================================================
   // Gnomonic Projection
   // ========================================================================
   template<uint32_t I>
   static void projectAndBound(const float32_t3 vertices[MAX_SILHOUETTE_VERTICES], float32_t3 projAxis1, float32_t3 projAxis2, float32_t3 projAxis3, NBL_REF_ARG(float32_t4) bound)
   {
      float32_t3 v = vertices[I];
      float32_t x = dot(v, projAxis1);
      float32_t y = dot(v, projAxis2);
      float32_t z = dot(v, projAxis3);
      float32_t rcpZ = (z > 0.0f) ? rcp(z) : 0.0f;
      float32_t projX = x * rcpZ;
      float32_t projY = y * rcpZ;
      bound.x = min(bound.x, projX);
      bound.y = min(bound.y, projY);
      bound.z = max(bound.z, projX);
      bound.w = max(bound.w, projY);
   }

   // Template-unrolled projection of all vertices.
   static void projectAllVertices(const ClippedSilhouette silhouette, float32_t3 projAxis1, float32_t3 projAxis2, float32_t3 projAxis3, NBL_REF_ARG(float32_t4) bound)
   {
      bound = float32_t4(1e10f, 1e10f, -1e10f, -1e10f);
      projectAndBound<0>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
      projectAndBound<1>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
      projectAndBound<2>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
      if (silhouette.count > 3)
      {
         projectAndBound<3>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
         if (silhouette.count > 4)
         {
            projectAndBound<4>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
            if (silhouette.count > 5)
            {
               projectAndBound<5>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
               if (silhouette.count > 6)
               {
                  projectAndBound<6>(silhouette.vertices, projAxis1, projAxis2, projAxis3, bound);
               }
            }
         }
      }
   }

   // ========================================================================
   // Adaptive Axis3
   // ========================================================================

   // t = max blend keeping dot(v, centroid*t + (0,0,1)) >= margin.
   template<uint32_t I>
   static float32_t blendLimit(const float32_t3 vertices[MAX_SILHOUETTE_VERTICES], float32_t3 center, float32_t margin, float32_t curMin)
   {
      float32_t cd = dot(vertices[I], center);
      float32_t tLimit = (cd < 0.0f) ? ((vertices[I].z - margin) / -cd) : 1e10f;
      return min(curMin, tLimit);
   }

   static float32_t computeBlendFactor(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, float32_t3 center, float32_t margin)
   {
      float32_t t = 1e10f;
      t = blendLimit<0>(silhouette.vertices, center, margin, t);
      t = blendLimit<1>(silhouette.vertices, center, margin, t);
      t = blendLimit<2>(silhouette.vertices, center, margin, t);
      if (silhouette.count > 3)
      {
         t = blendLimit<3>(silhouette.vertices, center, margin, t);
         if (silhouette.count > 4)
         {
            t = blendLimit<4>(silhouette.vertices, center, margin, t);
            if (silhouette.count > 5)
            {
               t = blendLimit<5>(silhouette.vertices, center, margin, t);
               if (silhouette.count > 6)
               {
                  t = blendLimit<6>(silhouette.vertices, center, margin, t);
               }
            }
         }
      }
      return max(t, 0.0f);
   }

   // ========================================================================
   // Rotating Calipers (fused edge normal computation)
   // ========================================================================
   template<uint32_t I, bool CheckCount = false>
   static void tryCaliperCandidate(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, float32_t3 fixedAxis3,
      NBL_REF_ARG(float32_t) bestArea, NBL_REF_ARG(float32_t3) bestAxis1,
      NBL_REF_ARG(float32_t3) bestAxis2, NBL_REF_ARG(float32_t4) bestBound,
      NBL_REF_ARG(uint32_t) bestEdge, NBL_REF_ARG(SilEdgeNormals) silEdgeNormals)
   {
      const uint32_t j = CheckCount ? ((I + 1 < silhouette.count) ? I + 1 : 0) : I + 1;
      float32_t3 vI = silhouette.vertices[I];
      float32_t3 vJ = silhouette.vertices[j];

      // Fused: edge normal from the same vertex pair (vertices already in registers)
      silEdgeNormals.edgeNormals[I] = cross(vI, vJ);

      float32_t3 edge = vJ - vI;

      // Project edge perpendicular to axis3. Skip edges nearly parallel to axis3.
      float32_t3 edgeInPlane = edge - fixedAxis3 * dot(edge, fixedAxis3);
      float32_t lenSq = dot(edgeInPlane, edgeInPlane);
      if (lenSq < 0.01f * dot(edge, edge))
         return;

      float32_t3 axis1Cand = edgeInPlane * rsqrt(lenSq);
      float32_t3 axis2Cand = cross(fixedAxis3, axis1Cand);

      float32_t4 bound;
      projectAllVertices(silhouette, axis1Cand, axis2Cand, fixedAxis3, bound);

      // Sticky selection: new edge must be meaningfully better (1% smaller area)
      // to prevent jitter when two edges have nearly identical bounding rects.
      float32_t rectArea = (bound.z - bound.x) * (bound.w - bound.y);
      if (rectArea < bestArea * (1.0f - 1e-2f))
      {
         bestArea = rectArea;
         bestAxis1 = axis1Cand;
         bestAxis2 = axis2Cand;
         bestBound = bound;
         bestEdge = I;
      }
   }

   // ========================================================================
   // Factory
   // ========================================================================

   static SphericalPyramid create(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, NBL_REF_ARG(SilEdgeNormals) silEdgeNormals)
   {
      SphericalPyramid self;
      silEdgeNormals = (SilEdgeNormals)0;

      // Step 1: Adaptive axis3 (local var, reconstructed via getAxis3() after construction).
      float32_t3 center = silhouette.getUnnormalizedCenter();
      const float32_t AXIS3_MARGIN = 0.15f;
      float32_t tBlend = computeBlendFactor(silhouette, center, AXIS3_MARGIN);
      float32_t3 axis3 = normalize(center * tBlend + float32_t3(0.0f, 0.0f, 1.0f));

      // Step 2: Rotating calipers, min-area gnomonic bounding rectangle.
      float32_t bestArea = 1e20f;
      self.axis1 = float32_t3(0.0f, 1.0f, 0.0f);
      self.axis2 = float32_t3(-1.0f, 0.0f, 0.0f);
      float32_t4 bounds = float32_t4(-0.1f, -0.1f, 0.1f, 0.1f);
      uint32_t bestEdge = 0;

      // Each candidate also computes cross(v[I], v[j]) for edge normals.
      // I=2 needs the wrap check because count can be exactly 3 (j must wrap to 0).
      tryCaliperCandidate<0>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
      tryCaliperCandidate<1>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
      tryCaliperCandidate<2, true>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
      if (silhouette.count > 3)
      {
         tryCaliperCandidate<3, true>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
         if (silhouette.count > 4)
         {
            tryCaliperCandidate<4, true>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
            if (silhouette.count > 5)
            {
               tryCaliperCandidate<5, true>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
               if (silhouette.count > 6)
               {
                  tryCaliperCandidate<6, true>(silhouette, axis3, bestArea, self.axis1, self.axis2, bounds, bestEdge, silEdgeNormals);
               }
            }
         }
      }

      // Step 3: Stabilize axis1 sign against a world-space reference.
      {
         float32_t3 worldRef = nbl::hlsl::select(abs(axis3.x) < 0.9f, float32_t3(1.0f, 0.0f, 0.0f), float32_t3(0.0f, 1.0f, 0.0f));
         float32_t3 axis1Ref = worldRef - axis3 * dot(worldRef, axis3);
         if (dot(self.axis1, axis1Ref) < 0.0f)
         {
            self.axis1 = -self.axis1;
            // axis2 also flips (recomputed below), so mirror both x and y bounds.
            bounds = float32_t4(-bounds.z, -bounds.w, -bounds.x, -bounds.y);
         }
      }

      // Step 4: Recompute axis2 so getAxis3() = cross(axis1, axis2) recovers axis3.
      self.axis2 = cross(axis3, self.axis1);

      // Degenerate bounds check
      if (bounds.x >= bounds.z || bounds.y >= bounds.w)
         bounds = float32_t4(-0.1f, -0.1f, 0.1f, 0.1f);

      self.rectR0 = bounds.xy;
      self.rectExtents = float32_t2(bounds.zw - bounds.xy);

      float32_t solidAngle;
      {
         nbl::hlsl::sampling::SphericalRectangle<float32_t> rectSampler = nbl::hlsl::sampling::SphericalRectangle<float32_t>::create(float32_t3x3(self.axis1, self.axis2, self.getAxis3()), float32_t3(self.rectR0, 1.0f), self.rectExtents);
         solidAngle = rectSampler.solidAngle;
      }

      VisContext::add(SphereDrawer::drawDot(normalize(center), 0.05f, 0.0f, float32_t3(1.0f, 0.0f, 1.0f)));
      VisContext::add(SphereDrawer::visualizeBestCaliperEdge(silhouette, bestEdge));
      self.visualize();

      DebugRecorder::recordPyramid(self.axis1, self.axis2, center, bounds, solidAngle, bestEdge);

      return self;
   }

   // ========================================================================
   // Visualization
   // ========================================================================

   void visualize()
   {
      // Colors for visualization
      float32_t3 boundColor1 = float32_t3(1.0f, 0.5f, 0.5f); // Light red for axis1 bounds
      float32_t3 boundColor2 = float32_t3(0.5f, 0.5f, 1.0f); // Light blue for axis2 bounds
      float32_t3 centerColor = float32_t3(1.0f, 1.0f, 0.0f); // Yellow for center

      float32_t3 a3 = getAxis3();
      float32_t x0 = rectR0.x;
      float32_t x1 = rectR0.x + rectExtents.x;
      float32_t y0 = rectR0.y;
      float32_t y1 = rectR0.y + rectExtents.y;
      const float32_t z = 1.0f;

      // Great circle normals for the 4 edges (in local frame, then transform to world)
      float32_t3 bottomNormalLocal = normalize(float32_t3(0, -z, y0));
      float32_t3 topNormalLocal = normalize(float32_t3(0, z, -y1));
      float32_t3 leftNormalLocal = normalize(float32_t3(-z, 0, x0));
      float32_t3 rightNormalLocal = normalize(float32_t3(z, 0, -x1));

      // Transform to world space
      float32_t3 bottomNormal = bottomNormalLocal.x * axis1 + bottomNormalLocal.y * axis2 + bottomNormalLocal.z * a3;
      float32_t3 topNormal = topNormalLocal.x * axis1 + topNormalLocal.y * axis2 + topNormalLocal.z * a3;
      float32_t3 leftNormal = leftNormalLocal.x * axis1 + leftNormalLocal.y * axis2 + leftNormalLocal.z * a3;
      float32_t3 rightNormal = rightNormalLocal.x * axis1 + rightNormalLocal.y * axis2 + rightNormalLocal.z * a3;

      // Draw center point (center of the rectangle projected onto sphere)
      float32_t centerX = (x0 + x1) * 0.5f;
      float32_t centerY = (y0 + y1) * 0.5f;
      float32_t3 centerLocal = normalize(float32_t3(centerX, centerY, z));
      float32_t3 centerWorld = centerLocal.x * axis1 + centerLocal.y * axis2 + centerLocal.z * a3;

      VisContext::add(SphereDrawer::drawCorner(centerWorld, 0.025f, 0.0f, centerColor));
      // Draw the 4 bounding great circles
      VisContext::add(SphereDrawer::drawGreatCircleHalf(bottomNormal, a3, boundColor2, 0.004f));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(topNormal, a3, boundColor2, 0.004f));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(leftNormal, a3, boundColor1, 0.004f));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(rightNormal, a3, boundColor1, 0.004f));

      VisContext::add(SphereDrawer::drawDot(axis1, 0.025f, 0.0f, float32_t3(1.0f, 0.0f, 0.0f)));
      VisContext::add(SphereDrawer::drawDot(axis2, 0.025f, 0.0f, float32_t3(0.0f, 1.0f, 0.0f)));
      VisContext::add(SphereDrawer::drawDot(a3, 0.025f, 0.0f, float32_t3(0.0f, 0.0f, 1.0f)));
   }
};


#include "pyramid_sampling/bilinear.hlsl"
#include "pyramid_sampling/biquadratic.hlsl"

#endif // _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
