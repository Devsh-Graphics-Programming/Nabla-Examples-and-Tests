//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_

#include "common.hlsl"

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl> // acos_csc_approx
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>

#include "silhouette.hlsl"
#include "drawing.hlsl"

// ============================================================================
// Spherical Pyramid: gnomonic bounding rectangle for silhouette sampling.
//
// Algorithm (SphericalPyramid::create):
// 1. Pass 1: walk the silhouette CCW, accumulating
//      unnormCentroid = sum(cross(v_i, v_{i+1}) * acos_csc_approx(dot(v_i, v_{i+1})))
//    which is the sum of normalized outward edge normals weighted by arc length
//    (Kelvin-Stokes form). This is the true spherical centroid of the polygon
//    and serves as a much better gnomonic-projection axis than blending the raw
//    vertex centroid toward (0,0,1). The cross products are also written into
//    silEdgeNormals.edgeNormals[i] (used later by the inside-polygon test).
// 2. axis3 = normalize(unnormCentroid).
// 3. Pass 2: Frisvad basis (u, v) orthogonal to axis3; project all silhouette
//    vertices to 2D gnomonic coordinates in (u, v) once, up front.
// 4. Pass 3: "guesstimate" calipers: pick the longest 2D edge as axis1, do
//    a single bound pass. O(N) edge-length compares + 1 bound pass, vs the old
//    O(N^2) cascade. The bound is slightly looser than the true min-area rect
//    but the rejection sampler tolerates that.
// 5. Reconstruct 3D axis1, axis2; sign-stabilize axis1 against a world ref.
//
// axis3 is not stored, reconstructed as cross(axis1, axis2).
// rectR0 is float2 (z is always 1.0 in the local gnomonic frame).
// ============================================================================
struct SphericalPyramid
{
   float32_t3 axis1; // edge-aligned, perpendicular to axis3
   float32_t3 axis2; // = cross(axis3, axis1); axis3 reconstructed via getAxis3()
   float32_t2 rectR0; // gnomonic bounding rect corner (z=1 implicit)
   float32_t2 rectExtents;

   float32_t3 getAxis3() NBL_CONST_MEMBER_FUNC { return cross(axis1, axis2); }

   // ========================================================================
   // Pass 1: per-edge cross + arc-length-weighted accumulate
   // ========================================================================
   template<uint32_t I, bool CheckCount = false>
   static void accumulateEdge(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, NBL_REF_ARG(float32_t3) unnormCentroid, NBL_REF_ARG(SilEdgeNormals) silEdgeNormals)
   {
      const uint32_t j              = CheckCount ? ((I + 1 < silhouette.count) ? I + 1 : 0) : I + 1;
      float32_t3     vI             = silhouette.vertices[I];
      float32_t3     vJ             = silhouette.vertices[j];
      float32_t3     c              = cross(vI, vJ);
      silEdgeNormals.edgeNormals[I] = c;
      // |c| = sin(arc) since vI, vJ are unit; so c/|c| * arc = c * acos(dot)/sin(arc) = c * acos_csc(dot).
      // Clamp away from -1: acos_csc_approx contains log2(1+arg), which goes -inf at arg=-1 and
      // produces inf-inf = NaN inside the order-2 polynomial for near-antipodal edges (which can
      // occur for "wide" silhouettes whose adjacent vertices sit far apart on the sphere).
      // TODO: will be moved to it's own namespace
      const float32_t cos_arc = max(dot(vI, vJ), -1.0f + 1e-5f);
      unnormCentroid += c * nbl::hlsl::shapes::acos_csc_approx<float32_t, 1>(cos_arc);
   }

   // ========================================================================
   // Pass 2: gnomonic project a single silhouette vertex into the (u,v) plane.
   // Skips the (w_dot > 0) guard, axis3 = normalize(unnormCentroid) is the
   // polygon's interior direction so all vertices have w_dot > 0 by construction.
   // ========================================================================
   template<uint32_t I>
   static float32_t2 projectVertex2D(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, float32_t3 axis_u, float32_t3 axis_v, float32_t3 axis3)
   {
      float32_t3 vert = silhouette.vertices[I];
      float32_t  rcpW = rcp(dot(vert, axis3));
      return float32_t2(dot(vert, axis_u), dot(vert, axis_v)) * rcpW;
   }

   // ========================================================================
   // Pass 3: 2D rotating-calipers helpers
   // ========================================================================
   template<uint32_t K>
   static void boundOne2D(const float32_t2 verts2d[MAX_SILHOUETTE_VERTICES], float32_t2 axis2d, float32_t2 perp2d, NBL_REF_ARG(float32_t4) bound)
   {
      float32_t2 v2 = verts2d[K];
      float32_t  x  = dot(v2, axis2d);
      float32_t  y  = dot(v2, perp2d);
      bound.x       = min(bound.x, x);
      bound.y       = min(bound.y, y);
      bound.z       = max(bound.z, x);
      bound.w       = max(bound.w, y);
   }

   static void computeBound2D(const float32_t2 verts2d[MAX_SILHOUETTE_VERTICES], uint32_t count, float32_t2 axis2d, float32_t2 perp2d, NBL_REF_ARG(float32_t4) bound)
   {
      bound = float32_t4(1e10f, 1e10f, -1e10f, -1e10f);
      boundOne2D<0>(verts2d, axis2d, perp2d, bound);
      boundOne2D<1>(verts2d, axis2d, perp2d, bound);
      boundOne2D<2>(verts2d, axis2d, perp2d, bound);
      if (count > 3)
      {
         boundOne2D<3>(verts2d, axis2d, perp2d, bound);
         if (count > 4)
         {
            boundOne2D<4>(verts2d, axis2d, perp2d, bound);
            if (count > 5)
            {
               boundOne2D<5>(verts2d, axis2d, perp2d, bound);
               if (count > 6)
                  boundOne2D<6>(verts2d, axis2d, perp2d, bound);
            }
         }
      }
   }

   // "Guesstimate" pass 3: pick the longest 2D edge as axis1 and do ONE bound
   // computation, instead of trying every edge as a caliper candidate. O(N) +
   // one bound pass, vs old O(N^2) of bound passes. The bound is slightly
   // looser than the true min-area rect (typically a few percent for OBB
   // silhouettes), but the rejection sampler tolerates that.
   template<uint32_t I, bool CheckCount = false>
   static void considerEdge(const float32_t2 verts2d[MAX_SILHOUETTE_VERTICES], uint32_t count, NBL_REF_ARG(float32_t) bestLenSq, NBL_REF_ARG(float32_t2) bestEdge2d, NBL_REF_ARG(uint32_t) bestEdge)
   {
      const uint32_t j      = CheckCount ? ((I + 1 < count) ? I + 1 : 0) : I + 1;
      float32_t2     edge2d = verts2d[j] - verts2d[I];
      float32_t      lenSq  = dot(edge2d, edge2d);
      // Sticky 1% threshold (in lenSq, ~0.5% in length) prevents axis1 from flipping
      // between two near-equal-length edges as the silhouette deforms.
      if (lenSq > bestLenSq * (1.0f + 1e-2f))
      {
         bestLenSq  = lenSq;
         bestEdge2d = edge2d;
         bestEdge   = I;
      }
   }

   // ========================================================================
   // Factory
   // ========================================================================

   static SphericalPyramid create(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, NBL_REF_ARG(SilEdgeNormals) silEdgeNormals)
   {
      SphericalPyramid self;
      silEdgeNormals = (SilEdgeNormals)0;

      // Pass 1: build unnormCentroid (true spherical centroid) and edgeNormals.
      // Seed with a tiny scaled vertex centroid so symmetric / near-cancelling
      // shapes don't degenerate to a zero direction on `normalize`.
      float32_t3 unnormCentroid = silhouette.getUnnormalizedCenter() * 1e-6f;

      // Count-cascade: silhouette.vertices[I] for I >= count is uninitialized in some
      // call sites (e.g. solid_angle_vis.frag.hlsl declares ClippedSilhouette without
      // zero-init), so we must NOT read past count. I=2 needs the wrap check because
      // count can be exactly 3 (j must wrap to 0).
      accumulateEdge<0>(silhouette, unnormCentroid, silEdgeNormals);
      accumulateEdge<1>(silhouette, unnormCentroid, silEdgeNormals);
      accumulateEdge<2, true>(silhouette, unnormCentroid, silEdgeNormals);
      if (silhouette.count > 3)
      {
         accumulateEdge<3, true>(silhouette, unnormCentroid, silEdgeNormals);
         if (silhouette.count > 4)
         {
            accumulateEdge<4, true>(silhouette, unnormCentroid, silEdgeNormals);
            if (silhouette.count > 5)
            {
               accumulateEdge<5, true>(silhouette, unnormCentroid, silEdgeNormals);
               if (silhouette.count > 6)
                  accumulateEdge<6, true>(silhouette, unnormCentroid, silEdgeNormals);
            }
         }
      }

      const float32_t3 axis3 = normalize(-unnormCentroid);

      // Pass 2: Frisvad basis + 2D gnomonic projection (one-time, before calipers).
      float32_t3 u, v;
      nbl::hlsl::math::frisvad<float32_t3>(axis3, u, v);

      // Project only the first `count` vertices; entries past `count` are unread by
      // try2DCaliper since its cascade is also count-gated.
      float32_t2 verts2d[MAX_SILHOUETTE_VERTICES];
      verts2d[0] = projectVertex2D<0>(silhouette, u, v, axis3);
      verts2d[1] = projectVertex2D<1>(silhouette, u, v, axis3);
      verts2d[2] = projectVertex2D<2>(silhouette, u, v, axis3);
      if (silhouette.count > 3)
      {
         verts2d[3] = projectVertex2D<3>(silhouette, u, v, axis3);
         if (silhouette.count > 4)
         {
            verts2d[4] = projectVertex2D<4>(silhouette, u, v, axis3);
            if (silhouette.count > 5)
            {
               verts2d[5] = projectVertex2D<5>(silhouette, u, v, axis3);
               if (silhouette.count > 6)
                  verts2d[6] = projectVertex2D<6>(silhouette, u, v, axis3);
            }
         }
      }

      // Pass 3: pick longest 2D edge as axis1 ("guesstimate" rotating calipers).
      // O(N) edge-length comparisons, then ONE bound pass after the winner is known.
      float32_t  bestLenSq  = 0.0f;
      float32_t2 bestEdge2d = float32_t2(1.0f, 0.0f);
      uint32_t   bestEdge   = 0;

      considerEdge<0>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
      considerEdge<1>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
      considerEdge<2, true>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
      if (silhouette.count > 3)
      {
         considerEdge<3, true>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
         if (silhouette.count > 4)
         {
            considerEdge<4, true>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
            if (silhouette.count > 5)
            {
               considerEdge<5, true>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
               if (silhouette.count > 6)
                  considerEdge<6, true>(verts2d, silhouette.count, bestLenSq, bestEdge2d, bestEdge);
            }
         }
      }

      // Single bound pass with the winning edge as axis1. Fall back to (1,0) if
      // every edge degenerated (silhouette projects to a single point).
      const float32_t2 bestAxis2d = bestLenSq > 1e-12f ? bestEdge2d * rsqrt(bestLenSq) : float32_t2(1.0f, 0.0f);
      const float32_t2 bestPerp2d = float32_t2(-bestAxis2d.y, bestAxis2d.x);
      float32_t4       bestBound;
      computeBound2D(verts2d, silhouette.count, bestAxis2d, bestPerp2d, bestBound);

      // Pass 4: reconstruct 3D, sign-stabilize axis1 against a world reference.
      // For right-handed (u, v, axis3) Frisvad basis, cross(axis3, u) = v and cross(axis3, v) = -u,
      // so axis1 = u*a + v*b => axis2 = cross(axis3, axis1) = v*a - u*b. Skip the 3D `cross`.
      const float32_t3 axis1Raw = u * bestAxis2d.x + v * bestAxis2d.y;
      const float32_t3 axis2Raw = v * bestAxis2d.x - u * bestAxis2d.y;
      {
         // Sign-stabilize axis1 against a world reference, branchless.
         // axis1 is already perpendicular to axis3, so dot(axis1, worldRef - axis3*dot(worldRef,axis3))
         // == dot(axis1, worldRef). Flipping axis1 also flips axis2 (both negate together since
         // axis2 = cross(axis3, axis1)); mirror both x and y bounds simultaneously.
         const float32_t3 worldRef = nbl::hlsl::select(abs(axis3.x) < 0.9f, float32_t3(1.0f, 0.0f, 0.0f), float32_t3(0.0f, 1.0f, 0.0f));
         const bool       flip     = dot(axis1Raw, worldRef) < 0.0f;
         self.axis1                = nbl::hlsl::select(flip, -axis1Raw, axis1Raw);
         self.axis2                = nbl::hlsl::select(flip, -axis2Raw, axis2Raw);
         bestBound                 = nbl::hlsl::select(flip, float32_t4(-bestBound.z, -bestBound.w, -bestBound.x, -bestBound.y), bestBound);
      }

      // Degenerate bounds fallback (branchless).
      const bool degenerateBounds = bestBound.x >= bestBound.z || bestBound.y >= bestBound.w;
      bestBound                   = nbl::hlsl::select(degenerateBounds, float32_t4(-0.1f, -0.1f, 0.1f, 0.1f), bestBound);

      self.rectR0      = bestBound.xy;
      self.rectExtents = float32_t2(bestBound.zw - bestBound.xy);
      
      VisContext::add(SphereDrawer::drawDot(normalize(-unnormCentroid), 0.05f, 0.0f, float32_t3(1.0f, 0.0f, 1.0f)));
      VisContext::add(SphereDrawer::visualizeBestCaliperEdge(silhouette, bestEdge));
      self.visualize();
      
      // DCE
      nbl::hlsl::sampling::SphericalRectangle<float32_t> rectSampler = nbl::hlsl::sampling::SphericalRectangle<float32_t>::create(float32_t3x3(self.axis1, self.axis2, axis3), float32_t3(self.rectR0, 1.0f), self.rectExtents);
      DebugRecorder::recordPyramid(self.axis1, self.axis2, -unnormCentroid, bestBound, rectSampler.solidAngle, bestEdge);

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

      float32_t3      a3 = getAxis3();
      float32_t       x0 = rectR0.x;
      float32_t       x1 = rectR0.x + rectExtents.x;
      float32_t       y0 = rectR0.y;
      float32_t       y1 = rectR0.y + rectExtents.y;
      const float32_t z  = 1.0f;

      // Great circle normals for the 4 edges (in local frame, then transform to world)
      float32_t3 bottomNormalLocal = normalize(float32_t3(0, -z, y0));
      float32_t3 topNormalLocal    = normalize(float32_t3(0, z, -y1));
      float32_t3 leftNormalLocal   = normalize(float32_t3(-z, 0, x0));
      float32_t3 rightNormalLocal  = normalize(float32_t3(z, 0, -x1));

      // Transform to world space
      float32_t3 bottomNormal = bottomNormalLocal.x * axis1 + bottomNormalLocal.y * axis2 + bottomNormalLocal.z * a3;
      float32_t3 topNormal    = topNormalLocal.x * axis1 + topNormalLocal.y * axis2 + topNormalLocal.z * a3;
      float32_t3 leftNormal   = leftNormalLocal.x * axis1 + leftNormalLocal.y * axis2 + leftNormalLocal.z * a3;
      float32_t3 rightNormal  = rightNormalLocal.x * axis1 + rightNormalLocal.y * axis2 + rightNormalLocal.z * a3;

      // Draw center point (center of the rectangle projected onto sphere)
      float32_t  centerX     = (x0 + x1) * 0.5f;
      float32_t  centerY     = (y0 + y1) * 0.5f;
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
