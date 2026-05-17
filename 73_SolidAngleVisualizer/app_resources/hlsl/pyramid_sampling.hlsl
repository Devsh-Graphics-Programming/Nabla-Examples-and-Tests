//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_

// Thin shim over the builtin SphericalPyramid. The builtin (in
// nbl/builtin/hlsl/sampling/spherical_pyramid.hlsl) is the source of truth;
// this file re-exports it at example-global scope, adds a buildInner overload
// for the example-local BilinearSampler, and adds a templated debug+visualize
// helper that re-derives the intermediates the builtin's debug-free
// createFromVertices() doesn't expose.
#include "common.hlsl"

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_pyramid.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>

#include "silhouette.hlsl"
#include "drawing.hlsl"
#include "pyramid_sampling/bilinear.hlsl"

// buildInner overload for the example-local BilinearSampler. Lives at global
// namespace so unqualified lookup from SphericalPyramid<_, BilinearSampler>::create
// (which the builtin defines in nbl::hlsl::sampling) finds it at instantiation.
inline BilinearSampler buildInner(float32_t3x3 basis, float32_t2 r0, float32_t2 ext, BilinearSampler /*tag*/)
{
   return BilinearSampler::create(basis, r0, ext);
}

// Re-export at example-global scope so existing SphericalPyramid<...> spellings
// in frag/benchmark/SelectSampler keep compiling without qualification.
template<bool UseCaliper, typename InnerSampler>
using SphericalPyramid = nbl::hlsl::sampling::SphericalPyramid<UseCaliper, InnerSampler>;

// PyramidDebugVis<SamplerT> is a no-op for non-pyramid samplers. The pyramid
// specialization re-materializes silhouette verts, recovers (rectR0, rectExtents)
// by re-running computeBound3D against the pyramid's frame, finds the chosen
// edge from the local-frame silEdgeNormals (matches the old findChosenEdge
// heuristic), records DebugRecorder::recordPyramid, and emits the bounding
// great-circle + axes overlay.
template<typename SamplerT>
struct PyramidDebugVis
{
   static void apply(SamplerT /*sampler*/, ClippedSilhouette /*silhouette*/, shapes::OBBView<float32_t> /*view*/) {}
};

template<bool UseCaliper, typename InnerSampler>
struct PyramidDebugVis<SphericalPyramid<UseCaliper, InnerSampler> >
{
   using PyramidT = SphericalPyramid<UseCaliper, InnerSampler>;

   // Cheap "which edge is most parallel to axis1" heuristic the original
   // visualize() used: smallest |edgeNormals[i].x| in the local frame.
   // silEdgeNormals are local-frame after createFromVertices transformToLocal.
   static uint32_t findChosenEdgeLocal(PyramidT pyramid, uint32_t count)
   {
      uint32_t  bestI   = 0;
      float32_t bestAbs = abs(pyramid.silEdgeNormals.edgeNormals[0].x);
      for (uint32_t i = 0; i < count; i++)
      {
         const float32_t v      = abs(pyramid.silEdgeNormals.edgeNormals[i].x);
         const bool      better = v < bestAbs;
         bestAbs                = nbl::hlsl::select(better, v, bestAbs);
         bestI                  = nbl::hlsl::select(better, i, bestI);
      }
      return bestI;
   }

   static void apply(PyramidT pyramid, ClippedSilhouette silhouette, shapes::OBBView<float32_t> view)
   {
      if (silhouette.count == 0)
         return;

      float32_t3 vertices[MAX_SILHOUETTE_VERTICES];
      silhouette.materialize(view, vertices);

      const float32_t3 axis3 = pyramid.getAxis3();

      // Recover (rectR0, rectExtents) from the pyramid frame.
      float32_t4 bestBound;
      PyramidT::computeBound3D(vertices, silhouette.count, pyramid.axis1, pyramid.axis2, axis3, bestBound);
      bestBound.zw = max(bestBound.zw, bestBound.xy + 1e-6f);
      const float32_t2 rectR0      = bestBound.xy;
      const float32_t2 rectExtents = float32_t2(bestBound.zw - bestBound.xy);

      // 4-edge spherical rectangle solid angle from bounds, for the debug overlay.
      const float32_t4 denorm_n_z             = float32_t4(-bestBound.y, bestBound.z, bestBound.w, -bestBound.x);
      const float32_t4 n_z                    = denorm_n_z * rsqrt(float32_t4(1.0f, 1.0f, 1.0f, 1.0f) + denorm_n_z * denorm_n_z);
      const float32_t4 cosGamma               = float32_t4(-n_z[0] * n_z[1], -n_z[1] * n_z[2], -n_z[2] * n_z[3], -n_z[3] * n_z[0]);
      math::sincos_accumulator<float32_t> acc = math::sincos_accumulator<float32_t>::create(cosGamma[0]);
      acc.addCosine(cosGamma[1]);
      acc.addCosine(cosGamma[2]);
      acc.addCosine(cosGamma[3]);
      const float32_t solidAngle = acc.getSumOfArccos() - 2.0f * numbers::pi<float32_t>;

      // bestEdge identification is post-hoc and approximate (the builtin
      // create() doesn't track it). The visualize() overlay's orange highlight
      // uses the local-frame |n.x| heuristic that's a reasonable proxy.
      const uint32_t bestEdge = findChosenEdgeLocal(pyramid, silhouette.count);

      // Approximate centroid sign for the debug recorder. The original tracked
      // -unnormCentroid during processEdge; -axis3 captures its direction.
      DebugRecorder::recordPyramid(pyramid.axis1, pyramid.axis2, -axis3, bestBound, solidAngle, bestEdge);

      // Bounding great circles + axis dots overlay.
      const float32_t  x0          = rectR0.x;
      const float32_t  x1          = rectR0.x + rectExtents.x;
      const float32_t  y0          = rectR0.y;
      const float32_t  y1          = rectR0.y + rectExtents.y;
      const float32_t  z           = 1.0f;
      const float32_t3 boundColor1 = float32_t3(1.0f, 0.5f, 0.5f);
      const float32_t3 boundColor2 = float32_t3(0.5f, 0.5f, 1.0f);
      const float32_t3 centerColor = float32_t3(1.0f, 1.0f, 0.0f);

      const float32_t3 bottomNormalLocal = normalize(float32_t3(0, -z, y0));
      const float32_t3 topNormalLocal    = normalize(float32_t3(0, z, -y1));
      const float32_t3 leftNormalLocal   = normalize(float32_t3(-z, 0, x0));
      const float32_t3 rightNormalLocal  = normalize(float32_t3(z, 0, -x1));

      const float32_t3 bottomNormal = bottomNormalLocal.x * pyramid.axis1 + bottomNormalLocal.y * pyramid.axis2 + bottomNormalLocal.z * axis3;
      const float32_t3 topNormal    = topNormalLocal.x * pyramid.axis1 + topNormalLocal.y * pyramid.axis2 + topNormalLocal.z * axis3;
      const float32_t3 leftNormal   = leftNormalLocal.x * pyramid.axis1 + leftNormalLocal.y * pyramid.axis2 + leftNormalLocal.z * axis3;
      const float32_t3 rightNormal  = rightNormalLocal.x * pyramid.axis1 + rightNormalLocal.y * pyramid.axis2 + rightNormalLocal.z * axis3;

      const float32_t  centerX     = (x0 + x1) * 0.5f;
      const float32_t  centerY     = (y0 + y1) * 0.5f;
      const float32_t3 centerLocal = normalize(float32_t3(centerX, centerY, z));
      const float32_t3 centerWorld = centerLocal.x * pyramid.axis1 + centerLocal.y * pyramid.axis2 + centerLocal.z * axis3;

      VisContext::add(SphereDrawer::drawCorner(centerWorld, 0.025f, 0.0f, centerColor));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(bottomNormal, axis3, boundColor2, 0.004f));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(topNormal, axis3, boundColor2, 0.004f));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(leftNormal, axis3, boundColor1, 0.004f));
      VisContext::add(SphereDrawer::drawGreatCircleHalf(rightNormal, axis3, boundColor1, 0.004f));

      const uint32_t   bestJ     = (bestEdge + 1u) % silhouette.count;
      float32_t3       chosen[2] = {vertices[bestEdge], vertices[bestJ]};
      VisContext::add(SphereDrawer::drawEdge(8u, chosen, 0.012f)); // colorLUT[8] = orange

      VisContext::add(SphereDrawer::drawDot(pyramid.axis1, 0.025f, 0.0f, float32_t3(1.0f, 0.0f, 0.0f)));
      VisContext::add(SphereDrawer::drawDot(pyramid.axis2, 0.025f, 0.0f, float32_t3(0.0f, 1.0f, 0.0f)));
      VisContext::add(SphereDrawer::drawDot(axis3, 0.025f, 0.0f, float32_t3(0.0f, 0.0f, 1.0f)));
   }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_PYRAMID_SAMPLING_HLSL_INCLUDED_
