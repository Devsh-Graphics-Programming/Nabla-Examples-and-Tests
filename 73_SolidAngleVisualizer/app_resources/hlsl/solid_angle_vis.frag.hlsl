//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

#include "drawing.hlsl"
#include "utils.hlsl"
#include "silhouette.hlsl"
#include "triangle_sampling.hlsl"
#include "parallelogram_sampling.hlsl"
#include "pyramid_sampling.hlsl"

[[vk::push_constant]] struct PushConstants pc;

static const SAMPLING_MODE_FLAGS samplingMode = SAMPLING_MODE_FLAGS_CONST;

// Mode -> Sampler type dispatch keyed on the dense ID (boost::wave can't
// evaluate enum-qualified `::` in #if, so we use a parallel numeric macro
// passed in by CMake). Dense IDs match the kAllModes ordering in common.hlsl:
//   0 SPH_RECT_FROM_CALIPER_PYRAMID    5 PROJECTED_PARALLELOGRAM_SOLID_ANGLE
//   1 SPH_RECT_FROM_PYRAMID            6 BILINEAR_FROM_PYRAMID
//   2 PROJ_SPH_RECT_FROM_PYRAMID       7 SILHOUETTE_CREATION_ONLY (early-exit)
//   3 TRIANGLE_SOLID_ANGLE             8 PYRAMID_CREATION_ONLY
//   4 TRIANGLE_PROJECTED_SOLID_ANGLE   9 CALIPER_PYRAMID_CREATION_ONLY
// PYRAMID_CREATION_ONLY / CALIPER_PYRAMID_CREATION_ONLY pick the sphrect inner
// so the bounding lunes still draw on screen; the inner is harmless extra work
// (those modes are timed in the compute benchmark, not the frag).
#if SAMPLING_MODE_DENSE_ID == 3
typedef TriangleFanSampler<false> SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 4
typedef TriangleFanSampler<true> SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 5
typedef Parallelogram SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 1 || SAMPLING_MODE_DENSE_ID == 8
typedef SphericalPyramid<false, sampling::SphericalRectangle<float32_t> > SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 0 || SAMPLING_MODE_DENSE_ID == 9
typedef SphericalPyramid<true, sampling::SphericalRectangle<float32_t> > SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 2
typedef SphericalPyramid<false, sampling::ProjectedSphericalRectangle<float32_t> > SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 6
typedef SphericalPyramid<false, BilinearSampler> SelectedSampler;
#elif SAMPLING_MODE_DENSE_ID == 7  // SILHOUETTE_CREATION_ONLY: alias any type so the
typedef Parallelogram SelectedSampler;  // unreachable post-early-return code parses.
#endif

void computeSpherePos(SVertexAttributes vx, out float32_t2 ndc, out float32_t3 spherePos)
{
   ndc              = vx.uv * 2.0f - 1.0f;
   float32_t aspect = pc.viewport.z / pc.viewport.w;
   ndc.x *= aspect;

   float32_t2 normalized = ndc / CIRCLE_RADIUS;
   float32_t  r2         = dot(normalized, normalized);

   if (r2 <= 1.0f)
   {
      spherePos = float32_t3(normalized.x, normalized.y, sqrt(1.0f - r2));
   }
   else
   {
      float32_t uv2Plus1 = r2 + 1.0f;
      spherePos          = float32_t3(normalized.x * 2.0f, normalized.y * 2.0f, 1.0f - r2) / uv2Plus1;
   }
   spherePos = normalize(spherePos);
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
   float32_t  aaWidth = length(float32_t2(ddx(vx.uv.x), ddy(vx.uv.y)));
   float32_t3 spherePos;
   float32_t2 ndc;
   computeSpherePos(vx, ndc, spherePos);
   VisContext::begin(ndc, spherePos, aaWidth);

   shapes::OBBView<float32_t> view       = shapes::OBBView<float32_t>::create(pc.modelMatrix);
   ClippedSilhouette          silhouette = ClippedSilhouette::create(view);

   if (SAMPLING_MODE_DENSE_ID == 7) // SILHOUETTE_CREATION_ONLY
   {
      // Sink that prevents DCE of the create+materialize cost.
      shapes::OBBView<float32_t> perturbedView = view;
      perturbedView.minCorner += float32_t3(ndc.x, ndc.y, 0.0f) * 1e-7f;
      ClippedSilhouette pSilhouette = ClippedSilhouette::create(perturbedView);
      float32_t3 pVerts[MAX_SILHOUETTE_VERTICES];
      pSilhouette.materialize(perturbedView, pVerts);

      uint32_t sink = pSilhouette.count;
      NBL_UNROLL
      for (uint32_t i = 0; i < MAX_SILHOUETTE_VERTICES; i++)
         sink ^= asuint(pVerts[i].x) ^ asuint(pVerts[i].y) ^ asuint(pVerts[i].z);
      return (float32_t4)asfloat(sink);
   }

   SelectedSampler sampler = SelectedSampler::create(silhouette, view);

   uint32_t validSampleCount = 0;
   for (uint32_t i = 0; i < pc.sampleCount; i++)
   {
      float32_t2 xi = float32_t2(
         (float32_t(i & 7u) + 0.5) / sqrt(pc.sampleCount) + ndc.x * 1e-9f,
         (float32_t(i >> 3u) + 0.5) / sqrt(pc.sampleCount) + ndc.y * 1e-9f);

      typename SelectedSampler::cache_type cache;
      const float32_t3                     sampleDir = sampler.generate(xi, cache);
      const float32_t                      pdf       = sampler.forwardPdf(xi, cache);

      if (pdf > 0.0f)
      {
         validSampleCount++;
         DebugRecorder::recordRay(i, sampleDir, pdf);
         if (VisContext::enabled())
            VisContext::add(SphereDrawer::visualizeSample(sampleDir, xi, sampler.selectedIdx(cache), vx.uv));
         else
            VisContext::add(float4(sampleDir * 0.02f / pdf, 1.0f));
      }
   }

   // Silhouette edges + debug recording. Re-materialize verts here -- the
   // sampler may have absorbed its own copy already, but `verts` is local to
   // this scope and dies at function end anyway.
   {
      float32_t3 vertices[MAX_SILHOUETTE_VERTICES];
      silhouette.materialize(view, vertices);
      silhouette.recordVertices(vertices);

      for (uint32_t i = 0; i < silhouette.count; i++)
      {
         const uint32_t   j       = (i + 1u < silhouette.count) ? i + 1u : 0u;
         const float32_t3 e0      = normalize(vertices[i]);
         const float32_t3 e1      = normalize(vertices[j]);
         const float32_t3 ePts[2] = {e0, e1};
         VisContext::add(SphereDrawer::drawEdge(0, ePts, aaWidth));
      }

      const uint32_t configIndex = silhouette.getConfigIndex();
      if (VisContext::enabled() && all(vx.uv >= float32_t2(0.f, 0.97f)) && all(vx.uv <= float32_t2(0.03f, 1.0f)))
         return float32_t4(colorLUT[configIndex], 1.0f);
      VisContext::add(SphereDrawer::drawRing(ndc));

      const BinSilhouette binSil = silhouette.getOriginalBinSilhouette();
      uint32_t            vertexIndices[6];
      for (uint32_t i = 0; i < 6; i++)
         vertexIndices[i] = uint32_t(binSil.getVertexIndex(i));
      DebugRecorder::recordFrameEnd(silhouette.getRegion(), configIndex, binSil.getVertexCount(), binSil.data, vertexIndices, validSampleCount, pc.sampleCount);
   }
   return VisContext::flush();
}
