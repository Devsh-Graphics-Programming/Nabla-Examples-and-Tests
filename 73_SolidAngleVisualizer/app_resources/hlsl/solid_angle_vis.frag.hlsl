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
#include "pyramid_sampling.hlsl"
#include "parallelogram_sampling.hlsl"

[[vk::push_constant]] struct PushConstants pc;

static const SAMPLING_MODE samplingMode = (SAMPLING_MODE)SAMPLING_MODE_CONST;

void computeSpherePos(SVertexAttributes vx, out float32_t2 ndc, out float32_t3 spherePos)
{
   ndc = vx.uv * 2.0f - 1.0f;
   float32_t aspect = pc.viewport.z / pc.viewport.w;
   ndc.x *= aspect;

   float32_t2 normalized = ndc / CIRCLE_RADIUS;
   float32_t r2 = dot(normalized, normalized);

   if (r2 <= 1.0f)
   {
      spherePos = float32_t3(normalized.x, normalized.y, sqrt(1.0f - r2));
   }
   else
   {
      float32_t uv2Plus1 = r2 + 1.0f;
      spherePos = float32_t3(normalized.x * 2.0f, normalized.y * 2.0f, 1.0f - r2) / uv2Plus1;
   }
   spherePos = normalize(spherePos);
}

// Sample a direction from a pyramid-based rectangle sampler, returning validity
template<typename Sampler>
float32_t3 sampleFromPyramid(inout Sampler sampler, SphericalPyramid pyramid, SilEdgeNormals silEdgeNormals, float32_t2 xi, out float32_t pdf, out bool valid)
{
   typename Sampler::cache_type cache;
   float32_t hitDist;
   float32_t3 localDir = sampler.generateNormalizedLocal(xi, cache, hitDist);
   float32_t3 dir = localDir.x * pyramid.axis1 + localDir.y * pyramid.axis2 + localDir.z * pyramid.getAxis3();
   float32_t localX = localDir.x / localDir.z;
   float32_t localY = localDir.y / localDir.z;
   valid = dir.z > 0.0f && silEdgeNormals.isInsideLocal(localX, localY);
   pdf = sampler.forwardPdf(xi, cache);
   return dir;
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
   float32_t aaWidth = length(float32_t2(ddx(vx.uv.x), ddy(vx.uv.y)));
   float32_t3 spherePos;
   float32_t2 ndc;
   computeSpherePos(vx, ndc, spherePos);
   VisContext::begin(ndc, spherePos, aaWidth);

   shapes::OBBView<float32_t> view = shapes::OBBView<float32_t>::create(pc.modelMatrix);
   uint32_t3 region;
   uint32_t configIndex;
   uint32_t vertexCount;
   BinSilhouette sil = ClippedSilhouette::computeRegionAndConfig(view, region, configIndex, vertexCount);

   ClippedSilhouette silhouette;
   silhouette.compute(view, vertexCount, sil);

   if (samplingMode == SAMPLING_MODE::SILHOUETTE_CREATION_ONLY)
   {
      shapes::OBBView<float32_t> perturbedView = view;
      perturbedView.minCorner += float32_t3(ndc.x, ndc.y, 0.0f) * 1e-7f;
      ClippedSilhouette pSilhouette = ClippedSilhouette::create(perturbedView);

      uint32_t sink = pSilhouette.count;
      NBL_UNROLL
      for (uint32_t i = 0; i < MAX_SILHOUETTE_VERTICES; i++)
         sink ^= asuint(pSilhouette.vertices[i].x) ^ asuint(pSilhouette.vertices[i].y) ^ asuint(pSilhouette.vertices[i].z);
      return (float32_t4)asfloat(sink);
   }

   // Draw silhouette edges on the sphere
   for (uint32_t ei = 0; ei < silhouette.count; ei++)
   {
      float32_t3 v0 = normalize(silhouette.vertices[ei]);
      float32_t3 v1 = normalize(silhouette.vertices[(ei + 1) % silhouette.count]);
      float32_t3 pts[2] = {v0, v1};
      VisContext::add(SphereDrawer::drawEdge(0, pts, aaWidth));
   }

   // =====================================================================
   // Build sampler
   // =====================================================================
   TriangleFanSampler samplingData;
   Parallelogram parallelogram;
   SphericalPyramid pyramid;
   sampling::SphericalRectangle<float32_t> rectSampler;
   sampling::ProjectedSphericalRectangle<float32_t> projRectSampler;
   BiquadraticSampler biquad;
   BilinearSampler bilin;
   SilEdgeNormals silEdgeNormals;

   if (samplingMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE ||
      samplingMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
   {
      samplingData = TriangleFanSampler::create(silhouette, samplingMode);
   }
   else if (samplingMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
   {
      silhouette.normalize();
      parallelogram = Parallelogram::create(silhouette, silEdgeNormals);
   }
   else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE ||
      samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC ||
      samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR ||
      samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_PROJECTED_SOLID_ANGLE_RECTANGLE ||
      samplingMode == SAMPLING_MODE::PYRAMID_CREATION_ONLY)
   {
      pyramid = SphericalPyramid::create(silhouette, silEdgeNormals);
      silEdgeNormals.transformToLocal(pyramid.axis1, pyramid.axis2, pyramid.getAxis3());

      if (samplingMode == SAMPLING_MODE::PYRAMID_CREATION_ONLY)
      {
         uint32_t sink = 0;
         for (uint32_t j = 0; j < pc.sampleCount; j++)
         {
            ClippedSilhouette pertSil = silhouette;
            float32_t pertScale = (float32_t(j) + ndc.x + ndc.y) * 0.001f;
            NBL_UNROLL
            for (uint32_t i = 0; i < MAX_SILHOUETTE_VERTICES; i++)
               pertSil.vertices[i] = normalize(pertSil.vertices[i] + float32_t3(pertScale * float32_t(i + 1), pertScale * 0.7f, 0.0f));

            SilEdgeNormals pertEdgeNormals;
            SphericalPyramid pertPyramid = SphericalPyramid::create(pertSil, pertEdgeNormals);
            sink ^= asuint(pertPyramid.axis1.x) ^ asuint(pertPyramid.axis2.x) ^ asuint(pertPyramid.rectR0.x) ^ asuint(pertPyramid.rectExtents.x) ^ asuint(float32_t(pertEdgeNormals.edgeNormals[0].x));
         }
         return (float32_t4)asfloat(sink);
      }

      if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE)
         rectSampler = sampling::SphericalRectangle<float32_t>::create(float32_t3x3(pyramid.axis1, pyramid.axis2, pyramid.getAxis3()), float32_t3(pyramid.rectR0, 1.0f), pyramid.rectExtents);
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_PROJECTED_SOLID_ANGLE_RECTANGLE)
      {
         shapes::CompressedSphericalRectangle<float32_t> compressed;
         compressed.origin = pyramid.axis1 * pyramid.rectR0.x + pyramid.axis2 * pyramid.rectR0.y + pyramid.getAxis3();
         compressed.right  = pyramid.axis1 * pyramid.rectExtents.x;
         compressed.up     = pyramid.axis2 * pyramid.rectExtents.y;
         projRectSampler = sampling::ProjectedSphericalRectangle<float32_t>::create(compressed, float32_t3(0.0f, 0.0f, 0.0f), float32_t3(0.0f, 0.0f, 1.0f), false);
      }
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC)
         biquad = BiquadraticSampler::create(pyramid);
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
         bilin = BilinearSampler::create(pyramid);
   }

   // =====================================================================
   // Sample loop
   // =====================================================================
   uint32_t validSampleCount = 0;
   DebugRecorder::recordSampleCount(pc.sampleCount);

   for (uint32_t i = 0; i < pc.sampleCount; i++)
   {
      float32_t2 xi = float32_t2(
         (float32_t(i & 7u) + 0.5) / sqrt(pc.sampleCount) + ndc.x * 1e-9f,
         (float32_t(i >> 3u) + 0.5) / sqrt(pc.sampleCount) + ndc.y * 1e-9f);

      float32_t pdf;
      uint32_t index = 0;
      float32_t3 sampleDir;
      bool valid;

      if (samplingMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE || samplingMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
         sampleDir = samplingData.sample(silhouette, xi, pdf, index);
      else if (samplingMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
         sampleDir = parallelogram.sample(silEdgeNormals, xi, pdf, valid);
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE)
         sampleDir = sampleFromPyramid(rectSampler, pyramid, silEdgeNormals, xi, pdf, valid);
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_PROJECTED_SOLID_ANGLE_RECTANGLE)
         sampleDir = sampleFromPyramid(projRectSampler, pyramid, silEdgeNormals, xi, pdf, valid);
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC)
         sampleDir = biquad.sample(pyramid, silEdgeNormals, xi, pdf, valid);
      else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
         sampleDir = bilin.sample(pyramid, silEdgeNormals, xi, pdf, valid);

      if (!valid)
         pdf = 0.0f;
      else
         validSampleCount++;

      DebugRecorder::recordRay(i, sampleDir, pdf);

      if (VisContext::enabled())
         VisContext::add(SphereDrawer::visualizeSample(sampleDir, xi, index, vx.uv));
      else if (pdf > 0.0f)
         VisContext::add(float4(sampleDir * 0.02f / pdf, 1.0f));
   }

   VisContext::add(SphereDrawer::drawRing(ndc));

   if (VisContext::enabled() && all(vx.uv >= float32_t2(0.f, 0.97f)) && all(vx.uv <= float32_t2(0.03f, 1.0f)))
      return float32_t4(colorLUT[configIndex], 1.0f);

   uint32_t vertexIndices[6];
   for (uint32_t i = 0; i < 6; i++)
      vertexIndices[i] = uint32_t(sil.getVertexIndex(i));
   DebugRecorder::recordFrameEnd(region, configIndex, sil.getSilhouetteSize(), sil.data, vertexIndices, validSampleCount);

   return VisContext::flush();
}
