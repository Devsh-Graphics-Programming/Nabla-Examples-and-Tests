//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/hlsl/common.hlsl"
#include "app_resources/hlsl/benchmark/common.hlsl"
#include "app_resources/hlsl/silhouette.hlsl"
#include "app_resources/hlsl/parallelogram_sampling.hlsl"
#include "app_resources/hlsl/pyramid_sampling.hlsl"
#include "app_resources/hlsl/triangle_sampling.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

static const SAMPLING_MODE benchmarkMode = (SAMPLING_MODE)SAMPLING_MODE_CONST;

float32_t2 stratifiedXi(uint32_t sampleIdx, uint32_t threadIdx)
{
   return float32_t2(
      (float32_t(sampleIdx & 7u) + 0.5f) / 8.0f + float32_t(threadIdx) * 1e-9f,
      (float32_t(sampleIdx >> 3u) + 0.5f) / 8.0f + float32_t(threadIdx) * 1e-9f);
}

struct PyramidSetup
{
   SphericalPyramid pyramid;
   SilEdgeNormals silEdgeNormals;

   static PyramidSetup create(ClippedSilhouette silhouette)
   {
      PyramidSetup s;
      s.pyramid = SphericalPyramid::create(silhouette, s.silEdgeNormals);
      s.silEdgeNormals.transformToLocal(s.pyramid.axis1, s.pyramid.axis2, s.pyramid.getAxis3());
      return s;
   }
};

// Per-thread input perturbation: scatters threads across the 27 OBB regions and
// generates a fresh silhouette per outer-loop iteration so creation work can't
// be hoisted out by the compiler.
ClippedSilhouette makePerturbedSilhouette(float32_t3 baseOffset, NBL_REF_ARG(random::PCG32) rng, float32_t rcpU32)
{
   const float32_t3 cJ = float32_t3(
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f,
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f,
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f);
   float32_t3x4 cM = pc.modelMatrix;
   cM[0][3] += baseOffset.x + cJ.x;
   cM[1][3] += baseOffset.y + cJ.y;
   cM[2][3] += baseOffset.z + cJ.z;
   shapes::OBBView<float32_t> cV = shapes::OBBView<float32_t>::create(cM);
   return ClippedSilhouette::create(cV);
}

[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)]
void main()
{
	const uint32_t invocationID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;

   // Scatter the OBB translation per invocation so threads span all 27 regions
   random::PCG32 rng = random::PCG32::construct(invocationID.x + 0x9e3779b9u);
   const float32_t rcpU32 = 1.0f / 4294967296.0f;
   const float32_t3 rndOffset = float32_t3(
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f,
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f,
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f);

   // XOR sink: every output XORs into this to prevent DCE.
   uint32_t sink = 0;

   bool sampleValid;

   // Sampling modes use a nested loop: outer iterates over `creations`, inner over
   // `samplesPerCreation`. Total samples per thread = sampleCount.
   const uint32_t creations = pc.sampleCount / pc.samplesPerCreation;

   if (benchmarkMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE ||
      benchmarkMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
   {
      for (uint32_t c = 0; c < creations; c++)
      {
         ClippedSilhouette silhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);
         TriangleFanSampler samplingData = TriangleFanSampler::create(silhouette, benchmarkMode);

         for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
         {
            float32_t2 xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
            float32_t pdf;
            uint32_t triIdx;
            float32_t3 dir = samplingData.sample(silhouette, xi, pdf, triIdx);
            sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ triIdx;
         }
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
   {
      for (uint32_t c = 0; c < creations; c++)
      {
         ClippedSilhouette silhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);
         silhouette.normalize();
         SilEdgeNormals silEdgeNormals;
         Parallelogram parallelogram = Parallelogram::create(silhouette, silEdgeNormals);

         for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
         {
            float32_t2 xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
            float32_t pdf;
            float32_t3 dir = parallelogram.sample(silEdgeNormals, xi, pdf, sampleValid);
            sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ (uint32_t)sampleValid;
         }
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE)
   {
      for (uint32_t c = 0; c < creations; c++)
      {
         ClippedSilhouette silhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);
         PyramidSetup ps = PyramidSetup::create(silhouette);
         sampling::SphericalRectangle<float32_t> rectSampler = sampling::SphericalRectangle<float32_t>::create(float32_t3x3(ps.pyramid.axis1, ps.pyramid.axis2, ps.pyramid.getAxis3()), float32_t3(ps.pyramid.rectR0, 1.0f), ps.pyramid.rectExtents);

         for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
         {
            float32_t2 xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
            sampling::SphericalRectangle<float32_t>::cache_type cache;
            float32_t hitDist;
            float32_t3 localDir = rectSampler.generateNormalizedLocal(xi, cache, hitDist);
            float32_t3 dir = localDir.x * ps.pyramid.axis1 + localDir.y * ps.pyramid.axis2 + localDir.z * ps.pyramid.getAxis3();
            float32_t localX = localDir.x / localDir.z;
            float32_t localY = localDir.y / localDir.z;
            sampleValid = dir.z > 0.0f && ps.silEdgeNormals.isInsideLocal(localX, localY);
            float32_t pdf = rectSampler.forwardPdf(xi, cache);
            sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ (uint32_t)sampleValid;
         }
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_PROJECTED_SOLID_ANGLE_RECTANGLE)
   {
      for (uint32_t c = 0; c < creations; c++)
      {
         ClippedSilhouette silhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);
         PyramidSetup ps = PyramidSetup::create(silhouette);

         const float32_t3 axis3 = ps.pyramid.getAxis3();
         shapes::CompressedSphericalRectangle<float32_t> compressed;
         compressed.origin = ps.pyramid.axis1 * ps.pyramid.rectR0.x + ps.pyramid.axis2 * ps.pyramid.rectR0.y + axis3;
         compressed.right  = ps.pyramid.axis1 * ps.pyramid.rectExtents.x;
         compressed.up     = ps.pyramid.axis2 * ps.pyramid.rectExtents.y;
         sampling::ProjectedSphericalRectangle<float32_t> projRectSampler = sampling::ProjectedSphericalRectangle<float32_t>::create(compressed, float32_t3(0.0f, 0.0f, 0.0f), float32_t3(0.0f, 0.0f, 1.0f), false);

         for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
         {
            float32_t2 xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
            sampling::ProjectedSphericalRectangle<float32_t>::cache_type cache;
            float32_t hitDist;
            float32_t3 localDir = projRectSampler.generateNormalizedLocal(xi, cache, hitDist);
            float32_t3 dir = localDir.x * ps.pyramid.axis1 + localDir.y * ps.pyramid.axis2 + localDir.z * ps.pyramid.getAxis3();
            float32_t localX = localDir.x / localDir.z;
            float32_t localY = localDir.y / localDir.z;
            sampleValid = dir.z > 0.0f && ps.silEdgeNormals.isInsideLocal(localX, localY);
            float32_t pdf = projRectSampler.forwardPdf(xi, cache);
            sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ (uint32_t)sampleValid;
         }
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC)
   {
      for (uint32_t c = 0; c < creations; c++)
      {
         ClippedSilhouette silhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);
         PyramidSetup ps = PyramidSetup::create(silhouette);
         BiquadraticSampler biquad = BiquadraticSampler::create(ps.pyramid);

         for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
         {
            float32_t2 xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
            float32_t pdf;
            float32_t3 dir = biquad.sample(ps.pyramid, ps.silEdgeNormals, xi, pdf, sampleValid);
            sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ (uint32_t)sampleValid;
         }
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
   {
      for (uint32_t c = 0; c < creations; c++)
      {
         ClippedSilhouette silhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);
         PyramidSetup ps = PyramidSetup::create(silhouette);
         BilinearSampler bilin = BilinearSampler::create(ps.pyramid);

         for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
         {
            float32_t2 xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
            float32_t pdf;
            float32_t3 dir = bilin.sample(ps.pyramid, ps.silEdgeNormals, xi, pdf, sampleValid);
            sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ (uint32_t)sampleValid;
         }
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::SILHOUETTE_CREATION_ONLY)
   {
      for (uint32_t i = 0; i < pc.sampleCount; i++)
      {
         ClippedSilhouette iterSilhouette = makePerturbedSilhouette(rndOffset, rng, rcpU32);

         sink ^= iterSilhouette.count;
         NBL_UNROLL
         for (uint32_t j = 0; j < MAX_SILHOUETTE_VERTICES; j++)
            sink ^= asuint(iterSilhouette.vertices[j].x) ^ asuint(iterSilhouette.vertices[j].y) ^ asuint(iterSilhouette.vertices[j].z);
      }
   }
   else if (benchmarkMode == SAMPLING_MODE::PYRAMID_CREATION_ONLY)
   {
      for (uint32_t i = 0; i < pc.sampleCount; i++)
      {
         ClippedSilhouette synthSil = (ClippedSilhouette)0;
         synthSil.count = 5;

         NBL_UNROLL
         for (uint32_t v = 0; v < 5; v++)
         {
            float32_t x = (float32_t(rng()) * rcpU32 - 0.5f) * 1.2f;
            float32_t y = (float32_t(rng()) * rcpU32 - 0.5f) * 1.2f;
            synthSil.vertices[v] = normalize(float32_t3(x, y, 1.0f));
         }

         SilEdgeNormals silEdgeNormals;
         SphericalPyramid pyramid = SphericalPyramid::create(synthSil, silEdgeNormals);

         uint32_t pyramidBits = asuint(pyramid.axis1.x) ^ asuint(pyramid.axis2.x) ^ asuint(pyramid.rectR0.x) ^ asuint(pyramid.rectR0.y) ^ asuint(pyramid.rectExtents.x) ^ asuint(pyramid.rectExtents.y);
         uint32_t edgeBits = asuint(float32_t(silEdgeNormals.edgeNormals[0].x)) ^ asuint(float32_t(silEdgeNormals.edgeNormals[1].x));
         sink ^= pyramidBits ^ edgeBits;
      }
   }

   const uint32_t offset = sizeof(uint32_t) * invocationID.x;
   outputBuffer.Store<uint32_t>(offset, sink);
}
