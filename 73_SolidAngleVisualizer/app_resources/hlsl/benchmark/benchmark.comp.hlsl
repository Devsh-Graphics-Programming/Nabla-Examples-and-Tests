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
#include "app_resources/hlsl/obb_face_sampling.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer    outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

static const SAMPLING_MODE_FLAGS benchmarkMode = SAMPLING_MODE_FLAGS_CONST;

float32_t2 stratifiedXi(uint32_t sampleIdx, uint32_t threadIdx)
{
   return float32_t2(
      (float32_t(sampleIdx & 7u) + 0.5f) / 8.0f + float32_t(threadIdx) * 1e-9f,
      (float32_t(sampleIdx >> 3u) + 0.5f) / 8.0f + float32_t(threadIdx) * 1e-9f);
}

// Per-thread input perturbation: scatters threads across the 27 OBB regions and
// generates a fresh OBBView per outer-loop iteration so creation work can't be
// hoisted out by the compiler. Returns just the view; callers build their own
// ClippedSilhouette + materialized verts from it as needed.
shapes::OBBView<float32_t> makePerturbedView(float32_t3 baseOffset, NBL_REF_ARG(Xoroshiro64Star) rng, float32_t rcpU32)
{
   const float32_t3 cJ = float32_t3(
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f,
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f,
      (float32_t(rng()) * rcpU32 - 0.5f) * 8.0f);
   float32_t3x4 cM = pc.modelMatrix;
   cM[0][3] += baseOffset.x + cJ.x;
   cM[1][3] += baseOffset.y + cJ.y;
   cM[2][3] += baseOffset.z + cJ.z;
   return shapes::OBBView<float32_t>::create(cM);
}

// Shared create-and-sample loop for any sampler with the standard
// `create(silhouette, view)` + `generate/forwardPdf/selectedIdx(cache)` shape.
// XORs all outputs into the returned sink to defeat DCE.
template<typename SamplerT>
uint32_t runCreateAndSample(uint32_t creations, NBL_REF_ARG(Xoroshiro64Star) rng, float32_t rcpU32, uint32_t invocationID, float32_t3 rndOffset)
{
   uint32_t sink = 0;
   for (uint32_t c = 0; c < creations; c++)
   {
      shapes::OBBView<float32_t> view       = makePerturbedView(rndOffset, rng, rcpU32);
      ClippedSilhouette          silhouette = ClippedSilhouette::create(view, pc.shadingPoint);
      SamplerT                   sampler    = SamplerT::create(silhouette, view);

      for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
      {
         float32_t2                    xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
         typename SamplerT::cache_type cache;
         float32_t3                    dir = sampler.generate(xi, cache);
         float32_t                     pdf = sampler.forwardPdf(xi, cache);
         sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ sampler.selectedIdx(cache);
      }
   }
   return sink;
}

// Variant for samplers whose `create(view)` works directly from the OBBView
// without needing a ClippedSilhouette upstream. Skips the ~25-30 ps silhouette
// build cost per creation.
template<typename SamplerT>
uint32_t runCreateAndSampleNoSilhouette(uint32_t creations, NBL_REF_ARG(Xoroshiro64Star) rng, float32_t rcpU32, uint32_t invocationID, float32_t3 rndOffset)
{
   uint32_t sink = 0;
   for (uint32_t c = 0; c < creations; c++)
   {
      shapes::OBBView<float32_t> view    = makePerturbedView(rndOffset, rng, rcpU32);
      SamplerT                   sampler = SamplerT::create(view, pc.shadingPoint);

      for (uint32_t s = 0; s < pc.samplesPerCreation; s++)
      {
         float32_t2                    xi = stratifiedXi(c * pc.samplesPerCreation + s, invocationID);
         typename SamplerT::cache_type cache;
         float32_t3                    dir = sampler.generate(xi, cache);
         float32_t                     pdf = sampler.forwardPdf(xi, cache);
         sink ^= asuint(dir.x) ^ asuint(dir.y) ^ asuint(dir.z) ^ asuint(pdf) ^ sampler.selectedIdx(cache);
      }
   }
   return sink;
}

// Pyramid-create-only benchmark using synthetic random vertices. Templated on
// UseCaliper so PYRAMID_CREATION_ONLY and CALIPER_PYRAMID_CREATION_ONLY share
// one body. Inner sampler is unused (no generate() calls), so default to SphRect.
template<bool UseCaliper>
uint32_t runPyramidCreationOnly(NBL_REF_ARG(Xoroshiro64Star) rng, float32_t rcpU32)
{
   typedef SphericalPyramid<UseCaliper, sampling::SphericalRectangle<float32_t> > PyramidT;
   uint32_t sink = 0;
   for (uint32_t i = 0; i < pc.sampleCount; i++)
   {
      float32_t3 synthVerts[MAX_SILHOUETTE_VERTICES];
      NBL_UNROLL
      for (uint32_t init = 0; init < MAX_SILHOUETTE_VERTICES; init++)
         synthVerts[init] = float32_t3(0, 0, 0);
      const uint32_t synthCount = 5;

      for (uint32_t v = 0; v < synthCount; v++)
      {
         float32_t x = (float32_t(rng()) * rcpU32 - 0.5f) * 1.2f;
         float32_t y = (float32_t(rng()) * rcpU32 - 0.5f) * 1.2f;
         // Diagnostic raw-rng sink: forces rng+normalize cost into the timing
         // even if the entire pyramid create() gets DCE'd downstream.
         sink ^= asuint(x) ^ asuint(y);
         synthVerts[v] = normalize(float32_t3(x, y, 1.0f));
         sink ^= asuint(synthVerts[v].x) ^ asuint(synthVerts[v].y) ^ asuint(synthVerts[v].z);
      }

      float32_t2 dummyR0, dummyExt;
      PyramidT   pyramid = PyramidT::createFromVertices(synthVerts, synthCount, dummyR0, dummyExt);

      const float32_t3 axis3 = pyramid.getAxis3();
      sink ^= asuint(pyramid.axis1.x) ^ asuint(pyramid.axis1.y) ^ asuint(pyramid.axis1.z);
      sink ^= asuint(pyramid.axis2.x) ^ asuint(pyramid.axis2.y) ^ asuint(pyramid.axis2.z);
      sink ^= asuint(axis3.x) ^ asuint(axis3.y) ^ asuint(axis3.z);
      NBL_UNROLL
      for (uint32_t e = 0; e < 5; e++)
      {
         const float32_t3 n = pyramid.silEdgeNormals.edgeNormals[e];
         sink ^= asuint(n.x) ^ asuint(n.y) ^ asuint(n.z);
      }
   }
   return sink;
}

[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)] 
void main()
{
   const uint32_t invocationID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;

   Xoroshiro64Star  rng       = Xoroshiro64Star::construct(uint32_t2(invocationID.x + 0x9e3779b9u, invocationID.x * 0x85ebca77u + 1u));
   const float32_t  rcpU32    = 1.0f / 4294967296.0f;
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

   if (benchmarkMode == SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY)
   {
      // Measure full silhouette-prep cost = create + materialize. The previous
      // ClippedSilhouette did both inline; the metadata-only ClippedSilhouette
      // splits them, so we exercise both here to keep this benchmark
      // apples-to-apples.
      for (uint32_t i = 0; i < pc.sampleCount; i++)
      {
         shapes::OBBView<float32_t> iterView       = makePerturbedView(rndOffset, rng, rcpU32);
         ClippedSilhouette          iterSilhouette = ClippedSilhouette::create(iterView, pc.shadingPoint);
         float32_t3                 iterVerts[MAX_SILHOUETTE_VERTICES];
         iterSilhouette.materialize(iterView, iterVerts);

         sink ^= iterSilhouette.count;
         NBL_UNROLL
         for (uint32_t j = 0; j < MAX_SILHOUETTE_VERTICES; j++)
            sink ^= asuint(iterVerts[j].x) ^ asuint(iterVerts[j].y) ^ asuint(iterVerts[j].z);
      }
   }
   else if ((benchmarkMode & SAMPLING_MODE_FLAGS::FLAG_PYRAMID) != 0u && (benchmarkMode & SAMPLING_MODE_FLAGS::FLAG_CREATE_ONLY) != 0u)
      sink ^= runPyramidCreationOnly<(benchmarkMode & SAMPLING_MODE_FLAGS::FLAG_CALIPER) != 0u>(rng, rcpU32);
   // Caliper variant: tighter rect → different rejection rate, only interesting when samplesPerCreation > 1.
   else if (benchmarkMode == SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID)
      sink ^= runCreateAndSample<SphericalPyramid<true, sampling::SphericalRectangle<float32_t> > >(creations, rng, rcpU32, invocationID, rndOffset);
   else if (benchmarkMode == SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID)
      sink ^= runCreateAndSample<SphericalPyramid<false, sampling::SphericalRectangle<float32_t> > >(creations, rng, rcpU32, invocationID, rndOffset);
   else if (benchmarkMode == SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID)
      sink ^= runCreateAndSample<SphericalPyramid<false, sampling::ProjectedSphericalRectangle<float32_t> > >(creations, rng, rcpU32, invocationID, rndOffset);
   else if ((benchmarkMode & SAMPLING_MODE_FLAGS::FLAG_TRIANGLE) != 0u)
      sink ^= runCreateAndSample<TriangleFanSampler<(benchmarkMode & SAMPLING_MODE_FLAGS::FLAG_PROJECTED) != 0u> >(creations, rng, rcpU32, invocationID, rndOffset);
   else if (benchmarkMode == SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
      sink ^= runCreateAndSample<Parallelogram>(creations, rng, rcpU32, invocationID, rndOffset);
   else if (benchmarkMode == SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID)
      sink ^= runCreateAndSample<SphericalPyramid<false, BilinearSampler> >(creations, rng, rcpU32, invocationID, rndOffset);
   else if (benchmarkMode == SAMPLING_MODE_FLAGS::OBB_FACE_DIRECT)
      sink ^= runCreateAndSampleNoSilhouette<OBBFaceSampler>(creations, rng, rcpU32, invocationID, rndOffset);
   else
   {
      assert(false);
   }
   const uint32_t offset = sizeof(uint32_t) * invocationID.x;
   outputBuffer.Store<uint32_t>(offset, sink);
}
