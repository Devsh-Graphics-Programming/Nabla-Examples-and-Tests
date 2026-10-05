#pragma shader_stage(compute)

#include "../common/projected_spherical_rectangle.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#ifdef BENCH_ITERS
#include "../common/sampler_bench_pc.hlsl"
[[vk::push_constant]] SamplerBenchPushConstants benchPC;
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedSphericalRectangleInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedSphericalRectangleTestResults> outputTestValues;
#endif

// Number of generate() calls per create(). Default = BENCH_ITERS (persistent: 1 create total).
// Set to 1 for 1:1, 16 for 1:16 multisampling, etc. Must divide BENCH_ITERS.
#if !defined(BENCH_SAMPLES_PER_CREATE) && defined(BENCH_ITERS)
#define BENCH_SAMPLES_PER_CREATE (BENCH_ITERS)
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
   const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
   // Perturb rectangle origin by invID so the sampler is non-uniform across threads.
   const float32_t perturbationBase = float32_t(invID) * 1.0e-7f;

   nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
   const float32_t toFloat = asfloat(0x2f800004u);
   uint32_t acc = 0u;
#ifdef BENCH_CREATE_ONLY
   for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
   {
      // Depend on i so the compiler can't hoist create() out of the loop.
      const float32_t perturbation = perturbationBase + float32_t(i) * 1.0e-9f;
      shapes::CompressedSphericalRectangle<float32_t> compressed;
      compressed.origin = float32_t3(perturbation, perturbation, -2.0f);
      compressed.right = float32_t3(1.0f, 0.0f, 0.0f);
      compressed.up = float32_t3(0.0f, 1.0f, 0.0f);
      shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
      sampling::ProjectedSphericalRectangle<float32_t> sampler = sampling::ProjectedSphericalRectangle<float32_t>::create(rect, float32_t3(0.0f, 0.0f, 0.0f), float32_t3(0.0f, 0.0f, perturbation + 0.5), false);
      // Read a cheap function of sampler state so create() can't be elided.
      sampling::ProjectedSphericalRectangle<float32_t>::cache_type pdfCache;
      sampler.generate(float32_t2(0.5f, 0.5f), pdfCache);
      acc ^= asuint(sampler.forwardPdf(float32_t2(0.5f, 0.5f), pdfCache));
   }
#else
   // Unified create:generate loop — one create per BENCH_SAMPLES_PER_CREATE generates.
   const uint32_t outerIters = uint32_t(BENCH_ITERS) / uint32_t(BENCH_SAMPLES_PER_CREATE);
   for (uint32_t j = 0u; j < outerIters; j++)
   {
      const float32_t perturbation = perturbationBase + float32_t(j) * 1.0e-9f;
      shapes::CompressedSphericalRectangle<float32_t> compressed;
      compressed.origin = float32_t3(perturbation, perturbation, -2.0f);
      compressed.right = float32_t3(1.0f, 0.0f, 0.0f);
      compressed.up = float32_t3(0.0f, 1.0f, 0.0f);
      shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
      sampling::ProjectedSphericalRectangle<float32_t> sampler = sampling::ProjectedSphericalRectangle<float32_t>::create(rect, float32_t3(0.0f, 0.0f, 0.0f), float32_t3(0.0f, 0.0f, perturbation + 0.5), false);
      for (uint32_t k = 0u; k < uint32_t(BENCH_SAMPLES_PER_CREATE); k++)
      {
         float32_t2 u = float32_t2(rng(), rng()) * toFloat;
         sampling::ProjectedSphericalRectangle<float32_t>::cache_type cache;
         float32_t3 generated = sampler.generate(u, cache);
         acc ^= asuint(generated.x) ^ asuint(generated.y) ^ asuint(generated.z);
         acc ^= asuint(sampler.forwardPdf(u, cache));
      }
   }
#endif
   vk::RawBufferStore<uint32_t>(benchPC.outputAddress + invID * 4u, acc);
#else
   ProjectedSphericalRectangleTestExecutor executor;
   executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
