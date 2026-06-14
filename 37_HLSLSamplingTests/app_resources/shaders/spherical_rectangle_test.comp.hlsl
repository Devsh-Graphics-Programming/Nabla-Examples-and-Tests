#pragma shader_stage(compute)

#include "../common/spherical_rectangle.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#ifdef BENCH_ITERS
[[vk::binding(1, 0)]] RWByteAddressBuffer benchOutput;
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<SphericalRectangleInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<SphericalRectangleTestResults> outputTestValues;
#endif

// Number of generate() calls per create(). Default = BENCH_ITERS (persistent: 1 create total).
// Set to 1 for 1:1 (create+generate per iter), 16 for 1:16 multisampling, etc. Must divide BENCH_ITERS.
#if !defined(BENCH_SAMPLES_PER_CREATE) && defined(BENCH_ITERS)
#define BENCH_SAMPLES_PER_CREATE (BENCH_ITERS)
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
   const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
   // Observer at origin so origin - observer = (p, p, -2) has no zero components:
   // keeps all 4 denorm_n_z components perturbation-dependent (no constant-folding).
   const float32_t perturbationBase = float32_t(invID) * 1.0e-7f;

#if (defined(BENCH_VARIANT_SA_EXTENTS) || defined(BENCH_VARIANT_R0_EXTENTS)) && !defined(BENCH_CREATE_ONLY)
   // variants 2/3 pre-build: produce a rect (for its basis, sa, extents) once per thread.
   shapes::CompressedSphericalRectangle<float32_t> compressedBase;
   compressedBase.origin = float32_t3(perturbationBase, perturbationBase, -2.0f);
   compressedBase.right = float32_t3(1.0f, 0.0f, 0.0f);
   compressedBase.up = float32_t3(0.0f, 1.0f, 0.0f);
   const shapes::SphericalRectangle<float32_t> rectBase = shapes::SphericalRectangle<float32_t>::create(compressedBase);
   const typename shapes::SphericalRectangle<float32_t>::solid_angle_type saBase = rectBase.solidAngle(float32_t3(0.0f, 0.0f, 0.0f));
   const float32_t2 extentsBase = rectBase.extents;
   const matrix<float32_t, 3, 3> basisBase = rectBase.basis;
#endif

   nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
   const float32_t toFloat = asfloat(0x2f800004u);
   uint32_t acc = 0u;
#ifdef BENCH_CREATE_ONLY
   for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
   {
      // Depend on i so the compiler can't hoist create() out of the loop.
      const float32_t perturbation = perturbationBase + float32_t(i) * 1.0e-9f;
      sampling::SphericalRectangle<float32_t> sampler;
  #if defined(BENCH_VARIANT_SA_EXTENTS)
      shapes::CompressedSphericalRectangle<float32_t> compressed;
      compressed.origin = float32_t3(perturbation, perturbation, -2.0f);
      compressed.right = float32_t3(1.0f, 0.0f, 0.0f);
      compressed.up = float32_t3(0.0f, 1.0f, 0.0f);
      shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
      typename shapes::SphericalRectangle<float32_t>::solid_angle_type sa = rect.solidAngle(float32_t3(0.0f, 0.0f, 0.0f));
      sampler = sampling::SphericalRectangle<float32_t>::create(rect.basis, sa, rect.extents);
  #elif defined(BENCH_VARIANT_R0_EXTENTS)
      // Build a basis from the same rect geometry so create(basis, r0, extents) has the right frame.
      shapes::CompressedSphericalRectangle<float32_t> compressedR0;
      compressedR0.origin = float32_t3(perturbation, perturbation, -2.0f);
      compressedR0.right = float32_t3(1.0f, 0.0f, 0.0f);
      compressedR0.up = float32_t3(0.0f, 1.0f, 0.0f);
      const shapes::SphericalRectangle<float32_t> rectR0 = shapes::SphericalRectangle<float32_t>::create(compressedR0);
      const float32_t3 r0 = float32_t3(perturbation, perturbation, -2.0f);
      const float32_t2 extents = float32_t2(1.0f, 1.0f);
      sampler = sampling::SphericalRectangle<float32_t>::create(rectR0.basis, r0, extents);
  #else
      shapes::CompressedSphericalRectangle<float32_t> compressed;
      compressed.origin = float32_t3(perturbation, perturbation, -2.0f);
      compressed.right = float32_t3(1.0f, 0.0f, 0.0f);
      compressed.up = float32_t3(0.0f, 1.0f, 0.0f);
      shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
      sampler = sampling::SphericalRectangle<float32_t>::create(rect, float32_t3(0.0f, 0.0f, 0.0f));
  #endif
      // Read a cheap function of sampler state so create() can't be elided.
      acc ^= asuint(sampler.backwardPdf(float32_t3(0.0f, 0.0f, 1.0f)));
   }
#else
   // Unified create:generate loop - one create per BENCH_SAMPLES_PER_CREATE generates.
   const uint32_t outerIters = uint32_t(BENCH_ITERS) / uint32_t(BENCH_SAMPLES_PER_CREATE);
   for (uint32_t j = 0u; j < outerIters; j++)
   {
      const float32_t perturbation = perturbationBase + float32_t(j) * 1.0e-9f;
      sampling::SphericalRectangle<float32_t> sampler;
  #if defined(BENCH_VARIANT_SA_EXTENTS)
      // variant 2: create(basis, sa, extents). Poison one cosGamma so the sincos_accumulator can't be hoisted.
      typename shapes::SphericalRectangle<float32_t>::solid_angle_type sa = saBase;
      sa.cosGamma[2] += perturbation;
      sampler = sampling::SphericalRectangle<float32_t>::create(basisBase, sa, extentsBase);
  #elif defined(BENCH_VARIANT_R0_EXTENTS)
      // variant 3: create(basis, r0, extents). r0 matches what variant 1 produces.
      const float32_t3 r0 = float32_t3(perturbation, perturbation, -2.0f);
      const float32_t2 extents = float32_t2(1.0f, 1.0f);
      sampler = sampling::SphericalRectangle<float32_t>::create(basisBase, r0, extents);
  #else
      // variant 1 (default): create(shape, observer).
      shapes::CompressedSphericalRectangle<float32_t> compressed;
      compressed.origin = float32_t3(perturbation, perturbation, -2.0f);
      compressed.right = float32_t3(1.0f, 0.0f, 0.0f);
      compressed.up = float32_t3(0.0f, 1.0f, 0.0f);
      shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
      sampler = sampling::SphericalRectangle<float32_t>::create(rect, float32_t3(0.0f, 0.0f, 0.0f));
  #endif
      for (uint32_t k = 0u; k < uint32_t(BENCH_SAMPLES_PER_CREATE); k++)
      {
         float32_t2 u = float32_t2(rng(), rng()) * toFloat;
         sampling::SphericalRectangle<float32_t>::cache_type cache;
         float32_t3 generated = sampler.generate(u, cache);
         acc ^= asuint(generated.x) ^ asuint(generated.y) ^ asuint(generated.z);
         acc ^= asuint(sampler.forwardPdf(u, cache));
      }
   }
#endif
   benchOutput.Store(invID * 4u, acc);
#else
   SphericalRectangleTestExecutor executor;
   executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
