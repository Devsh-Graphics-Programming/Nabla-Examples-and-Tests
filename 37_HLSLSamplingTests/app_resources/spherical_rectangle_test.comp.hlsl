#pragma shader_stage(compute)

#include "common/spherical_rectangle.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<SphericalRectangleInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<SphericalRectangleTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")] void
main()
{
   const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
   // Hardcode a valid non-degenerate rectangle: observer at origin, rect at z=-2.
   // Use invID for baseU so u is a runtime value — prevents loop DCE after unrolling.
   shapes::CompressedSphericalRectangle<float32_t> compressed;
   compressed.origin = float32_t3(0.0f, 0.0f, -2.0f);
   compressed.right = float32_t3(1.0f, 0.0f, 0.0f);
   compressed.up = float32_t3(0.0f, 1.0f, 0.0f);
   shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
   sampling::SphericalRectangle<float32_t> sampler = sampling::SphericalRectangle<float32_t>::create(rect, float32_t3(0.0f, 0.0f, 0.0f));

   nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
   const float32_t toFloat = asfloat(0x2f800004u);
   uint32_t2 acc = (uint32_t2)0;
   uint32_t accPdf = 0;
   for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
   {
      float32_t2 u = float32_t2(rng(), rng()) * toFloat;
      sampling::SphericalRectangle<float32_t>::cache_type cache;
      float32_t2 generated = sampler.generate(u, cache);
      acc ^= asuint(generated);
      accPdf ^= asuint(sampler.forwardPdf(generated, cache));
   }
   SphericalRectangleTestResults result = (SphericalRectangleTestResults)0;
   result.generated = asfloat(acc);
   result.forwardPdf = asfloat(accPdf);
   outputTestValues[invID] = result;
#else
   SphericalRectangleTestExecutor executor;
   executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
