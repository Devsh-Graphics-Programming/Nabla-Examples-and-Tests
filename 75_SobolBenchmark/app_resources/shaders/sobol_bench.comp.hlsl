// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/macros.h>
#include <nbl/builtin/hlsl/sampling/sobol.hlsl>

// both alrady defined from `examples_tests\75_SobolBenchmark\CMakeLists.txt` so you can remove these (or vice versa)
#ifndef DEPTH
#define DEPTH 8
#endif
#ifndef METHOD
#define METHOD 0
#endif

namespace nbl
{
namespace hlsl
{

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t Depth      = _static_cast<uint16_t>(DEPTH);
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t Samples    = _static_cast<uint16_t>(BENCH_ITERS);
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t Triplets   = _static_cast<uint16_t>(2);
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t Components = _static_cast<uint16_t>(3);

} // namespace hlsl
} // namespace nbl

// TODO: real direction-number conversion from include/nbl/core/sampling/SobolSampler.h.
// Each entry below is a 16x16 GF(2) direction matrix. Tally is Depth * Triplets * Components.
#define _NBL_SOBOL_PLACEHOLDER_MATRIX {0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu}

#if METHOD == 0
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR nbl::hlsl::sampling::RowMajorSobolMatrix rowMatrices[DEPTH * 6] = {
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX};
#elif METHOD == 1
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR nbl::hlsl::sampling::ColMajorSobolMatrix colMatrices[DEPTH * 6] = {
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
   _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX};
#else
#error "METHOD must be 0 (row-major) or 1 (col-major)"
#endif

[[vk::binding(1, 0)]] RWByteAddressBuffer benchOutput;

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
   using namespace nbl::hlsl;

   const uint32_t invID = glsl::gl_GlobalInvocationID().x;
    // simulate ranking by Heitz (but not the scramble)
   const uint16_t rank = _static_cast<uint16_t>(bitReverse<uint32_t>(invID) >> 16);

   uint16_t seed = _static_cast<uint16_t>(0xdeadu);
   // simulate rays running paths to exhaustion in a working set
   [[loop]]
   for (uint16_t s = _static_cast<uint16_t>(0u); s < Samples; s++)
   {
      [[loop]]
      for (uint16_t d = _static_cast<uint16_t>(0u); d < Depth; d++)
      {
         uint16_t3 triples[2];
         NBL_UNROLL
         for (uint16_t t = _static_cast<uint16_t>(0u); t < Triplets; t++)
         {
            NBL_UNROLL
            for (uint16_t c = _static_cast<uint16_t>(0u); c < Components; c++)
            {
#if METHOD == 0
               triples[t][c] = rowMatrices[(d * Triplets + t) * Components + c](s ^ rank); // implemented in `include\nbl\builtin\hlsl\sampling\sobol.hlsl`
#elif METHOD == 1
               triples[t][c] = colMatrices[(d * Triplets + t) * Components + c](s ^ rank);
#endif
            }
         }
         seed ^= triples[0].x;
         seed += triples[0].y;
         seed ^= triples[0].z;
         seed += triples[1].x;
         seed ^= triples[1].y;
         seed += triples[1].z;
      }
   }
   benchOutput.Store(invID * 4u, _static_cast<uint32_t>(seed));
}
