// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/macros.h>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/sampling/quantized_sequence.hlsl>
#include <nbl/builtin/hlsl/sampling/sobol.hlsl>

// METHOD 0: Cached+Quantized Sobol from BDA, scrambled by xoroshiro, no float decode.
// METHOD 1: 16x16 GF(2) row-major matrix-mul (bitcount + strided 5-bit mask).
// METHOD 2: 16x16 GF(2) col-major matrix-mul (two's-complement broadcast mask).
// BENCH_ITERS, WORKGROUP_SIZE, DEPTH come from CMakeLists.txt -DBENCH_ITERS, etc.

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

struct PushConstants
{
    uint64_t pSampleBuffer;        // BDA to QuantizedSequence<uint32_t2,3>[]
    uint32_t sequenceSamplesLog2;  // log2 of samples-per-dimension stride
    // Lane-divergence + early-termination knobs. Both default to "off" if zeroed.
    uint32_t depthStaggerMask;     // power-of-2 minus 1; lane offset = invID & this
    float    probAtDepth0;         // linear PDF of breaking out of the depth loop
    float    probAtDepthMax;       // (compared against triples[0].z each iteration)
};

[[vk::push_constant]] PushConstants pc;

// Binding 0 is a 512x512 R32G32_UINT storage image holding xoroshiro seeds (METHOD 0).
// METHOD 1 and 2 don't read it; the harness still binds it because the descriptor
// set layout is shared across all three benches. Storage image (vs SSBO) so the loads
// go through the texture cache hierarchy with tiled memory layout, matching how
// the path tracer (ex 31, ex 40) reads its scramble image.
[[vk::binding(0, 0)]] RWTexture2D<uint32_t2> seedTexture;
[[vk::binding(1, 0)]] RWByteAddressBuffer benchOutput;
//[[vk::binding(2, 0)]] StructuredBuffer<uint32_t>  inputBuf; // unused

#if METHOD == 1 || METHOD == 2
// TODO: real direction-number conversion from include/nbl/core/sampling/SobolSampler.h.
// Each entry below is a 16x16 GF(2) direction matrix. Tally is Depth * Triplets * Components.
#define _NBL_SOBOL_PLACEHOLDER_MATRIX {0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu, 0xdeadu}
#endif

#if METHOD == 1
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR nbl::hlsl::sampling::RowMajorSobolMatrix rowMatrices[DEPTH * 6] = {
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX};
#elif METHOD == 2
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR nbl::hlsl::sampling::ColMajorSobolMatrix colMatrices[DEPTH * 6] = {
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX,
    _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX, _NBL_SOBOL_PLACEHOLDER_MATRIX};
#endif

#if METHOD != 0 && METHOD != 1 && METHOD != 2
#error "METHOD must be 0 (quantized BDA), 1 (row-major), or 2 (col-major)"
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
    using namespace nbl::hlsl;

    const uint32_t invID = glsl::gl_GlobalInvocationID().x;
    // simulate ranking by Heitz (but not the scramble)
    const uint16_t rank = _static_cast<uint16_t>(bitReverse<uint32_t>(invID) >> 16);

    // Per-invocation depth offset to model SER/wavefront-style subgroup divergence:
    // lanes within a subgroup walk d in lockstep but pull a different matrix index,
    // so loads can no longer broadcast and the subgroup has to "strip" the divergence.
    // DEPTH must be a power of two so the wrap can be a bit-and, a runtime uint16
    // modulo in the inner loop was 10x slower in practice.
    const uint16_t laneOffset = _static_cast<uint16_t>(invID & pc.depthStaggerMask) & _static_cast<uint16_t>(DEPTH - 1u);

    // Linear-PDF early-break threshold over the depth loop (models miss / absorption /
    // Russian roulette terminating paths and not getting refilled). When DEPTH==1 the
    // delta is unused, threshold stays at the depth-0 value.
    const float thresholdScale = 65535.0f;
    float threshold = pc.probAtDepth0 * thresholdScale;
#if DEPTH > 1
    const float thresholdDelta = (pc.probAtDepthMax - pc.probAtDepth0) * thresholdScale / float(DEPTH - 1);
#else
    const float thresholdDelta = 0.0f;
#endif

    uint16_t seed = _static_cast<uint16_t>(0xdeadu);

#if METHOD == 0
    using sequence_type = sampling::QuantizedSequence<uint32_t2, 3>;
    NBL_CONSTEXPR uint32_t SeedDim = 512u;
    NBL_CONSTEXPR uint32_t SeedDimMask = SeedDim - 1u;
    const uint32_t invX = invID & SeedDimMask;
    const uint32_t invY = (invID >> 9) & SeedDimMask;
#endif

    [[loop]]
    for (uint16_t s = _static_cast<uint16_t>(0u); s < Samples; s++)
    {
#if METHOD == 0
        // Re-read the per-path xoroshiro seed every sample (REPEAT wrap), in the real
        // path tracer we'd read once per pixel, but here we want to amortize the texture
        // fetch cost across the matrix-mul work like SER/wavefront would.
        const uint32_t coordX = (invX + uint32_t(s)) & SeedDimMask;
        const uint32_t coordY = (invY + uint32_t(s)) & SeedDimMask;
        Xoroshiro64Star rng = Xoroshiro64Star::construct(seedTexture[uint32_t2(coordX, coordY)]);
#endif

        // Reset per-sample so each "ray" sees the same depth-PDF.
        float perSampleThreshold = threshold;

        [[loop]]
        for (uint16_t d = _static_cast<uint16_t>(0u); d < Depth; d++)
        {
            // Stagger the depth index per lane so neighbouring invocations load different
            // matrices/dimensions for the same iteration of the rolled loop.
            const uint16_t dIdx = (d + laneOffset) & _static_cast<uint16_t>(DEPTH - 1u);

            uint16_t3 triples[2];
#if METHOD == 0
            // Two triplets per depth. 4 rng() calls per depth, a 64-bit
            // scramble key per triplet. Stay in unorm bits, no float decode, for
            // apples-to-apples throughput against the matrix-mul methods.
            NBL_UNROLL
            for (uint16_t t = _static_cast<uint16_t>(0u); t < Triplets; t++)
            {
                const uint32_t baseDim = uint32_t(dIdx) * uint32_t(Triplets) + uint32_t(t);
                const uint32_t address = uint32_t(s) | (baseDim << pc.sequenceSamplesLog2);
                sequence_type tmpSeq = vk::RawBufferLoad<sequence_type>(pc.pSampleBuffer + uint64_t(address) * uint64_t(sizeof(sequence_type)));
                sequence_type scrambleKey;
                scrambleKey.data[0] = rng();
                scrambleKey.data[1] = rng();
                tmpSeq.data ^= scrambleKey.data;
                triples[t].x = _static_cast<uint16_t>(tmpSeq.get(_static_cast<uint16_t>(0u)));
                triples[t].y = _static_cast<uint16_t>(tmpSeq.get(_static_cast<uint16_t>(1u)));
                triples[t].z = _static_cast<uint16_t>(tmpSeq.get(_static_cast<uint16_t>(2u)));
            }
#else
            NBL_UNROLL
            for (uint16_t t = _static_cast<uint16_t>(0u); t < Triplets; t++)
            {
                NBL_UNROLL
                for (uint16_t c = _static_cast<uint16_t>(0u); c < Components; c++)
                {
#if METHOD == 1
                    triples[t][c] = rowMatrices[(dIdx * Triplets + t) * Components + c](s ^ rank);  // implemented in `include\nbl\builtin\hlsl\sampling\sobol.hlsl`
#elif METHOD == 2
                    triples[t][c] = colMatrices[(dIdx * Triplets + t) * Components + c](s ^ rank);
#endif
                }
            }
#endif

            // Current bounce's contribution always lands. Models BSDF sample + NEE +
            // contribution being computed before any termination decision is made.
            seed ^= triples[0].x;
            seed += triples[0].y;
            seed ^= triples[0].z;
            seed += triples[1].x;
            seed ^= triples[1].y;
            seed += triples[1].z;

            // Russian-roulette-style early break: decides whether the NEXT bounce happens.
            // Skips the next iteration's matrix loads (which is the real cost). Random
            // source is triples[0].z, threshold is a linear PDF over depth. With both
            // probs at 0 the comparison is always false (0 < 0 fails) so the loop runs
            // to Depth.
            if (triples[0].z < _static_cast<uint16_t>(perSampleThreshold))
                break;
            perSampleThreshold += thresholdDelta;
        }
    }

    benchOutput.Store(invID * 4u, _static_cast<uint32_t>(seed));
}
