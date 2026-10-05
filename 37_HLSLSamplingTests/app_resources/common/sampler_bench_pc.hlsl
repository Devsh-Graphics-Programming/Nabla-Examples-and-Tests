#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SAMPLER_BENCH_PC_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SAMPLER_BENCH_PC_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

// Implicit-output benchmark push constants. Every sampler bench shader writes
// one uint32_t accumulator per thread to outputAddress[invID]; nothing reads it
// back -- the goal is to keep the optimiser from eliding the sampling work.
// Mirrors the BDA convention from discrete_sampler_bench.hlsl.
struct SamplerBenchPushConstants
{
	uint64_t outputAddress;
};

#endif
