#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_DISCRETE_SAMPLER_BENCH_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_DISCRETE_SAMPLER_BENCH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t WorkgroupSize = WORKGROUP_SIZE;

struct CumProbPushConstants
{
	uint64_t cumProbAddress;	// float cumProb[N-1]
	uint64_t outputAddress;		// uint32_t acc[threadCount]
	uint32_t tableSize;			// N
};

// Variants A and B both take the entry array plus a separate pdf[] array
// (A: 4 B words, B: 8 B {packedWord, ownPdf}; pdf[] has the same contents in
// both but is tapped independently by the sampler).
struct PackedAliasABPushConstants
{
	uint64_t entriesAddress;	// A: uint32_t words[N] (4 B); B: PackedAliasEntryB<float>[N] (8 B)
	uint64_t pdfAddress;		// float pdf[N]
	uint64_t outputAddress;		// uint32_t acc[threadCount]
	uint32_t tableSize;			// N
};

#endif
