#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_DISCRETE_SAMPLER_BENCH_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_DISCRETE_SAMPLER_BENCH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 64
#endif
NBL_CONSTEXPR uint32_t WorkgroupSize = WORKGROUP_SIZE;

struct AliasTablePushConstants
{
	uint64_t probAddress;		// float probability[N]
	uint64_t aliasAddress;		// uint32_t alias[N]
	uint64_t pdfAddress;		// float pdf[N]
	uint64_t outputAddress;		// uint32_t acc[threadCount]
	uint32_t tableSize;			// N
};

struct CumProbPushConstants
{
	uint64_t cumProbAddress;	// float cumProb[N-1]
	uint64_t outputAddress;		// uint32_t acc[threadCount]
	uint32_t tableSize;			// N
};

#endif
