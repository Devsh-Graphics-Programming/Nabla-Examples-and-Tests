#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

#ifdef BENCH_ITERS
#include "../common/discrete_sampler_bench.hlsl"
#include <nbl/builtin/hlsl/sampling/alias_table.hlsl>

[[vk::push_constant]] PackedAliasABPushConstants pc;

// Log2N bucket. Covers all sweep sizes up to 2^LOG2N buckets without precision
// loss. The same value must be passed to the host-side packA<Log2N>() /
// packB<Log2N>() call so the bit layouts match.
NBL_CONSTEXPR uint32_t LOG2N_BUCKET = 26;

// Variant A accessor: 4 B packed words.
struct BdaPackedWordAccessor
{
	using value_type = uint32_t;

	template<typename V, typename I NBL_FUNC_REQUIRES(is_integral_v<V> && is_integral_v<I>)
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
	{
		val = vk::RawBufferLoad<V>(addr + uint64_t(sizeof(V)) * uint64_t(i), sizeof(V));
	}

	uint64_t addr;
};

// Variant B accessor: 8 B PackedAliasEntryB. Loads a uint2 and decomposes it
// into the POD entry so DXC never sees a bitfield — avoids the Insert/Extract
// round-trip we observed when the sampler read from a bitfield struct.
struct BdaPackedAliasBAccessor
{
	using value_type = nbl::hlsl::sampling::PackedAliasEntryB<float32_t>;

	template<typename V, typename I NBL_FUNC_REQUIRES(is_integral_v<I>)
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
	{
		const uint64_t loadAddr = addr + uint64_t(8u) * uint64_t(i);
		const uint2 raw = vk::RawBufferLoad<uint2>(loadAddr, 8u);
		val.packedWord = raw.x;
		val.ownPdf = asfloat(raw.y);
	}

	uint64_t addr;
};

// Separate 4 B pdf[] accessor.
struct BdaPdfAccessor
{
	using value_type = float32_t;

	template<typename V, typename I NBL_FUNC_REQUIRES(is_floating_point_v<V> && is_integral_v<I>)
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
	{
		val = vk::RawBufferLoad<V>(addr + uint64_t(sizeof(V)) * uint64_t(i), sizeof(V));
	}

	uint64_t addr;
};

#ifdef NBL_PACKED_ALIAS_B
using BenchPackedAlias = nbl::hlsl::sampling::PackedAliasTableB<float32_t, float32_t, uint32_t, BdaPackedAliasBAccessor, BdaPdfAccessor, LOG2N_BUCKET>;
#else
using BenchPackedAlias = nbl::hlsl::sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, BdaPackedWordAccessor, BdaPdfAccessor, LOG2N_BUCKET>;
#endif

#else
#include "../common/alias_table.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<AliasTableInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<AliasTableTestResults> outputTestValues;
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;

#ifdef BENCH_ITERS
#ifdef NBL_PACKED_ALIAS_B
	BdaPackedAliasBAccessor entryAcc;
#else
	BdaPackedWordAccessor entryAcc;
#endif
	entryAcc.addr = pc.entriesAddress;
	BdaPdfAccessor pdfAcc;
	pdfAcc.addr = pc.pdfAddress;
	BenchPackedAlias sampler = BenchPackedAlias::create(entryAcc, pdfAcc, pc.tableSize);

	float32_t xi = float32_t(nbl::hlsl::glsl::bitfieldReverse(invID)) / float32_t(~0u);
	NBL_CONSTEXPR float32_t goldenRatio = 0.6180339887498949f;
	uint32_t acc = 0u;

	[loop]
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		xi = frac(xi + goldenRatio);
		BenchPackedAlias::cache_type cache;
		uint32_t generated = sampler.generate(xi, cache);
		acc ^= generated ^ asuint(sampler.forwardPdf(xi, cache));
	}

	vk::RawBufferStore<uint32_t>(pc.outputAddress + uint64_t(sizeof(uint32_t)) * uint64_t(invID), acc);
#else
#ifdef NBL_PACKED_ALIAS_B
	PackedAliasBTestExecutor executor;
#else
	PackedAliasATestExecutor executor;
#endif
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
