#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

#ifdef BENCH_ITERS
#include "../common/discrete_sampler_bench.hlsl"
// Benchmark the descent in the path tracer's power-only mode (1), matching how the
// renderer actually uses the sampler, rather than the library default (mode 0).
#ifndef NBL_LIGHTCUT_TREE_WEIGHT_MODE
#define NBL_LIGHTCUT_TREE_WEIGHT_MODE 1
#endif
// No real subtree alias table here, so disable the early-stop and measure the full
// weighted descent. With it enabled the NoSubtreeAliasAccessor stub would short-circuit
// the descent to a degenerate truncated path instead of the traversal we mean to time.
#ifndef NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
#define NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED 0
#endif
#include <nbl/builtin/hlsl/sampling/stochastic_lightcut_tree.hlsl>

[[vk::push_constant]] LightcutTreePushConstants pc;

// CWBVH-4 packed wide-node: 32 B per heap entry, the library LightcutTreePackedWideNode layout
// (same one ex40 ships). We fetch the whole node in two batched uint32_t4 taps (32 B, always
// 16-byte aligned), then hand it to the library unpack.
//
//   bytes  0-15 (uint32_t4 a): originXYZ (3 fp32) + (parentPowerF16 low 16 | sharedExp<<16 | mask<<24)
//   bytes 16-31 (uint32_t4 b): childPacked[0..3]
NBL_CONSTEXPR uint32_t kNodeRecordSize  = 32u;
NBL_CONSTEXPR uint32_t kLeafRecordSize  = 32u;

struct BdaLightcutTreeNodeAccessor
{
	uint64_t addr;

	template<typename V, typename I>
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
	{
		const uint64_t base = addr + uint64_t(i) * uint64_t(kNodeRecordSize);
		const uint32_t4 a = vk::RawBufferLoad<uint32_t4>(base + 0ull,  16u);
		const uint32_t4 b = vk::RawBufferLoad<uint32_t4>(base + 16ull, 16u);

		nbl::hlsl::sampling::LightcutTreePackedWideNode packed;
		packed.origin      = asfloat(a.xyz);
		packed.powExpMask  = a.w;
		packed.childPacked = b;
		val = nbl::hlsl::sampling::lightcutTreeUnpackWideNode<float32_t>(packed);
	}
};

struct BdaLightcutTreeLeafAccessor
{
	uint64_t addr;

	template<typename V, typename I>
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
	{
		const uint64_t base = addr + uint64_t(i) * uint64_t(kLeafRecordSize);
		const uint32_t4 lo = vk::RawBufferLoad<uint32_t4>(base + 0ull,  16u);
		const uint32_t4 hi = vk::RawBufferLoad<uint32_t4>(base + 16ull, 16u);
		nbl::hlsl::sampling::LightcutTreePackedLeaf packed;
		packed.bboxMin   = asfloat(lo.xyz);
		packed.bboxMax   = float32_t3(asfloat(lo.w), asfloat(hi.x), asfloat(hi.y));
		packed.emitterID = hi.z;
		val = nbl::hlsl::sampling::lightcutTreeUnpackLeaf<float32_t>(packed);
	}
};

using BenchSubAcc = nbl::hlsl::sampling::NoSubtreeAliasAccessor<float32_t, uint32_t>;
using BenchLightcutTree = nbl::hlsl::sampling::StochasticLightcutTreeSampler<
	float32_t, uint32_t, BdaLightcutTreeNodeAccessor, BdaLightcutTreeLeafAccessor, BenchSubAcc, NBL_LIGHTCUT_TREE_WEIGHT_MODE>;

#else
// Weight mode swept by the geometric scenarios (one shader variant per mode). The
// multi/single consistency scenarios ignore it and use the library default.
#ifndef NBL_LIGHTCUT_TREE_TEST_MODE
#define NBL_LIGHTCUT_TREE_TEST_MODE NBL_LIGHTCUT_TREE_WEIGHT_MODE
#endif
#include "../common/stochastic_lightcut_tree.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<LightcutTreeInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<LightcutTreeTestResults> outputTestValues;
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;

#ifdef BENCH_ITERS
	BdaLightcutTreeNodeAccessor nodeAcc;  nodeAcc.addr = pc.nodesAddress;
	BdaLightcutTreeLeafAccessor leafAcc;  leafAcc.addr = pc.leavesAddress;
	BenchLightcutTree sampler = BenchLightcutTree::create(
		nodeAcc, leafAcc, BenchSubAcc::create(), pc.firstLeafIdx, pc.shadingPoint, pc.shadingNormal);

	float32_t xi = float32_t(nbl::hlsl::glsl::bitfieldReverse(invID)) / float32_t(~0u);
	NBL_CONSTEXPR float32_t goldenRatio = 0.6180339887498949f;
	uint32_t acc = 0u;

	NBL_HLSL_LOOP
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		xi = frac(xi + goldenRatio);
		BenchLightcutTree::cache_type cache;
		uint32_t generated = sampler.generate(xi, cache);
		acc ^= generated ^ asuint(sampler.forwardPdf(xi, cache));
	}

	vk::RawBufferStore<uint32_t>(pc.outputAddress + uint64_t(sizeof(uint32_t)) * uint64_t(invID), acc);
#else
#if   defined(NBL_LIGHTCUT_TREE_SINGLE_LEAF)
	LightcutTreeSingleLeafExecutor                                     executor;
#elif defined(NBL_LIGHTCUT_TREE_BELOW_PLANE)
	LightcutTreeBelowPlaneExecutor<NBL_LIGHTCUT_TREE_TEST_MODE>        executor;
#elif defined(NBL_LIGHTCUT_TREE_DISTANCE_FALLOFF)
	LightcutTreeDistanceFalloffExecutor<NBL_LIGHTCUT_TREE_TEST_MODE>   executor;
#elif defined(NBL_LIGHTCUT_TREE_INFLATED_BBOX)
	LightcutTreeInflatedBboxExecutor<NBL_LIGHTCUT_TREE_TEST_MODE>      executor;
#elif defined(NBL_LIGHTCUT_TREE_DEPTH2)
	LightcutTreeDepth2Executor<NBL_LIGHTCUT_TREE_TEST_MODE>            executor;
#else
	LightcutTreeMultiLeafExecutor                                     executor;
#endif
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
