#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"
#include <nbl/builtin/hlsl/sampling/alias_table_builder.h>
#include <nbl/builtin/hlsl/sampling/cumulative_probability_builder.h>
#include <nbl/builtin/hlsl/sampling/stochastic_lightcut_tree.hlsl> // shared 32 B pack contract
#include "app_resources/common/discrete_sampler_bench.hlsl"
#include "nbl/examples/Benchmark/IBenchmark.h"
#include "nbl/examples/Benchmark/GPUBenchmarkHelper.h"

#include <random>

using namespace nbl;

class CDiscreteSamplerBenchmark : public GPUBenchmark
{
   public:
   // Declared up-front because it's used as the index domain for m_pipelineIdx[]
   // (a member-array bound needs the type complete in declaration order).
   enum class SamplerKind : uint32_t
   {
      AliasPackedA = 0,
      AliasPackedB,
      CumProbCompare,
      CumProbYolo,
      CumProbEytzinger,
      LightcutTree,
      Count
   };

   // CWBVH-4 packed records: same canonical 32 B layout the renderer uses, from the builtin library. 
   // The bench fills them via the library pack helpers
   // so it benchmarks the exact encode/decode the renderer ships, with no drifting copy.
   using LightcutTreeWideNodeRecord = nbl::hlsl::sampling::LightcutTreePackedWideNode;
   using LightcutTreeLeafRecord     = nbl::hlsl::sampling::LightcutTreePackedLeaf;
   static_assert(sizeof(LightcutTreeWideNodeRecord) == 32, "Wide-node record must be 32 B (CWBVH-4)");
   static_assert(sizeof(LightcutTreeLeafRecord) == 32, "Leaf record must be 32 B");

   struct SetupData
   {
      core::smart_refctd_ptr<IAssetManager> assetMgr;
      // Each pipeline is independent; main.cpp can pick precompiled or runtime per
      // pipeline by passing ShaderVariant::Precompiled(get_spirv_key<...>()) or
      // ShaderVariant::FromSource(path, defines) respectively.
      GPUBenchmarkHelper::ShaderVariant packedAliasAVariant;
      GPUBenchmarkHelper::ShaderVariant packedAliasBVariant;
      GPUBenchmarkHelper::ShaderVariant cumProbVariant;
      GPUBenchmarkHelper::ShaderVariant cumProbYoloVariant;
      GPUBenchmarkHelper::ShaderVariant cumProbEytzingerVariant;
      GPUBenchmarkHelper::ShaderVariant lightcutTreeVariant;
      hlsl::uint32_t3                   dispatchGroupCount;
      uint64_t                          targetBudgetMs = 400; // wall-clock budget per sweep row
      // N values the sweep cycles through. Dispatch count per row is auto-sized
      // by runTimedBudgeted to hit the budget.
      std::span<const uint32_t> sweepNs;
   };

   // Shape is derivable from SetupData; expose it so the caller can use it
   // both to configure the bench and to build the matching RunContext for the
   // span that runs this bench
   static WorkloadShape shapeFor(const SetupData& data)
   {
      const uint32_t totalThreads       = data.dispatchGroupCount.x * data.dispatchGroupCount.y * data.dispatchGroupCount.z * WORKGROUP_SIZE;
      const uint64_t samplesPerDispatch = uint64_t(totalThreads) * uint64_t(BENCH_ITERS);
      return {
         .workgroupSize      = {WORKGROUP_SIZE, 1u, 1u},
         .dispatchGroupCount = data.dispatchGroupCount,
         .samplesPerDispatch = samplesPerDispatch,
      };
   }

   CDiscreteSamplerBenchmark(Aggregator& aggregator, const SetupData& data)
      : GPUBenchmark(aggregator, GPUBenchmark::SetupData{
                                    .name             = {}, // per-row names synthesized at run time
                                    .warmupDispatches = 0,
                                    .shape            = shapeFor(data),
                                    .targetBudgetMs   = data.targetBudgetMs,
                                 })
   {
      const uint32_t totalThreads = data.dispatchGroupCount.x * data.dispatchGroupCount.y * data.dispatchGroupCount.z * WORKGROUP_SIZE;

      m_assetMgr = data.assetMgr;
      m_sweepNs  = data.sweepNs;

      for (const uint32_t N : m_sweepNs)
      {
         const std::string nStr = std::format("N={}", N);
         for (const auto& v : kSweepVariants)
            registerVariant({nStr, v.family, v.leaf});
      }

      // Shared output buffer (size only depends on thread count). GPU writes via BDA and
      // nothing reads it on the CPU.
      m_outputBuf = createBdaOutputBuffer(totalThreads * sizeof(uint32_t)).buf;

      // Pipelines (N-independent; only push constants change per run). Indices
      // into m_pipelines (GPUBenchmarkHelper) are stored in the same order as SamplerKind
      // so the sweep's variant table can index by enum directly.
      m_pipelineIdx[static_cast<size_t>(SamplerKind::AliasPackedA)]     = createPipeline(data.packedAliasAVariant, m_assetMgr, sizeof(PackedAliasABPushConstants), "alias-packed-A");
      m_pipelineIdx[static_cast<size_t>(SamplerKind::AliasPackedB)]     = createPipeline(data.packedAliasBVariant, m_assetMgr, sizeof(PackedAliasABPushConstants), "alias-packed-B");
      m_pipelineIdx[static_cast<size_t>(SamplerKind::CumProbCompare)]   = createPipeline(data.cumProbVariant, m_assetMgr, sizeof(CumProbPushConstants), "cumprob-comparator");
      m_pipelineIdx[static_cast<size_t>(SamplerKind::CumProbYolo)]      = createPipeline(data.cumProbYoloVariant, m_assetMgr, sizeof(CumProbPushConstants), "cumprob-yolo");
      m_pipelineIdx[static_cast<size_t>(SamplerKind::CumProbEytzinger)] = createPipeline(data.cumProbEytzingerVariant, m_assetMgr, sizeof(CumProbPushConstants), "cumprob-eytzinger");
      m_pipelineIdx[static_cast<size_t>(SamplerKind::LightcutTree)]     = createPipeline(data.lightcutTreeVariant, m_assetMgr, sizeof(LightcutTreePushConstants), "lightcut-tree");
   }

   // Rows are synthesized per (N, variant), not a single named entry, so
   // each row checks cli.focusVariants individually. The aggregator's silent
   // flag selects which half (focused / unfocused) we contribute to.
   void run() override
   {
      const bool focusedPhase = isFocusPhase();
      // Warmup is small and fixed; budgeted measurement auto-sizes the
      // measured-dispatch count to hit getTargetBudgetMs().
      constexpr uint32_t kWarmupDispatches = 64;

      for (const uint32_t N : m_sweepNs)
      {
         const std::string nStr = std::format("N={}", N);
         bool              built = false;
         for (const auto& [family, leaf, kind] : kSweepVariants)
         {
            core::vector<core::string> name      = {nStr, family, leaf};
            const bool                 inFocus   = isFocused(name);
            const bool                 shouldRun = focusedPhase ? inFocus : !inFocus;
            if (!shouldRun)
               continue;
            if (!built)
            {
               buildAndUpload(N);
               built = true;
            }
            runSingle(N, std::move(name), kind, kWarmupDispatches);
         }
         if (built)
            releaseTables();
      }
   }

   private:
   // (family, leaf, kind) for every variant the sweep runs.
   struct SweepVariant
   {
      const char* family; // e.g. "AliasTable"
      const char* leaf;   // e.g. "packed A, 4 B"
      SamplerKind kind;
   };
   static constexpr SweepVariant kSweepVariants[] = {
      {"AliasTable", "packed A, 4 B", SamplerKind::AliasPackedA},
      {"AliasTable", "packed B, 8 B", SamplerKind::AliasPackedB},
      {"CumulativeProbability", "comparator", SamplerKind::CumProbCompare},
      {"CumulativeProbability", "YOLO", SamplerKind::CumProbYolo},
      {"CumulativeProbability", "Eytzinger", SamplerKind::CumProbEytzinger},
      {"StochasticLightcutTree", "4-ary heap, unpacked BDA", SamplerKind::LightcutTree},
   };

   void buildAndUpload(const uint32_t N)
   {
      m_currentN = N;

      std::vector<float>                    weights(N);
      std::mt19937                          rng(42u + N);
      std::uniform_real_distribution<float> dist(0.001f, 100.0f);
      for (uint32_t i = 0; i < N; i++)
         weights[i] = dist(rng);

      // Build the alias table SoA (intermediate form), then pack it for variants A and B.
      // Builder may pad PoT N to N+1 for cache-friendly stride; returned size drives
      // every downstream buffer / push-constant value.
      std::vector<float>    aliasProb;
      std::vector<uint32_t> aliasIdx;
      std::vector<float>    aliasPdf;
      m_aliasTableN = sampling::AliasTableBuilder<float>::build({weights}, aliasProb, aliasIdx, aliasPdf);

      constexpr uint32_t                              kPackedLog2N = 26u;
      std::vector<uint32_t>                           packedA(m_aliasTableN);
      std::vector<sampling::PackedAliasEntryB<float>> packedB(m_aliasTableN);
      sampling::AliasTableBuilder<float>::packA<kPackedLog2N>({aliasProb}, {aliasIdx}, packedA.data());
      sampling::AliasTableBuilder<float>::packB<kPackedLog2N>({aliasProb}, {aliasIdx}, {aliasPdf}, packedB.data());

      // Cumulative probability (N-1 entries, last bucket implicitly 1.0)
      std::vector<float> cumProb(N - 1u);
      sampling::computeNormalizedCumulativeHistogram({weights}, cumProb.data());

      // Eytzinger level-order tree: 2*P entries where P = nextPot(N)
      const uint32_t     eytzingerP        = sampling::eytzingerLeafCount(N);
      const uint32_t     eytzingerTreeSize = 2u * eytzingerP;
      std::vector<float> cumProbEytzinger(eytzingerTreeSize);
      sampling::buildEytzinger({weights}, cumProbEytzinger.data());

      m_aliasPdfBuf         = createBdaBuffer(aliasPdf.data(), m_aliasTableN * sizeof(float));
      m_packedAliasABuf     = createBdaBuffer(packedA.data(), m_aliasTableN * sizeof(uint32_t));
      m_packedAliasBBuf     = createBdaBuffer(packedB.data(), m_aliasTableN * sizeof(sampling::PackedAliasEntryB<float>));
      m_cumProbBuf          = createBdaBuffer(cumProb.data(), (N - 1u) * sizeof(float));
      m_cumProbEytzingerBuf = createBdaBuffer(cumProbEytzinger.data(), eytzingerTreeSize * sizeof(float));

      // Build a 4-ary light-cut tree with N synthetic leaves at random positions
      // in a unit cube around the origin. Powers reuse the alias table's weight
      // distribution so the same N gives comparable signal scale. Leaves are
      // padded to the next power of 4 so the heap is complete.
      buildLightcutTree(N, weights);
   }

   void releaseTables()
   {
      m_aliasPdfBuf         = nullptr;
      m_packedAliasABuf     = nullptr;
      m_packedAliasBBuf     = nullptr;
      m_cumProbBuf          = nullptr;
      m_cumProbEytzingerBuf = nullptr;
      m_lightcutNodesBuf    = nullptr;
      m_lightcutLeavesBuf   = nullptr;
   }

   // Pad to next power of 4; min 1.
   static uint32_t nextPowerOf4(uint32_t n)
   {
      if (n <= 1) return 1;
      uint32_t p = std::bit_ceil(n);
      if ((std::countr_zero(p) & 1u) != 0u)
         p <<= 1u;
      return p;
   }

   // Build a 4-ary CWBVH-4 packed light-cut tree. Two phases:
   //  1) bottom-up bbox-union + power-sum into a temporary decoded heap;
   //  2) per-wide-node, quantise the 4 children's bboxes / powers into a 32 B
   //     packed record and emit. Mirrors ex40's CLightTree.cpp builder.
   void buildLightcutTree(uint32_t N, const std::vector<float>& powers)
   {
      const uint32_t numLeavesPadded = nextPowerOf4(N);
      const uint32_t numNodes        = (numLeavesPadded - 1u) / 3u;
      m_lightcutFirstLeafIdx         = numNodes;
      m_lightcutNumLeaves            = N;

      // Decoded heap: leaves at [numNodes, numNodes+numLeavesPadded); internals at [0, numNodes).
      struct Decoded { float mn[3]; float mx[3]; float power; };
      const uint32_t totalHeap = numNodes + numLeavesPadded;
      std::vector<Decoded> heap(totalHeap, Decoded{});

      std::vector<LightcutTreeLeafRecord> leaves(numLeavesPadded, LightcutTreeLeafRecord{});
      std::mt19937 rng(0xC0FFEEu + N);
      std::uniform_real_distribution<float> posDist(-1.0f, 1.0f);
      constexpr float kHalfExt = 0.025f;
      for (uint32_t i = 0; i < numLeavesPadded; ++i)
      {
         Decoded& d = heap[numNodes + i];
         if (i < N)
         {
            const float cx = posDist(rng), cy = posDist(rng), cz = posDist(rng);
            d.mn[0] = cx - kHalfExt; d.mn[1] = cy - kHalfExt; d.mn[2] = cz - kHalfExt;
            d.mx[0] = cx + kHalfExt; d.mx[1] = cy + kHalfExt; d.mx[2] = cz + kHalfExt;
            d.power = powers[i];
            leaves[i].bboxMin = hlsl::float32_t3(d.mn[0], d.mn[1], d.mn[2]);
            leaves[i].bboxMax = hlsl::float32_t3(d.mx[0], d.mx[1], d.mx[2]);
            leaves[i].emitterID  = i;
         }
         else
         {
            d.power = 0.f;
            // Padding: bbox stays zero; lightcutTreeChildWeight short-circuits on power<=0.
            leaves[i].emitterID = nbl::hlsl::sampling::LightcutTreePackedNoEmitter;
         }
      }

      // Bottom-up merge: each internal node is the bbox-union + power-sum of its 4 children.
      for (int32_t W = int32_t(numNodes) - 1; W >= 0; --W)
      {
         Decoded& p = heap[uint32_t(W)];
         p.mn[0] = p.mn[1] = p.mn[2] = +std::numeric_limits<float>::infinity();
         p.mx[0] = p.mx[1] = p.mx[2] = -std::numeric_limits<float>::infinity();
         p.power = 0.f;
         for (uint32_t s = 0; s < 4; ++s)
         {
            const Decoded& c = heap[4u * uint32_t(W) + 1u + s];
            if (!(c.power > 0.f)) continue;
            for (uint32_t a = 0; a < 3; ++a)
            {
               p.mn[a] = std::min(p.mn[a], c.mn[a]);
               p.mx[a] = std::max(p.mx[a], c.mx[a]);
            }
            p.power += c.power;
         }
         if (!(p.power > 0.f))
         {
            p.mn[0] = p.mn[1] = p.mn[2] = 0.f;
            p.mx[0] = p.mx[1] = p.mx[2] = 0.f;
         }
      }

      // Encode each wide-node in CWBVH-4 packed form.
      std::vector<LightcutTreeWideNodeRecord> wideNodes(numNodes, LightcutTreeWideNodeRecord{});
      for (uint32_t W = 0; W < numNodes; ++W)
      {
         namespace pk = nbl::hlsl::sampling;
         const Decoded& parent = heap[W];
         const float ext[3] = { parent.mx[0] - parent.mn[0], parent.mx[1] - parent.mn[1], parent.mx[2] - parent.mn[2] };
         // One shared exponent for all 3 axes (largest extent picks the grid). Same library pack
         // contract as the renderer, including ceil+floor-to-1 relative power.
         const uint32_t expS  = pk::lightcutTreePickBiasedExp(std::max({ext[0], ext[1], ext[2]}));
         const float    scale = pk::lightcutTreeBiasedExpToScale(expS);

         const float parentPowerSafe = parent.power > 0.f ? parent.power : 1.f;
         uint32_t     childLeafMask  = 0u;
         uint32_t     childPacked[4] = {0u, 0u, 0u, 0u};
         for (uint32_t s = 0; s < 4; ++s)
         {
            const uint32_t childHeap = 4u * W + 1u + s;
            if (childHeap >= numNodes)
               childLeafMask |= (1u << s);
            const Decoded& ch = heap[childHeap];
            const hlsl::float32_t3 loRel(ch.mn[0] - parent.mn[0], ch.mn[1] - parent.mn[1], ch.mn[2] - parent.mn[2]);
            const hlsl::float32_t3 hiRel(ch.mx[0] - parent.mn[0], ch.mx[1] - parent.mn[1], ch.mx[2] - parent.mn[2]);
            childPacked[s] = pk::lightcutTreePackChild(loRel, hiRel, scale, ch.power, parentPowerSafe);
         }

         LightcutTreeWideNodeRecord& wn = wideNodes[W];
         wn.origin      = hlsl::float32_t3(parent.mn[0], parent.mn[1], parent.mn[2]);
         wn.powExpMask  = pk::lightcutTreePackPowExpMask(parent.power, expS, childLeafMask);
         wn.childPacked = hlsl::uint32_t4(childPacked[0], childPacked[1], childPacked[2], childPacked[3]);
      }

      m_lightcutNodesBuf  = createBdaBuffer(wideNodes.data(), wideNodes.size() * sizeof(LightcutTreeWideNodeRecord));
      m_lightcutLeavesBuf = createBdaBuffer(leaves.data(),    leaves.size()    * sizeof(LightcutTreeLeafRecord));
   }

   void runSingle(uint32_t N, core::vector<core::string> name, SamplerKind kind, uint32_t warmupIterations)
   {
      // Pipeline + push constants are bound *once* in bindOnce, the inner loop is just
      // dispatch(...). Putting binds inside dispatchOne would inflate ps/sample on the
      // tighter samplers.
      const PipelineEntry* pe = getPipelineEntry(m_pipelineIdx[size_t(kind)], joinName(name));
      if (!pe)
         return;

      const TimingResult timingResult = runTimedBudgeted(warmupIterations, getTargetBudgetMs(),
         [&](IGPUCommandBuffer* cb)
         {
            if (kind == SamplerKind::AliasPackedA || kind == SamplerKind::AliasPackedB)
            {
               PackedAliasABPushConstants pc = {};
               pc.entriesAddress             = (kind == SamplerKind::AliasPackedA ? m_packedAliasABuf : m_packedAliasBBuf)->getDeviceAddress();
               pc.pdfAddress                 = m_aliasPdfBuf->getDeviceAddress();
               pc.outputAddress              = m_outputBuf->getDeviceAddress();
               pc.tableSize                  = m_aliasTableN;
               defaultBindAndPush(cb, *pe, pc);
            }
            else if (kind == SamplerKind::LightcutTree)
            {
               LightcutTreePushConstants pc = {};
               pc.nodesAddress              = m_lightcutNodesBuf->getDeviceAddress();
               pc.leavesAddress             = m_lightcutLeavesBuf->getDeviceAddress();
               pc.outputAddress             = m_outputBuf->getDeviceAddress();
               pc.firstLeafIdx              = m_lightcutFirstLeafIdx;
               pc.numLeaves                 = m_lightcutNumLeaves;
               // Shading point at origin, normal +Y; matches the synthetic tree's
               // leaf cube so weights are well-conditioned for every variant of N.
               pc.shadingPoint              = hlsl::float32_t3(0.f, 0.f, 0.f);
               pc.shadingNormal             = hlsl::float32_t3(0.f, 1.f, 0.f);
               defaultBindAndPush(cb, *pe, pc);
            }
            else
            {
               CumProbPushConstants pc  = {};
               const auto&          buf = (kind == SamplerKind::CumProbEytzinger) ? m_cumProbEytzingerBuf : m_cumProbBuf;
               pc.cumProbAddress        = buf->getDeviceAddress();
               pc.outputAddress         = m_outputBuf->getDeviceAddress();
               pc.tableSize             = N;
               defaultBindAndPush(cb, *pe, pc);
            }
         },
         [this](IGPUCommandBuffer* cb) { defaultDispatch(cb); },
         samplesForCurrentRow());

      record(std::move(name), timingResult, pe->stats);
   }

   core::smart_refctd_ptr<IAssetManager> m_assetMgr;

   // Indices into m_pipelines (GPUBenchmarkHelper), indexed by SamplerKind.
   uint32_t m_pipelineIdx[size_t(SamplerKind::Count)] = {};

   // Per-N data buffers (rebuilt each sweep step). pdf[] is shared between A and B.
   core::smart_refctd_ptr<IGPUBuffer> m_aliasPdfBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_packedAliasABuf;
   core::smart_refctd_ptr<IGPUBuffer> m_packedAliasBBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_cumProbBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_cumProbEytzingerBuf;

   // Light-cut tree buffers + heap metadata (firstLeafIdx + unpadded N).
   core::smart_refctd_ptr<IGPUBuffer> m_lightcutNodesBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_lightcutLeavesBuf;
   uint32_t                           m_lightcutFirstLeafIdx = 0;
   uint32_t                           m_lightcutNumLeaves    = 0;

   // Shared
   core::smart_refctd_ptr<IGPUBuffer> m_outputBuf;
   uint32_t                           m_currentN    = 0;
   uint32_t                           m_aliasTableN = 0;
   std::span<const uint32_t>          m_sweepNs;
};

#endif
