#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"
#include <nbl/builtin/hlsl/sampling/alias_table_builder.h>
#include <nbl/builtin/hlsl/sampling/cumulative_probability_builder.h>
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
      Count
   };

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
   }

   void releaseTables()
   {
      m_aliasPdfBuf         = nullptr;
      m_packedAliasABuf     = nullptr;
      m_packedAliasBBuf     = nullptr;
      m_cumProbBuf          = nullptr;
      m_cumProbEytzingerBuf = nullptr;
   }

   void runSingle(uint32_t N, core::vector<core::string> name, SamplerKind kind, uint32_t warmupIterations)
   {
      // Pipeline + push constants are bound *once* in bindOnce, the inner loop is just
      // dispatch(...). Putting binds inside dispatchOne would inflate ps/sample on the
      // tighter samplers.
      const PipelineEntry& pe = m_pipelines[m_pipelineIdx[size_t(kind)]];

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
               defaultBindAndPush(cb, pe, pc);
            }
            else
            {
               CumProbPushConstants pc  = {};
               const auto&          buf = (kind == SamplerKind::CumProbEytzinger) ? m_cumProbEytzingerBuf : m_cumProbBuf;
               pc.cumProbAddress        = buf->getDeviceAddress();
               pc.outputAddress         = m_outputBuf->getDeviceAddress();
               pc.tableSize             = N;
               defaultBindAndPush(cb, pe, pc);
            }
         },
         [this](IGPUCommandBuffer* cb) { defaultDispatch(cb); },
         samplesForCurrentRow());

      record(std::move(name), timingResult, pe.stats);
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

   // Shared
   core::smart_refctd_ptr<IGPUBuffer> m_outputBuf;
   uint32_t                           m_currentN    = 0;
   uint32_t                           m_aliasTableN = 0;
   std::span<const uint32_t>          m_sweepNs;
};

#endif
