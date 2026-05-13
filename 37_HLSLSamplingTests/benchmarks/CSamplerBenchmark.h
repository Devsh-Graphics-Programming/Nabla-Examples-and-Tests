// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_SAMPLER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_SAMPLER_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"
#include "nbl/examples/Benchmark/IBenchmark.h"
#include "nbl/examples/Benchmark/GPUBenchmarkHelper.h"
#include "app_resources/common/sampler_bench_pc.hlsl"

using namespace nbl;

// Measures GPU execution time of a sampler shader using GPU timestamp queries.
// Output is implicit BDA addressed via SamplerBenchPushConstants. GPU plumbing
// (pipeline / buffer / timestamp queries) comes from GPUBenchmarkHelper; the
// bench-side glue here is PC layout + per-run dispatch + result recording.
class CSamplerBenchmark : public GPUBenchmark
{
   public:
   struct SetupData : GPUBenchmark::SetupData
   {
      core::smart_refctd_ptr<asset::IAssetManager> assetMgr;
      GPUBenchmarkHelper::ShaderVariant            variant; // precompiled key OR source path + defines
      size_t                                       outputBufferBytes; // sizeof(uint32_t) * threadsPerDispatch
   };

   CSamplerBenchmark(Aggregator& aggregator, const SetupData& data)
      : GPUBenchmark(aggregator, data) // slicing-copy of the GPUBenchmark::SetupData base
   {
      auto bda        = createBdaOutputBuffer(data.outputBufferBytes);
      m_outputBuf     = std::move(bda.buf);
      m_outputAddress = bda.address;

      m_pipelineIdx = createPipeline(data.variant, data.assetMgr, sizeof(SamplerBenchPushConstants), joinName(data.name));
   }

   void doRun() override
   {
      const PipelineEntry&      pe = m_pipelines[m_pipelineIdx];
      SamplerBenchPushConstants pc = {};
      pc.outputAddress             = m_outputAddress;

      const TimingResult t = runTimedBudgeted(getWarmupDispatches(), getTargetBudgetMs(),
         [&](video::IGPUCommandBuffer* cb) { defaultBindAndPush(cb, pe, pc); },
         [this](video::IGPUCommandBuffer* cb) { defaultDispatch(cb); },
         samplesForCurrentRow());

      record(m_name, t, pe.stats);
   }

   private:
   core::smart_refctd_ptr<video::IGPUBuffer> m_outputBuf;
   uint64_t                                  m_outputAddress = 0;
   uint32_t                                  m_pipelineIdx   = 0;
};

#endif
