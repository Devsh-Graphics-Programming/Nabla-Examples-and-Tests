// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_40_PATHTRACER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_40_PATHTRACER_BENCHMARK_INCLUDED_

#include "nbl/examples/Benchmark/IBenchmark.h"

#include "renderer/CRenderer.h"
#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <vector>


// Timing model per row:
//   1. Run getWarmupDispatches() untimed frames (first-frame compile, caches).
//   2. Pilot a fixed N to get ns/frame, then converge N up or down so the
//      window covers ~getTargetBudgetMs().
//   3. If samplesForCurrentRow() > 1 (focus mode), take K windows at that N
//      and return the median.
class CPathTracerBenchmark : public IBenchmark
{
public:
   struct SetupData
   {
      nbl::core::vector<nbl::core::string> name;
      WorkloadShape                        shape;
      uint64_t                             targetBudgetMs   = 50000;
      uint32_t                             warmupDispatches = 5;

      // When non-zero, run exactly this many accumulating timed frames instead of
      // converging to targetBudgetMs. Gives every row the SAME spp (= frames *
      // samplesPerDispatch), so FLIP isolates per-sample variance (selection quality)
      // rather than throughput, a faster selector no longer banks extra samples.
      uint32_t                             fixedTimedFrames = 0;

      nbl::this_example::CRenderer* renderer = nullptr;
      nbl::this_example::CSession*  session  = nullptr;

      // Optional per-row override of the renderer's NEE-technique toggle.
      // -1 = leave the renderer's current value untouched.
      //  0 = light-tree descent.    1 = alias-table emitter pick.
      int useAliasNEE = -1;

      // Optional per-row override of the Beauty MIS-mode pipeline variant (separate SPIR-V each).
      // -1 = leave untouched; otherwise a CSession::MisMode value (0=NEEOnly, 1=BxDFOnly, 2=Both).
      int misMode = -1;

      int leafSampler = -1;

      // Optional per-row NEE-architecture override: -1 = leave untouched.
      int deferredNEE = -1;

      // Optional per-row batched band count (the request-buffer memory knob); -1 = leave untouched.
      int neeBands = -1;

      // Fires once per row, AFTER the K-sample median is collected and BEFORE
      // cooldown (so the image still reflects the timed configuration). Used
      // by main.cpp to dump a beauty EXR alongside the bench JSON for offline
      // image comparisons (FLIP).
      std::function<void()> onAfterTimedFrames = nullptr;
   };

   CPathTracerBenchmark(Aggregator& aggregator, const SetupData& data)
      : IBenchmark(aggregator, data.name, data.warmupDispatches, data.shape, data.targetBudgetMs)
      , m_renderer(data.renderer)
      , m_session(data.session)
      , m_fixedTimedFrames(data.fixedTimedFrames)
      , m_useAliasNEE(data.useAliasNEE)
      , m_misMode(data.misMode)
      , m_leafSampler(data.leafSampler)
      , m_deferredNEE(data.deferredNEE)
      , m_neeBands(data.neeBands)
      , m_onAfterTimedFrames(data.onAfterTimedFrames)
      , m_device(aggregator.getLogicalDevice().get())
      , m_physicalDevice(aggregator.getPhysicalDevice())
      , m_logger(aggregator.getLogger())
   {
      nbl::video::IQueryPool::SCreationParams qparams = {};
      qparams.queryType                               = nbl::video::IQueryPool::TYPE::TIMESTAMP;
      qparams.queryCount                              = 2;
      qparams.pipelineStatisticsFlags                 = nbl::video::IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
      m_queryPool                                     = m_device->createQueryPool(qparams);
      if (!m_queryPool)
         m_logger->log("CPathTracerBenchmark: failed to create timestamp query pool", nbl::system::ILogger::ELL_ERROR);

      m_timestampPeriodNs = double(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);

      // Pull executable stats off the RT pipeline that handles this session's
      // render mode. Requires the device's pipelineExecutableInfo feature to
      // have been enabled at pipeline-create time.
      if (m_session)
      {
         const auto mode = m_session->getConstructionParams().mode;
         // Report stats for THIS row's (MIS-mode, alias/tree) variant pipeline (separate SPIR-V), so the
         // per-variant register/occupancy difference is visible, the reason these are distinct pipelines.
         const auto statMisMode  = m_misMode >= 0 ? static_cast<nbl::this_example::CSession::MisMode>(m_misMode) : nbl::this_example::CSession::MisMode::Both;
         const bool statUseAlias = m_useAliasNEE >= 0 ? (m_useAliasNEE != 0) : m_renderer->getUseAliasNEE();
         const auto statSampler  = m_leafSampler >= 0 ? static_cast<nbl::this_example::CSession::LightSampler>(m_leafSampler) : m_renderer->getLeafSampler();
         if (auto* const scene = m_session->getConstructionParams().scene.get())
            if (auto* const pipeline = scene->getPipeline(mode, statMisMode, statUseAlias, statSampler))
            {
               const auto infos = pipeline->getExecutableInfo();
               benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_INFO, "CPathTracerBenchmark: pipeline getExecutableInfo() -> {} entries", infos.size());
               extractPipelineStats(pipeline, m_stats);
            }
      }

      if (!m_stats.raw.empty())
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE, "{} pipeline executable report:\n{}", joinName(m_name), m_stats.raw);
   }

   uint32_t getSubgroupSize() const { return m_stats.subgroupSize; }
   const PipelineStats& getStats() const { return m_stats; }

protected:
   void doRun() override
   {
      if (!m_renderer || !m_session || !m_queryPool)
      {
         m_logger->log("CPathTracerBenchmark::doRun: missing renderer/session/queryPool", nbl::system::ILogger::ELL_ERROR);
         return;
      }

      // Apply per-row renderer toggles before any frames run (including warmup). The leaf sampler only
      // selects the pipeline variant here; its scene/buffers were already rebuilt by the caller.
      if (m_useAliasNEE >= 0)
         m_renderer->setUseAliasNEE(m_useAliasNEE != 0);
      if (m_misMode >= 0)
         m_renderer->setMisMode(static_cast<nbl::this_example::CSession::MisMode>(m_misMode));
      if (m_leafSampler >= 0)
         m_renderer->setLeafSampler(static_cast<nbl::this_example::CSession::LightSampler>(m_leafSampler));
      if (m_deferredNEE >= 0)
         m_renderer->setDeferredNEEMode(static_cast<nbl::this_example::CRenderer::DeferredNEEMode>(m_deferredNEE));
      if (m_neeBands >= 1)
         m_renderer->setNeeBandCount(uint32_t(m_neeBands));

      // 1) Warmup
      for (uint32_t i = 0; i < getWarmupDispatches(); ++i)
         renderOnce(false);

      // 2) Either run a fixed frame count (equal-spp comparison) or render for a wall-time budget.
      uint32_t lastN;
      double   elapsed_ns;
      const auto wallStart = std::chrono::steady_clock::now();
      if (m_fixedTimedFrames > 0u)
      {
         lastN      = m_fixedTimedFrames;
         elapsed_ns = runN(lastN);
      }
      else
      {
         const double   targetNs = double(getTargetBudgetMs()) * 1'000'000.0;
         // Past MaxSPP/spp frames accumulation saturates and every further frame is a no-op phantom
         // frame at near-zero GPU time, which spins this loop without progress. Cap there, and to run
         // longer raise MAX_SPP_LOG2 rather than the budget.
         const uint32_t spp   = std::max<uint32_t>(1u, m_renderer->getMaxSppPerDispatch());
         const uint32_t kMaxN = std::max<uint32_t>(1u, (nbl::this_example::MaxSPP + spp - 1u) / spp);
         lastN      = 0u;
         elapsed_ns = 0.0;
         while (elapsed_ns < targetNs && lastN < kMaxN)
         {
            elapsed_ns += renderOnce(true, lastN == 0u); // first frame clears, the rest accumulate
            ++lastN;
         }
      }

      // A low summed-interval / wall ratio means the timestamp bracket under-measures frames, which
      // inflates the dispatch count and deflates ps/sample. Wall includes CPU record and fence waits,
      // so expect ~0.85-1.0, not exactly 1.
      {
         const double wall_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wallStart).count());
         m_logger->log("%s: timed window %u frames, gpu sum %.1f ms, wall %.1f ms, gpu/wall %.3f",
            nbl::system::ILogger::ELL_INFO, m_name, lastN, elapsed_ns * 1e-6, wall_ns * 1e-6, wall_ns > 0.0 ? elapsed_ns / wall_ns : 0.0);
      }

      // 3) K-sample median (focus mode only, regular runs return samples=1)
      const uint32_t samples = samplesForCurrentRow();
      double         result_ns = elapsed_ns;
      if (samples > 1u)
      {
         std::vector<double> ns;
         ns.reserve(samples);
         ns.push_back(elapsed_ns);
         for (uint32_t i = 1u; i < samples; ++i)
            ns.push_back(runN(lastN));
         std::sort(ns.begin(), ns.end());
         result_ns = ns[ns.size() / 2];
      }

      // Beauty image still reflects the timed configuration here, the cooldown
      // pass below would issue another fresh-frame render and clobber it. Hook
      // fires synchronously; main.cpp uses it to dump a per-row EXR.
      if (m_onAfterTimedFrames)
         m_onAfterTimedFrames();

      // 4) Cooldown (matches helper's symmetric warmup/cooldown shape)
      for (uint32_t i = 0; i < getWarmupDispatches(); ++i)
         renderOnce(false);

      const uint64_t samplesPerDispatch = getShape().samplesPerDispatch;
      TimingResult   t {};
      t.elapsed_ns                     = result_ns;
      t.totalSamples                   = uint64_t(lastN) * samplesPerDispatch;
      t.ps_per_sample                  = t.totalSamples ? result_ns * 1e3 / double(t.totalSamples) : 0.0;
      t.gsamples_per_s                 = result_ns > 0.0 ? double(t.totalSamples) / result_ns : 0.0;
      t.ms_total                       = result_ns * 1e-6;

      record(m_name, t, m_stats);
   }

private:
   double runN(uint32_t N)
   {
      double total_ns = 0.0;
      for (uint32_t i = 0u; i < N; ++i)
         total_ns += renderOnce(true, i == 0u); // first frame of the window clears, the rest accumulate onto it
      return total_ns;
   }

   double renderOnce(bool timed, bool clearAccumulation = false)
   {
      using namespace nbl;

      nbl::this_example::CRenderer::STimingScope scope = {};
      if (timed)
      {
         scope.queryPool      = m_queryPool.get();
         scope.startQueryIdx  = 0;
         scope.endQueryIdx    = 1;
         scope.forceFreshFrame = clearAccumulation;
         scope.forceAccumulate = !clearAccumulation;
         scope.maxSPPOverride  = nbl::this_example::MaxSPP;
      }
      else
      {
         scope.forceFreshFrame = true;
      }

      auto submit = m_renderer->render(m_session, scope);
      if (!submit)
      {
         m_logger->log("CPathTracerBenchmark: render() returned empty SSubmit", system::ILogger::ELL_ERROR);
         return 0.0;
      }
      const auto sem = submit({});
      if (!sem.semaphore)
      {
         m_logger->log("CPathTracerBenchmark: SSubmit::operator() failed", system::ILogger::ELL_ERROR);
         return 0.0;
      }

      const video::ISemaphore::SWaitInfo waits[] = {{.semaphore = sem.semaphore, .value = sem.value}};
      if (m_device->blockForSemaphores(waits) != video::ISemaphore::WAIT_RESULT::SUCCESS)
         return 0.0;

      if (!timed)
         return 0.0;

      uint64_t ts[2] = {};
      const auto flags = core::bitflag(video::IQueryPool::RESULTS_FLAGS::_64_BIT) | core::bitflag(video::IQueryPool::RESULTS_FLAGS::WAIT_BIT);
      if (!m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, ts, sizeof(uint64_t), flags))
         return 0.0;

      return double(ts[1] - ts[0]) * m_timestampPeriodNs;
   }

   nbl::this_example::CRenderer*                       m_renderer        = nullptr;
   nbl::this_example::CSession*                        m_session         = nullptr;
   uint32_t                                            m_fixedTimedFrames = 0;
   int                                                 m_useAliasNEE     = -1;
   int                                                 m_misMode         = -1;
   int                                                 m_leafSampler     = -1;
   int                                                 m_deferredNEE     = -1;
   int                                                 m_neeBands        = -1;
   std::function<void()>                               m_onAfterTimedFrames;
   nbl::video::ILogicalDevice*                         m_device          = nullptr;
   nbl::video::IPhysicalDevice*                        m_physicalDevice  = nullptr;
   nbl::core::smart_refctd_ptr<nbl::system::ILogger>   m_logger;
   nbl::core::smart_refctd_ptr<nbl::video::IQueryPool> m_queryPool;
   double                                              m_timestampPeriodNs = 1.0;
   PipelineStats                                       m_stats {};
};

#endif
