// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_COMMON_I_BENCHMARK_INCLUDED_
#define _NBL_COMMON_I_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/Benchmark/BenchmarkTypes.h"
#include "nbl/examples/Benchmark/BenchmarkConsole.h"
#include "nbl/examples/Benchmark/GPUBenchmarkHelper.h"
#include "nbl/examples/Benchmark/BenchmarkJson.h"
#include "nbl/examples/Benchmark/BenchmarkCli.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <concepts>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>


struct RunContext
{
   WorkloadShape shape;
   uint64_t      targetBudgetMs = 400; // wall-clock budget per row
   std::string   sectionLabel   = "Benchmarks";
};

// Typical use:
//
//   Aggregator agg(logger, logicalDevice, physicalDevice, computeFamilyIndex);
//   agg.applyCli({.argv = argv, .defaultOutputPath = "Bench.json"});
//   const RunContext myCtx{.shape = ..., .targetBudgetMs = 400, .sectionLabel = "..."};
//   std::vector<MyBench> benches;
//   for (...) benches.emplace_back(agg, MyBench::SetupData{...});
//   MyOtherBench other(agg, MyOtherBench::SetupData{...});
//   agg.runSessionAndReport(
//      Aggregator::Span<MyBench>{std::span(benches), myCtx},
//      Aggregator::Span<MyOtherBench>{std::span(&other, 1), otherCtx});
class Aggregator
{
   friend class IBenchmark;

public:
   Aggregator() = default;

   Aggregator(nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger,
      nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>   logicalDevice,
      nbl::video::IPhysicalDevice*                              physicalDevice,
      uint32_t                                                  computeFamilyIndex)
   {
      m_console.setLogger(std::move(logger));
      m_logicalDevice      = std::move(logicalDevice);
      m_physicalDevicePtr  = physicalDevice;
      m_computeFamilyIndex = computeFamilyIndex;
      setDevice(physicalDevice);
   }

   void setSilent(bool silent) { m_console.setSilent(silent); }

   const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& getLogicalDevice() const { return m_logicalDevice; }
   nbl::video::IPhysicalDevice*                                   getPhysicalDevice() const { return m_physicalDevicePtr; }
   uint32_t                                                       getComputeFamilyIndex() const { return m_computeFamilyIndex; }
   nbl::core::smart_refctd_ptr<nbl::system::ILogger>              getLogger() const
   {
      return nbl::core::smart_refctd_ptr<nbl::system::ILogger>(m_console.getLogger());
   }

   bool loadBaseline(std::string label, const std::string& path)
   {
      auto b = benchmark_json::loadBaselineFile(label, path);
      if (!b)
         return false;

      for (const auto& [_, row] : b->rowsByName)
         m_console.growForBaseline(row);

      // Vector (not map) so delta columns print in load order.
      auto it = std::find_if(m_baselines.begin(), m_baselines.end(),
         [&](const Baseline& existing) { return existing.label == label; });
      if (it != m_baselines.end())
         *it = std::move(*b);
      else
         m_baselines.push_back(std::move(*b));
      return true;
   }

   bool loadBaseline(const std::string& path) { return loadBaseline("baseline", path); }

   bool writeReport(const std::string& path)
   {
      size_t preservedCount = 0;
      if (!benchmark_json::writeReportFile(path, m_device, m_baselines, m_results, m_console.getLogger(), &preservedCount))
         return false;

      if (preservedCount > 0)
         benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_INFO,
            "Wrote benchmark report to {} ({} new + {} preserved from prior file)",
            path, m_results.size(), preservedCount);
      else
         benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_INFO,
            "Wrote benchmark report to {} ({} rows)", path, m_results.size());
      return true;
   }

   // Captured for the UUID-mismatch warning in applyCli.
   void setDevice(const nbl::video::IPhysicalDevice* dev) { m_device = benchmark_json::buildDeviceMetadata(dev); }

   struct CliResult
   {
      std::string                                             outputPath;
      nbl::core::vector<nbl::core::vector<nbl::core::string>> focusVariants;
      uint32_t                                                focusSamples = 3; // --focus-samples, see samplesForCurrentRow

      bool isFocused(const nbl::core::vector<nbl::core::string>& name) const
      {
         return std::ranges::find(focusVariants, name) != focusVariants.end();
      }
   };

   template<typename T>
   struct Span
   {
      std::span<T> benches;
      RunContext   context;
   };

   // Two overloads so a single bench doesn't need `std::span<T>(&bench, 1)`.
   template<typename Range>
      requires requires (Range& r) { std::data(r); std::size(r); }
   static auto makeSpan(Range& benches, RunContext context)
   {
      using T = std::remove_reference_t<decltype(*std::data(benches))>;
      return Span<T>{std::span<T>(std::data(benches), std::size(benches)), std::move(context)};
   }

   template<typename T>
      requires std::derived_from<T, IBenchmark>
   static Span<T> makeSpan(T& bench, RunContext context)
   {
      return Span<T>{std::span<T>(&bench, 1), std::move(context)};
   }

   static std::string describe(const RunContext& ctx)
   {
      const auto&    sh             = ctx.shape;
      const uint32_t wgThreads      = sh.workgroupSize.x * sh.workgroupSize.y * sh.workgroupSize.z;
      const uint32_t threadsPerDisp = sh.dispatchGroupCount.x * sh.dispatchGroupCount.y * sh.dispatchGroupCount.z * wgThreads;
      const uint64_t itersPerThread = threadsPerDisp ? sh.samplesPerDispatch / threadsPerDisp : 0;
      const double   budgetMs       = double(ctx.targetBudgetMs);
      return std::format("=== {} (~{:.0f}ms/row, {} threads/dispatch, {} iters/thread; wg={}x{}x{}; ps/sample is per all GPU threads) ===",
         ctx.sectionLabel, budgetMs, threadsPerDisp, itersPerThread, sh.workgroupSize.x, sh.workgroupSize.y, sh.workgroupSize.z);
   }

   // Order: banner -> focus(spans...) -> comparison table -> banner ->
   //        column header -> rest(spans...) -> writeReport.
   // All focus rows print globally first, then all rest rows; banner printed
   // twice so each chunk reads in isolation when scrolling back.
   template<typename... Benches>
      requires(std::derived_from<Benches, IBenchmark> && ...)
   void runSessionAndReport(Span<Benches>... spans)
   {
      // Templated lambda (not `auto& s`) so only Span<T> deduces -- a future
      // signature change can't silently start passing arbitrary types through.
      auto runSpan = [this]<typename T>(Span<T>& s, bool silent)
      {
         if (s.benches.empty())
            return;
         if (!silent)
         {
            m_console.logSectionBanner(describe(s.context));
            m_console.logHeader(m_baselines);
         }
         for (auto& e : s.benches)
            e.run();
         // Flush after each rest span: if span N+1 dies mid-way, span N's
         // rows are already on disk. Trailing flush is also the final write.
         if (!silent)
            writeReport(m_cli.outputPath);
      };

      m_console.logBannerNotes(m_baselines);
      if (!m_cli.focusVariants.empty())
      {
         m_console.setSilent(true); // benches read this to know they're in the focused-rows half
         (runSpan(spans, true), ...);
         m_console.setSilent(false);
         m_console.printBaselineComparison(std::span<const nbl::core::vector<nbl::core::string>>(m_focusNames), m_baselines, m_results);
      }
      (runSpan(spans, false), ...);
   }

   struct CliConfig
   {
      std::span<const std::string> argv; // feed from IApplicationFramework::argv
      std::string                  defaultOutputPath = "Bench.json";
      std::string                  appName           = "benchmark";
   };

   CliResult applyCli(const CliConfig& cfg)
   {
      auto parsed = benchmark_cli::parseArgs(cfg.argv, cfg.defaultOutputPath);
      if (parsed.helpRequested)
      {
         benchmark_cli::printHelp(m_console.getLogger(), cfg.appName, cfg.defaultOutputPath);
         exit(0);
      }
      if (parsed.noColor)
         m_console.setColorEnabled(false);

      CliResult res;
      res.outputPath = parsed.outputPath;

      if (!parsed.baselines.empty())
      {
         size_t succeeded = 0;
         for (const auto& [label, path] : parsed.baselines)
         {
            if (loadBaseline(label, path))
            {
               ++succeeded;
               benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_INFO,
                  "Loaded baseline '{}' from {} ({} rows)", label, path, m_baselines.back().rowsByName.size());
            }
            else
               benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_WARNING,
                  "Failed to load baseline '{}' from {}, skipped", label, path);
         }
         if (succeeded == 0)
            benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_WARNING,
               "All {} --baseline load(s) failed. delta columns and --focus will be empty. "
               "Check the paths above; default auto-load of '{}' is suppressed once any --baseline is specified, "
               "drop the --baseline flag(s) or use --no-baseline to silence this warning.",
               parsed.baselines.size(), res.outputPath);
         else if (succeeded < parsed.baselines.size())
            benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_WARNING,
               "{} of {} --baseline load(s) failed; continuing with {} loaded.",
               parsed.baselines.size() - succeeded, parsed.baselines.size(), succeeded);
      }
      else if (!parsed.noBaseline)
      {
         if (loadBaseline(res.outputPath))
            benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_INFO,
               "Loaded baseline from {} ({} rows)", res.outputPath,
               m_baselines.empty() ? size_t {0} : m_baselines.back().rowsByName.size());
         else
            benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_INFO,
               "No baseline at {}, delta column will read 'n/a'", res.outputPath);
      }

      warnDeviceMismatch();

      res.focusVariants = std::move(parsed.focus);
      res.focusSamples  = parsed.focusSamples;
      m_cli             = res;
      return res;
   }

private:
   void warnDeviceMismatch() const
   {
      if (!m_device.is_object() || !m_device.contains("deviceUUID"))
         return;
      const auto& currentUUID = m_device["deviceUUID"];
      for (const auto& b : m_baselines)
      {
         if (!b.device.is_object() || !b.device.contains("deviceUUID"))
            continue;
         if (b.device["deviceUUID"] == currentUUID)
            continue;
         const std::string baselineDevName = b.device.value("name", std::string {"<unknown>"});
         const std::string currentDevName  = m_device.value("name", std::string {"<unknown>"});
         benchLogFmt(m_console.getLogger(), nbl::system::ILogger::ELL_WARNING,
            "Baseline '{}' (from {}) was measured on a different GPU ('{}' vs current '{}'). "
            "Delta values will be apples-to-oranges.",
            b.label, b.path, baselineDevName, currentDevName);
      }
   }

   // In focus phase (silent), captures the row's name into m_focusNames so
   // runSessionAndReport can build the comparison table without main.cpp
   // threading names back through each bench class.
   void appendAndLog(Result&& r)
   {
      const std::string joined = joinName(r.name);
      if (!m_baselines.empty())
      {
         const std::string key = makeKey(r.name);
         for (const auto& b : m_baselines)
         {
            auto it = b.rowsByName.find(key);
            if (it == b.rowsByName.end())
               continue;
            const bool shapeMismatch = r.workload.present() && it->second.workload.present() && (r.workload.shape != it->second.workload.shape);
            r.baselines[b.label] = {it->second.psPerSample, shapeMismatch};
         }
      }
      m_console.growWidthFor(joined);
      if (m_console.silent())
         m_focusNames.push_back(r.name);
      m_results.push_back(std::move(r));
      m_console.logRow(std::span<const std::string>(m_results.back().name), joined, m_results.back().timing, m_results.back().stats, m_results.back().baselines, m_baselines);
   }

   std::vector<Result>                                     m_results;
   std::vector<Baseline>                                   m_baselines;
   nbl::core::vector<nbl::core::vector<nbl::core::string>> m_focusNames;
   nlohmann::json                                          m_device;
   CliResult                                               m_cli;
   BenchmarkConsole                                        m_console;
   nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_logicalDevice;
   nbl::video::IPhysicalDevice*                            m_physicalDevicePtr  = nullptr;
   uint32_t                                                m_computeFamilyIndex = 0;
};

class IBenchmark
{
public:
   virtual ~IBenchmark() = default;

   // Single-named benches override doRun() and inherit this default filter.
   // Sweep-style benches synthesize per-row names; they override run() and
   // do per-row filtering themselves.
   virtual void run()
   {
      const bool silent    = isFocusPhase();
      const bool inFocus   = isFocused(m_name);
      const bool shouldRun = silent ? inFocus : !inFocus;
      if (shouldRun)
         doRun();
   }

   uint32_t             getWarmupDispatches() const { return m_warmupDispatches; }
   uint64_t             getTargetBudgetMs() const { return m_targetBudgetMs; }
   const WorkloadShape& getShape() const { return m_workloadShape; }

   // Pass this to runTimedBudgeted so only --focus rows pay the K * budget cost.
   uint32_t samplesForCurrentRow() const { return isFocusPhase() ? m_aggregator.m_cli.focusSamples : 1u; }

protected:
   // Banner label is NOT taken here; it belongs to the span (see Aggregator::Span).
   IBenchmark(Aggregator& aggregator, core::vector<core::string> name, uint32_t warmupDispatches, const WorkloadShape& shape, uint64_t targetBudgetMs)
      : m_name(std::move(name))
      , m_aggregator(aggregator)
      , m_warmupDispatches(warmupDispatches)
      , m_targetBudgetMs(targetBudgetMs)
      , m_workloadShape(shape)
   {
      registerVariant(m_name);
   }

   virtual void doRun() {}

   bool isFocusPhase() const { return m_aggregator.m_console.silent(); }
   bool isFocused(const core::vector<core::string>& name) const { return m_aggregator.m_cli.isFocused(name); }
   void registerVariant(std::span<const std::string> name) { m_aggregator.m_console.registerVariant(name); }
   void registerVariant(std::initializer_list<std::string_view> name) { m_aggregator.m_console.registerVariant(name); }

   void record(core::vector<core::string> name, const TimingResult& t, const PipelineStats& s)
   {
      Workload w{.shape = m_workloadShape};
      w.benchDispatches = w.shape.samplesPerDispatch ? uint32_t(t.totalSamples / w.shape.samplesPerDispatch) : 0;

      Result r;
      r.name     = std::move(name);
      r.timing   = t;
      r.stats    = s;
      r.workload = w;
      m_aggregator.appendAndLog(std::move(r));
   }

   core::vector<core::string> m_name;
   Aggregator&                m_aggregator; // non-owning, outlives this bench
   uint32_t                   m_warmupDispatches;
   uint64_t                   m_targetBudgetMs;
   WorkloadShape              m_workloadShape;
};

class GPUBenchmark : public IBenchmark, public GPUBenchmarkHelper
{
public:
   struct SetupData
   {
      core::vector<core::string> name;
      uint32_t                   warmupDispatches = 0;
      WorkloadShape              shape            = {};
      uint64_t                   targetBudgetMs   = 400;
   };

protected:
   GPUBenchmark(Aggregator& aggregator, const SetupData& data)
      : IBenchmark(aggregator, data.name, data.warmupDispatches, data.shape, data.targetBudgetMs)
   {
      GPUBenchmarkHelper::init({
         .device             = aggregator.getLogicalDevice(),
         .logger             = aggregator.getLogger(),
         .physicalDevice     = aggregator.getPhysicalDevice(),
         .computeFamilyIndex = aggregator.getComputeFamilyIndex(),
         .dispatchGroupCount = data.shape.dispatchGroupCount,
         .samplesPerDispatch = data.shape.samplesPerDispatch,
      });
   }
};

#endif
