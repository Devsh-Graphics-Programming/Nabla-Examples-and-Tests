// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_COMMON_BENCHMARK_TYPES_INCLUDED_
#define _NBL_COMMON_BENCHMARK_TYPES_INCLUDED_

#include <nabla.h>
#include "nlohmann/json.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct PipelineStats
{
   uint64_t    registerCount   = 0;
   uint64_t    codeSizeBytes   = 0;
   uint64_t    sharedMemBytes  = 0;
   uint64_t    privateMemBytes = 0;
   uint64_t    stackBytes      = 0;
   uint32_t    subgroupSize    = 0;
   std::string raw;

   // Driver stats matchStat didn't recognise. Structured (not lossy-stringified
   // into `raw`) so JSON round-trips the correct numeric type.
   std::vector<nbl::video::IGPUPipelineBase::SExecutableStatistic> unknowns;
};

// Match one driver-reported executable stat into the appropriate PipelineStats
// slot, accumulating max. VGPR/SGPR are returned separately so callers can
// fall back to (vgpr + sgpr) when no combined `register` stat exists (AMD).
inline void matchPipelineStat(const nbl::video::IGPUPipelineBase::SExecutableStatistic& stat, PipelineStats& out, uint64_t& vgpr, uint64_t& sgpr)
{
   const uint64_t v        = stat.asUint();
   auto           contains = [&](std::string_view kw)
   {
      const auto it = std::ranges::search(stat.name, kw, [&](char a, char b)
         { return std::tolower(a) == std::tolower(b); }).begin();
      return it != stat.name.end();
   };

   // Order matters: more specific keys first.
   if (contains("subgroup size") || contains("subgroupsize") || contains("warp size") || contains("wave size"))
      out.subgroupSize = std::max<uint32_t>(out.subgroupSize, uint32_t(v));
   else if (contains("vgpr"))
      vgpr = std::max(vgpr, v);
   else if (contains("sgpr"))
      sgpr = std::max(sgpr, v);
   else if (contains("register"))
      out.registerCount = std::max(out.registerCount, v);
   else if (contains("binary size") || contains("binarysize") || contains("codesize") || contains("code size") || contains("isa size"))
      out.codeSizeBytes = std::max(out.codeSizeBytes, v);
   else if (contains("instructioncount") || contains("instruction count") || contains("numinstructions"))
      out.codeSizeBytes = std::max(out.codeSizeBytes, v); // proxy when no byte size
   else if (contains("shared memory") || contains("sharedmemory") || contains("groupshared") || contains("lds"))
      out.sharedMemBytes = std::max(out.sharedMemBytes, v);
   else if (contains("stack size") || contains("stacksize"))
      out.stackBytes = std::max(out.stackBytes, v);
   else if (contains("local memory") || contains("localmemory") || contains("scratch") || contains("private memory") || contains("privatememory") || contains("stack"))
      out.privateMemBytes = std::max(out.privateMemBytes, v);
   else
      out.unknowns.push_back(stat);
}

// Pull executable stats off any pipeline (compute, RT, graphics) into a
// PipelineStats. Requires the device-level pipelineExecutableInfo feature to
// have been enabled at pipeline-create time (otherwise getExecutableInfo()
// returns empty and `out` keeps its default zeros).
inline void extractPipelineStats(const nbl::video::IGPUPipelineBase* pipeline, PipelineStats& out)
{
   if (!pipeline)
      return;
   auto infos = pipeline->getExecutableInfo();
   out.raw    = nbl::system::to_string(infos);

   uint64_t vgpr = 0, sgpr = 0;
   for (const auto& info : infos)
   {
      if (info.subgroupSize)
         out.subgroupSize = std::max<uint32_t>(out.subgroupSize, info.subgroupSize);
      for (const auto& stat : info.structuredStatistics)
         matchPipelineStat(stat, out, vgpr, sgpr);
   }
   // AMD-style drivers expose VGPR/SGPR separately without a combined count.
   if (out.registerCount == 0 && (vgpr || sgpr))
      out.registerCount = vgpr + sgpr;
}

struct TimingResult
{
   float64_t elapsed_ns     = 0.0;
   uint64_t  totalSamples   = 0;
   float64_t ps_per_sample  = 0.0;
   float64_t gsamples_per_s = 0.0;
   float64_t ms_total       = 0.0;
};

struct Format
{
   struct Widths
   {
      size_t name     = std::string_view("Name").size();
      size_t psSample = std::string_view("ps/sample").size();
      size_t gsamples = std::string_view("GSamples/s").size();
      size_t regs     = std::string_view("regs").size();
      size_t code     = std::string_view("code(B)").size();
      size_t shared   = std::string_view("shared(B)").size();
      size_t local    = std::string_view("local(B)").size();

      void grow(std::string_view joinedName) { name = std::max(name, joinedName.size()); }
   };

   static std::string headerBase(const Widths& w = {})
   {
      return std::format("{:<{}} | {:>12} | {:>12} | {:>6} | {:>8} | {:>12} | {:>12}",
         "Name", w.name, "ps/sample", "GSamples/s", "regs", "code(B)", "shared(B)", "local(B)");
   }

   static std::string dataBase(const Widths& w, std::string_view joinedName, const TimingResult& t, const PipelineStats& s)
   {
      return std::format("{:<{}} | {:>12.3f} | {:>12.3f} | {:>6} | {:>8} | {:>12} | {:>12}",
         joinedName, w.name, t.ps_per_sample, t.gsamples_per_s, s.registerCount, s.codeSizeBytes, s.sharedMemBytes, s.privateMemBytes);
   }
};

// The "what was measured" part of a workload. Workload (adds benchDispatches)
// and RunContext (adds banner label + budget) both embed a WorkloadShape, so
// the shape can be sliced into either from the other.
struct WorkloadShape
{
   nbl::hlsl::uint32_t3 workgroupSize      = {0, 0, 0};
   nbl::hlsl::uint32_t3 dispatchGroupCount = {0, 0, 0};
   uint64_t             samplesPerDispatch = 0;

   inline bool operator==(const WorkloadShape& other) const
   {
      return workgroupSize == other.workgroupSize && dispatchGroupCount == other.dispatchGroupCount && samplesPerDispatch == other.samplesPerDispatch;
   }

   inline bool operator!=(const WorkloadShape& other) const
   {
      return !(*this == other);
   }
};

struct Workload
{
   WorkloadShape shape;
   uint32_t      benchDispatches = 0;

   // Default-constructed (all zeros) signals "not recorded".
   bool present() const { return shape.samplesPerDispatch != 0; }
};

struct BaselineRow
{
   // UINT64_MAX sentinel: no real pipeline stat reaches that magnitude, so an
   // "absent" field can't collide with a real value. The current run can also
   // produce kAbsent when a driver doesn't expose a given stat.
   static constexpr uint64_t kAbsent = std::numeric_limits<uint64_t>::max();

   float64_t psPerSample     = 0.0;
   uint64_t  registerCount   = kAbsent;
   uint64_t  codeSizeBytes   = kAbsent;
   uint64_t  sharedMemBytes  = kAbsent;
   uint64_t  privateMemBytes = kAbsent;
   uint64_t  stackBytes      = kAbsent;
   uint64_t  subgroupSize    = kAbsent; // uint64_t (not 32) to share kAbsent semantics
   Workload  workload {};
};

// Per-baseline reference for a single row: the baseline's ps/sample plus
// whether its recorded workload shape differs from this run (renders the
// "[WG!]" marker so the reader knows the comparison is questionable).
struct BaselineRef
{
   float64_t psPerSample   = 0.0;
   bool      shapeMismatch = false;
};

struct Result
{
   // Hierarchical name, outermost first. Tooling can group by any prefix; the
   // console joins with " > ".
   nbl::core::vector<nbl::core::string>         name;
   TimingResult                                 timing {};
   PipelineStats                                stats {};
   Workload                                     workload {};
   std::unordered_map<std::string, BaselineRef> baselines;
};

inline std::string joinName(std::span<const std::string> name, std::string_view sep = " > ")
{
   std::string out;
   for (size_t i = 0; i < name.size(); ++i)
   {
      if (i)
         out.append(sep);
      out.append(name[i]);
   }
   return out;
}

// Unit-separator (\x1f) between segments so makeKey can't collide with any
// user-supplied content.
inline std::string makeKey(std::span<const std::string> name)
{
   std::string k;
   size_t      total = 0;
   for (const auto& s : name)
      total += s.size() + 1;
   k.reserve(total);
   for (size_t i = 0; i < name.size(); ++i)
   {
      if (i)
         k.push_back('\x1f');
      k.append(name[i]);
   }
   return k;
}

inline nbl::core::vector<nbl::core::string> splitFocusSpec(std::string_view spec)
{
   auto trim = [](std::string_view s)
   {
      while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
         s.remove_prefix(1);
      while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
         s.remove_suffix(1);
      return s;
   };
   nbl::core::vector<nbl::core::string> out;
   size_t                               start = 0;
   while (start <= spec.size())
   {
      size_t end = spec.find('>', start);
      if (end == std::string_view::npos)
         end = spec.size();
      const auto seg = trim(spec.substr(start, end - start));
      if (!seg.empty())
         out.emplace_back(seg);
      if (end == spec.size())
         break;
      start = end + 1;
   }
   return out;
}

struct Baseline
{
   std::string                                  label;
   std::string                                  path;
   nlohmann::json                               device; // top-level "device" field from the file, or null if absent
   std::unordered_map<std::string, BaselineRow> rowsByName; // makeKey(name) -> stats
};

template<typename... Args>
inline void benchLogFmt(nbl::system::ILogger* logger, nbl::system::ILogger::E_LOG_LEVEL level, std::string_view fmt, const Args&... args)
{
   if (!logger)
      return;
   logger->log("%s", level, std::vformat(fmt, std::make_format_args(args...)).c_str());
}

#endif
