// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_COMMON_BENCHMARK_CONSOLE_INCLUDED_
#define _NBL_COMMON_BENCHMARK_CONSOLE_INCLUDED_

#include <nabla.h>
#include "nbl/examples/Benchmark/BenchmarkTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Methods templated on the baselines range must expose `.label` and `.rowsByName`.
class BenchmarkConsole
{
   public:
   BenchmarkConsole()
   {
      // https://no-color.org
      if (const char* nc = std::getenv("NO_COLOR"); nc && nc[0] != '\0')
         m_useAnsi = false;
   }
   explicit BenchmarkConsole(nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger)
      : BenchmarkConsole()
   {
      m_logger = std::move(logger);
   }

   void                  setLogger(nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger) { m_logger = std::move(logger); }
   nbl::system::ILogger* getLogger() const { return m_logger.get(); }

   void setSilent(bool s) { m_silent = s; }
   bool silent() const { return m_silent; }

   void setColorEnabled(bool e) { m_useAnsi = e; }
   bool colorEnabled() const { return m_useAnsi; }

   // `neutral` is ELL_PERFORMANCE blue (not a full reset) so uncolored cell
   // parts inherit the logger's line-wrap color. Only correct because rows /
   // banners are all logged at ELL_PERFORMANCE.
   struct Ansi
   {
      static constexpr std::string_view neutral = "\033[34m";
      static constexpr std::string_view reset   = "\033[0m";
      static constexpr std::string_view red     = "\033[31m";
      static constexpr std::string_view green   = "\033[32m";
      static constexpr std::string_view yellow  = "\033[33m";
      static constexpr std::string_view cyan    = "\033[36m";
      static constexpr std::string_view bold    = "\033[1m";
   };

   // visualWidth excludes ANSI escape bytes (std::format's `{:>{}}` counts
   // bytes), so colored cells must be padded manually via padCell.
   struct CellOut
   {
      std::string text;
      size_t      visualWidth = 0;
   };

   const Format::Widths& widths() const { return m_widths; }
   void                  growWidthFor(std::string_view joined) { m_widths.grow(joined); }

   // Sizes int columns to unchanged-value width, float columns to "value
   // (+/-delta)" with delta=0. Changed-int rows overflow; padding every row
   // for worst-case wastes ~40% horizontal space on stable runs.
   void growForBaseline(const BaselineRow& b)
   {
      const auto growInt = [&](size_t& w, uint64_t v)
      {
         if (v == BaselineRow::kAbsent)
            return;
         w = std::max(w, std::format("{}", v).size());
      };
      growInt(m_widths.regs,   b.registerCount);
      growInt(m_widths.code,   b.codeSizeBytes);
      growInt(m_widths.shared, b.sharedMemBytes);
      growInt(m_widths.local,  b.privateMemBytes);

      if (b.psPerSample > 0.0)
      {
         m_widths.psSample = std::max(m_widths.psSample, floatCellPlainText(b.psPerSample, 0.0).size());
         const double gsBase = 1000.0 / b.psPerSample;
         m_widths.gsamples = std::max(m_widths.gsamples, floatCellPlainText(gsBase, 0.0).size());
      }
   }

   // Pre-register so the header (logged once up front) doesn't stay narrower than later rows.
   void registerVariant(std::span<const std::string> name) { m_widths.grow(joinName(name)); }
   void registerVariant(std::initializer_list<std::string_view> name)
   {
      std::vector<std::string> tmp;
      tmp.reserve(name.size());
      for (auto s : name)
         tmp.emplace_back(s);
      m_widths.grow(joinName(tmp));
   }

   void logSectionBanner(std::string_view banner) const
   {
      if (banner.empty())
         return;
      if (m_useAnsi)
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE, "{}{}{}{}", Ansi::bold, Ansi::cyan, banner, Ansi::reset);
      else
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE, "{}", banner);
   }

   // Once per session, not per span, otherwise readers see the same text N times.
   template<typename Baselines>
   void logBannerNotes(const Baselines& baselines) const
   {
      if (std::empty(baselines))
         return;
      const auto&       primary      = *std::begin(baselines);
      const bool        multi        = std::distance(std::begin(baselines), std::end(baselines)) > 1;
      const std::string primaryLabel = primary.label;
      benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE,
         "Note: ps/sample lower = faster; GSamples/s higher = faster. Inline annotations compare to primary baseline '{}': "
         "floats show 'value (+/-delta)' always; ints show 'old -> new' only when changed.",
         primaryLabel);
      if (multi)
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE,
            "Note: trailing 'vs LABEL' columns carry raw ps/sample deltas against secondary baselines (primary skipped, shown inline).");
      benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE,
         "Note: '[WG!]' on a delta = baseline's workload shape (workgroup / dispatch / samplesPerDispatch) differs from this run, comparison is apples-to-oranges.");
      benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE,
         "Note: float deltas only get green/red coloring when the relative change is >= {:.0f}% (typical GPU jitter is 1-2%); smaller deltas stay neutral.",
         kFloatColorThreshold * 100.0);
   }

   template<typename Baselines>
   void logHeader(const Baselines& baselines) const
   {
      std::string line = std::format("{:<{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}}",
         "Name",       m_widths.name,
         "ps/sample",  m_widths.psSample,
         "GSamples/s", m_widths.gsamples,
         "regs",       m_widths.regs,
         "code(B)",    m_widths.code,
         "shared(B)",  m_widths.shared,
         "local(B)",   m_widths.local);
      // Primary is shown inline on every value column; only secondaries get trailing columns.
      size_t idx = 0;
      for (const auto& b : baselines)
      {
         if (idx++ == 0)
            continue;
         const std::string col = std::format("vs {}", b.label);
         line += std::format(" | {:>{}}", col, baselineColWidth(b.label));
      }
      benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE, "{}", line);
   }

   template<typename Baselines>
   void logRow(std::span<const std::string> name, std::string_view joinedName,
      const TimingResult& t, const PipelineStats& s,
      const std::unordered_map<std::string, BaselineRef>& rowBaselines,
      const Baselines&                                    baselines) const
   {
      if (!m_logger || m_silent)
         return;

      const BaselineRow* primary = nullptr;
      if (!std::empty(baselines))
      {
         const std::string key = makeKey(name);
         const auto&       b0  = *std::begin(baselines);
         if (auto it = b0.rowsByName.find(key); it != b0.rowsByName.end())
            primary = &it->second;
      }

      // ps_per_sample * GSamples/s == 1000 (see runTimed), so GSamples is derived not stored.
      const auto baselineGSamples = primary ? std::optional<double>{primary->psPerSample > 0.0 ? 1000.0 / primary->psPerSample : 0.0} : std::nullopt;

      std::string line = std::format("{:<{}}", joinedName, m_widths.name);
      line += " | " + padCell(formatFloatCell(t.ps_per_sample,   primary ? std::optional<double>{primary->psPerSample} : std::nullopt, true),  m_widths.psSample);
      line += " | " + padCell(formatFloatCell(t.gsamples_per_s,  baselineGSamples,                                                    false), m_widths.gsamples);
      line += " | " + padCell(formatIntCell(s.registerCount,     primary ? primary->registerCount   : BaselineRow::kAbsent),                                     m_widths.regs);
      line += " | " + padCell(formatIntCell(s.codeSizeBytes,     primary ? primary->codeSizeBytes   : BaselineRow::kAbsent),                                     m_widths.code);
      line += " | " + padCell(formatIntCell(s.sharedMemBytes,    primary ? primary->sharedMemBytes  : BaselineRow::kAbsent),                                     m_widths.shared);
      line += " | " + padCell(formatIntCell(s.privateMemBytes,   primary ? primary->privateMemBytes : BaselineRow::kAbsent),                                     m_widths.local);

      size_t idx = 0;
      for (const auto& b : baselines)
      {
         if (idx++ == 0)
            continue;
         std::string plain;
         bool        better      = false;
         bool        significant = false;
         bool        haveValue   = false;
         bool        flagShape   = false;
         if (auto it = rowBaselines.find(b.label); it != rowBaselines.end() && it->second.psPerSample > 0.0)
         {
            const double delta = t.ps_per_sample - it->second.psPerSample;
            plain       = std::format("{:+.3f}", delta);
            better      = delta < 0.0;
            significant = std::abs(delta) / it->second.psPerSample >= kFloatColorThreshold;
            haveValue   = true;
            flagShape   = it->second.shapeMismatch;
         }
         else
         {
            plain = "n/a";
         }
         std::string suffix = flagShape ? std::string(" [WG!]") : std::string();
         CellOut cell;
         cell.visualWidth = plain.size() + suffix.size();
         if (!m_useAnsi)
         {
            cell.text = plain + suffix;
         }
         else
         {
            const bool        paint        = haveValue && significant;
            const std::string_view col     = paint ? (better ? Ansi::green : Ansi::red) : std::string_view{};
            std::string       coloredPlain = paint
                                                ? std::format("{}{}{}", col, plain, Ansi::neutral)
                                                : plain;
            std::string       coloredSuffix = flagShape
                                                ? std::format("{}{}{}{}", Ansi::bold, Ansi::red, suffix, Ansi::neutral)
                                                : std::string();
            cell.text = coloredPlain + coloredSuffix;
         }
         line += " | " + padCell(cell, baselineColWidth(b.label));
      }
      benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE, "{}", line);
   }

   // Flat table, one row per (variant, stat); each baseline gets one delta column:
   //
   //   Name  | stat        | current | vs iter47 | vs iter48
   //   X     | ps/sample   |   2.151 |   -0.044  |   +0.123
   //   X     | GSamples/s  |   464.9 |   +9.456  |   -7.234
   //   X     | regs        |      40 |     +0    |     +0
   //   X     | code(B)     |    4992 |   +128    |      0
   template<typename Baselines, typename Results>
   void printBaselineComparison(std::span<const nbl::core::vector<nbl::core::string>> names,
      const Baselines& baselines, const Results& results) const
   {
      if (!m_logger || names.empty())
         return;
      if (std::empty(baselines))
      {
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_WARNING,
            "--focus requested {} variant(s) but no baselines are loaded, nothing to compare against. "
            "Did your --baseline paths fail to load?",
            names.size());
         return;
      }

      struct Current
      {
         TimingResult  t;
         PipelineStats s;
         Workload      w;
         bool          present = false;
      };
      std::unordered_map<std::string, Current> currentByKey;
      currentByKey.reserve(std::size(results));
      for (const auto& r : results)
         currentByKey[makeKey(r.name)] = {r.timing, r.stats, r.workload, true};

      const size_t baselineCount = static_cast<size_t>(std::distance(std::begin(baselines), std::end(baselines)));

      std::vector<std::vector<CellOut>> rows;
      rows.reserve(1 + names.size() * 6);

      {
         auto plainCell = [](std::string s) -> CellOut { const size_t w = s.size(); return {std::move(s), w}; };
         std::vector<CellOut> header;
         header.reserve(3 + baselineCount);
         header.push_back(plainCell("Name"));
         header.push_back(plainCell("stat"));
         header.push_back(plainCell("current"));
         for (const auto& b : baselines)
            header.push_back(plainCell(std::format("vs {}", b.label)));
         rows.push_back(std::move(header));
      }

      auto floatStatRow = [&](const char* label, std::string_view joined, bool have, double curV,
                               const Workload& curW, const std::string& key,
                               auto baselineLookup /*BaselineRow -> double*/, bool lowerIsBetter)
      {
         auto plainCell = [](std::string s) -> CellOut { const size_t w = s.size(); return {std::move(s), w}; };
         std::vector<CellOut> row;
         row.reserve(3 + baselineCount);
         row.push_back(plainCell(std::string(joined)));
         row.push_back(plainCell(label));
         row.push_back(have ? plainCell(formatFloat5(curV)) : plainCell("n/a"));

         for (const auto& b : baselines)
         {
            auto bit = b.rowsByName.find(key);
            if (!have || bit == b.rowsByName.end())
            {
               row.push_back(plainCell("n/a"));
               continue;
            }
            const double baseV = baselineLookup(bit->second);
            if (baseV <= 0.0)
            {
               row.push_back(plainCell("n/a"));
               continue;
            }
            const bool        shapeMismatch = curW.present() && bit->second.workload.present() && (curW.shape != bit->second.workload.shape);
            const double      delta         = curV - baseV;
            const std::string deltaStr      = std::format("{}{}", delta >= 0 ? "+" : "-", formatFloat5(std::abs(delta)));
            const bool        significant   = std::abs(delta) / baseV >= kFloatColorThreshold;
            const std::string suffix        = shapeMismatch ? std::string(" [WG!]") : std::string();
            CellOut           cell;
            cell.visualWidth = deltaStr.size() + suffix.size();
            if (!m_useAnsi || !significant)
            {
               cell.text = m_useAnsi && shapeMismatch
                              ? std::format("{}{}{}{}{}", deltaStr, Ansi::bold, Ansi::red, suffix, Ansi::neutral)
                              : deltaStr + suffix;
            }
            else
            {
               const bool             better = (lowerIsBetter && delta < 0.0) || (!lowerIsBetter && delta > 0.0);
               const std::string_view col    = better ? Ansi::green : Ansi::red;
               std::string            coloredDelta  = std::format("{}{}{}", col, deltaStr, Ansi::neutral);
               std::string            coloredSuffix = shapeMismatch
                                                         ? std::format("{}{}{}{}", Ansi::bold, Ansi::red, suffix, Ansi::neutral)
                                                         : std::string();
               cell.text = coloredDelta + coloredSuffix;
            }
            row.push_back(std::move(cell));
         }
         rows.push_back(std::move(row));
      };

      auto intStatRow = [&](const char* label, std::string_view joined, bool have, uint64_t curV,
                              const Workload& curW, const std::string& key, uint64_t BaselineRow::* baseField)
      {
         auto plainCell = [](std::string s) -> CellOut { const size_t w = s.size(); return {std::move(s), w}; };
         std::vector<CellOut> row;
         row.reserve(3 + baselineCount);
         row.push_back(plainCell(std::string(joined)));
         row.push_back(plainCell(label));
         row.push_back(have ? plainCell(std::format("{}", curV)) : plainCell("n/a"));

         for (const auto& b : baselines)
         {
            auto bit = b.rowsByName.find(key);
            if (!have || bit == b.rowsByName.end())
            {
               row.push_back(plainCell("n/a"));
               continue;
            }
            const uint64_t baseV = bit->second.*baseField;
            if (baseV == BaselineRow::kAbsent)
            {
               row.push_back(plainCell("n/a"));
               continue;
            }
            const bool        shapeMismatch = curW.present() && bit->second.workload.present() && (curW.shape != bit->second.workload.shape);
            const int64_t     delta         = int64_t(curV) - int64_t(baseV);
            const std::string deltaStr      = std::format("{:+d}", delta);
            const std::string suffix        = shapeMismatch ? std::string(" [WG!]") : std::string();
            CellOut           cell;
            cell.visualWidth = deltaStr.size() + suffix.size();
            if (!m_useAnsi)
            {
               cell.text = deltaStr + suffix;
            }
            else
            {
               std::string coloredDelta  = delta != 0
                                              ? std::format("{}{}{}", Ansi::yellow, deltaStr, Ansi::neutral)
                                              : deltaStr;
               std::string coloredSuffix = shapeMismatch
                                              ? std::format("{}{}{}{}", Ansi::bold, Ansi::red, suffix, Ansi::neutral)
                                              : std::string();
               cell.text = coloredDelta + coloredSuffix;
            }
            row.push_back(std::move(cell));
         }
         rows.push_back(std::move(row));
      };

      for (const auto& nameVec : names)
      {
         const std::string joined = joinName(nameVec);
         const std::string key    = makeKey(nameVec);
         const auto        cit    = currentByKey.find(key);
         const bool        have   = (cit != currentByKey.end()) && cit->second.present;
         const auto&       t      = have ? cit->second.t : TimingResult {};
         const auto&       s      = have ? cit->second.s : PipelineStats {};
         const auto&       w      = have ? cit->second.w : Workload {};

         floatStatRow("ps/sample",  joined, have, t.ps_per_sample,  w, key,
            [](const BaselineRow& b) { return b.psPerSample; }, true);
         floatStatRow("GSamples/s", joined, have, t.gsamples_per_s, w, key,
            [](const BaselineRow& b) { return b.psPerSample > 0.0 ? 1000.0 / b.psPerSample : 0.0; }, false);
         intStatRow("regs",      joined, have, s.registerCount,   w, key, &BaselineRow::registerCount);
         intStatRow("code(B)",   joined, have, s.codeSizeBytes,   w, key, &BaselineRow::codeSizeBytes);
         intStatRow("shared(B)", joined, have, s.sharedMemBytes,  w, key, &BaselineRow::sharedMemBytes);
         intStatRow("local(B)",  joined, have, s.privateMemBytes, w, key, &BaselineRow::privateMemBytes);
      }

      const size_t        nCols = 3 + baselineCount;
      std::vector<size_t> colWidths(nCols, 0);
      for (const auto& r : rows)
         for (size_t i = 0; i < r.size() && i < nCols; ++i)
            colWidths[i] = std::max(colWidths[i], r[i].visualWidth);

      benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE,
         "=== Focus comparison ({} variant(s) vs {} baseline(s); ps/sample lower is better, integer deltas are absolute) ===",
         names.size(), baselineCount);
      auto leftPad = [](const CellOut& c, size_t targetWidth) -> std::string
      {
         if (c.visualWidth >= targetWidth)
            return c.text;
         return c.text + std::string(targetWidth - c.visualWidth, ' ');
      };
      for (size_t ri = 0; ri < rows.size(); ++ri)
      {
         std::string line;
         for (size_t ci = 0; ci < rows[ri].size(); ++ci)
         {
            if (ci)
               line.append(" | ");
            if (ci <= 1)
               line += leftPad(rows[ri][ci], colWidths[ci]);
            else
               line += padCell(rows[ri][ci], colWidths[ci]);
         }
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_PERFORMANCE, "{}", line);
      }
   }

   private:
   static constexpr size_t kBaselineMinColWidth = 10;
   size_t                  baselineColWidth(std::string_view label) const
   {
      return std::max<size_t>(kBaselineMinColWidth, std::string_view("vs ").size() + label.size());
   }

   // Typical GPU jitter is 1-2%; coloring below 5% would mostly highlight noise.
   static constexpr double kFloatColorThreshold = 0.05;

   // std::format counts ANSI escape bytes, so `{:>N}` can't pad colored cells.
   std::string padCell(const CellOut& c, size_t targetWidth) const
   {
      if (c.visualWidth >= targetWidth)
         return c.text;
      return std::string(targetWidth - c.visualWidth, ' ') + c.text;
   }

   // "regs 40 -> 54" is more useful than "+14 from somewhere", show both endpoints.
   CellOut formatIntCell(uint64_t current, uint64_t baseline) const
   {
      if (baseline == BaselineRow::kAbsent || baseline == current)
      {
         auto s = std::format("{}", current);
         const size_t w = s.size();
         return {std::move(s), w};
      }
      const std::string baseStr = std::format("{}", baseline);
      const std::string curStr  = std::format("{}", current);
      const std::string plain   = std::format("{} -> {}", baseStr, curStr);
      const size_t      visW    = plain.size();
      if (!m_useAnsi)
         return {plain, visW};
      auto colored = std::format("{}{} -> {}{}", Ansi::yellow, baseStr, curStr, Ansi::neutral);
      return {std::move(colored), visW};
   }

   // ~5 chars including the decimal point, so column widths stay predictable
   // across ps/sample (0.5..100) and GSamples/s (0.03..1000+).
   static std::string formatFloat5(double v)
   {
      const double mag = std::abs(v);
      if (mag >= 10000.0) return std::format("{:.0f}", v);
      if (mag >= 1000.0)  return std::format("{:.1f}", v);
      if (mag >= 100.0)   return std::format("{:.1f}", v);
      if (mag >= 10.0)    return std::format("{:.2f}", v);
      return std::format("{:.3f}", v);
   }

   static std::string floatCellPlainText(double value, double delta)
   {
      const std::string deltaStr = std::format("{}{}", delta >= 0 ? "+" : "-", formatFloat5(std::abs(delta)));
      return std::format("{} ({})", formatFloat5(value), deltaStr);
   }

   CellOut formatFloatCell(double current, std::optional<double> baseline, bool lowerIsBetter) const
   {
      if (!baseline.has_value() || *baseline <= 0.0)
      {
         auto s = formatFloat5(current);
         const size_t w = s.size();
         return {std::move(s), w};
      }
      const double      delta    = current - *baseline;
      const std::string plain    = floatCellPlainText(current, delta);
      const size_t      visW     = plain.size();
      const bool        significant = std::abs(delta) / *baseline >= kFloatColorThreshold;
      if (!m_useAnsi || !significant)
         return {plain, visW};
      const std::string      valStr   = formatFloat5(current);
      const std::string      deltaStr = std::format("{}{}", delta >= 0 ? "+" : "-", formatFloat5(std::abs(delta)));
      const bool             better   = (lowerIsBetter && delta < 0.0) || (!lowerIsBetter && delta > 0.0);
      const std::string_view color    = better ? Ansi::green : Ansi::red;
      auto                   colored = std::format("{} ({}{}{})", valStr, color, deltaStr, Ansi::neutral);
      return {std::move(colored), visW};
   }

   nbl::core::smart_refctd_ptr<nbl::system::ILogger> m_logger;
   Format::Widths                                    m_widths;
   bool                                              m_silent  = false;
   bool                                              m_useAnsi = true;
};

#endif
