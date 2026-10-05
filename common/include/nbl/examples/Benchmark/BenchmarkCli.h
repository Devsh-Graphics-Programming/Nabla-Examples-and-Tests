// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_COMMON_BENCHMARK_CLI_INCLUDED_
#define _NBL_COMMON_BENCHMARK_CLI_INCLUDED_

#include <nabla.h>
#include "nbl/examples/Benchmark/BenchmarkTypes.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace benchmark_cli
{

struct ParsedArgs
{
   std::string                                             outputPath;
   bool                                                    noBaseline    = false;
   bool                                                    noColor       = false;
   bool                                                    helpRequested = false;
   std::vector<std::pair<std::string, std::string>>        baselines; // (label, path)
   nbl::core::vector<nbl::core::vector<nbl::core::string>> focus;
   // Median-of-K window count used for focused rows (see
   // IBenchmark::samplesForCurrentRow). Default 3 trades 3 * targetBudgetMs
   // wall time for jitter-robust comparisons.
   uint32_t focusSamples = 3;
};

// Pure: parse argv into a ParsedArgs. Unknown flags are silently ignored;
// the caller decides what to do on help / no-baseline / per-load failure.
inline ParsedArgs parseArgs(std::span<const std::string> argv, std::string defaultOutputPath)
{
   ParsedArgs out;
   out.outputPath = std::move(defaultOutputPath);

   for (size_t i = 1; i < argv.size(); ++i)
   {
      if (argv[i] == "--output" && i + 1 < argv.size())
         out.outputPath = argv[++i];
      else if (argv[i] == "--no-baseline")
         out.noBaseline = true;
      else if (argv[i] == "--no-color")
         out.noColor = true;
      else if (argv[i] == "--baseline" && i + 1 < argv.size())
      {
         const std::string& spec = argv[++i];
         const auto         eq   = spec.find('=');
         std::string        label, path;
         if (eq == std::string::npos)
         {
            path            = spec;
            const auto stem = std::filesystem::path(path).stem().string();
            label           = stem.empty() ? std::string("baseline") : stem;
         }
         else
         {
            label = spec.substr(0, eq);
            path  = spec.substr(eq + 1);
         }
         out.baselines.emplace_back(std::move(label), std::move(path));
      }
      else if (argv[i] == "--focus" && i + 1 < argv.size())
      {
         out.focus.push_back(splitFocusSpec(argv[++i]));
      }
      else if (argv[i] == "--focus-samples" && i + 1 < argv.size())
      {
         // Clamp to [1, 32]: 1 disables the median+outlier path, 32 is well past
         // the point of diminishing returns (variance of the trimmed mean drops
         // ~1/sqrt(K)). from_chars instead of stol to stay no-exceptions per
         // Nabla style; malformed input leaves the default in place.
         const std::string& s = argv[++i];
         long v = 0;
         const auto [_, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
         if (ec == std::errc{})
            out.focusSamples = uint32_t(std::clamp<long>(v, 1, 32));
      }
      else if (argv[i] == "--help" || argv[i] == "-h")
      {
         out.helpRequested = true;
      }
   }
   return out;
}

inline void printHelp(nbl::system::ILogger* logger, std::string_view appName, std::string_view defaultOutputPath)
{
   benchLogFmt(logger, nbl::system::ILogger::ELL_INFO,
      "{} CLI:\n"
      "  --output PATH              write this run's report to PATH (default: {})\n"
      "  --baseline [LABEL=]PATH    load PATH as a baseline; LABEL becomes the column header ('vs LABEL').\n"
      "                             repeatable. If LABEL= is omitted, the file's stem is used\n"
      "                             (e.g. main.json -> 'main'). '=' is used instead of ':' so Windows\n"
      "                             drive letters in paths don't collide with the separator.\n"
      "  --no-baseline              skip the default auto-load of the output path\n"
      "  --no-color                 disable ANSI color in the live table (also honored: NO_COLOR=1 env var)\n"
      "  --focus NAME               print a focused baseline-comparison table for NAME before the run.\n"
      "                             NAME is the hierarchical name with '>' between segments (whitespace\n"
      "                             around '>' is optional). Repeatable; one row per --focus. The first\n"
      "                             loaded baseline is the reference for inline deltas in this table.\n"
      "                             Example: --focus \"Linear > Linear > 1:1\"\n"
      "  --focus-samples N          run each focused row N times (median + outlier rejection) for\n"
      "                             jitter-robust comparisons. Default 3; clamped to [1, 32]. N=1\n"
      "                             matches the rest-phase single-shot path. Wall time per focused\n"
      "                             row scales linearly with N.\n"
      "  --help, -h                 print this help\n"
      "\n"
      "Default behaviour: with no flags, the prior run's output (if present) is loaded as the single\n"
      "  'baseline', and a fresh one is written at the end; iterate-and-compare with no flags needed.\n"
      "\n"
      "Failed loads (missing/corrupt file) log a warning and continue; the corresponding column reads 'n/a'.",
      appName, defaultOutputPath);
}

}

#endif
