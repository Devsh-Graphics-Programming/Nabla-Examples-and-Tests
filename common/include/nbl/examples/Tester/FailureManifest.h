#ifndef _NBL_COMMON_TESTER_FAILURE_MANIFEST_INCLUDED_
#define _NBL_COMMON_TESTER_FAILURE_MANIFEST_INCLUDED_

#include <nabla.h>

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nbl::examples::testing
{

struct FailureCase
{
   std::string check;
   std::string side;
   uint64_t iteration = 0;
   uint32_t seed = 0;
   double maxRelative = 0.0;
   double maxAbsolute = 0.0;
};

struct FailureGroup
{
   std::string phase;
   std::string id;
   std::string name;
   std::string logFile;
   std::vector<FailureCase> cases;
   uint32_t omittedCases = 0;
};

class FailureManifest
{
   public:
   explicit FailureManifest(std::string suite = {}) : m_suite(std::move(suite)) {}

   void setSuite(std::string suite) { m_suite = std::move(suite); }

   void addGroupFailure(std::string_view phase, std::string_view id, std::string_view name, std::string_view logFile = {})
   {
      auto& group = groupFor(phase, id, name);
      if (!logFile.empty())
         group.logFile = std::string(logFile);
   }

   void addCase(std::string_view phase, std::string_view id, std::string_view name, std::string_view check, std::string_view side,
      uint64_t iteration, uint32_t seed, double maxRelative, double maxAbsolute)
   {
      auto& group = groupFor(phase, id, name);
      if (group.cases.size() >= MaxCasesPerGroup)
      {
         ++group.omittedCases;
         return;
      }

      group.cases.push_back(FailureCase{
         .check = std::string(check),
         .side = std::string(side),
         .iteration = iteration,
         .seed = seed,
         .maxRelative = maxRelative,
         .maxAbsolute = maxAbsolute,
      });
   }

   const std::vector<FailureGroup>& failures() const { return m_failures; }

   nlohmann::json toJson() const
   {
      nlohmann::json doc;
      doc["version"] = 1;
      doc["suite"] = m_suite;
      auto& failures = doc["failures"] = nlohmann::json::array();

      for (const auto& group : m_failures)
      {
         nlohmann::json g;
         g["phase"] = group.phase;
         g["id"] = group.id;
         g["name"] = group.name;
         if (!group.logFile.empty())
            g["log_file"] = group.logFile;

         auto& cases = g["cases"] = nlohmann::json::array();
         for (const auto& c : group.cases)
         {
            nlohmann::json row;
            row["check"] = c.check;
            row["side"] = c.side;
            row["iteration"] = c.iteration;
            row["seed"] = c.seed;
            row["max_relative"] = c.maxRelative;
            row["max_absolute"] = c.maxAbsolute;
            cases.push_back(std::move(row));
         }

         if (group.omittedCases > 0)
            g["omitted_cases"] = group.omittedCases;

         failures.push_back(std::move(g));
      }

      return doc;
   }

   private:
   static constexpr size_t MaxCasesPerGroup = 64;

   FailureGroup& groupFor(std::string_view phase, std::string_view id, std::string_view name)
   {
      const std::string idString(id);
      auto it = std::find_if(m_failures.begin(), m_failures.end(), [&](const FailureGroup& g) { return g.id == idString; });
      if (it != m_failures.end())
      {
         if (it->name.empty())
            it->name = std::string(name);
         if (it->phase.empty())
            it->phase = std::string(phase);
         return *it;
      }

      m_failures.push_back(FailureGroup{
         .phase = std::string(phase),
         .id = idString,
         .name = std::string(name),
      });
      return m_failures.back();
   }

   std::string m_suite;
   std::vector<FailureGroup> m_failures;
};

class TestFilter
{
   public:
   bool enabled() const { return m_enabled; }

   void enable() { m_enabled = true; }

   bool shouldRun(std::string_view id) const
   {
      return !m_enabled || m_ids.contains(std::string(id));
   }

   void add(std::string_view id)
   {
      m_enabled = true;
      const auto first = id.find_first_not_of(" \t\r\n");
      if (first == std::string_view::npos)
         return;
      const auto last = id.find_last_not_of(" \t\r\n");
      m_ids.insert(std::string(id.substr(first, last - first + 1)));
   }

   void addSeed(std::string_view id, uint32_t seed)
   {
      add(id);
      m_seeds[std::string(id)] = seed;
   }

   void addList(std::string_view ids)
   {
      m_enabled = true;
      while (!ids.empty())
      {
         const auto comma = ids.find(',');
         add(ids.substr(0, comma));
         if (comma == std::string_view::npos)
            return;
         ids.remove_prefix(comma + 1);
      }
   }

   std::optional<uint32_t> seedFor(std::string_view id) const
   {
      auto it = m_seeds.find(std::string(id));
      if (it == m_seeds.end())
         return {};
      return it->second;
   }

   private:
   bool m_enabled = false;
   std::set<std::string> m_ids;
   std::map<std::string, uint32_t> m_seeds;
};

struct RunControl
{
   bool valid = true;
   bool skipBenchmarks = false;
   std::string failedOutPath;
   TestFilter filter;
};

inline bool addFailedIdsFromFile(TestFilter& filter, const std::string& path, nbl::system::ILogger* logger)
{
   filter.enable();
   std::ifstream in(path);
   if (!in.is_open())
   {
      if (logger)
         logger->log("Failed to open failed-test manifest '%s'", nbl::system::ILogger::ELL_ERROR, path.c_str());
      return false;
   }

   nlohmann::json doc;
   try
   {
      in >> doc;
   }
   catch (const std::exception& e)
   {
      if (logger)
         logger->log("Failed to parse failed-test manifest '%s': %s", nbl::system::ILogger::ELL_ERROR, path.c_str(), e.what());
      return false;
   }

   const auto failuresIt = doc.find("failures");
   if (failuresIt == doc.end() || !failuresIt->is_array())
   {
      if (logger)
         logger->log("Failed-test manifest '%s' does not contain a failures array", nbl::system::ILogger::ELL_ERROR, path.c_str());
      return false;
   }

   for (const auto& failure : *failuresIt)
   {
      if (!failure.is_object())
         continue;
      const auto idIt = failure.find("id");
      if (idIt != failure.end() && idIt->is_string())
      {
         const std::string id = idIt->get<std::string>();
         const auto casesIt = failure.find("cases");
         if (casesIt != failure.end() && casesIt->is_array())
         {
            const auto seedIt = std::find_if(casesIt->begin(), casesIt->end(), [](const nlohmann::json& row) {
               if (!row.is_object())
                  return false;
               const auto it = row.find("seed");
               return it != row.end() && it->is_number_integer();
            });
            if (seedIt != casesIt->end())
            {
               filter.addSeed(id, (*seedIt)["seed"].get<uint32_t>());
               continue;
            }
         }
         filter.add(id);
      }
   }

   return true;
}

inline RunControl parseRunControl(std::span<const std::string> argv, nbl::system::ILogger* logger)
{
   RunControl out;

   for (size_t i = 1; i < argv.size(); ++i)
   {
      const std::string& arg = argv[i];
      if (arg == "--skip-benchmarks")
         out.skipBenchmarks = true;
      else if (arg == "--failed-out" && i + 1 < argv.size())
         out.failedOutPath = argv[++i];
      else if (arg.starts_with("--failed-out="))
         out.failedOutPath = arg.substr(std::string("--failed-out=").size());
      else if (arg == "--test" && i + 1 < argv.size())
         out.filter.addList(argv[++i]);
      else if (arg.starts_with("--test="))
         out.filter.addList(std::string_view(arg).substr(std::string_view("--test=").size()));
      else if (arg == "--rerun-failed" && i + 1 < argv.size())
      {
         if (!addFailedIdsFromFile(out.filter, argv[++i], logger))
            out.valid = false;
      }
      else if (arg.starts_with("--rerun-failed="))
      {
         if (!addFailedIdsFromFile(out.filter, arg.substr(std::string("--rerun-failed=").size()), logger))
            out.valid = false;
      }
   }

   if (out.filter.enabled())
      out.skipBenchmarks = true;

   return out;
}

inline bool writeFailureManifestFile(const FailureManifest& manifest, const std::string& path, nbl::system::ILogger* logger)
{
   std::ofstream out(path, std::ios::out | std::ios::trunc);
   if (!out.is_open())
   {
      if (logger)
         logger->log("Failed to open failed-test manifest '%s' for writing", nbl::system::ILogger::ELL_ERROR, path.c_str());
      return false;
   }

   out << manifest.toJson().dump(3) << '\n';
   if (!out.good())
   {
      if (logger)
         logger->log("Failed to write failed-test manifest '%s'", nbl::system::ILogger::ELL_ERROR, path.c_str());
      return false;
   }

   if (logger)
      logger->log("Wrote failed-test manifest '%s' with %llu failed groups", nbl::system::ILogger::ELL_INFO,
         path.c_str(), static_cast<unsigned long long>(manifest.failures().size()));
   return true;
}

} // namespace nbl::examples::testing

#endif
