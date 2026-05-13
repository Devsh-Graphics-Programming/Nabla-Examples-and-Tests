// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_COMMON_BENCHMARK_JSON_INCLUDED_
#define _NBL_COMMON_BENCHMARK_JSON_INCLUDED_

#include <nabla.h>
#include "nbl/examples/Benchmark/BenchmarkTypes.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace benchmark_json
{

// Builds the "device" JSON object from a physical device, or null if dev is null.
inline nlohmann::json buildDeviceMetadata(const nbl::video::IPhysicalDevice* dev)
{
   if (!dev)
      return nullptr;
   const auto&    p     = dev->getProperties();
   nlohmann::json out   = nlohmann::json::object();
   out["name"]          = std::string(p.deviceName);
   out["vendorID"]      = p.vendorID;
   out["deviceID"]      = p.deviceID;
   out["driverID"]      = static_cast<int>(p.driverID);
   out["driverName"]    = std::string(p.driverName);
   out["driverInfo"]    = std::string(p.driverInfo);
   out["driverVersion"] = p.driverVersion;
   out["deviceUUID"]    = std::vector<uint8_t>(p.deviceUUID, p.deviceUUID + 16);
   out["driverUUID"]    = std::vector<uint8_t>(p.driverUUID, p.driverUUID + 16);
   return out;
}

// Parses a JSON report file into a Baseline. Returns nullopt on missing /
// unparseable / empty file. Caller is responsible for appending / replacing
// in their baseline store and for feeding rows into BenchmarkConsole widths.
inline std::optional<Baseline> loadBaselineFile(std::string label, const std::string& path)
{
   std::ifstream f(path);
   if (!f.is_open())
      return std::nullopt;

   nlohmann::json j;
   try
   {
      f >> j;
   }
   catch (const std::exception&)
   {
      return std::nullopt;
   }

   const auto resultsIt = j.find("results");
   if (resultsIt == j.end() || !resultsIt->is_array())
      return std::nullopt;

   std::unordered_map<std::string, BaselineRow> rowsByName;
   for (const auto& r : *resultsIt)
   {
      const auto n  = r.find("name");
      const auto ps = r.find("ps_per_sample");
      if (n == r.end() || ps == r.end())
         continue;
      if (!n->is_array() || !ps->is_number())
         continue;
      std::vector<std::string> nameVec;
      nameVec.reserve(n->size());
      for (const auto& seg : *n)
      {
         if (!seg.is_string())
         {
            nameVec.clear();
            break;
         }
         nameVec.emplace_back(seg.get<std::string>());
      }
      if (nameVec.empty())
         continue;
         
      BaselineRow row;
      row.psPerSample     = ps->get<double>();
      row.registerCount   = r.at("regs").get<uint64_t>();
      row.codeSizeBytes   = r.at("code_bytes").get<uint64_t>();
      row.sharedMemBytes  = r.at("shared_mem_bytes").get<uint64_t>();
      row.privateMemBytes = r.at("local_mem_bytes").get<uint64_t>();
      row.stackBytes      = r.at("stack_bytes").get<uint64_t>();
      row.subgroupSize    = r.at("subgroup_size").get<uint64_t>();

      auto readUvec3 = [&](const char* key, nbl::hlsl::uint32_t3& out)
      {
         const auto& a = r.at(key);
         out.x         = a[0].get<uint32_t>();
         out.y         = a[1].get<uint32_t>();
         out.z         = a[2].get<uint32_t>();
      };
      readUvec3("workgroup_size", row.workload.shape.workgroupSize);
      readUvec3("dispatch_groups", row.workload.shape.dispatchGroupCount);
      row.workload.shape.samplesPerDispatch = r.at("samples_per_dispatch").get<uint64_t>();
      row.workload.benchDispatches          = r.at("bench_dispatches").get<uint32_t>();

      rowsByName[makeKey(nameVec)] = row;
   }
   if (rowsByName.empty())
      return std::nullopt;

   return Baseline {std::move(label), path, j.at("device"), std::move(rowsByName)};
}

// Writes a JSON report. Preserves rows in the prior file whose names weren't
// re-measured this run, so writeReportFile can be an intermediate checkpoint
// during a multi-bench-class session. Returns preservedCount via out-param.
inline bool writeReportFile(const std::string& path, const nlohmann::json& deviceMetadata, const std::vector<Baseline>& baselines, const std::vector<Result>& results, nbl::system::ILogger* logger, size_t* outPreservedCount = nullptr)
{
   nlohmann::json doc;
   doc["version"] = 1;

   if (!deviceMetadata.is_null())
      doc["device"] = deviceMetadata;

   if (!baselines.empty())
   {
      auto& baselinesNode = doc["baselines"] = nlohmann::json::object();
      for (const auto& b : baselines)
         baselinesNode[b.label] = b.path;
   }
   auto& resultsNode = doc["results"] = nlohmann::json::array();

   std::unordered_set<std::string> currentKeys;
   currentKeys.reserve(results.size());
   for (const auto& r : results)
      currentKeys.insert(makeKey(r.name));

   for (const auto& r : results)
   {
      nlohmann::json row;
      row["name"]             = r.name;
      row["ps_per_sample"]    = r.timing.ps_per_sample;
      row["gsamples_per_s"]   = r.timing.gsamples_per_s;
      row["ms_total"]         = r.timing.ms_total;
      row["regs"]             = r.stats.registerCount;
      row["code_bytes"]       = r.stats.codeSizeBytes;
      row["shared_mem_bytes"] = r.stats.sharedMemBytes;
      row["local_mem_bytes"]  = r.stats.privateMemBytes;
      row["stack_bytes"]      = r.stats.stackBytes;
      row["subgroup_size"]    = r.stats.subgroupSize;

      // Structured so JSON preserves the exact numeric type.
      if (!r.stats.unknowns.empty())
      {
         using F   = nbl::video::IGPUPipelineBase::SExecutableStatistic::FORMAT;
         auto& arr = row["unknown_stats"] = nlohmann::json::array();
         for (const auto& s : r.stats.unknowns)
         {
            nlohmann::json entry;
            entry["name"] = s.name;
            switch (s.format)
            {
               case F::BOOL32:
                  entry["type"]  = "bool";
                  entry["value"] = s.value.b32;
                  break;
               case F::INT64:
                  entry["type"]  = "int";
                  entry["value"] = s.value.i64;
                  break;
               case F::UINT64:
                  entry["type"]  = "uint";
                  entry["value"] = s.value.u64;
                  break;
               case F::FLOAT64:
                  entry["type"]  = "float";
                  entry["value"] = s.value.f64;
                  break;
            }
            arr.push_back(std::move(entry));
         }
      }

      row["workgroup_size"]       = {r.workload.shape.workgroupSize.x, r.workload.shape.workgroupSize.y, r.workload.shape.workgroupSize.z};
      row["dispatch_groups"]      = {r.workload.shape.dispatchGroupCount.x, r.workload.shape.dispatchGroupCount.y, r.workload.shape.dispatchGroupCount.z};
      row["samples_per_dispatch"] = r.workload.shape.samplesPerDispatch;
      row["bench_dispatches"]     = r.workload.benchDispatches;

      resultsNode.push_back(std::move(row));
   }

   // Caveat: renamed/removed variants linger forever. Delete the output JSON
   // to get a clean slate.
   size_t preservedCount = 0;
   {
      std::ifstream in(path);
      if (in.is_open())
      {
         nlohmann::json existing;
         try
         {
            in >> existing;
         }
         catch (const std::exception&)
         {
            existing = nullptr;
         }
         const auto rIt = existing.find("results");
         if (rIt != existing.end() && rIt->is_array())
         {
            for (const auto& priorRow : *rIt)
            {
               const auto n = priorRow.find("name");
               if (n == priorRow.end() || !n->is_array())
                  continue;
               std::vector<std::string> nameVec;
               bool                     ok = true;
               for (const auto& seg : *n)
               {
                  if (!seg.is_string())
                  {
                     ok = false;
                     break;
                  }
                  nameVec.emplace_back(seg.get<std::string>());
               }
               if (!ok || nameVec.empty())
                  continue;
               if (currentKeys.find(makeKey(nameVec)) != currentKeys.end())
                  continue; // re-measured this run

               resultsNode.push_back(priorRow);
               ++preservedCount;
            }
         }
      }
   }

   std::ofstream f(path, std::ios::out | std::ios::trunc);
   if (!f.is_open())
   {
      benchLogFmt(logger, nbl::system::ILogger::ELL_ERROR, "benchmark_json::writeReportFile: failed to open '{}'", path);
      return false;
   }

   // One result per line keeps `git diff` showing one row per change instead
   // of N lines per row.
   f << "{\n";
   f << "  \"version\": " << doc["version"].dump() << ",\n";
   if (doc.contains("device"))
   {
      // Compact value render so byte arrays (deviceUUID etc.) stay inline.
      const auto& dev = doc["device"];
      f << "  \"device\": {\n";
      bool first = true;
      for (auto it = dev.begin(); it != dev.end(); ++it)
      {
         if (!first)
            f << ",\n";
         first = false;
         f << "    \"" << it.key() << "\": " << it.value().dump();
      }
      f << "\n  },\n";
   }
   if (doc.contains("baselines"))
      f << "  \"baselines\": " << doc["baselines"].dump() << ",\n";
   f << "  \"results\": [";
   for (size_t i = 0; i < resultsNode.size(); ++i)
   {
      f << (i ? ",\n    " : "\n    ");
      f << resultsNode[i].dump();
   }
   f << (resultsNode.empty() ? "]\n" : "\n  ]\n");
   f << "}\n";

   if (outPreservedCount)
      *outPreservedCount = preservedCount;
   return true;
}

} // namespace benchmark_json

#endif
