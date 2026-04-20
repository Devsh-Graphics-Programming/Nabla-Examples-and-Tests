// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"

#include "nbl/examples/git/info.h"

#include "nlohmann/json.hpp"
#include "nbl/core/hash/blake.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace
{

using perf_json_t = nlohmann::ordered_json;

constexpr std::string_view PerfSchemaVersion = "meshloaders-perf-run-v1";
constexpr std::string_view PerfProtocolVersion = "meshloaders-roundtrip-v1";

std::string normalizeProfileComponent(std::string_view value)
{
    std::string normalized;
    normalized.reserve(value.size());
    bool lastWasSeparator = false;
    for (const auto ch : value)
    {
        if (std::isalnum(static_cast<unsigned char>(ch)))
        {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            lastWasSeparator = false;
        }
        else if (!lastWasSeparator)
        {
            normalized.push_back('-');
            lastWasSeparator = true;
        }
    }
    while (!normalized.empty() && normalized.back() == '-')
        normalized.pop_back();
    if (normalized.empty())
        normalized = "unknown";
    return normalized;
}

std::string hashToHex(const nbl::core::blake3_hash_t& hash)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (const auto byte : hash.data)
        oss << std::setw(2) << static_cast<uint32_t>(byte);
    return oss.str();
}

std::string currentTimestampTag()
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string currentTimestampIsoUtc()
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string runModeName(const uint32_t modeValue)
{
    switch (modeValue)
    {
    case 0u:
        return "interactive";
    case 1u:
        return "batch";
    case 2u:
        return "ci";
    default:
        return "unknown";
    }
}

std::string runtimeTuningModeName(const asset::SFileIOPolicy::SRuntimeTuning::Mode mode)
{
    switch (mode)
    {
    case asset::SFileIOPolicy::SRuntimeTuning::Mode::Sequential:
        return "sequential";
    case asset::SFileIOPolicy::SRuntimeTuning::Mode::Heuristic:
        return "heuristic";
    case asset::SFileIOPolicy::SRuntimeTuning::Mode::Hybrid:
        return "hybrid";
    default:
        return "unknown";
    }
}

std::string platformName()
{
#if defined(_WIN32)
    return "windows";
#elif defined(__linux__)
    return "linux";
#elif defined(__ANDROID__)
    return "android";
#else
    return "unknown";
#endif
}

std::string buildConfigName()
{
#if defined(NBL_MESHLOADERS_BUILD_CONFIG)
    return NBL_MESHLOADERS_BUILD_CONFIG;
#elif defined(_DEBUG)
    return "Debug";
#else
    return "Release";
#endif
}

perf_json_t dirtyStateJson(const std::optional<bool>& dirty)
{
    if (!dirty.has_value())
        return nullptr;
    return perf_json_t(dirty.value());
}

std::filesystem::path normalizePathForPerf(const std::filesystem::path& path)
{
    if (path.empty())
        return {};

    std::error_code ec;
    auto normalized = path.lexically_normal();
    if (std::filesystem::exists(normalized, ec) && !ec)
    {
        auto canonical = std::filesystem::weakly_canonical(normalized, ec);
        if (!ec)
            return canonical.lexically_normal();
        ec.clear();
    }
    else
        ec.clear();

    if (!normalized.is_absolute())
    {
        auto absolute = std::filesystem::absolute(normalized, ec);
        if (!ec)
            return absolute.lexically_normal();
        ec.clear();
    }

    return normalized;
}

std::optional<std::string> tryRelativePerfPath(const std::filesystem::path& targetPath, const std::filesystem::path& basePath)
{
    if (targetPath.empty() || basePath.empty())
        return std::nullopt;

    std::error_code ec;
    const auto normalizedTarget = normalizePathForPerf(targetPath);
    const auto normalizedBase = normalizePathForPerf(basePath);
    const auto relative = std::filesystem::relative(normalizedTarget, normalizedBase, ec);
    if (ec || relative.empty() || relative.is_absolute())
        return std::nullopt;

    return relative.lexically_normal().generic_string();
}

std::string makePortablePerfPath(const std::filesystem::path& path, const std::optional<std::filesystem::path>& preferredBase = std::nullopt)
{
    if (path.empty())
        return {};

    auto normalized = path.lexically_normal();
    if (!normalized.is_absolute())
        return normalized.generic_string();

    if (preferredBase.has_value())
        if (auto relative = tryRelativePerfPath(path, *preferredBase); relative.has_value())
            return *relative;

    std::error_code ec;
    const auto cwd = std::filesystem::current_path(ec);
    if (!ec)
        if (auto relative = tryRelativePerfPath(path, cwd); relative.has_value())
            return *relative;

    normalized = normalizePathForPerf(path);
    if (!normalized.filename().empty())
        return normalized.filename().generic_string();

    return normalizeProfileComponent(normalized.generic_string());
}

perf_json_t toJson(const MeshLoadersApp::LoadStageMetrics& metrics)
{
    return perf_json_t{
        {"valid", metrics.valid},
        {"input_size", metrics.inputSize},
        {"get_asset_ms", metrics.getAssetMs},
        {"extract_ms", metrics.extractMs},
        {"total_ms", metrics.totalMs},
        {"non_loader_ms", metrics.nonLoaderMs}
    };
}

perf_json_t toJson(const MeshLoadersApp::WriteStageMetrics& metrics)
{
    return perf_json_t{
        {"valid", metrics.valid},
        {"output_size", metrics.outputSize},
        {"open_ms", metrics.openMs},
        {"write_ms", metrics.writeMs},
        {"stat_ms", metrics.statMs},
        {"total_ms", metrics.totalMs},
        {"non_writer_ms", metrics.nonWriterMs},
        {"used_memory_transport", metrics.usedMemoryTransport},
        {"used_disk_fallback", metrics.usedDiskFallback},
        {"persisted_disk_artifact", metrics.persistedDiskArtifact}
    };
}

bool metricRegression(const double current, const double reference, const double relativeThreshold, const double absoluteThreshold)
{
    if (reference <= 0.0)
        return current > absoluteThreshold;
    const double allowed = std::max(reference * (1.0 + relativeThreshold), reference + absoluteThreshold);
    return current > allowed;
}

void compareMetric(core::vector<std::string>& failures, const std::string& caseName, const std::string_view metricName, const double current, const double reference, const double relativeThreshold, const double absoluteThreshold)
{
    if (!metricRegression(current, reference, relativeThreshold, absoluteThreshold))
        return;

    std::ostringstream oss;
    oss << caseName << ": " << metricName << " regressed from " << reference << " ms to " << current << " ms";
    failures.push_back(oss.str());
}

perf_json_t buildCaseJson(const MeshLoadersApp::CasePerformanceMetrics& metrics, const std::optional<std::filesystem::path>& testListDir)
{
    return perf_json_t{
        {"name", metrics.caseName},
        {"input_path", makePortablePerfPath(metrics.inputPath, testListDir)},
        {"original_load", toJson(metrics.originalLoad)},
        {"write", toJson(metrics.write)},
        {"written_load", toJson(metrics.writtenLoad)}
    };
}

bool writePerfJson(system::ISystem* const system, const system::path& path, const perf_json_t& json)
{
    if (!system)
        return false;

    const auto parentDir = path.parent_path();
    if (!parentDir.empty())
        std::filesystem::create_directories(parentDir);
    system->deleteFile(path);

    system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> writeFileFuture;
    system->createFile(writeFileFuture, path, system::IFile::ECF_WRITE);
    core::smart_refctd_ptr<system::IFile> writeFile;
    writeFileFuture.acquire().move_into(writeFile);
    if (!writeFile)
        return false;

    const auto serialized = json.dump(2);
    size_t written = 0ull;
    while (written < serialized.size())
    {
        system::IFile::success_t success;
        writeFile->write(success, serialized.data() + written, written, serialized.size() - written);
        const auto processed = success.getBytesProcessed();
        if (!success || processed == 0ull)
            return false;
        written += processed;
    }
    return true;
}

}

bool MeshLoadersApp::performanceEnabled() const
{
    return m_perf.enabled;
}

void MeshLoadersApp::beginPerformanceRun()
{
    if (!performanceEnabled())
        return;

    m_perf.finalized = false;
    m_perf.completedCases.clear();
    m_perf.completedCases.reserve(m_runtime.cases.size());
    m_perf.comparisonFailures.clear();
    m_perf.runStart = std::chrono::steady_clock::now();

    const auto systemInfo = m_system->getSystemInfo();
    const auto normalizedCpuName = normalizeProfileComponent(systemInfo.cpuName);
    if (m_perf.options.profileOverride.has_value())
        m_perf.profileId = normalizeProfileComponent(*m_perf.options.profileOverride);
    else
    {
        std::ostringstream profile;
        profile << platformName()
            << "__" << normalizedCpuName
            << "__thr-" << std::thread::hardware_concurrency()
            << "__" << normalizeProfileComponent(buildConfigName());
        m_perf.profileId = normalizeProfileComponent(profile.str());
    }

    nbl::core::blake3_hasher workloadHasher;
    workloadHasher.update(PerfProtocolVersion.data(), PerfProtocolVersion.size());
    workloadHasher << static_cast<uint32_t>(m_runtime.mode);
    workloadHasher << static_cast<uint32_t>(m_runtimeTuningMode);
    workloadHasher << m_runtime.rowViewEnabled;
    if (!m_output.testListPath.empty() && std::filesystem::exists(m_output.testListPath))
    {
        std::ifstream stream(m_output.testListPath, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        workloadHasher.update(content.data(), content.size());
    }
    else
    {
        workloadHasher << m_runtime.cases.size();
        for (const auto& testCase : m_runtime.cases)
        {
            workloadHasher << testCase.name;
            workloadHasher << makePortablePerfPath(testCase.path);
        }
    }
    m_perf.workloadId = hashToHex(static_cast<nbl::core::blake3_hash_t>(workloadHasher));
}

void MeshLoadersApp::beginPerformanceCase(const TestCase& testCase)
{
    if (!performanceEnabled())
        return;

    m_perf.currentCaseIndex = m_perf.completedCases.size();
    m_perf.completedCases.push_back(CasePerformanceMetrics{
        .caseName = testCase.name,
        .inputPath = testCase.path
    });
}

void MeshLoadersApp::recordOriginalLoadMetrics(const LoadStageMetrics& metrics)
{
    if (!performanceEnabled() || m_perf.currentCaseIndex >= m_perf.completedCases.size())
        return;
    m_perf.completedCases[m_perf.currentCaseIndex].originalLoad = metrics;
}

void MeshLoadersApp::recordWrittenLoadMetrics(const LoadStageMetrics& metrics)
{
    if (!performanceEnabled() || m_perf.currentCaseIndex >= m_perf.completedCases.size())
        return;
    m_perf.completedCases[m_perf.currentCaseIndex].writtenLoad = metrics;
}

void MeshLoadersApp::recordWriteMetrics(const WriteStageMetrics& metrics)
{
    if (!performanceEnabled() || m_perf.currentCaseIndex >= m_perf.completedCases.size())
        return;
    m_perf.completedCases[m_perf.currentCaseIndex].write = metrics;
}

void MeshLoadersApp::recordWriteMetrics(const WrittenAssetResult& result)
{
    WriteStageMetrics metrics = {};
    metrics.openMs = result.openMs;
    metrics.writeMs = result.writeMs;
    metrics.statMs = result.statMs;
    metrics.totalMs = result.totalWriteMs;
    metrics.nonWriterMs = result.nonWriterMs;
    metrics.outputSize = result.outputSize;
    metrics.usedMemoryTransport = result.usedMemoryTransport;
    metrics.usedDiskFallback = result.usedDiskFallback;
    metrics.persistedDiskArtifact = result.persistedDiskArtifact;
    metrics.valid = true;
    recordWriteMetrics(metrics);
}

void MeshLoadersApp::endPerformanceCase()
{
    if (!performanceEnabled() || m_perf.currentCaseIndex >= m_perf.completedCases.size())
        return;

    m_perf.currentCaseIndex = ~size_t(0u);
}

void MeshLoadersApp::finalizePerformanceRun()
{
    if (!performanceEnabled() || m_perf.finalized)
        return;

    m_perf.finalized = true;

    perf_json_t root = {};
    root["schema_version"] = PerfSchemaVersion;
    root["protocol_version"] = PerfProtocolVersion;
    root["profile_id"] = m_perf.profileId;
    root["workload_id"] = m_perf.workloadId;
    root["run_mode"] = runModeName(static_cast<uint32_t>(m_runtime.mode));
    root["runtime_tuning"] = runtimeTuningModeName(m_runtimeTuningMode);
    root["provenance"] = {
        {"created_at_utc", currentTimestampIsoUtc()},
        {"nabla_commit", std::string(nbl::gtml::nabla_git_info.commitHash())},
        {"nabla_dirty", dirtyStateJson(nbl::gtml::nabla_git_info.hasUncommittedChanges())},
        {"examples_commit", std::string(nbl::examples::gtml::examples_git_info.commitHash())},
        {"examples_dirty", dirtyStateJson(nbl::examples::gtml::examples_git_info.hasUncommittedChanges())}
    };

    const auto systemInfo = m_system->getSystemInfo();
    root["environment"] = {
        {"platform", platformName()},
        {"os_full_name", systemInfo.OSFullName},
        {"cpu_name", systemInfo.cpuName},
        {"physical_core_count", systemInfo.physicalCoreCount},
        {"gpu_name", m_physicalDevice ? m_physicalDevice->getProperties().deviceName : "unknown"},
        {"total_memory_bytes", systemInfo.totalMemory},
        {"available_memory_bytes", systemInfo.availableMemory},
        {"hardware_concurrency", std::thread::hardware_concurrency()},
        {"build_config", buildConfigName()}
    };
    const auto testListDir = m_output.testListPath.empty() ? std::optional<std::filesystem::path>{} : std::optional<std::filesystem::path>{m_output.testListPath.parent_path()};
    const auto testListName = m_output.testListPath.empty() ? std::string{} : m_output.testListPath.filename().generic_string();
    root["inputs"] = {
        {"test_list_name", testListName},
        {"row_view_enabled", m_runtime.rowViewEnabled},
        {"case_count", m_runtime.cases.size()}
    };

    root["cases"] = perf_json_t::array();
    for (const auto& metrics : m_perf.completedCases)
        root["cases"].push_back(buildCaseJson(metrics, testListDir));
    root["totals"] = {
        {"run_wall_ms", toMs(std::chrono::steady_clock::now() - m_perf.runStart)}
    };

    if (m_perf.options.referenceDir)
    {
        m_perf.referencePath = *m_perf.options.referenceDir / m_perf.workloadId / (m_perf.profileId + ".json");
        root["reference"]["lookup_key"] = m_perf.workloadId + "/" + m_perf.profileId + ".json";
        if (!m_perf.options.updateReference && std::filesystem::exists(m_perf.referencePath))
        {
            m_perf.referenceMatched = true;
            std::ifstream stream(m_perf.referencePath);
            perf_json_t reference;
            stream >> reference;

            if (!reference.contains("cases") || !reference["cases"].is_array())
                m_perf.comparisonFailures.push_back("Reference file does not contain a valid cases array.");
            else
            {
                std::unordered_map<std::string, perf_json_t> referenceCases;
                for (const auto& caseJson : reference["cases"])
                    referenceCases.emplace(caseJson.value("name", ""), caseJson);

                if (referenceCases.size() != m_perf.completedCases.size())
                    m_perf.comparisonFailures.push_back("Reference case count does not match the current run.");

                for (const auto& metrics : m_perf.completedCases)
                {
                    const auto refIt = referenceCases.find(metrics.caseName);
                    if (refIt == referenceCases.end())
                    {
                        m_perf.comparisonFailures.push_back(metrics.caseName + ": missing reference case.");
                        continue;
                    }

                    const auto& refCase = refIt->second;
                    compareMetric(m_perf.comparisonFailures, metrics.caseName, "original_load.total_ms", metrics.originalLoad.totalMs, refCase["original_load"].value("total_ms", 0.0), 0.20, 5.0);
                    compareMetric(m_perf.comparisonFailures, metrics.caseName, "write.total_ms", metrics.write.totalMs, refCase["write"].value("total_ms", 0.0), 0.20, 5.0);
                    compareMetric(m_perf.comparisonFailures, metrics.caseName, "written_load.total_ms", metrics.writtenLoad.totalMs, refCase["written_load"].value("total_ms", 0.0), 0.20, 5.0);

                    const bool refUsedMemoryTransport = refCase["write"].value("used_memory_transport", false);
                    if (metrics.write.valid && metrics.write.usedMemoryTransport != refUsedMemoryTransport)
                        m_perf.comparisonFailures.push_back(metrics.caseName + ": memory transport usage does not match the reference.");
                }
            }
        }
    }
    root["reference"]["matched"] = m_perf.referenceMatched;
    root["reference"]["strict"] = m_perf.options.strict;
    root["reference"]["updated"] = m_perf.options.updateReference;
    root["reference"]["comparison_failures"] = m_perf.comparisonFailures;

    if (m_perf.options.dumpDir)
    {
        const auto dumpDir = *m_perf.options.dumpDir / m_perf.workloadId;
        std::filesystem::create_directories(dumpDir);
        m_perf.dumpPath = dumpDir / (currentTimestampTag() + "__" + m_perf.profileId + ".json");
        if (!writePerfJson(m_system.get(), m_perf.dumpPath, root))
            failExit("Failed to write performance dump file: %s", m_perf.dumpPath.string().c_str());
    }
    if (m_perf.options.updateReference)
    {
        if (!writePerfJson(m_system.get(), m_perf.referencePath, root))
            failExit("Failed to write performance reference file: %s", m_perf.referencePath.string().c_str());
    }

    if (m_logger)
    {
        if (m_perf.options.updateReference)
            m_logger->log("Performance reference updated for workload=%s profile=%s.", ILogger::ELL_INFO, m_perf.workloadId.c_str(), m_perf.profileId.c_str());
        else if (!m_perf.referenceMatched)
            m_logger->log("Performance reference not found for workload=%s profile=%s.", ILogger::ELL_INFO, m_perf.workloadId.c_str(), m_perf.profileId.c_str());
        else if (m_perf.comparisonFailures.empty())
            m_logger->log("Performance reference comparison passed for workload=%s profile=%s.", ILogger::ELL_INFO, m_perf.workloadId.c_str(), m_perf.profileId.c_str());
        else
            for (const auto& failure : m_perf.comparisonFailures)
                m_logger->log("%s", ILogger::ELL_ERROR, failure.c_str());
    }

    if (m_perf.options.strict && m_perf.referenceMatched && !m_perf.comparisonFailures.empty())
        failExit("Structured performance comparison failed.");
}
