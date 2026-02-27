// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "MeshLoadersApp.hpp"

#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/asset/interchange/SGeometryContentHashCommon.h"
#include "nbl/core/hash/blake.h"

namespace
{

core::blake3_hash_t meshloadersHashBufferLegacySequential(const asset::ICPUBuffer* const buffer)
{
    if (!buffer)
        return static_cast<core::blake3_hash_t>(core::blake3_hasher{});
    const auto* const ptr = buffer->getPointer();
    const size_t size = buffer->getSize();
    if (!ptr || size == 0ull)
        return static_cast<core::blake3_hash_t>(core::blake3_hasher{});
    core::blake3_hasher hasher;
    hasher.update(ptr, size);
    return static_cast<core::blake3_hash_t>(hasher);
}

}

std::string MeshLoadersApp::makeUniqueCaseName(const system::path& path)
{
    auto base = path.stem().string();
    if (base.empty())
        base = "case";
    auto& counter = m_caseNameCounts[base];
    std::string name = (counter == 0u) ? base : (base + "_" + std::to_string(counter));
    ++counter;
    return name;
}

double MeshLoadersApp::toMs(const std::chrono::high_resolution_clock::duration& d)
{
    return std::chrono::duration<double, std::milli>(d).count();
}

std::string MeshLoadersApp::makeCacheKey(const system::path& path) const
{
    return path.lexically_normal().generic_string();
}

void MeshLoadersApp::logRowViewPerf(const RowViewPerfStats& stats) const
{
    if (!m_logger)
        return;
    m_logger->log(
        "RowView perf: mode=%s cases=%llu cpuHit=%llu cpuMiss=%llu gpuHit=%llu gpuMiss=%llu convert=%llu add=%llu total=%.3f ms",
        ILogger::ELL_INFO,
        stats.incremental ? "inc" : "full",
        static_cast<unsigned long long>(stats.cases),
        static_cast<unsigned long long>(stats.cpuHits),
        static_cast<unsigned long long>(stats.cpuMisses),
        static_cast<unsigned long long>(stats.gpuHits),
        static_cast<unsigned long long>(stats.gpuMisses),
        static_cast<unsigned long long>(stats.convertCount),
        static_cast<unsigned long long>(stats.addCount),
        stats.totalMs);
    m_logger->log(
        "RowView perf: clear=%.3f load=%.3f extract=%.3f aabb=%.3f convert=%.3f add=%.3f layout=%.3f inst=%.3f cam=%.3f",
        ILogger::ELL_INFO,
        stats.clearMs,
        stats.loadMs,
        stats.extractMs,
        stats.aabbMs,
        stats.convertMs,
        stats.addGeoMs,
        stats.layoutMs,
        stats.instanceMs,
        stats.cameraMs);
}

void MeshLoadersApp::logRowViewAssetLoad(const system::path& path, const double ms, const bool cached) const
{
    if (!m_logger)
        return;
    m_logger->log(
        "RowView perf: asset %s load=%.3f ms%s",
        ILogger::ELL_INFO,
        path.string().c_str(),
        ms,
        cached ? " (cached)" : "");
}

void MeshLoadersApp::logRowViewLoadTotal(const double ms, const size_t hits, const size_t misses) const
{
    if (!m_logger)
        return;
    m_logger->log(
        "RowView perf: asset load total=%.3f ms hits=%llu misses=%llu",
        ILogger::ELL_INFO,
        ms,
        static_cast<unsigned long long>(hits),
        static_cast<unsigned long long>(misses));
}

core::blake3_hash_t MeshLoadersApp::hashGeometry(const ICPUPolygonGeometry* geo)
{
    return CPolygonGeometryManipulator::computeDeterministicContentHash(geo);
}

bool MeshLoadersApp::runHashConsistencyChecks()
{
    using clock_t = std::chrono::high_resolution_clock;

    if (m_cases.empty())
        return logFail("Hash test requires at least one test case.");

    IAssetLoader::SAssetLoadParams params = makeLoadParams();
    params.logger = nullptr;
    params.loaderFlags = static_cast<IAssetLoader::E_LOADER_PARAMETER_FLAGS>(params.loaderFlags | IAssetLoader::ELPF_DONT_COMPUTE_CONTENT_HASHES);

    double totalLoadMs = 0.0;
    double totalLegacySequentialMs = 0.0;
    double totalNewSequentialMs = 0.0;
    double totalNewParallelMs = 0.0;
    uint64_t totalGeometryCount = 0ull;
    uint64_t totalBufferCount = 0ull;

    for (const auto& testCase : m_cases)
    {
        m_assetMgr->clearAllAssetCache();

        AssetLoadCallResult loadResult = {};
        if (!loadAssetCallFromPath(testCase.path, params, loadResult))
            failExit("Hash test failed to load input %s.", testCase.path.string().c_str());
        totalLoadMs += loadResult.getAssetMs;

        if (loadResult.bundle.getContents().empty())
            failExit("Hash test loaded empty asset for %s.", testCase.path.string().c_str());

        core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
        if (!appendGeometriesFromBundle(loadResult.bundle, geometries))
            failExit("Hash test found no polygon geometry in %s.", testCase.path.string().c_str());

        double caseLegacySequentialMs = 0.0;
        double caseNewSequentialMs = 0.0;
        double caseNewParallelMs = 0.0;
        uint64_t caseBufferCount = 0ull;

        for (size_t geoIx = 0u; geoIx < geometries.size(); ++geoIx)
        {
            auto* geometry = const_cast<ICPUPolygonGeometry*>(geometries[geoIx].get());
            if (!geometry)
                failExit("Hash test failed to access geometry %llu in %s.", static_cast<unsigned long long>(geoIx), testCase.path.string().c_str());

            core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
            asset::collectGeometryBuffers(geometry, buffers);
            if (buffers.empty())
                continue;

            core::vector<core::blake3_hash_t> legacySequentialHashes;
            core::vector<core::blake3_hash_t> newSequentialHashes;
            core::vector<core::blake3_hash_t> newParallelHashes;
            legacySequentialHashes.reserve(buffers.size());
            newSequentialHashes.reserve(buffers.size());
            newParallelHashes.reserve(buffers.size());

            for (const auto& buffer : buffers)
            {
                if (!buffer)
                    continue;

                const auto* const ptr = buffer->getPointer();
                const size_t size = buffer->getSize();

                const auto legacyStart = clock_t::now();
                const auto legacyHash = meshloadersHashBufferLegacySequential(buffer.get());
                caseLegacySequentialMs += toMs(clock_t::now() - legacyStart);
                legacySequentialHashes.push_back(legacyHash);

                const auto newSeqStart = clock_t::now();
                const auto newSeqHash = core::blake3_hash_buffer_sequential(ptr, size);
                caseNewSequentialMs += toMs(clock_t::now() - newSeqStart);
                newSequentialHashes.push_back(newSeqHash);

                const auto newParStart = clock_t::now();
                const auto newParHash = core::blake3_hash_buffer(ptr, size);
                caseNewParallelMs += toMs(clock_t::now() - newParStart);
                newParallelHashes.push_back(newParHash);
            }

            if (legacySequentialHashes.size() != newSequentialHashes.size() || legacySequentialHashes.size() != newParallelHashes.size())
                failExit("Hash test buffer count mismatch for %s geo=%llu.", testCase.path.string().c_str(), static_cast<unsigned long long>(geoIx));

            for (size_t hashIx = 0u; hashIx < legacySequentialHashes.size(); ++hashIx)
            {
                if (legacySequentialHashes[hashIx] == newSequentialHashes[hashIx] && legacySequentialHashes[hashIx] == newParallelHashes[hashIx])
                    continue;
                failExit(
                    "Hash mismatch for %s geo=%llu buffer=%llu legacy_seq=%s new_seq=%s new_parallel=%s",
                    testCase.path.string().c_str(),
                    static_cast<unsigned long long>(geoIx),
                    static_cast<unsigned long long>(hashIx),
                    geometryHashToHex(legacySequentialHashes[hashIx]).c_str(),
                    geometryHashToHex(newSequentialHashes[hashIx]).c_str(),
                    geometryHashToHex(newParallelHashes[hashIx]).c_str());
            }

            caseBufferCount += legacySequentialHashes.size();
            ++totalGeometryCount;
        }

        totalLegacySequentialMs += caseLegacySequentialMs;
        totalNewSequentialMs += caseNewSequentialMs;
        totalNewParallelMs += caseNewParallelMs;
        totalBufferCount += caseBufferCount;

        if (m_logger)
        {
            m_logger->log(
                "Hash test case: %s load=%.3f ms geos=%llu buffers=%llu legacy_seq=%.3f ms new_seq=%.3f ms new_parallel=%.3f ms",
                ILogger::ELL_INFO,
                testCase.path.string().c_str(),
                loadResult.getAssetMs,
                static_cast<unsigned long long>(geometries.size()),
                static_cast<unsigned long long>(caseBufferCount),
                caseLegacySequentialMs,
                caseNewSequentialMs,
                caseNewParallelMs);
        }
    }

    if (m_logger)
    {
        m_logger->log(
            "Hash test summary: cases=%llu geos=%llu buffers=%llu load=%.3f ms legacy_seq=%.3f ms new_seq=%.3f ms new_parallel=%.3f ms",
            ILogger::ELL_INFO,
            static_cast<unsigned long long>(m_cases.size()),
            static_cast<unsigned long long>(totalGeometryCount),
            static_cast<unsigned long long>(totalBufferCount),
            totalLoadMs,
            totalLegacySequentialMs,
            totalNewSequentialMs,
            totalNewParallelMs);
    }

    return true;
}

bool MeshLoadersApp::validateWrittenAsset(const system::path& path)
{
    if (!std::filesystem::exists(path))
        return false;

    m_assetMgr->clearAllAssetCache();

    IAssetLoader::SAssetLoadParams params = makeLoadParams();
    auto asset = m_assetMgr->getAsset(path.string(), params);
    if (asset.getContents().empty())
        return false;

    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
    switch (asset.getAssetType())
    {
    case IAsset::E_TYPE::ET_GEOMETRY:
        for (const auto& item : asset.getContents())
            if (auto polyGeo = IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
                geometries.push_back(polyGeo);
        break;
    default:
        return false;
    }
    return !geometries.empty();
}

bool MeshLoadersApp::captureScreenshot(const system::path& path, core::smart_refctd_ptr<asset::ICPUImageView>& outImage)
{
    if (!m_device || !m_surface || !m_assetMgr)
        return false;

    m_device->waitIdle();

    auto* scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
    auto* fb = scRes ? scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex) : nullptr;
    if (!fb)
        return false;

    auto colorView = fb->getCreationParameters().colorAttachments[0u];
    if (!colorView)
        return false;

    auto cpuView = ext::ScreenShot::createScreenShot(
        m_device.get(),
        getGraphicsQueue(),
        nullptr,
        colorView.get(),
        asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
        asset::IImage::LAYOUT::PRESENT_SRC);
    if (!cpuView)
        return false;

    if (!path.empty())
        std::filesystem::create_directories(path.parent_path());

    IAssetWriter::SAssetWriteParams params(cpuView.get());
    if (!m_assetMgr->writeAsset(path.string(), params))
        return false;

    outImage = cpuView;
    return true;
}

bool MeshLoadersApp::appendGeometriesFromBundle(const asset::SAssetBundle& bundle, core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>>& out) const
{
    if (bundle.getContents().empty())
        return false;

    switch (bundle.getAssetType())
    {
    case IAsset::E_TYPE::ET_GEOMETRY:
        for (const auto& item : bundle.getContents())
        {
            if (auto polyGeo = IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
                out.push_back(polyGeo);
        }
        break;
    case IAsset::E_TYPE::ET_GEOMETRY_COLLECTION:
        for (const auto& item : bundle.getContents())
        {
            auto collection = IAsset::castDown<ICPUGeometryCollection>(item);
            if (!collection)
                continue;
            auto* refs = collection->getGeometries();
            if (!refs)
                continue;
            for (const auto& ref : *refs)
            {
                if (!ref.geometry)
                    continue;
                if (ref.geometry->getPrimitiveType() != IGeometryBase::EPrimitiveType::Polygon)
                    continue;
                auto poly = core::smart_refctd_ptr_static_cast<ICPUPolygonGeometry>(ref.geometry);
                if (poly)
                    out.push_back(poly);
            }
        }
        break;
    default:
        return false;
    }

    return !out.empty();
}

bool MeshLoadersApp::compareImages(const asset::ICPUImageView* a, const asset::ICPUImageView* b, uint64_t& diffCount, uint8_t& maxDiff)
{
    diffCount = 0u;
    maxDiff = 0u;
    if (!a || !b)
        return false;

    const auto* imgA = a->getCreationParameters().image.get();
    const auto* imgB = b->getCreationParameters().image.get();
    if (!imgA || !imgB)
        return false;

    const auto paramsA = imgA->getCreationParameters();
    const auto paramsB = imgB->getCreationParameters();
    if (paramsA.format != paramsB.format)
        return false;
    if (paramsA.extent != paramsB.extent)
        return false;

    const auto* bufA = imgA->getBuffer();
    const auto* bufB = imgB->getBuffer();
    if (!bufA || !bufB)
        return false;

    const size_t sizeA = bufA->getSize();
    if (sizeA != bufB->getSize())
        return false;

    const auto* dataA = static_cast<const uint8_t*>(bufA->getPointer());
    const auto* dataB = static_cast<const uint8_t*>(bufB->getPointer());
    if (!dataA || !dataB)
        return false;

    for (size_t i = 0; i < sizeA; ++i)
    {
        const uint8_t va = dataA[i];
        const uint8_t vb = dataB[i];
        const uint8_t diff = va > vb ? static_cast<uint8_t>(va - vb) : static_cast<uint8_t>(vb - va);
        if (diff)
        {
            ++diffCount;
            if (diff > maxDiff)
                maxDiff = diff;
        }
    }

    return true;
}

void MeshLoadersApp::advanceCase()
{
    if (m_runMode == RunMode::Interactive || m_cases.empty())
        return;
    if (isRowViewActive())
        return;

    const uint32_t frameLimit = m_runMode == RunMode::CI ? CiFramesBeforeCapture : NonCiFramesPerCase;
    ++m_phaseFrameCounter;
    if (m_phaseFrameCounter < frameLimit)
        return;

    if (m_phase == Phase::RenderOriginal)
    {
        if (!captureScreenshot(m_loadedScreenshotPath, m_loadedScreenshot))
            failExit("Failed to capture loaded screenshot.");

        if (m_saveGeom)
        {
            if (!m_currentCpuGeom)
                failExit("No geometry to write.");
            if (!writeGeometry(m_currentCpuGeom, m_writtenPath.string()))
                failExit("Geometry write failed.");
        }

        if (m_runMode == RunMode::CI)
        {
            if (!loadModel(m_writtenPath, false, false))
                failExit("Failed to load written asset %s.", m_writtenPath.string().c_str());
            if (!m_currentCpuGeom)
                failExit("Written geometry missing.");
            m_phase = Phase::RenderWritten;
            m_phaseFrameCounter = 0u;
            return;
        }

        if (m_saveGeom)
        {
            if (!validateWrittenAsset(m_writtenPath))
                failExit("Failed to load written asset %s.", m_writtenPath.string().c_str());
        }

        advanceToNextCase();
        return;
    }

    if (m_phase == Phase::RenderWritten)
    {
        if (!captureScreenshot(m_writtenScreenshotPath, m_writtenScreenshot))
            failExit("Failed to capture written screenshot.");

        if (m_hasReferenceGeometryHash)
        {
            const auto writtenHash = hashGeometry(m_currentCpuGeom.get());
            if (writtenHash != m_referenceGeometryHash)
                failExit("Geometry hash mismatch for %s. Current=%s Reference=%s ReferenceFile=%s", m_caseName.c_str(), geometryHashToHex(writtenHash).c_str(), geometryHashToHex(m_referenceGeometryHash).c_str(), m_caseGeometryHashReferencePath.empty() ? "<none>" : m_caseGeometryHashReferencePath.string().c_str());
        }

        uint64_t diffCount = 0u;
        uint8_t maxDiff = 0u;
        if (!compareImages(m_loadedScreenshot.get(), m_writtenScreenshot.get(), diffCount, maxDiff))
            failExit("Image compare failed for %s.", m_caseName.c_str());
        if (diffCount > MaxImageDiffBytes || maxDiff > MaxImageDiffValue)
            failExit("Image diff detected for %s. Bytes: %llu MaxDiff: %u", m_caseName.c_str(), static_cast<unsigned long long>(diffCount), maxDiff);
        if (diffCount != 0u)
            m_logger->log("Image diff within tolerance for %s. Bytes: %llu MaxDiff: %u", ILogger::ELL_WARNING, m_caseName.c_str(), static_cast<unsigned long long>(diffCount), maxDiff);

        advanceToNextCase();
    }
}


