// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "MeshLoadersApp.hpp"

#include "nbl/ext/ScreenShot/ScreenShot.h"

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


