// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "MeshLoadersApp.hpp"

#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/examples/common/ImageComparison.h"

std::string MeshLoadersApp::makeUniqueCaseName(const system::path& path)
{
    auto base = path.stem().string();
    if (base.empty())
        base = "case";
    auto& counter = m_runtime.caseNameCounts[base];
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
    if (!appendGeometriesFromBundle(asset, geometries))
        return false;
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

    auto appendCollection = [&](const ICPUGeometryCollection* collection) -> void
    {
        if (!collection)
            return;
        const auto& refs = collection->getGeometries();
        for (const auto& ref : refs)
        {
            if (!ref.geometry)
                continue;
            if (ref.geometry->getPrimitiveType() != IGeometryBase::EPrimitiveType::Polygon)
                continue;
            const auto assetRef = core::smart_refctd_ptr_static_cast<const IAsset>(ref.geometry);
            auto poly = IAsset::castDown<const ICPUPolygonGeometry>(assetRef);
            if (poly)
                out.push_back(poly);
        }
    };

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
            auto collection = IAsset::castDown<const ICPUGeometryCollection>(item);
            appendCollection(collection.get());
        }
        break;
    case IAsset::E_TYPE::ET_SCENE:
        for (const auto& item : bundle.getContents())
        {
            auto scene = IAsset::castDown<const ICPUScene>(item);
            if (!scene)
                continue;
            const auto& instances = scene->getInstances().getMorphTargets();
            for (const auto& morphTargets : instances)
            {
                if (!morphTargets)
                    continue;
                const auto& targets = *morphTargets->getTargets();
                for (const auto& target : targets)
                    appendCollection(target.geoCollection.get());
            }
        }
        break;
    default:
        return false;
    }

    return !out.empty();
}

bool MeshLoadersApp::compareImages(
    const asset::ICPUImageView* a,
    const asset::ICPUImageView* b,
    uint64_t& diffCodeUnitCount,
    uint32_t& maxDiffCodeUnitValue)
{
    return nbl::examples::image::compareCpuImageViewsByCodeUnit(a, b, diffCodeUnitCount, maxDiffCodeUnitValue);
}

void MeshLoadersApp::advanceCase()
{
    if (m_runtime.mode == RunMode::Interactive || m_runtime.cases.empty())
        return;
    if (isRowViewActive())
        return;

    const uint32_t frameLimit = m_runtime.mode == RunMode::CI ? CiFramesBeforeCapture : NonCiFramesPerCase;
    ++m_runtime.phaseFrameCounter;
    if (m_runtime.phaseFrameCounter < frameLimit)
        return;

    if (m_runtime.phase == Phase::RenderOriginal)
    {
        if (!captureScreenshot(m_output.loadedScreenshotPath, m_render.loadedScreenshot))
            failExit("Failed to capture loaded screenshot.");

        const bool canWriteCurrentAsset = m_output.saveGeom && static_cast<bool>(m_render.currentCpuAsset);
        if (m_output.saveGeom)
        {
            if (!canWriteCurrentAsset)
                m_logger->log("Skipping write/reload for %s because the loaded case expands to multiple root geometries.", ILogger::ELL_INFO, m_caseName.c_str());
            else if (!writeAssetRoot(m_render.currentCpuAsset, m_output.writtenPath.string()))
                failExit("Geometry write failed.");
        }

        if (m_runtime.mode == RunMode::CI)
        {
            if (!canWriteCurrentAsset)
            {
                advanceToNextCase();
                return;
            }
            if (!loadModel(m_output.writtenPath, false, false))
                failExit("Failed to load written asset %s.", m_output.writtenPath.string().c_str());
            if (!m_render.currentCpuGeom)
                failExit("Written geometry missing.");
            m_runtime.phase = Phase::RenderWritten;
            m_runtime.phaseFrameCounter = 0u;
            return;
        }

        if (canWriteCurrentAsset)
        {
            if (!validateWrittenAsset(m_output.writtenPath))
                failExit("Failed to load written asset %s.", m_output.writtenPath.string().c_str());
        }

        advanceToNextCase();
        return;
    }

    if (m_runtime.phase == Phase::RenderWritten)
    {
        if (!captureScreenshot(m_output.writtenScreenshotPath, m_render.writtenScreenshot))
            failExit("Failed to capture written screenshot.");

        uint64_t diffCodeUnitCount = 0u;
        uint32_t maxDiffCodeUnitValue = 0u;
        if (!compareImages(m_render.loadedScreenshot.get(), m_render.writtenScreenshot.get(), diffCodeUnitCount, maxDiffCodeUnitValue))
            failExit("Image compare failed for %s.", m_caseName.c_str());
        if (diffCodeUnitCount > MaxImageDiffCodeUnits || maxDiffCodeUnitValue > MaxImageDiffCodeUnitValue)
            failExit("Image diff detected for %s. CodeUnits: %llu MaxCodeUnitDiff: %u", m_caseName.c_str(), static_cast<unsigned long long>(diffCodeUnitCount), maxDiffCodeUnitValue);
        if (diffCodeUnitCount != 0u)
            m_logger->log("Image diff within tolerance for %s. CodeUnits: %llu MaxCodeUnitDiff: %u", ILogger::ELL_WARNING, m_caseName.c_str(), static_cast<unsigned long long>(diffCodeUnitCount), maxDiffCodeUnitValue);

        advanceToNextCase();
    }
}



