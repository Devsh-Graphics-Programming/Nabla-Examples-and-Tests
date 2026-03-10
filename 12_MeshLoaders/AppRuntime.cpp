// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "BundleGeometryItems.h"

#include "nbl/examples/common/ImageComparison.h"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

#include <cstring>

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

    core::vector<meshloaders::BundleGeometryItem> geometries;
    if (!meshloaders::collectBundleGeometryItems(asset, geometries, false))
        return false;
    return !geometries.empty();
}

bool MeshLoadersApp::requestScreenshotCapture(const system::path& path)
{
    if (!m_device || !m_surface || !m_assetMgr)
        return false;
    if (m_render.pendingScreenshot.active())
        return false;

    auto* const scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
    auto* const fb = scRes ? scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex) : nullptr;
    if (!fb)
        return false;

    auto colorView = fb->getCreationParameters().colorAttachments[0u];
    if (!colorView)
        return false;

    auto gpuImage = colorView->getCreationParameters().image;
    if (!gpuImage)
        return false;

    const auto imageParams = gpuImage->getCreationParameters();
    if (!imageParams.usage.hasFlags(asset::IImage::EUF_TRANSFER_SRC_BIT))
        return false;
    if (asset::isBlockCompressionFormat(imageParams.format))
        return false;

    auto commandPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
    if (!commandPool)
        return false;

    core::smart_refctd_ptr<IGPUCommandBuffer> commandBuffer;
    if (!commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &commandBuffer, 1u }) || !commandBuffer)
        return false;
    if (!commandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
        return false;

    const auto imageViewParams = colorView->getCreationParameters();
    const auto extent = gpuImage->getMipSize();

    IGPUImage::SBufferCopy copyRegion = {};
    copyRegion.imageSubresource.aspectMask = imageViewParams.subresourceRange.aspectMask;
    copyRegion.imageSubresource.mipLevel = imageViewParams.subresourceRange.baseMipLevel;
    copyRegion.imageSubresource.baseArrayLayer = imageViewParams.subresourceRange.baseArrayLayer;
    copyRegion.imageSubresource.layerCount = imageViewParams.subresourceRange.layerCount;
    copyRegion.imageExtent = { extent.x, extent.y, extent.z };

    IGPUBuffer::SCreationParams bufferParams = {};
    bufferParams.size = static_cast<size_t>(extent.x) * static_cast<size_t>(extent.y) * static_cast<size_t>(extent.z) * asset::getTexelOrBlockBytesize(imageParams.format);
    bufferParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
    auto texelBuffer = m_device->createBuffer(std::move(bufferParams));
    if (!texelBuffer)
        return false;

    auto texelBufferMemReqs = texelBuffer->getMemoryReqs();
    texelBufferMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
    if (!texelBufferMemReqs.memoryTypeBits)
        return false;
    auto texelBufferMemory = m_device->allocate(texelBufferMemReqs, texelBuffer.get());
    if (!texelBufferMemory.isValid())
        return false;

    IGPUCommandBuffer::SPipelineBarrierDependencyInfo dependencyInfo = {};
    IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarrier = {};
    dependencyInfo.imgBarriers = { &imageBarrier, &imageBarrier + 1 };

    imageBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
    imageBarrier.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
    imageBarrier.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT;
    imageBarrier.oldLayout = asset::IImage::LAYOUT::PRESENT_SRC;
    imageBarrier.newLayout = asset::IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
    imageBarrier.image = gpuImage.get();
    imageBarrier.subresourceRange.aspectMask = imageViewParams.subresourceRange.aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = imageViewParams.subresourceRange.baseMipLevel;
    imageBarrier.subresourceRange.levelCount = 1u;
    imageBarrier.subresourceRange.baseArrayLayer = imageViewParams.subresourceRange.baseArrayLayer;
    imageBarrier.subresourceRange.layerCount = imageViewParams.subresourceRange.layerCount;
    commandBuffer->pipelineBarrier(asset::EDF_NONE, dependencyInfo);

    commandBuffer->copyImageToBuffer(gpuImage.get(), asset::IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, texelBuffer.get(), 1u, &copyRegion);

    imageBarrier.barrier.dep.srcStageMask = imageBarrier.barrier.dep.dstStageMask;
    imageBarrier.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::NONE;
    imageBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
    imageBarrier.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::NONE;
    imageBarrier.oldLayout = imageBarrier.newLayout;
    imageBarrier.newLayout = asset::IImage::LAYOUT::PRESENT_SRC;
    commandBuffer->pipelineBarrier(asset::EDF_NONE, dependencyInfo);

    if (!commandBuffer->end())
        return false;

    auto completionSemaphore = m_device->createSemaphore(0u);
    if (!completionSemaphore)
        return false;

    IQueue::SSubmitInfo::SCommandBufferInfo commandBufferInfo = { .cmdbuf = commandBuffer.get() };
    IQueue::SSubmitInfo::SSemaphoreInfo waitInfo = {
        .semaphore = m_render.semaphore.get(),
        .value = m_render.realFrameIx,
        .stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT
    };
    IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
        .semaphore = completionSemaphore.get(),
        .value = 1u,
        .stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT
    };
    IQueue::SSubmitInfo submitInfo = {
        .waitSemaphores = { &waitInfo, 1u },
        .commandBuffers = { &commandBufferInfo, 1u },
        .signalSemaphores = { &signalInfo, 1u }
    };
    if (getGraphicsQueue()->submit({ &submitInfo, 1u }) != IQueue::RESULT::SUCCESS)
        return false;

    m_render.pendingScreenshot.path = path;
    m_render.pendingScreenshot.sourceView = std::move(colorView);
    m_render.pendingScreenshot.commandBuffer = std::move(commandBuffer);
    m_render.pendingScreenshot.texelBuffer = std::move(texelBuffer);
    m_render.pendingScreenshot.completionSemaphore = std::move(completionSemaphore);
    m_render.pendingScreenshot.imageParams = imageParams;
    m_render.pendingScreenshot.subresourceRange = imageViewParams.subresourceRange;
    m_render.pendingScreenshot.viewFormat = imageViewParams.format;
    m_render.pendingScreenshot.completionValue = 1u;
    return true;
}

bool MeshLoadersApp::finalizeScreenshotCapture(core::smart_refctd_ptr<asset::ICPUImageView>& outImage, bool& ready)
{
    ready = false;
    if (!m_render.pendingScreenshot.active())
        return false;

    if (m_render.pendingScreenshot.completionSemaphore->getCounterValue() < m_render.pendingScreenshot.completionValue)
        return true;

    const auto texelBufferSize = m_render.pendingScreenshot.texelBuffer->getSize();
    auto* const allocation = m_render.pendingScreenshot.texelBuffer->getBoundMemory().memory;
    if (!allocation)
    {
        m_render.pendingScreenshot = {};
        return false;
    }

    bool mappedHere = false;
    if (!allocation->getMappedPointer())
    {
        const IDeviceMemoryAllocation::MemoryRange range = { 0u, texelBufferSize };
        if (!allocation->map(range, IDeviceMemoryAllocation::EMCAF_READ))
        {
            m_render.pendingScreenshot = {};
            return false;
        }
        mappedHere = true;
    }

    if (allocation->haveToMakeVisible())
    {
        const ILogicalDevice::MappedMemoryRange mappedRange(allocation, 0u, texelBufferSize);
        m_device->invalidateMappedMemoryRanges(1u, &mappedRange);
    }

    auto cpuImage = asset::ICPUImage::create(m_render.pendingScreenshot.imageParams);
    auto cpuBuffer = asset::ICPUBuffer::create({ texelBufferSize });
    if (!cpuImage || !cpuBuffer)
    {
        if (mappedHere)
            allocation->unmap();
        m_render.pendingScreenshot = {};
        return false;
    }

    std::memcpy(cpuBuffer->getPointer(), allocation->getMappedPointer(), texelBufferSize);

    auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
    auto& region = regions->front();
    region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
    region.imageSubresource.mipLevel = 0u;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.bufferOffset = 0u;
    region.bufferRowLength = m_render.pendingScreenshot.imageParams.extent.width;
    region.bufferImageHeight = 0u;
    region.imageOffset = { 0u, 0u, 0u };
    region.imageExtent = m_render.pendingScreenshot.imageParams.extent;
    cpuImage->setBufferAndRegions(core::smart_refctd_ptr(cpuBuffer), regions);

    if (mappedHere)
        allocation->unmap();

    asset::ICPUImageView::SCreationParams viewParams = {};
    viewParams.image = std::move(cpuImage);
    viewParams.format = m_render.pendingScreenshot.viewFormat;
    viewParams.viewType = asset::ICPUImageView::ET_2D;
    viewParams.subresourceRange = m_render.pendingScreenshot.subresourceRange;

    auto cpuView = asset::ICPUImageView::create(std::move(viewParams));
    if (!cpuView)
    {
        m_render.pendingScreenshot = {};
        return false;
    }

    if (!m_render.pendingScreenshot.path.empty())
    {
        const auto parentPath = m_render.pendingScreenshot.path.parent_path();
        if (!parentPath.empty())
            std::filesystem::create_directories(parentPath);
        IAssetWriter::SAssetWriteParams params(cpuView.get());
        if (!m_assetMgr->writeAsset(m_render.pendingScreenshot.path.string(), params))
        {
            m_render.pendingScreenshot = {};
            return false;
        }
    }

    outImage = std::move(cpuView);
    ready = true;
    m_render.pendingScreenshot = {};
    return true;
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

    const auto finalizePendingCapture = [this](core::smart_refctd_ptr<asset::ICPUImageView>& outImage, const char* const failureMessage) -> bool
    {
        bool ready = false;
        if (!finalizeScreenshotCapture(outImage, ready))
            failExit("%s", failureMessage);
        return ready;
    };

    if (m_runtime.phase == Phase::CaptureOriginalPending)
    {
        if (!finalizePendingCapture(m_render.loadedScreenshot, "Failed to finalize loaded screenshot."))
            return;

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

    if (m_runtime.phase == Phase::CaptureWrittenPending)
    {
        if (!finalizePendingCapture(m_render.writtenScreenshot, "Failed to finalize written screenshot."))
            return;

        uint64_t diffCodeUnitCount = 0u;
        uint32_t maxDiffCodeUnitValue = 0u;
        if (!compareImages(m_render.loadedScreenshot.get(), m_render.writtenScreenshot.get(), diffCodeUnitCount, maxDiffCodeUnitValue))
            failExit("Image compare failed for %s.", m_caseName.c_str());
        if (diffCodeUnitCount > MaxImageDiffCodeUnits || maxDiffCodeUnitValue > MaxImageDiffCodeUnitValue)
            failExit("Image diff detected for %s. CodeUnits: %llu MaxCodeUnitDiff: %u", m_caseName.c_str(), static_cast<unsigned long long>(diffCodeUnitCount), maxDiffCodeUnitValue);
        if (diffCodeUnitCount != 0u)
            m_logger->log("Image diff within tolerance for %s. CodeUnits: %llu MaxCodeUnitDiff: %u", ILogger::ELL_WARNING, m_caseName.c_str(), static_cast<unsigned long long>(diffCodeUnitCount), maxDiffCodeUnitValue);

        advanceToNextCase();
        return;
    }

    const uint32_t frameLimit = m_runtime.mode == RunMode::CI ? CiFramesBeforeCapture : NonCiFramesPerCase;
    const uint32_t captureRequestThreshold = (frameLimit > 1u) ? (frameLimit - 1u) : frameLimit;
    ++m_runtime.phaseFrameCounter;
    if (m_runtime.phaseFrameCounter < captureRequestThreshold)
        return;

    if (m_runtime.phase == Phase::RenderOriginal)
    {
        if (!requestScreenshotCapture(m_output.loadedScreenshotPath))
            failExit("Failed to request loaded screenshot capture.");
        m_runtime.phase = Phase::CaptureOriginalPending;
        return;
    }

    if (m_runtime.phase == Phase::RenderWritten)
    {
        if (!requestScreenshotCapture(m_output.writtenScreenshotPath))
            failExit("Failed to request written screenshot capture.");
        m_runtime.phase = Phase::CaptureWrittenPending;
    }
}
