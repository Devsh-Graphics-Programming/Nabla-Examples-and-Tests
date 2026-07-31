// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "BundleGeometryItems.h"

#include "nbl/examples/common/ImageComparison.h"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/system/CGrowableMemoryFile.h"

#include <cstring>

namespace
{

bool persistMemoryFileToDisk(system::ISystem* const system, const system::path& path, const system::CGrowableMemoryFile* const file)
{
    if (!system || !file)
        return false;

    system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> writeFileFuture;
    system->createFile(writeFileFuture, path, system::IFile::ECF_WRITE);
    core::smart_refctd_ptr<system::IFile> writeFile;
    writeFileFuture.acquire().move_into(writeFile);
    if (!writeFile)
        return false;

    const auto* const data = file->data();
    const auto size = file->getSize();
    system::IFile::success_t success;
    writeFile->write(success, data, 0ull, size);
    return static_cast<bool>(success);
}

}

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
    return validateWrittenBundle(asset);
}

bool MeshLoadersApp::validateWrittenBundle(const asset::SAssetBundle& bundle)
{
    core::vector<meshloaders::BundleGeometryItem> geometries;
    if (!meshloaders::collectBundleGeometryItems(bundle, geometries, false))
        return false;
    return !geometries.empty();
}

bool MeshLoadersApp::startBackgroundAssetWorker()
{
    auto& worker = m_backgroundAssetWorker;
    if (worker.thread.joinable())
        return true;

    worker.stop = false;
    worker.thread = std::thread(&MeshLoadersApp::backgroundAssetWorkerMain, this);
    return true;
}

void MeshLoadersApp::stopBackgroundAssetWorker()
{
    auto& worker = m_backgroundAssetWorker;
    {
        std::lock_guard lock(worker.mutex);
        worker.stop = true;
    }
    worker.cv.notify_all();
    if (worker.thread.joinable())
        worker.thread.join();

    std::lock_guard lock(worker.mutex);
    worker.request.reset();
    worker.result.reset();
    worker.busy = false;
    worker.stop = false;
}

bool MeshLoadersApp::startBackgroundLoadWorker()
{
    auto& worker = m_backgroundLoadWorker;
    if (worker.thread.joinable())
        return true;

    worker.stop = false;
    worker.thread = std::thread(&MeshLoadersApp::backgroundLoadWorkerMain, this);
    return true;
}

void MeshLoadersApp::stopBackgroundLoadWorker()
{
    auto& worker = m_backgroundLoadWorker;
    {
        std::lock_guard lock(worker.mutex);
        worker.stop = true;
    }
    worker.cv.notify_all();
    if (worker.thread.joinable())
        worker.thread.join();

    std::lock_guard lock(worker.mutex);
    worker.requestCaseIndex.reset();
    worker.requestPath.clear();
    worker.result.reset();
    worker.busy = false;
    worker.stop = false;
}

void MeshLoadersApp::backgroundAssetWorkerMain()
{
    auto workerSystem = core::smart_refctd_ptr(m_system);
    auto workerAssetMgr = workerSystem ? core::make_smart_refctd_ptr<asset::IAssetManager>(core::smart_refctd_ptr(workerSystem)) : nullptr;

    for (;;)
    {
        WrittenAssetRequest request = {};
        {
            std::unique_lock lock(m_backgroundAssetWorker.mutex);
            m_backgroundAssetWorker.cv.wait(lock, [this] {
                return m_backgroundAssetWorker.stop || m_backgroundAssetWorker.request.has_value();
            });
            if (m_backgroundAssetWorker.stop && !m_backgroundAssetWorker.request.has_value())
                break;
            request = std::move(*m_backgroundAssetWorker.request);
            m_backgroundAssetWorker.request.reset();
        }

        WrittenAssetResult result = {};
        result.path = request.path;
        result.extension = normalizeExtension(request.path);

        if (!workerSystem || !workerAssetMgr)
        {
            result.error = "Background asset worker is unavailable.";
        }
        else if (!request.asset)
        {
            result.error = "Background asset worker received an empty asset.";
        }
        else
        {
            using clock_t = std::chrono::high_resolution_clock;
            const auto writeOuterStart = clock_t::now();
            auto* const assetPtr = const_cast<IAsset*>(request.asset.get());
            const auto flags = getWriterFlagsForPath(request.asset.get(), request.path);
            IAssetWriter::SAssetWriteParams writeParams{ assetPtr, flags };
            writeParams.logger = request.loadParams.logger;

            bool useDiskTransport = !request.useMemoryTransport;
            result.usedMemoryTransport = request.useMemoryTransport;
            if (request.useMemoryTransport)
            {
                auto memoryFile = core::make_smart_refctd_ptr<system::CGrowableMemoryFile>(system::path(request.path));
                bool memoryTransportSucceeded = false;

                if (memoryFile)
                {
                    const auto writeStart = clock_t::now();
                    if (!workerAssetMgr->writeAsset(memoryFile.get(), writeParams))
                    {
                        if (request.allowDiskFallback)
                            useDiskTransport = true;
                        else
                            result.error = "Background asset worker failed to write the asset to the in-memory transport.";
                    }
                    else
                    {
                        result.writeMs = toMs(clock_t::now() - writeStart);
                        result.outputSize = memoryFile->getSize();
                        result.totalWriteMs = toMs(clock_t::now() - writeOuterStart);
                        result.nonWriterMs = std::max(0.0, result.totalWriteMs - result.writeMs);

                        workerAssetMgr->clearAllAssetCache();
                        result.loadResult.inputSize = memoryFile->getSize();
                        const auto loadStart = clock_t::now();
                        result.loadResult.bundle = workerAssetMgr->getAsset(memoryFile.get(), request.path.string(), request.loadParams);
                        result.loadResult.getAssetMs = toMs(clock_t::now() - loadStart);
                        if (result.loadResult.bundle.getContents().empty())
                        {
                            if (request.allowDiskFallback)
                                useDiskTransport = true;
                            else
                                result.error = "Background asset worker failed to load the in-memory written asset.";
                        }
                        else
                            memoryTransportSucceeded = true;
                    }

                    if (memoryTransportSucceeded && request.persistDiskArtifact)
                    {
                        if (!persistMemoryFileToDisk(workerSystem.get(), request.path, memoryFile.get()))
                        {
                            if (request.allowDiskFallback)
                            {
                                useDiskTransport = true;
                                result.usedDiskFallback = true;
                            }
                            else
                                result.error = "Background asset worker failed to persist the in-memory written asset.";
                            if (useDiskTransport)
                            {
                                result.loadResult = {};
                                result.outputSize = 0u;
                                result.totalWriteMs = 0.0;
                                result.nonWriterMs = 0.0;
                            }
                        }
                        else
                            result.persistedDiskArtifact = true;
                    }
                }
                else
                {
                    if (request.allowDiskFallback)
                    {
                        useDiskTransport = true;
                        result.usedDiskFallback = true;
                    }
                    else
                        result.error = "Background asset worker could not create the in-memory transport.";
                }
            }

            if (useDiskTransport && result.error.empty())
            {
                result.usedDiskFallback = result.usedDiskFallback || request.useMemoryTransport;
                result.persistedDiskArtifact = true;
                const auto openStart = clock_t::now();
                system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> writeFileFuture;
                workerSystem->createFile(writeFileFuture, request.path, system::IFile::ECF_WRITE);
                core::smart_refctd_ptr<system::IFile> writeFile;
                writeFileFuture.acquire().move_into(writeFile);
                result.openMs = toMs(clock_t::now() - openStart);
                if (!writeFile)
                {
                    result.error = "Background asset worker failed to open the output file.";
                }
                else
                {
                    const auto writeStart = clock_t::now();
                    if (!workerAssetMgr->writeAsset(writeFile.get(), writeParams))
                    {
                        result.error = "Background asset worker failed to write the asset.";
                    }
                    result.writeMs = toMs(clock_t::now() - writeStart);
                    writeFile = nullptr;
                }

                if (result.error.empty())
                {
                    const auto statStart = clock_t::now();
                    if (std::filesystem::exists(request.path))
                        result.outputSize = std::filesystem::file_size(request.path);
                    result.statMs = toMs(clock_t::now() - statStart);
                    result.totalWriteMs = toMs(clock_t::now() - writeOuterStart);
                    result.nonWriterMs = std::max(0.0, result.totalWriteMs - result.writeMs);

                    workerAssetMgr->clearAllAssetCache();
                    if (std::filesystem::exists(request.path))
                        result.loadResult.inputSize = std::filesystem::file_size(request.path);
                    else
                        result.loadResult.inputSize = 0u;

                    const auto loadStart = clock_t::now();
                    result.loadResult.bundle = workerAssetMgr->getAsset(request.path.string(), request.loadParams);
                    result.loadResult.getAssetMs = toMs(clock_t::now() - loadStart);
                    if (result.loadResult.bundle.getContents().empty())
                        result.error = "Background asset worker failed to load the written asset.";
                }
            }
        }

        result.success = result.error.empty();
        {
            std::lock_guard lock(m_backgroundAssetWorker.mutex);
            m_backgroundAssetWorker.result = std::move(result);
            m_backgroundAssetWorker.busy = false;
        }
        m_backgroundAssetWorker.cv.notify_all();
    }
}

void MeshLoadersApp::backgroundLoadWorkerMain()
{
    auto workerSystem = core::smart_refctd_ptr(m_system);
    auto workerAssetMgr = workerSystem ? core::make_smart_refctd_ptr<asset::IAssetManager>(core::smart_refctd_ptr(workerSystem)) : nullptr;

    for (;;)
    {
        std::optional<size_t> caseIndex;
        system::path requestPath;
        IAssetLoader::SAssetLoadParams requestParams = {};
        {
            std::unique_lock lock(m_backgroundLoadWorker.mutex);
            m_backgroundLoadWorker.cv.wait(lock, [this] {
                return m_backgroundLoadWorker.stop || m_backgroundLoadWorker.requestCaseIndex.has_value();
            });
            if (m_backgroundLoadWorker.stop && !m_backgroundLoadWorker.requestCaseIndex.has_value())
                break;
            caseIndex = m_backgroundLoadWorker.requestCaseIndex;
            requestPath = m_backgroundLoadWorker.requestPath;
            requestParams = m_backgroundLoadWorker.requestParams;
            m_backgroundLoadWorker.requestCaseIndex.reset();
            m_backgroundLoadWorker.requestPath.clear();
        }

        PreparedAssetLoad result = {};
        result.caseIndex = *caseIndex;
        result.path = requestPath;
        if (!workerAssetMgr)
        {
            result.error = "Background load worker is unavailable.";
        }
        else if (!std::filesystem::exists(requestPath))
        {
            result.error = "Background load worker did not find the requested input.";
        }
        else
        {
            workerAssetMgr->clearAllAssetCache();
            result.loadResult.inputSize = std::filesystem::file_size(requestPath);
            const auto loadStart = std::chrono::high_resolution_clock::now();
            result.loadResult.bundle = workerAssetMgr->getAsset(requestPath.string(), requestParams);
            result.loadResult.getAssetMs = toMs(std::chrono::high_resolution_clock::now() - loadStart);
            if (result.loadResult.bundle.getContents().empty())
                result.error = "Background load worker failed to load the requested asset.";
        }
        result.success = result.error.empty();
        {
            std::lock_guard lock(m_backgroundLoadWorker.mutex);
            m_backgroundLoadWorker.result = std::move(result);
            m_backgroundLoadWorker.busy = false;
        }
        m_backgroundLoadWorker.cv.notify_all();
    }
}

bool MeshLoadersApp::startWrittenAssetWork(smart_refctd_ptr<const IAsset> asset, const system::path& path)
{
    if (!asset || path.empty())
        return false;
    if (!startBackgroundAssetWorker())
        return false;

    std::unique_lock lock(m_backgroundAssetWorker.mutex);
    if (m_backgroundAssetWorker.busy || m_backgroundAssetWorker.request.has_value() || m_backgroundAssetWorker.result.has_value())
        return false;

    m_backgroundAssetWorker.request = WrittenAssetRequest{
        .asset = std::move(asset),
        .path = path,
        .loadParams = makeLoadParams(),
        .useMemoryTransport = true,
        .allowDiskFallback = (m_runtime.mode != RunMode::CI),
        .persistDiskArtifact = (m_runtime.mode != RunMode::CI)
    };
    m_backgroundAssetWorker.busy = true;
    lock.unlock();
    m_backgroundAssetWorker.cv.notify_one();
    return true;
}

bool MeshLoadersApp::finalizeWrittenAssetWork(WrittenAssetResult& result, bool& ready, bool waitForCompletion)
{
    ready = false;
    if (!m_backgroundAssetWorker.thread.joinable())
        return false;

    std::unique_lock lock(m_backgroundAssetWorker.mutex);
    if (waitForCompletion)
    {
        m_backgroundAssetWorker.cv.wait(lock, [this] {
            return !m_backgroundAssetWorker.busy && !m_backgroundAssetWorker.request.has_value();
        });
    }
    if (m_backgroundAssetWorker.result.has_value())
    {
        result = std::move(*m_backgroundAssetWorker.result);
        m_backgroundAssetWorker.result.reset();
        ready = true;
        return true;
    }
    return m_backgroundAssetWorker.busy || m_backgroundAssetWorker.request.has_value();
}

bool MeshLoadersApp::startPreparedAssetLoad(const size_t caseIndex, const system::path& path)
{
    if (path.empty())
        return false;
    if (!startBackgroundLoadWorker())
        return false;

    std::unique_lock lock(m_backgroundLoadWorker.mutex);
    if (m_backgroundLoadWorker.busy || m_backgroundLoadWorker.requestCaseIndex.has_value())
        return false;

    m_backgroundLoadWorker.result.reset();
    m_backgroundLoadWorker.requestCaseIndex = caseIndex;
    m_backgroundLoadWorker.requestPath = path;
    m_backgroundLoadWorker.requestParams = makeLoadParams();
    m_backgroundLoadWorker.busy = true;
    lock.unlock();
    m_backgroundLoadWorker.cv.notify_one();
    return true;
}

bool MeshLoadersApp::finalizePreparedAssetLoad(PreparedAssetLoad& result, bool& ready, bool waitForCompletion)
{
    ready = false;
    if (!m_backgroundLoadWorker.thread.joinable())
        return false;

    std::unique_lock lock(m_backgroundLoadWorker.mutex);
    if (waitForCompletion)
    {
        m_backgroundLoadWorker.cv.wait(lock, [this] {
            return !m_backgroundLoadWorker.busy && !m_backgroundLoadWorker.requestCaseIndex.has_value();
        });
    }
    if (m_backgroundLoadWorker.result.has_value())
    {
        result = std::move(*m_backgroundLoadWorker.result);
        m_backgroundLoadWorker.result.reset();
        ready = true;
        return true;
    }
    return m_backgroundLoadWorker.busy || m_backgroundLoadWorker.requestCaseIndex.has_value();
}

void MeshLoadersApp::logWrittenAssetWork(const WrittenAssetResult& result) const
{
    m_logger->log(
        "Asset write call perf: path=%s ext=%s time=%.3f ms size=%llu",
        ILogger::ELL_INFO,
        result.path.string().c_str(),
        result.extension.c_str(),
        result.writeMs,
        static_cast<unsigned long long>(result.outputSize));
    m_logger->log(
        "Asset write outer perf: path=%s ext=%s open=%.3f ms writeAsset=%.3f ms stat=%.3f ms total=%.3f ms non_writer=%.3f ms size=%llu",
        ILogger::ELL_INFO,
        result.path.string().c_str(),
        result.extension.c_str(),
        result.openMs,
        result.writeMs,
        result.statMs,
        result.totalWriteMs,
        result.nonWriterMs,
        static_cast<unsigned long long>(result.outputSize));
    m_logger->log(
        "Writer perf: path=%s ext=%s time=%.3f ms size=%llu",
        ILogger::ELL_INFO,
        result.path.string().c_str(),
        result.extension.c_str(),
        result.writeMs,
        static_cast<unsigned long long>(result.outputSize));
    m_logger->log("Mesh successfully saved!", ILogger::ELL_INFO);
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

bool MeshLoadersApp::finalizeScreenshotCapture(core::smart_refctd_ptr<asset::ICPUImageView>& outImage, bool& ready, bool waitForCompletion)
{
    ready = false;
    if (!m_render.pendingScreenshot.active())
        return false;

    if (waitForCompletion)
    {
        const ISemaphore::SWaitInfo waitInfo = {
            .semaphore = m_render.pendingScreenshot.completionSemaphore.get(),
            .value = m_render.pendingScreenshot.completionValue
        };
        if (m_device->blockForSemaphores({ &waitInfo, 1u }) != ISemaphore::WAIT_RESULT::SUCCESS)
        {
            m_render.pendingScreenshot = {};
            return false;
        }
    }

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

    const auto finalizePendingCapture = [this](core::smart_refctd_ptr<asset::ICPUImageView>& outImage, const char* const failureMessage, const bool waitForCompletion = false) -> bool
    {
        bool ready = false;
        if (!finalizeScreenshotCapture(outImage, ready, waitForCompletion))
            failExit("%s", failureMessage);
        return ready;
    };

    const auto handleWrittenAssetReady = [this](WrittenAssetResult&& result) -> void
    {
        if (!result.success)
            failExit("%s", result.error.c_str());
        logWrittenAssetWork(result);
        if (performanceEnabled())
            recordWriteMetrics(result);

        if (m_runtime.mode == RunMode::CI)
        {
            LoadStageMetrics writtenLoadMetrics = {};
            if (!loadPreparedModel(m_output.writtenPath, std::move(result.loadResult), false, false, &writtenLoadMetrics))
                failExit("Failed to load written asset %s.", m_output.writtenPath.string().c_str());
            if (performanceEnabled() && writtenLoadMetrics.valid)
                recordWrittenLoadMetrics(writtenLoadMetrics);
            if (!m_render.currentCpuGeom)
                failExit("Written geometry missing.");
            m_runtime.phase = Phase::RenderWritten;
            m_runtime.phaseFrameCounter = 0u;
            return;
        }

        if (!validateWrittenBundle(result.loadResult.bundle))
            failExit("Failed to load written asset %s.", m_output.writtenPath.string().c_str());

        advanceToNextCase();
    };

    if (m_runtime.mode == RunMode::CI && m_runtime.phase == Phase::RenderOriginal)
    {
        ++m_runtime.phaseFrameCounter;
        if (m_runtime.phaseFrameCounter == 0u)
            return;

        if (!requestScreenshotCapture(m_output.loadedScreenshotPath))
            failExit("Failed to request loaded screenshot capture.");
        if (!finalizePendingCapture(m_render.loadedScreenshot, "Failed to finalize loaded screenshot.", true))
            failExit("Loaded screenshot capture did not complete.");

        const bool canWriteCurrentAsset = m_output.saveGeom && static_cast<bool>(m_render.currentCpuAsset);
        if (m_output.saveGeom && !canWriteCurrentAsset)
            m_logger->log("Skipping write/reload for %s because the loaded case expands to multiple root geometries.", ILogger::ELL_INFO, m_caseName.c_str());
        if (!canWriteCurrentAsset)
        {
            advanceToNextCase();
            return;
        }

        WrittenAssetResult result = {};
        bool ready = false;
        if (!finalizeWrittenAssetWork(result, ready, true) || !ready)
            failExit("Written asset preparation did not complete.");
        handleWrittenAssetReady(std::move(result));
        return;
    }

    if (m_runtime.mode == RunMode::CI && m_runtime.phase == Phase::RenderWritten)
    {
        ++m_runtime.phaseFrameCounter;
        if (m_runtime.phaseFrameCounter == 0u)
            return;

        if (!requestScreenshotCapture(m_output.writtenScreenshotPath))
            failExit("Failed to request written screenshot capture.");
        if (!finalizePendingCapture(m_render.writtenScreenshot, "Failed to finalize written screenshot.", true))
            failExit("Written screenshot capture did not complete.");

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

    if (m_runtime.phase == Phase::CaptureOriginalPending)
    {
        if (!finalizePendingCapture(m_render.loadedScreenshot, "Failed to finalize loaded screenshot."))
            return;

        const bool canWriteCurrentAsset = m_output.saveGeom && static_cast<bool>(m_render.currentCpuAsset);
        if (m_output.saveGeom)
        {
            if (!canWriteCurrentAsset)
                m_logger->log("Skipping write/reload for %s because the loaded case expands to multiple root geometries.", ILogger::ELL_INFO, m_caseName.c_str());
            else
            {
                WrittenAssetResult result = {};
                bool ready = false;
                const bool workerStateValid = finalizeWrittenAssetWork(result, ready);
                if (!workerStateValid)
                {
                    if (m_runtime.mode == RunMode::CI)
                        failExit("Background written asset preparation is unavailable.");
                    WriteStageMetrics writeMetrics = {};
                    if (!writeAssetRoot(m_render.currentCpuAsset, m_output.writtenPath.string(), &writeMetrics))
                        failExit("Geometry write failed.");
                    if (performanceEnabled() && writeMetrics.valid)
                        recordWriteMetrics(writeMetrics);

                    if (m_runtime.mode == RunMode::CI)
                    {
                        LoadStageMetrics writtenLoadMetrics = {};
                        if (!loadModel(m_output.writtenPath, false, false, &writtenLoadMetrics))
                            failExit("Failed to load written asset %s.", m_output.writtenPath.string().c_str());
                        if (performanceEnabled() && writtenLoadMetrics.valid)
                            recordWrittenLoadMetrics(writtenLoadMetrics);
                        if (!m_render.currentCpuGeom)
                            failExit("Written geometry missing.");
                        m_runtime.phase = Phase::RenderWritten;
                        m_runtime.phaseFrameCounter = 0u;
                        return;
                    }

                    if (!validateWrittenAsset(m_output.writtenPath))
                        failExit("Failed to load written asset %s.", m_output.writtenPath.string().c_str());

                    advanceToNextCase();
                    return;
                }

                if (!ready)
                {
                    m_runtime.phase = Phase::WrittenAssetPending;
                    return;
                }

                handleWrittenAssetReady(std::move(result));
                return;
            }
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

    if (m_runtime.phase == Phase::WrittenAssetPending)
    {
        WrittenAssetResult result = {};
        bool ready = false;
        if (!finalizeWrittenAssetWork(result, ready))
            failExit("Background written asset work failed unexpectedly.");
        if (!ready)
            return;
        handleWrittenAssetReady(std::move(result));
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
