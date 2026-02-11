// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "MeshLoadersApp.hpp"

#include <algorithm>
#include <cmath>

#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

bool MeshLoadersApp::loadModel(const system::path& modelPath, bool updateCamera, bool storeCamera)
{
    if (modelPath.empty())
        failExit("Empty model path.");
    if (!std::filesystem::exists(modelPath))
        failExit("Missing input: %s", modelPath.string().c_str());
    using clock_t = std::chrono::high_resolution_clock;
    const auto loadOuterStart = clock_t::now();

    m_modelPath = modelPath.string();

    // free up
    m_renderer->m_instances.clear();
    m_renderer->clearGeometries({ .semaphore = m_semaphore.get(),.value = m_realFrameIx });
    m_assetMgr->clearAllAssetCache();

    //! load the geometry
    IAssetLoader::SAssetLoadParams params = makeLoadParams();
    AssetLoadCallResult loadResult = {};
    if (!loadAssetCallFromPath(modelPath, params, loadResult))
        failExit("Failed to open input file %s.", modelPath.string().c_str());
    const auto loadMs = loadResult.getAssetMs;
    auto asset = std::move(loadResult.bundle);
    m_logger->log(
        "Asset load call perf: path=%s time=%.3f ms size=%llu",
        ILogger::ELL_INFO,
        m_modelPath.c_str(),
        loadMs,
        static_cast<unsigned long long>(loadResult.inputSize));
    if (asset.getContents().empty())
        failExit("Failed to load asset %s.", m_modelPath.c_str());

    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
    const auto extractStart = clock_t::now();
    if (!appendGeometriesFromBundle(asset, geometries))
        failExit("Asset loaded but not a supported type for %s.", m_modelPath.c_str());
    const auto extractMs = toMs(clock_t::now() - extractStart);
    if (geometries.empty())
        failExit("No geometry found in asset %s.", m_modelPath.c_str());
    const auto outerMs = toMs(clock_t::now() - loadOuterStart);
    const auto nonLoaderMs = std::max(0.0, outerMs - loadMs);
    m_logger->log(
        "Asset load outer perf: path=%s getAsset=%.3f ms extract=%.3f ms total=%.3f ms non_loader=%.3f ms",
        ILogger::ELL_INFO,
        m_modelPath.c_str(),
        loadMs,
        extractMs,
        outerMs,
        nonLoaderMs);

    m_currentCpuGeom = geometries[0];

    using aabb_t = hlsl::shapes::AABB<3, double>;
    auto printAABB = [&](const aabb_t& aabb, const char* extraMsg = "")->void
        {
            m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)", ILogger::ELL_INFO, extraMsg, aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
        };
    auto bound = aabb_t::create();
    // convert the geometries
    {
        smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get() });

        const auto transferFamily = getTransferUpQueue()->getFamilyIndex();

        struct SInputs : CAssetConverter::SInputs
        {
            virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const CAssetConverter::patch_t<asset::ICPUBuffer>& patch) const
            {
                return sharedBufferOwnership;
            }

            core::vector<uint32_t> sharedBufferOwnership;
        } inputs = {};
        core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(), CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
        {
            inputs.logger = m_logger.get();
            std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = { &geometries.front().get(),geometries.size() };
            std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
            // set up shared ownership so we don't have to 
            core::unordered_set<uint32_t> families;
            families.insert(transferFamily);
            families.insert(getGraphicsQueue()->getFamilyIndex());
            if (families.size() > 1)
                for (const auto fam : families)
                    inputs.sharedBufferOwnership.push_back(fam);
        }

        // reserve
        auto reservation = converter->reserve(inputs);
        if (!reservation)
        {
            failExit("Failed to reserve GPU objects for CPU->GPU conversion.");
        }

        // convert
        {
            auto semaphore = m_device->createSemaphore(0u);

            constexpr auto MultiBuffering = 2;
            std::array<smart_refctd_ptr<IGPUCommandBuffer>, MultiBuffering> commandBuffers = {};
            {
                auto pool = m_device->createCommandPool(transferFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
                pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, commandBuffers, smart_refctd_ptr(m_logger));
            }
            commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

            std::array<IQueue::SSubmitInfo::SCommandBufferInfo, MultiBuffering> commandBufferSubmits;
            for (auto i = 0; i < MultiBuffering; i++)
                commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

            SIntendedSubmitInfo transfer = {};
            transfer.queue = getTransferUpQueue();
            transfer.scratchCommandBuffers = commandBufferSubmits;
            transfer.scratchSemaphore = {
                .semaphore = semaphore.get(),
                .value = 0u,
                .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
            };

            CAssetConverter::SConvertParams cpar = {};
            cpar.utilities = m_utils.get();
            cpar.transfer = &transfer;

            auto future = reservation.convert(cpar);
            if (future.copy() != IQueue::RESULT::SUCCESS)
                failExit("Failed to await submission feature.");
        }

        auto tmp = hlsl::float32_t4x3(
            hlsl::float32_t3(1, 0, 0),
            hlsl::float32_t3(0, 1, 0),
            hlsl::float32_t3(0, 0, 1),
            hlsl::float32_t3(0, 0, 0));
        core::vector<hlsl::float32_t3x4> worldTforms;
        const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
        m_aabbInstances.resize(converted.size());
        if (m_drawBBMode == DBBM_OBB)
            m_obbInstances.resize(converted.size());
        for (uint32_t i = 0; i < converted.size(); i++)
        {
            const auto& cpuGeom = geometries[i].get();
            const auto promoted = getGeometryAABB(cpuGeom);
            printAABB(promoted, "Geometry");
            const auto promotedWorld = hlsl::float64_t3x4(worldTforms.emplace_back(hlsl::transpose(tmp)));
            const auto translation = hlsl::float64_t3(
                static_cast<double>(tmp[3].x),
                static_cast<double>(tmp[3].y),
                static_cast<double>(tmp[3].z));
            const auto transformed = translateAABB(promoted, translation);
            printAABB(transformed, "Transformed");
            bound = hlsl::shapes::util::union_(transformed, bound);

#ifdef NBL_BUILD_DEBUG_DRAW
            auto& aabbInst = m_aabbInstances[i];
            const auto tmpAabb = shapes::AABB<3, float>(promoted.minVx, promoted.maxVx);

            hlsl::float32_t3x4 aabbTransform = ext::debug_draw::DrawAABB::getTransformFromAABB(tmpAabb);
            const auto tmpWorld = hlsl::float32_t3x4(promotedWorld);
            const auto world4x4 = float32_t4x4{
                tmpWorld[0],
                tmpWorld[1],
                tmpWorld[2],
                float32_t4(0, 0, 0, 1)
            };

            aabbInst.color = { 1, 1, 1, 1 };
            aabbInst.transform = math::linalg::promoted_mul(world4x4, aabbTransform);

            if (m_drawBBMode == DBBM_OBB)
            {
                auto& obbInst = m_obbInstances[i];
                const auto obb = CPolygonGeometryManipulator::calculateOBB(
                    cpuGeom->getPositionView().getElementCount(),
                    [geo = cpuGeom, &world4x4](size_t vertex_i) {
                        hlsl::float32_t3 pt;
                        geo->getPositionView().decodeElement(vertex_i, pt);
                        return pt;
                    });
                obbInst.color = { 0, 0, 1, 1 };
                obbInst.transform = math::linalg::promoted_mul(world4x4, obb.transform);
            }
#endif
        }

        printAABB(bound, "Total");
        if (!m_renderer->addGeometries({ &converted.front().get(),converted.size() }))
            failExit("Failed to add geometries to renderer.");
        if (m_logger)
        {
            const auto& gpuGeos = m_renderer->getGeometries();
            for (size_t geoIx = 0u; geoIx < gpuGeos.size(); ++geoIx)
            {
                const auto& gpuGeo = gpuGeos[geoIx];
                m_logger->log(
                    "Renderer geo state: idx=%llu elem=%u posView=%u normalView=%u indexType=%u",
                    ILogger::ELL_DEBUG,
                    static_cast<unsigned long long>(geoIx),
                    gpuGeo.elementCount,
                    static_cast<uint32_t>(gpuGeo.positionView),
                    static_cast<uint32_t>(gpuGeo.normalView),
                    static_cast<uint32_t>(gpuGeo.indexType));
            }
        }

        auto worlTformsIt = worldTforms.begin();
        for (const auto& geo : m_renderer->getGeometries())
            m_renderer->m_instances.push_back({
                .world = *(worlTformsIt++),
                .packedGeo = &geo
            });
    }

    if (updateCamera)
    {
        setupCameraFromAABB(bound);
        if (storeCamera)
            storeCameraState();
    }
    else if (m_referenceCamera)
        applyCameraState(*m_referenceCamera);
    else
        setupCameraFromAABB(bound);

    return true;
}

bool MeshLoadersApp::loadRowView(const RowViewReloadMode mode)
{
    if (m_cases.empty())
        failExit("No test cases loaded for row view.");

    using clock_t = std::chrono::high_resolution_clock;
    RowViewPerfStats stats = {};
    stats.incremental = (mode == RowViewReloadMode::Incremental);
    stats.cases = m_cases.size();
    const auto totalStart = clock_t::now();

    const auto clearStart = clock_t::now();
    if (mode == RowViewReloadMode::Full)
    {
        m_renderer->m_instances.clear();
        m_renderer->clearGeometries({ .semaphore = m_semaphore.get(),.value = m_realFrameIx });
    }
    stats.clearMs = toMs(clock_t::now() - clearStart);

    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
    core::vector<hlsl::shapes::AABB<3, double>> aabbs;
    geometries.reserve(m_cases.size());
    aabbs.reserve(m_cases.size());

    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> cpuToConvert;
    core::vector<CachedGeometryEntry*> convertEntries;

    m_rowViewCache.reserve(m_cases.size());

    IAssetLoader::SAssetLoadParams params = makeLoadParams();

    for (const auto& testCase : m_cases)
    {
        const auto& path = testCase.path;
        if (!std::filesystem::exists(path))
            failExit("Missing input: %s", path.string().c_str());

        const auto cacheKey = makeCacheKey(path);
        auto& entry = m_rowViewCache[cacheKey];
        double assetLoadMs = 0.0;
        bool cached = true;
        if (!entry.cpu)
        {
            stats.cpuMisses++;
            cached = false;
            AssetLoadCallResult loadResult = {};
            if (!loadAssetCallFromPath(path, params, loadResult))
                failExit("Failed to open input file %s.", path.string().c_str());
            auto asset = std::move(loadResult.bundle);
            assetLoadMs = loadResult.getAssetMs;
            stats.loadMs += assetLoadMs;
            if (asset.getContents().empty())
                failExit("Failed to load asset %s.", path.string().c_str());

            const auto extractStart = clock_t::now();
            core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> found;
            if (appendGeometriesFromBundle(asset, found))
            {
                if (!found.empty())
                    entry.cpu = found.front();
            }
            stats.extractMs += toMs(clock_t::now() - extractStart);
            if (!entry.cpu)
                failExit("No geometry found in asset %s.", path.string().c_str());

            const auto aabbStart = clock_t::now();
            entry.aabb = getGeometryAABB(entry.cpu.get());
            entry.hasAabb = isValidAABB(entry.aabb);
            stats.aabbMs += toMs(clock_t::now() - aabbStart);
        }
        else
        {
            stats.cpuHits++;
            if (!entry.hasAabb)
            {
                const auto aabbStart = clock_t::now();
                entry.aabb = getGeometryAABB(entry.cpu.get());
                entry.hasAabb = isValidAABB(entry.aabb);
                stats.aabbMs += toMs(clock_t::now() - aabbStart);
            }
        }
        logRowViewAssetLoad(path, assetLoadMs, cached);

        if (!entry.gpu)
        {
            stats.gpuMisses++;
            cpuToConvert.push_back(entry.cpu);
            convertEntries.push_back(&entry);
        }
        else
        {
            stats.gpuHits++;
        }

        geometries.push_back(entry.cpu);
        aabbs.push_back(entry.aabb);
    }

    if (geometries.empty())
        failExit("No geometry found for row view.");
    logRowViewLoadTotal(stats.loadMs, stats.cpuHits, stats.cpuMisses);

    if (!cpuToConvert.empty())
    {
        stats.convertCount = cpuToConvert.size();
        const auto convertStart = clock_t::now();

        smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get() });
        const auto transferFamily = getTransferUpQueue()->getFamilyIndex();

        struct SInputs : CAssetConverter::SInputs
        {
            virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t, const asset::ICPUBuffer*, const CAssetConverter::patch_t<asset::ICPUBuffer>&) const
            {
                return sharedBufferOwnership;
            }

            core::vector<uint32_t> sharedBufferOwnership;
        } inputs = {};
        core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(cpuToConvert.size(), CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
        {
            inputs.logger = m_logger.get();
            std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = { &cpuToConvert.front().get(),cpuToConvert.size() };
            std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
            core::unordered_set<uint32_t> families;
            families.insert(transferFamily);
            families.insert(getGraphicsQueue()->getFamilyIndex());
            if (families.size() > 1)
                for (const auto fam : families)
                    inputs.sharedBufferOwnership.push_back(fam);
        }

        auto reservation = converter->reserve(inputs);
        if (!reservation)
            failExit("Failed to reserve GPU objects for CPU->GPU conversion.");

        {
            auto semaphore = m_device->createSemaphore(0u);

            constexpr auto MultiBuffering = 2;
            std::array<smart_refctd_ptr<IGPUCommandBuffer>, MultiBuffering> commandBuffers = {};
            {
                auto pool = m_device->createCommandPool(transferFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
                pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, commandBuffers, smart_refctd_ptr(m_logger));
            }
            commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

            std::array<IQueue::SSubmitInfo::SCommandBufferInfo, MultiBuffering> commandBufferSubmits;
            for (auto i = 0; i < MultiBuffering; i++)
                commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

            SIntendedSubmitInfo transfer = {};
            transfer.queue = getTransferUpQueue();
            transfer.scratchCommandBuffers = commandBufferSubmits;
            transfer.scratchSemaphore = {
                .semaphore = semaphore.get(),
                .value = 0u,
                .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
            };

            CAssetConverter::SConvertParams cpar = {};
            cpar.utilities = m_utils.get();
            cpar.transfer = &transfer;

            auto future = reservation.convert(cpar);
            if (future.copy() != IQueue::RESULT::SUCCESS)
                failExit("Failed to await submission feature.");
        }

        const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
        for (size_t i = 0u; i < converted.size(); ++i)
            convertEntries[i]->gpu = converted[i];

        stats.convertMs = toMs(clock_t::now() - convertStart);
    }

    size_t existingCount = m_renderer->getGeometries().size();
    const bool incremental = (mode == RowViewReloadMode::Incremental) && (existingCount <= m_cases.size());
    if (!incremental && mode == RowViewReloadMode::Incremental)
        return loadRowView(RowViewReloadMode::Full);

    if (mode == RowViewReloadMode::Full)
    {
        core::vector<const IGPUPolygonGeometry*> allGeometries;
        allGeometries.reserve(m_cases.size());
        for (const auto& testCase : m_cases)
        {
            const auto& entry = m_rowViewCache[makeCacheKey(testCase.path)];
            if (!entry.gpu)
                failExit("Missing GPU geometry for %s.", testCase.path.string().c_str());
            allGeometries.push_back(entry.gpu.get());
        }
        stats.addCount = allGeometries.size();
        const auto addStart = clock_t::now();
        if (!allGeometries.empty())
            if (!m_renderer->addGeometries({ allGeometries.data(),allGeometries.size() }))
                failExit("Failed to add geometries to renderer.");
        stats.addGeoMs = toMs(clock_t::now() - addStart);
    }
    else
    {
        const size_t addCount = (existingCount < m_cases.size()) ? (m_cases.size() - existingCount) : 0u;
        stats.addCount = addCount;
        if (addCount > 0u)
        {
            core::vector<const IGPUPolygonGeometry*> newGeometries;
            newGeometries.reserve(addCount);
            for (size_t i = existingCount; i < m_cases.size(); ++i)
            {
                const auto& entry = m_rowViewCache[makeCacheKey(m_cases[i].path)];
                if (!entry.gpu)
                    failExit("Missing GPU geometry for %s.", m_cases[i].path.string().c_str());
                newGeometries.push_back(entry.gpu.get());
            }
            const auto addStart = clock_t::now();
            if (!m_renderer->addGeometries({ newGeometries.data(),newGeometries.size() }))
                failExit("Failed to add geometries to renderer.");
            stats.addGeoMs = toMs(clock_t::now() - addStart);
        }
    }

    using aabb_t = hlsl::shapes::AABB<3, double>;
    auto printAABB = [&](const aabb_t& aabb, const char* extraMsg = "")->void
        {
            m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)", ILogger::ELL_INFO, extraMsg, aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
        };
    auto bound = aabb_t::create();

    const auto layoutStart = clock_t::now();
    double targetExtent = 0.0;
    core::vector<double> maxDims;
    maxDims.reserve(aabbs.size());
    for (const auto& aabb : aabbs)
    {
        const auto extent = aabb.getExtent();
        const double maxDim = std::max({ extent.x, extent.y, extent.z, 0.001 });
        maxDims.push_back(maxDim);
        if (maxDim > targetExtent)
            targetExtent = maxDim;
    }

    core::vector<double> scales;
    scales.reserve(aabbs.size());
    for (const auto maxDim : maxDims)
        scales.push_back(targetExtent / maxDim);

    double maxWidth = 0.0;
    double totalWidth = 0.0;
    core::vector<double> widths;
    widths.reserve(aabbs.size());
    for (size_t i = 0; i < aabbs.size(); ++i)
    {
        const auto extent = aabbs[i].getExtent();
        const double width = std::max(0.001, extent.x * scales[i]);
        widths.push_back(width);
        totalWidth += width;
        if (width > maxWidth)
            maxWidth = width;
    }
    const double spacing = std::max(0.05 * maxWidth, 0.01);
    const double totalSpan = totalWidth + spacing * double(widths.size() > 0 ? widths.size() - 1 : 0);
    double cursor = -0.5 * totalSpan;
    stats.layoutMs = toMs(clock_t::now() - layoutStart);

    const auto instanceStart = clock_t::now();
    auto tmp = hlsl::float32_t4x3(
        hlsl::float32_t3(1, 0, 0),
        hlsl::float32_t3(0, 1, 0),
        hlsl::float32_t3(0, 0, 1),
        hlsl::float32_t3(0, 0, 0)
    );
    core::vector<hlsl::float32_t3x4> worldTforms;
    worldTforms.reserve(geometries.size());
    m_aabbInstances.resize(geometries.size());
    if (m_drawBBMode == DBBM_OBB)
        m_obbInstances.resize(geometries.size());
    m_renderer->m_instances.clear();

    for (uint32_t i = 0; i < geometries.size(); i++)
    {
        const auto& cpuGeom = geometries[i].get();
        const auto aabb = aabbs[i];
        printAABB(aabb, "Geometry");

        const double scale = scales[i];
        const auto center = (aabb.minVx + aabb.maxVx) * 0.5;
        const double width = widths[i];
        const double targetCenterX = cursor + 0.5 * width;
        cursor += width + spacing;

        const double tx = targetCenterX - scale * center.x;
        const double ty = -scale * center.y;
        const double tz = -scale * center.z;
        tmp[0] = hlsl::float32_t3(static_cast<float>(scale), 0.f, 0.f);
        tmp[1] = hlsl::float32_t3(0.f, static_cast<float>(scale), 0.f);
        tmp[2] = hlsl::float32_t3(0.f, 0.f, static_cast<float>(scale));
        tmp[3] = hlsl::float32_t3(static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz));

        const auto promotedWorld = hlsl::float64_t3x4(worldTforms.emplace_back(hlsl::transpose(tmp)));
        const auto translation = hlsl::float64_t3(tx, ty, tz);
        const auto scaled = scaleAABB(aabb, scale);
        const auto transformed = translateAABB(scaled, translation);
        printAABB(transformed, "Transformed");
        bound = hlsl::shapes::util::union_(transformed, bound);

#ifdef NBL_BUILD_DEBUG_DRAW
        auto& aabbInst = m_aabbInstances[i];
        const auto tmpAabb = shapes::AABB<3, float>(aabb.minVx, aabb.maxVx);
        hlsl::float32_t3x4 aabbTransform = ext::debug_draw::DrawAABB::getTransformFromAABB(tmpAabb);
        const auto tmpWorld = hlsl::float32_t3x4(promotedWorld);
        const auto world4x4 = float32_t4x4{
            tmpWorld[0],
            tmpWorld[1],
            tmpWorld[2],
            float32_t4(0, 0, 0, 1)
        };
        aabbInst.color = { 1,1,1,1 };
        aabbInst.transform = math::linalg::promoted_mul(world4x4, aabbTransform);

        if (m_drawBBMode == DBBM_OBB)
        {
            auto& obbInst = m_obbInstances[i];
            const auto obb = CPolygonGeometryManipulator::calculateOBB(
                cpuGeom->getPositionView().getElementCount(),
                [geo = cpuGeom](size_t vertex_i) {
                    hlsl::float32_t3 pt;
                    geo->getPositionView().decodeElement(vertex_i, pt);
                    return pt;
                });
            obbInst.color = { 0, 0, 1, 1 };
            obbInst.transform = math::linalg::promoted_mul(world4x4, obb.transform);
        }
#endif
    }

    printAABB(bound, "Total");
    for (uint32_t i = 0; i < worldTforms.size(); i++)
    {
        m_renderer->m_instances.push_back({
            .world = worldTforms[i],
            .packedGeo = &m_renderer->getGeometry(i)
            });
    }
    stats.instanceMs = toMs(clock_t::now() - instanceStart);

    const auto cameraStart = clock_t::now();
    setupCameraFromAABB(bound);
    stats.cameraMs = toMs(clock_t::now() - cameraStart);

    m_modelPath = "Row view (all meshes)";
    m_rowViewScreenshotPath = m_screenshotPrefixPath / "meshloaders_row_view.png";
    m_rowViewScreenshotCaptured = false;
    stats.totalMs = toMs(clock_t::now() - totalStart);
    logRowViewPerf(stats);
    return true;
}

bool MeshLoadersApp::writeGeometry(smart_refctd_ptr<const ICPUPolygonGeometry> geometry, const std::string& savePath)
{
    using clock_t = std::chrono::high_resolution_clock;
    const auto writeOuterStart = clock_t::now();
    IAsset* assetPtr = const_cast<IAsset*>(static_cast<const IAsset*>(geometry.get()));
    const auto ext = normalizeExtension(system::path(savePath));
    auto flags = asset::EWF_MESH_IS_RIGHT_HANDED;
    if (ext != ".obj")
        flags = static_cast<asset::E_WRITER_FLAGS>(flags | asset::EWF_BINARY);
    IAssetWriter::SAssetWriteParams params{ assetPtr, flags };
    params.logger = getAssetLoadLogger();
    m_logger->log("Saving mesh to %s", ILogger::ELL_INFO, savePath.c_str());
    const auto openStart = clock_t::now();
    system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> writeFileFuture;
    m_system->createFile(writeFileFuture, system::path(savePath), system::IFile::ECF_WRITE);
    core::smart_refctd_ptr<system::IFile> writeFile;
    writeFileFuture.acquire().move_into(writeFile);
    const auto openMs = toMs(clock_t::now() - openStart);
    if (!writeFile)
    {
        m_logger->log("Failed to open output file %s", ILogger::ELL_ERROR, savePath.c_str());
        return false;
    }
    const auto start = clock_t::now();
    if (!m_assetMgr->writeAsset(writeFile.get(), params))
    {
        const auto ms = toMs(clock_t::now() - start);
        m_logger->log("Failed to save %s after %.3f ms", ILogger::ELL_ERROR, savePath.c_str(), ms);
        return false;
    }
    const auto writeMs = toMs(clock_t::now() - start);
    const auto statStart = clock_t::now();
    uintmax_t size = 0u;
    if (std::filesystem::exists(savePath))
        size = std::filesystem::file_size(savePath);
    const auto statMs = toMs(clock_t::now() - statStart);
    const auto outerMs = toMs(clock_t::now() - writeOuterStart);
    const auto nonWriterMs = std::max(0.0, outerMs - writeMs);
    m_logger->log("Asset write call perf: path=%s ext=%s time=%.3f ms size=%llu", ILogger::ELL_INFO, savePath.c_str(), ext.c_str(), writeMs, static_cast<unsigned long long>(size));
    m_logger->log(
        "Asset write outer perf: path=%s ext=%s open=%.3f ms writeAsset=%.3f ms stat=%.3f ms total=%.3f ms non_writer=%.3f ms size=%llu",
        ILogger::ELL_INFO,
        savePath.c_str(),
        ext.c_str(),
        openMs,
        writeMs,
        statMs,
        outerMs,
        nonWriterMs,
        static_cast<unsigned long long>(size));
    m_logger->log("Writer perf: path=%s ext=%s time=%.3f ms size=%llu", ILogger::ELL_INFO, savePath.c_str(), ext.c_str(), writeMs, static_cast<unsigned long long>(size));
    m_logger->log("Mesh successfully saved!", ILogger::ELL_INFO);
    return true;
}

void MeshLoadersApp::setupCameraFromAABB(const hlsl::shapes::AABB<3, double>& bound)
{
    const auto extent = bound.getExtent();
    const auto aspectRatio = double(m_window->getWidth()) / double(m_window->getHeight());
    const double fovY = 1.2;
    const double fovX = 2.0 * std::atan(std::tan(fovY * 0.5) * aspectRatio);
    const auto center = (bound.minVx + bound.maxVx) * 0.5;
    const auto halfExtent = extent * 0.5;
    const double halfX = std::max(halfExtent.x, 0.001);
    const double halfY = std::max(halfExtent.y, 0.001);
    const double halfZ = std::max(halfExtent.z, 0.001);
    const double safeRadius = std::max({ halfX, halfY, halfZ });

    const double distY = halfY / std::tan(fovY * 0.5);
    const double distX = halfX / std::tan(fovX * 0.5);
    double dist = std::max(distX, distY) + halfZ;
    dist *= 1.1;

    const auto dir = hlsl::float64_t3(0.0, 0.0, 1.0);
    const auto pos = center + dir * dist;

    const double margin = halfZ * 0.1 + 0.01;
    const double nearPlane = std::max(0.001, dist - halfZ - margin);
    const double farPlane = dist + halfZ + margin;

    const auto projection = nbl::hlsl::buildProjectionMatrixPerspectiveFovRH<nbl::hlsl::float32_t>(
        static_cast<float>(fovY),
        static_cast<float>(aspectRatio),
        static_cast<float>(nearPlane),
        static_cast<float>(farPlane));
    camera.setProjectionMatrix(projection);
    camera.setMoveSpeed(static_cast<float>(safeRadius * 0.1));
    camera.setPosition(vectorSIMDf(pos.x, pos.y, pos.z));
    camera.setTarget(vectorSIMDf(center.x, center.y, center.z));
}

hlsl::shapes::AABB<3, double> MeshLoadersApp::translateAABB(const hlsl::shapes::AABB<3, double>& aabb, const hlsl::float64_t3& translation)
{
    auto out = aabb;
    out.minVx += translation;
    out.maxVx += translation;
    return out;
}

hlsl::shapes::AABB<3, double> MeshLoadersApp::scaleAABB(const hlsl::shapes::AABB<3, double>& aabb, const double scale)
{
    auto out = aabb;
    out.minVx *= scale;
    out.maxVx *= scale;
    return out;
}

void MeshLoadersApp::storeCameraState()
{
    m_referenceCamera = CameraState{
        camera.getPosition(),
        camera.getTarget(),
        camera.getProjectionMatrix(),
        camera.getMoveSpeed()
    };
}

void MeshLoadersApp::applyCameraState(const CameraState& state)
{
    camera.setProjectionMatrix(state.projection);
    camera.setPosition(state.position);
    camera.setTarget(state.target);
    camera.setMoveSpeed(state.moveSpeed);
}

bool MeshLoadersApp::isValidAABB(const hlsl::shapes::AABB<3, double>& aabb)
{
    return
        (aabb.minVx.x <= aabb.maxVx.x) &&
        (aabb.minVx.y <= aabb.maxVx.y) &&
        (aabb.minVx.z <= aabb.maxVx.z);
}

hlsl::shapes::AABB<3, double> MeshLoadersApp::getGeometryAABB(const ICPUPolygonGeometry* geometry) const
{
    if (!geometry)
        return hlsl::shapes::AABB<3, double>::create();
    auto aabb = geometry->getAABB<hlsl::shapes::AABB<3, double>>();
    if (!isValidAABB(aabb))
    {
        CPolygonGeometryManipulator::recomputeAABB(geometry);
        aabb = geometry->getAABB<hlsl::shapes::AABB<3, double>>();
    }
    return aabb;
}

system::ILogger* MeshLoadersApp::getAssetLoadLogger() const
{
    if (m_assetLoadLogger)
        return m_assetLoadLogger.get();
    return m_logger.get();
}

IAssetLoader::SAssetLoadParams MeshLoadersApp::makeLoadParams() const
{
    IAssetLoader::SAssetLoadParams params = {};
    params.logger = getAssetLoadLogger();
    if ((m_runMode == RunMode::CI || isRowViewActive()) && !m_loaderPerfLogger)
        params.logger = nullptr;
    params.cacheFlags = IAssetLoader::ECF_DUPLICATE_TOP_LEVEL;
    params.ioPolicy.runtimeTuning.mode = m_runtimeTuningMode;
    if (m_forceLoaderContentHashes)
        params.loaderFlags = static_cast<IAssetLoader::E_LOADER_PARAMETER_FLAGS>(params.loaderFlags | IAssetLoader::ELPF_COMPUTE_CONTENT_HASHES);
    return params;
}

bool MeshLoadersApp::loadAssetCallFromPath(const system::path& modelPath, const IAssetLoader::SAssetLoadParams& params, AssetLoadCallResult& out)
{
    using clock_t = std::chrono::high_resolution_clock;
    if (std::filesystem::exists(modelPath))
        out.inputSize = std::filesystem::file_size(modelPath);
    else
        out.inputSize = 0u;

    const auto loadStart = clock_t::now();
    out.bundle = m_assetMgr->getAsset(modelPath.string(), params);
    out.getAssetMs = toMs(clock_t::now() - loadStart);
    return true;
}

bool MeshLoadersApp::initLoaderPerfLogger(const system::path& logPath)
{
    if (!m_system)
        return logFail("Could not initialize loader perf logger because system is unavailable.");
    if (logPath.empty())
        return false;
    const auto parent = logPath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec)
            return logFail("Could not create loader perf log directory %s", parent.string().c_str());
    }
    system::ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
    m_system->createFile(future, logPath, system::IFile::ECF_READ_WRITE);
    if (!future.wait() || !future.get())
        return logFail("Could not create loader perf log file %s", logPath.string().c_str());
    const auto logMask = core::bitflag(system::ILogger::ELL_ALL);
    m_loaderPerfLogger = core::make_smart_refctd_ptr<system::CFileLogger>(future.copy(), false, logMask);
    m_assetLoadLogger = m_loaderPerfLogger;
    return true;
}


