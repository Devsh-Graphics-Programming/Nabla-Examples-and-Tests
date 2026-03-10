// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "BundleGeometryItems.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>
#include <nbl/builtin/hlsl/math/linalg/transform.hlsl>
#include "nbl/examples/common/GeometryAABBUtilities.h"

namespace
{
using display_aabb_t = hlsl::shapes::AABB<3, double>;
struct DisplayLayout
{
    core::vector<hlsl::float32_t3x4> worldTransforms = {};
    display_aabb_t bound = display_aabb_t::create();
};
struct RowLayoutGroup
{
    size_t firstGeometry = 0u;
    size_t geometryCount = 0u;
    display_aabb_t layoutAABB = display_aabb_t::create();
    bool preserveInternalTransforms = false;
    bool addAggregateDebugAABB = false;
};
struct PreparedGeometryBatch
{
    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
    core::vector<hlsl::float32_t3x4> worlds;
};
static std::optional<core::vector<video::asset_cached_t<asset::ICPUPolygonGeometry>>> convertPolygonGeometries(
    video::ILogicalDevice* device,
    video::IQueue* transferQueue,
    video::IQueue* graphicsQueue,
    video::IUtilities* utilities,
    system::ILogger* logger,
    const core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>>& geometries)
{
    if (geometries.empty())
        return core::vector<video::asset_cached_t<asset::ICPUPolygonGeometry>>{};

    smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = device });
    const auto transferFamily = transferQueue->getFamilyIndex();

    struct SInputs : CAssetConverter::SInputs
    {
        virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t, const asset::ICPUBuffer*, const CAssetConverter::patch_t<asset::ICPUBuffer>&) const
        {
            return sharedBufferOwnership;
        }

        core::vector<uint32_t> sharedBufferOwnership;
    } inputs = {};
    core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(), CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
    {
        inputs.logger = logger;
        std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = { &geometries.front().get(),geometries.size() };
        std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
        core::unordered_set<uint32_t> families;
        families.insert(transferFamily);
        families.insert(graphicsQueue->getFamilyIndex());
        if (families.size() > 1)
            for (const auto fam : families)
                inputs.sharedBufferOwnership.push_back(fam);
    }

    auto reservation = converter->reserve(inputs);
    if (!reservation)
        return std::nullopt;

    {
        auto semaphore = device->createSemaphore(0u);

        constexpr auto MultiBuffering = 2;
        std::array<smart_refctd_ptr<IGPUCommandBuffer>, MultiBuffering> commandBuffers = {};
        {
            auto pool = device->createCommandPool(transferFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
            pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, commandBuffers, core::smart_refctd_ptr<system::ILogger>(logger));
        }
        commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

        std::array<IQueue::SSubmitInfo::SCommandBufferInfo, MultiBuffering> commandBufferSubmits;
        for (auto i = 0; i < MultiBuffering; i++)
            commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

        SIntendedSubmitInfo transfer = {};
        transfer.queue = transferQueue;
        transfer.scratchCommandBuffers = commandBufferSubmits;
        transfer.scratchSemaphore = {
            .semaphore = semaphore.get(),
            .value = 0u,
            .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
        };

        CAssetConverter::SConvertParams cpar = {};
        cpar.utilities = utilities;
        cpar.transfer = &transfer;

        auto future = reservation.convert(cpar);
        if (future.copy() != IQueue::RESULT::SUCCESS)
            return std::nullopt;
    }

    const auto convertedSpan = reservation.getGPUObjects<ICPUPolygonGeometry>();
    return core::vector<video::asset_cached_t<asset::ICPUPolygonGeometry>>(convertedSpan.begin(), convertedSpan.end());
}
static hlsl::float32_t3x4 makeIdentityWorld()
{
    auto tmp = hlsl::float32_t4x3(
        hlsl::float32_t3(1, 0, 0),
        hlsl::float32_t3(0, 1, 0),
        hlsl::float32_t3(0, 0, 1),
        hlsl::float32_t3(0, 0, 0));
    return hlsl::transpose(tmp);
}
static hlsl::float32_t4x4 makeAffine4x4(const hlsl::float32_t3x4& world)
{
    return hlsl::float32_t4x4{
        world[0],
        world[1],
        world[2],
        hlsl::float32_t4(0, 0, 0, 1)
    };
}
#ifdef NBL_BUILD_DEBUG_DRAW
static ext::debug_draw::InstanceData makeAABBInstance(
    const display_aabb_t& aabb,
    const hlsl::float32_t3x4& world,
    const hlsl::float32_t4& color)
{
    ext::debug_draw::InstanceData instance = {};
    instance.color = color;
    instance.transform = math::linalg::promoted_mul(
        makeAffine4x4(world),
        ext::debug_draw::DrawAABB::getTransformFromAABB(shapes::AABB<3, float>(aabb.minVx, aabb.maxVx)));
    return instance;
}
static ext::debug_draw::InstanceData makeOBBInstance(
    const ICPUPolygonGeometry* geometry,
    const hlsl::float32_t3x4& world,
    const hlsl::float32_t4& color)
{
    ext::debug_draw::InstanceData instance = {};
    instance.color = color;
    const auto obb = CPolygonGeometryManipulator::calculateOBB(
        geometry->getPositionView().getElementCount(),
        [geometry](size_t vertex_i) {
            hlsl::float32_t3 pt;
            geometry->getPositionView().decodeElement(vertex_i, pt);
            return pt;
        });
    instance.transform = math::linalg::promoted_mul(makeAffine4x4(world), obb.transform);
    return instance;
}
#endif
static DisplayLayout buildDisplayLayout(const core::vector<display_aabb_t>& aabbs, const bool arrangeInRow)
{
    DisplayLayout retval = {};
    retval.worldTransforms.reserve(aabbs.size());
    if (!arrangeInRow)
    {
        const auto identity = makeIdentityWorld();
        for (const auto& aabb : aabbs)
        {
            retval.worldTransforms.push_back(identity);
            retval.bound = hlsl::shapes::util::union_(aabb, retval.bound);
        }
        return retval;
    }
    double targetExtent = 0.0;
    core::vector<double> maxDims;
    maxDims.reserve(aabbs.size());
    for (const auto& aabb : aabbs)
    {
        const auto extent = aabb.getExtent();
        const double maxDim = std::max({extent.x, extent.y, extent.z, 0.001});
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
    for (size_t i = 0u; i < aabbs.size(); ++i)
    {
        const auto extent = aabbs[i].getExtent();
        const double width = std::max(0.001, extent.x * scales[i]);
        widths.push_back(width);
        totalWidth += width;
        if (width > maxWidth)
            maxWidth = width;
    }
    const double spacing = std::max(0.05 * maxWidth, 0.01);
    const double totalSpan = totalWidth + spacing * double(widths.size() > 0u ? widths.size() - 1u : 0u);
    double cursor = -0.5 * totalSpan;
    auto tmp = hlsl::float32_t4x3(
        hlsl::float32_t3(1, 0, 0),
        hlsl::float32_t3(0, 1, 0),
        hlsl::float32_t3(0, 0, 1),
        hlsl::float32_t3(0, 0, 0));
    for (size_t i = 0u; i < aabbs.size(); ++i)
    {
        const auto& aabb = aabbs[i];
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
        const auto world = hlsl::transpose(tmp);
        retval.worldTransforms.push_back(world);
        retval.bound = hlsl::shapes::util::union_(nbl::hlsl::math::linalg::pseudo_mul(hlsl::float64_t3x4(world), aabb), retval.bound);
    }
    return retval;
}
static bool extractPreparedGeometryBatch(const asset::SAssetBundle& bundle, PreparedGeometryBatch& out, const bool preserveTransforms)
{
    core::vector<meshloaders::BundleGeometryItem> items;
    if (!meshloaders::collectBundleGeometryItems(bundle, items, preserveTransforms))
        return false;
    out.geometries.reserve(items.size());
    out.worlds.reserve(items.size());
    for (auto& item : items)
    {
        out.worlds.push_back(item.world);
        out.geometries.push_back(std::move(item.geometry));
    }
    return !out.geometries.empty();
}
template<typename GetAABB, typename WarnInvalid>
static void collectGeometryAABBs(
    const core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>>& geometries,
    core::vector<display_aabb_t>& out,
    GetAABB&& getAABB,
    WarnInvalid&& warnInvalid)
{
    out.clear();
    out.reserve(geometries.size());
    for (uint32_t i = 0u; i < geometries.size(); ++i)
    {
        auto aabb = getAABB(geometries[i].get());
        if (!nbl::examples::geometry::isValidAABB(aabb))
        {
            warnInvalid(i);
            aabb = nbl::examples::geometry::fallbackUnitAABB();
        }
        out.push_back(aabb);
    }
}
}

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
    m_render.renderer->m_instances.clear();
    m_render.renderer->clearGeometries({ .semaphore = m_render.semaphore.get(),.value = m_render.realFrameIx });
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
	m_render.currentCpuAsset = (asset.getContents().size() == 1u) ? asset.getContents()[0] : nullptr;

    PreparedGeometryBatch batch = {};
    const auto extractStart = clock_t::now();
    const bool renderAsScene = asset.getAssetType() == IAsset::E_TYPE::ET_SCENE;
    if (!extractPreparedGeometryBatch(asset, batch, renderAsScene))
        failExit("Asset loaded but not a supported type for %s.", m_modelPath.c_str());
    const auto extractMs = toMs(clock_t::now() - extractStart);
    if (batch.geometries.empty())
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

    m_render.currentCpuGeom = batch.geometries[0];

    using aabb_t = hlsl::shapes::AABB<3, double>;
    auto printAABB = [&](const aabb_t& aabb, const char* extraMsg = "")->void
        {
            m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)", ILogger::ELL_INFO, extraMsg, aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
        };
    core::vector<aabb_t> aabbs;
    collectGeometryAABBs(
        batch.geometries,
        aabbs,
        [this](const ICPUPolygonGeometry* geometry) { return getGeometryAABB(geometry); },
        [this](const uint32_t geoIx) {
            m_logger->log("Invalid geometry AABB for %s (geo=%u). Using fallback unit AABB for framing.", ILogger::ELL_WARNING, m_modelPath.c_str(), geoIx);
        });
    core::vector<hlsl::float32_t3x4> worldTforms;
    worldTforms.reserve(batch.geometries.size());
    auto bound = display_aabb_t::create();
    if (renderAsScene)
    {
        for (uint32_t i = 0u; i < batch.worlds.size(); ++i)
        {
            const auto& world = batch.worlds[i];
            worldTforms.push_back(world);
            bound = hlsl::shapes::util::union_(nbl::hlsl::math::linalg::pseudo_mul(hlsl::float64_t3x4(world), aabbs[i]), bound);
        }
    }
    else
    {
        const auto layout = buildDisplayLayout(aabbs, batch.geometries.size() > 1u);
        worldTforms = layout.worldTransforms;
        bound = layout.bound;
    }
    // convert the geometries
    {
        const auto converted = convertPolygonGeometries(
            m_device.get(),
            getTransferUpQueue(),
            getGraphicsQueue(),
            m_utils.get(),
            m_logger.get(),
            batch.geometries);
        if (!converted.has_value())
            failExit("Failed to convert CPU geometries to GPU.");
        const auto& convertedGeometries = *converted;
        m_aabbInstances.resize(convertedGeometries.size());
        if (m_drawBBMode == DBBM_OBB)
            m_obbInstances.resize(convertedGeometries.size());
        for (uint32_t i = 0; i < convertedGeometries.size(); i++)
        {
            const auto& cpuGeom = batch.geometries[i].get();
            const auto& promoted = aabbs[i];
            printAABB(promoted, "Geometry");
            const auto promotedWorld = hlsl::float64_t3x4(worldTforms[i]);
            const auto transformed = nbl::hlsl::math::linalg::pseudo_mul(promotedWorld, promoted);
            printAABB(transformed, "Transformed");

#ifdef NBL_BUILD_DEBUG_DRAW
            m_aabbInstances[i] = makeAABBInstance(promoted, worldTforms[i], hlsl::float32_t4(1, 1, 1, 1));

            if (m_drawBBMode == DBBM_OBB)
            {
                m_obbInstances[i] = makeOBBInstance(cpuGeom, worldTforms[i], hlsl::float32_t4(0, 0, 1, 1));
            }
#endif
        }

        printAABB(bound, "Total");
        if (!m_render.renderer->addGeometries({ &convertedGeometries.front().get(),convertedGeometries.size() }))
            failExit("Failed to add geometries to renderer.");
        if (m_logger)
        {
            const auto& gpuGeos = m_render.renderer->getGeometries();
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

        auto worldTformsIt = worldTforms.begin();
        for (const auto& geo : m_render.renderer->getGeometries())
            m_render.renderer->m_instances.push_back({
                .world = *(worldTformsIt++),
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
    if (m_runtime.cases.empty())
        failExit("No test cases loaded for row view.");

    using clock_t = std::chrono::high_resolution_clock;
    RowViewPerfStats stats = {};
    stats.incremental = (mode == RowViewReloadMode::Incremental);
    stats.cases = m_runtime.cases.size();
    const auto totalStart = clock_t::now();

    const auto clearStart = clock_t::now();
    if (mode == RowViewReloadMode::Full)
    {
        m_render.renderer->m_instances.clear();
        m_render.renderer->clearGeometries({ .semaphore = m_render.semaphore.get(),.value = m_render.realFrameIx });
    }
    stats.clearMs = toMs(clock_t::now() - clearStart);

    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
    core::vector<hlsl::shapes::AABB<3, double>> aabbs;
    core::vector<hlsl::float32_t3x4> sourceWorlds;
    core::vector<RowLayoutGroup> layoutGroups;

    core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> cpuToConvert;
    struct ConvertTarget { CachedGeometryEntry* entry = nullptr; size_t geometryIx = 0u; };
    core::vector<ConvertTarget> convertTargets;

    m_rowView.cache.reserve(m_runtime.cases.size());

    IAssetLoader::SAssetLoadParams params = makeLoadParams();
    auto rebuildTileAABB = [&](CachedGeometryEntry& entry, const system::path& path) -> void
    {
        entry.tileAABB = display_aabb_t::create();
        if (!entry.layoutAsSingleTile)
            return;
        for (uint32_t geoIx = 0u; geoIx < entry.aabbs.size(); ++geoIx)
            entry.tileAABB = hlsl::shapes::util::union_(
                nbl::hlsl::math::linalg::pseudo_mul(hlsl::float64_t3x4(entry.world[geoIx]), entry.aabbs[geoIx]),
                entry.tileAABB);
        if (!isValidAABB(entry.tileAABB))
        {
            m_logger->log("Invalid row-view scene AABB for %s. Using fallback unit AABB.", ILogger::ELL_WARNING, path.string().c_str());
            entry.tileAABB = nbl::examples::geometry::fallbackUnitAABB();
        }
    };
    auto assignCachedEntryGeometryBatch = [&](CachedGeometryEntry& entry, PreparedGeometryBatch&& batch) -> void
    {
        entry.cpu = std::move(batch.geometries);
        entry.world = std::move(batch.worlds);
        if (!entry.layoutAsSingleTile)
            entry.world.assign(entry.cpu.size(), hlsl::math::linalg::identity<hlsl::float32_t3x4>());
        entry.gpu.resize(entry.cpu.size());
    };
    auto refreshCachedEntryAABBs = [&](CachedGeometryEntry& entry, const system::path& path) -> void
    {
        collectGeometryAABBs(
            entry.cpu,
            entry.aabbs,
            [this](const ICPUPolygonGeometry* geometry) { return getGeometryAABB(geometry); },
            [this, &path](const uint32_t geoIx) {
                m_logger->log("Invalid row-view geometry AABB for %s (geo=%u). Using fallback unit AABB.", ILogger::ELL_WARNING, path.string().c_str(), geoIx);
            });
        rebuildTileAABB(entry, path);
    };
    auto appendCachedEntryToLayoutInputs = [&](
        const CachedGeometryEntry& entry,
        core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>>& geometryOut,
        core::vector<hlsl::shapes::AABB<3, double>>& aabbOut,
        core::vector<hlsl::float32_t3x4>& worldOut,
        core::vector<RowLayoutGroup>& groupOut) -> void
    {
        const size_t firstGeometry = geometryOut.size();
        geometryOut.insert(geometryOut.end(), entry.cpu.begin(), entry.cpu.end());
        aabbOut.insert(aabbOut.end(), entry.aabbs.begin(), entry.aabbs.end());
        worldOut.insert(worldOut.end(), entry.world.begin(), entry.world.end());
        if (entry.layoutAsSingleTile)
        {
            groupOut.push_back({
                .firstGeometry = firstGeometry,
                .geometryCount = entry.cpu.size(),
                .layoutAABB = entry.tileAABB,
                .preserveInternalTransforms = true,
                .addAggregateDebugAABB = true
                });
            return;
        }
        for (size_t geoIx = 0u; geoIx < entry.cpu.size(); ++geoIx)
        {
            groupOut.push_back({
                .firstGeometry = firstGeometry + geoIx,
                .geometryCount = 1u,
                .layoutAABB = entry.aabbs[geoIx],
                .preserveInternalTransforms = false,
                .addAggregateDebugAABB = false
                });
        }
    };

    for (const auto& testCase : m_runtime.cases)
    {
        const auto& path = testCase.path;
        if (!std::filesystem::exists(path))
            failExit("Missing input: %s", path.string().c_str());

        const auto cacheKey = makeCacheKey(path);
        auto& entry = m_rowView.cache[cacheKey];
        double assetLoadMs = 0.0;
        bool cached = true;
        if (entry.cpu.empty())
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
            entry.world.clear();
            entry.layoutAsSingleTile = (asset.getAssetType() == IAsset::E_TYPE::ET_SCENE);
            PreparedGeometryBatch batch = {};
            if (extractPreparedGeometryBatch(asset, batch, entry.layoutAsSingleTile))
                assignCachedEntryGeometryBatch(entry, std::move(batch));
            stats.extractMs += toMs(clock_t::now() - extractStart);
            if (entry.cpu.empty())
                failExit("No geometry found in asset %s.", path.string().c_str());

            const auto aabbStart = clock_t::now();
            refreshCachedEntryAABBs(entry, path);
            stats.aabbMs += toMs(clock_t::now() - aabbStart);
        }
        else
        {
            stats.cpuHits++;
            if (entry.gpu.size() != entry.cpu.size())
                entry.gpu.resize(entry.cpu.size());
            if (entry.world.size() != entry.cpu.size())
                entry.world.assign(entry.cpu.size(), hlsl::math::linalg::identity<hlsl::float32_t3x4>());
            if (entry.aabbs.size() != entry.cpu.size())
            {
                const auto aabbStart = clock_t::now();
                refreshCachedEntryAABBs(entry, path);
                stats.aabbMs += toMs(clock_t::now() - aabbStart);
            }
        }
        logRowViewAssetLoad(path, assetLoadMs, cached);

        for (size_t geoIx = 0u; geoIx < entry.cpu.size(); ++geoIx)
        {
            if (!entry.gpu[geoIx])
            {
                stats.gpuMisses++;
                cpuToConvert.push_back(entry.cpu[geoIx]);
                convertTargets.push_back({.entry = &entry,.geometryIx = geoIx});
            }
            else
                stats.gpuHits++;
        }

        appendCachedEntryToLayoutInputs(entry, geometries, aabbs, sourceWorlds, layoutGroups);
    }

    if (geometries.empty())
        failExit("No geometry found for row view.");
    logRowViewLoadTotal(stats.loadMs, stats.cpuHits, stats.cpuMisses);

    if (!cpuToConvert.empty())
    {
        stats.convertCount = cpuToConvert.size();
        const auto convertStart = clock_t::now();
        const auto converted = convertPolygonGeometries(
            m_device.get(),
            getTransferUpQueue(),
            getGraphicsQueue(),
            m_utils.get(),
            m_logger.get(),
            cpuToConvert);
        if (!converted.has_value())
            failExit("Failed to convert CPU geometries to GPU.");
        const auto& convertedGeometries = *converted;
        for (size_t i = 0u; i < convertedGeometries.size(); ++i)
            convertTargets[i].entry->gpu[convertTargets[i].geometryIx] = convertedGeometries[i];

        stats.convertMs = toMs(clock_t::now() - convertStart);
    }

    const size_t totalGeometryCount = geometries.size();
    size_t existingCount = m_render.renderer->getGeometries().size();
    const bool incremental = (mode == RowViewReloadMode::Incremental) && (existingCount <= totalGeometryCount);
    if (!incremental && mode == RowViewReloadMode::Incremental)
        return loadRowView(RowViewReloadMode::Full);

    core::vector<const IGPUPolygonGeometry*> allGeometries;
    allGeometries.reserve(totalGeometryCount);
    for (const auto& testCase : m_runtime.cases)
    {
        const auto& entry = m_rowView.cache[makeCacheKey(testCase.path)];
        for (size_t geoIx = 0u; geoIx < entry.gpu.size(); ++geoIx)
        {
            if (!entry.gpu[geoIx])
                failExit("Missing GPU geometry for %s.", testCase.path.string().c_str());
            allGeometries.push_back(entry.gpu[geoIx].get());
        }
    }

    if (mode == RowViewReloadMode::Full)
    {
        stats.addCount = allGeometries.size();
        const auto addStart = clock_t::now();
        if (!allGeometries.empty())
            if (!m_render.renderer->addGeometries({ allGeometries.data(),allGeometries.size() }))
                failExit("Failed to add geometries to renderer.");
        stats.addGeoMs = toMs(clock_t::now() - addStart);
    }
    else
    {
        const size_t addCount = (existingCount < totalGeometryCount) ? (totalGeometryCount - existingCount) : 0u;
        stats.addCount = addCount;
        if (addCount > 0u)
        {
            const auto addStart = clock_t::now();
            if (!m_render.renderer->addGeometries({ allGeometries.data() + existingCount,addCount }))
                failExit("Failed to add geometries to renderer.");
            stats.addGeoMs = toMs(clock_t::now() - addStart);
        }
    }

    using aabb_t = hlsl::shapes::AABB<3, double>;
    auto printAABB = [&](const aabb_t& aabb, const char* extraMsg = "")->void
        {
            m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)", ILogger::ELL_INFO, extraMsg, aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
        };
    const auto layoutStart = clock_t::now();
    core::vector<hlsl::float32_t3x4> worldTforms;
    worldTforms.resize(geometries.size());
    auto bound = aabb_t::create();
    core::vector<aabb_t> tileAABBs;
    tileAABBs.reserve(layoutGroups.size());
    for (const auto& group : layoutGroups)
        tileAABBs.push_back(group.layoutAABB);
    const auto layout = buildDisplayLayout(tileAABBs, true);
#ifdef NBL_BUILD_DEBUG_DRAW
    struct AggregateDebugAABB
    {
        aabb_t aabb = aabb_t::create();
        hlsl::float32_t3x4 world = makeIdentityWorld();
    };
    core::vector<AggregateDebugAABB> sceneDebugAABBs;
    sceneDebugAABBs.reserve(layoutGroups.size());
#endif
    for (uint32_t groupIx = 0u; groupIx < layoutGroups.size(); ++groupIx)
    {
        const auto& group = layoutGroups[groupIx];
        const auto& tileWorld = layout.worldTransforms[groupIx];
#ifdef NBL_BUILD_DEBUG_DRAW
        if (group.addAggregateDebugAABB)
            sceneDebugAABBs.push_back({.aabb = group.layoutAABB,.world = tileWorld});
#endif
        for (uint32_t localIx = 0u; localIx < group.geometryCount; ++localIx)
        {
            const uint32_t geometryIx = static_cast<uint32_t>(group.firstGeometry + localIx);
            worldTforms[geometryIx] = group.preserveInternalTransforms ?
                hlsl::math::linalg::promoted_mul(tileWorld, sourceWorlds[geometryIx]) :
                tileWorld;
            bound = hlsl::shapes::util::union_(
                nbl::hlsl::math::linalg::pseudo_mul(hlsl::float64_t3x4(worldTforms[geometryIx]), aabbs[geometryIx]),
                bound);
        }
    }
    stats.layoutMs = toMs(clock_t::now() - layoutStart);

    const auto instanceStart = clock_t::now();
#ifdef NBL_BUILD_DEBUG_DRAW
    m_aabbInstances.clear();
    m_aabbInstances.reserve(geometries.size() + sceneDebugAABBs.size());
    if (m_drawBBMode == DBBM_OBB)
        m_obbInstances.resize(geometries.size());
#endif
    m_render.renderer->m_instances.clear();

    for (uint32_t i = 0; i < geometries.size(); i++)
    {
        const auto& cpuGeom = geometries[i].get();
        const auto aabb = aabbs[i];
        printAABB(aabb, "Geometry");

        const auto promotedWorld = hlsl::float64_t3x4(worldTforms[i]);
        const auto transformed = nbl::hlsl::math::linalg::pseudo_mul(promotedWorld, aabb);
        printAABB(transformed, "Transformed");

#ifdef NBL_BUILD_DEBUG_DRAW
        m_aabbInstances.push_back(makeAABBInstance(aabb, worldTforms[i], hlsl::float32_t4(1, 1, 1, 1)));

        if (m_drawBBMode == DBBM_OBB)
            m_obbInstances[i] = makeOBBInstance(cpuGeom, worldTforms[i], hlsl::float32_t4(0, 0, 1, 1));
#endif
    }
#ifdef NBL_BUILD_DEBUG_DRAW
    for (const auto& sceneAABB : sceneDebugAABBs)
        m_aabbInstances.push_back(makeAABBInstance(sceneAABB.aabb, sceneAABB.world, hlsl::float32_t4(1, 0.65f, 0, 1)));
#endif

    printAABB(bound, "Total");
    for (uint32_t i = 0; i < worldTforms.size(); i++)
    {
        m_render.renderer->m_instances.push_back({
            .world = worldTforms[i],
            .packedGeo = &m_render.renderer->getGeometry(i)
            });
    }
    stats.instanceMs = toMs(clock_t::now() - instanceStart);

    const auto cameraStart = clock_t::now();
    setupCameraFromAABB(bound);
    stats.cameraMs = toMs(clock_t::now() - cameraStart);

    m_modelPath = "Row view (all meshes)";
    m_output.rowViewScreenshotPath = m_output.screenshotPrefixPath / "meshloaders_row_view.png";
    m_runtime.rowViewScreenshotCaptured = false;
    stats.totalMs = toMs(clock_t::now() - totalStart);
    logRowViewPerf(stats);
    return true;
}

bool MeshLoadersApp::writeAssetRoot(smart_refctd_ptr<const IAsset> asset, const std::string& savePath)
{
    using clock_t = std::chrono::high_resolution_clock;
    const auto writeOuterStart = clock_t::now();
    if (!asset)
        return false;

    IAsset* assetPtr = const_cast<IAsset*>(asset.get());
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
    auto validBound = bound;
    if (!isValidAABB(validBound))
    {
        m_logger->log("Total AABB invalid; using fallback unit AABB for camera setup.", ILogger::ELL_WARNING);
        validBound = nbl::examples::geometry::fallbackUnitAABB();
    }
    const auto extent = validBound.getExtent();
    const auto aspectRatio = double(m_window->getWidth()) / double(m_window->getHeight());
    const double fovY = 1.2;
    const double fovX = 2.0 * std::atan(std::tan(fovY * 0.5) * aspectRatio);
    const auto center = (validBound.minVx + validBound.maxVx) * 0.5;
    const auto halfExtent = extent * 0.5;
    const double halfX = std::max(halfExtent.x, 0.001);
    const double halfY = std::max(halfExtent.y, 0.001);
    const double halfZ = std::max(halfExtent.z, 0.001);
    const double safeRadius = std::max({ halfX, halfY, halfZ });

    const hlsl::float64_t3 dir(0.0, 0.0, -1.0);
    const double planeHalfX = halfX;
    const double planeHalfY = halfY;
    const double depthHalf = halfZ;
    const double distY = planeHalfY / std::tan(fovY * 0.5);
    const double distX = planeHalfX / std::tan(fovX * 0.5);
    const double framingMargin = std::max(0.1, safeRadius * 0.35);
    const double dist = std::max(distX, distY) + depthHalf + framingMargin;
    const double eyeHeightOffset = std::max(halfY * 0.2, 0.05);
    const auto eyeCenter = center + hlsl::float64_t3(0.0, eyeHeightOffset, 0.0);
    const auto pos = eyeCenter + dir * dist;

    const double tightNear = std::max(0.0, dist - depthHalf - framingMargin);
    const double tightFar = dist + depthHalf + framingMargin;
    const double nearByTight = tightNear * 0.01;
    const double nearByRadius = safeRadius * 0.002;
    const double nearPlane = std::max(0.001, std::min({ nearByTight, nearByRadius, 1.0 }));
    const double farPlane = std::max({ tightFar * 16.0, nearPlane + safeRadius * 24.0 + 10.0, dist + safeRadius * 24.0 });

    const auto projection = nbl::hlsl::math::thin_lens::rhPerspectiveFovMatrix<nbl::hlsl::float32_t>(
        static_cast<float>(fovY),
        static_cast<float>(aspectRatio),
        static_cast<float>(nearPlane),
        static_cast<float>(farPlane));
    camera.setProjectionMatrix(projection);
    const double moveSpeed = std::clamp(safeRadius * 0.015, 0.2, 40.0);
    camera.setMoveSpeed(static_cast<float>(moveSpeed));
    camera.setPosition(vectorSIMDf(pos.x, pos.y, pos.z));
    camera.setTarget(vectorSIMDf(eyeCenter.x, eyeCenter.y, eyeCenter.z));
}

void MeshLoadersApp::storeCameraState()
{
    const auto position = camera.getPosition();
    const auto target = camera.getTarget();
    m_referenceCamera = CameraState{
        hlsl::float32_t3(position.x, position.y, position.z),
        hlsl::float32_t3(target.x, target.y, target.z),
        camera.getProjectionMatrix(),
        camera.getMoveSpeed()
    };
}

void MeshLoadersApp::applyCameraState(const CameraState& state)
{
    camera.setProjectionMatrix(state.projection);
    camera.setPosition(vectorSIMDf(state.position.x, state.position.y, state.position.z));
    camera.setTarget(vectorSIMDf(state.target.x, state.target.y, state.target.z));
    camera.setMoveSpeed(state.moveSpeed);
}

bool MeshLoadersApp::isValidAABB(const hlsl::shapes::AABB<3, double>& aabb)
{
    return nbl::examples::geometry::isValidAABB(aabb);
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
        if (!isValidAABB(aabb))
            aabb = nbl::examples::geometry::computeFiniteUsedPositionAABB(geometry);
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
    if ((m_runtime.mode == RunMode::CI || isRowViewActive()) && !m_loaderPerfLogger)
        params.logger = nullptr;
    params.cacheFlags = IAssetLoader::ECF_DUPLICATE_TOP_LEVEL;
    params.ioPolicy.runtimeTuning.mode = m_runtimeTuningMode;
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



