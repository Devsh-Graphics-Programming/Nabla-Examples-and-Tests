#ifndef _NBL_EXAMPLES_12_MESHLOADERS_APP_H_INCLUDED_
#define _NBL_EXAMPLES_12_MESHLOADERS_APP_H_INCLUDED_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "nbl/examples/common/MonoWindowApplication.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#ifdef NBL_BUILD_DEBUG_DRAW
#include "nbl/ext/DebugDraw/CDrawAABB.h"
#endif

class MeshLoadersWindowedApplication : public virtual nbl::examples::MonoWindowApplication
{
    using base_t = nbl::examples::MonoWindowApplication;

public:
    template<typename... Args>
    MeshLoadersWindowedApplication(const hlsl::uint16_t2 initialResolution, const asset::E_FORMAT depthFormat, Args&&... args)
        : base_t(initialResolution, depthFormat, std::forward<Args>(args)...) {}

protected:
    inline const char* getWindowCaption() const override
    {
        return "MeshLoaders";
    }
    inline void amendSwapchainCreateParams(video::ISwapchain::SCreationParams& swapchainParams) const override
    {
        swapchainParams.sharedParams.imageUsage |= IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT;
    }
};

class MeshLoadersApp final : public MeshLoadersWindowedApplication, public BuiltinResourcesApplication
{
    using device_base_t = MeshLoadersWindowedApplication;
    using asset_base_t = BuiltinResourcesApplication;

    enum DrawBoundingBoxMode
    {
        DBBM_NONE,
        DBBM_AABB,
        DBBM_OBB
    };

    enum class RunMode
    {
        Interactive,
        Batch,
        CI
    };

    enum class Phase
    {
        RenderOriginal,
        RenderWritten
    };

    enum class RowViewReloadMode
    {
        Full,
        Incremental
    };

    struct TestCase
    {
        std::string name;
        nbl::system::path path;
    };

    struct CachedGeometryEntry
    {
        core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> cpu;
        core::vector<video::asset_cached_t<asset::ICPUPolygonGeometry>> gpu;
        core::vector<hlsl::shapes::AABB<3, double>> aabbs;
        core::vector<hlsl::float32_t3x4> world;
        hlsl::shapes::AABB<3, double> tileAABB = hlsl::shapes::AABB<3, double>::create();
        bool layoutAsSingleTile = false;
    };

    struct RowViewPerfStats
    {
        double totalMs = 0.0;
        double clearMs = 0.0;
        double loadMs = 0.0;
        double extractMs = 0.0;
        double aabbMs = 0.0;
        double convertMs = 0.0;
        double addGeoMs = 0.0;
        double layoutMs = 0.0;
        double instanceMs = 0.0;
        double cameraMs = 0.0;
        size_t cases = 0u;
        size_t cpuHits = 0u;
        size_t cpuMisses = 0u;
        size_t gpuHits = 0u;
        size_t gpuMisses = 0u;
        size_t convertCount = 0u;
        size_t addCount = 0u;
        bool incremental = false;
    };

    struct CameraState
    {
        hlsl::float32_t3 position = hlsl::float32_t3(0.0f, 0.0f, 0.0f);
        hlsl::float32_t3 target = hlsl::float32_t3(0.0f, 0.0f, -1.0f);
        nbl::hlsl::float32_t4x4 projection;
        float moveSpeed = 1.0f;
    };

    struct AssetLoadCallResult
    {
        asset::SAssetBundle bundle = {};
        double getAssetMs = 0.0;
        uintmax_t inputSize = 0u;
    };

    struct RuntimeState
    {
        bool nonInteractiveTest = false;
        bool rowViewEnabled = true;
        bool forceRowViewForCurrentTestList = false;
        bool rowViewScreenshotCaptured = false;
        bool fileDialogOpen = false;

        RunMode mode = RunMode::Batch;
        Phase phase = Phase::RenderOriginal;
        uint32_t phaseFrameCounter = 0u;
        size_t caseIndex = 0u;
        core::vector<TestCase> cases;
        std::unordered_map<std::string, uint32_t> caseNameCounts;
        bool shouldQuit = false;
    };

    struct OutputState
    {
        bool saveGeom = true;
        std::optional<const std::string> specifiedGeomSavePath;
        nbl::system::path saveGeomPrefixPath;
        nbl::system::path screenshotPrefixPath;
        nbl::system::path rowViewScreenshotPath;
        nbl::system::path testListPath;
        std::optional<nbl::system::path> loaderPerfLogPath;
        std::optional<nbl::system::path> rowAddPath;
        uint32_t rowDuplicateCount = 0u;

        nbl::system::path writtenPath;
        nbl::system::path loadedScreenshotPath;
        nbl::system::path writtenScreenshotPath;
    };

    struct RenderState
    {
        smart_refctd_ptr<CSimpleDebugRenderer> renderer;
        smart_refctd_ptr<ISemaphore> semaphore;
        uint64_t realFrameIx = 0u;
        std::array<smart_refctd_ptr<IGPUCommandBuffer>, 3u> cmdBufs;

        core::smart_refctd_ptr<const IAsset> currentCpuAsset;
        core::smart_refctd_ptr<const ICPUPolygonGeometry> currentCpuGeom;
        core::smart_refctd_ptr<asset::ICPUImageView> loadedScreenshot;
        core::smart_refctd_ptr<asset::ICPUImageView> writtenScreenshot;
    };

    struct RowViewState
    {
        std::unordered_map<std::string, CachedGeometryEntry> cache;
    };

public:
    MeshLoadersApp(const path& localInputCWD, const path& localOutputCWD, const path& sharedInputCWD, const path& sharedOutputCWD);

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override;
    IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override;
    bool onAppTerminated() override;

protected:
    core::bitflag<system::ILogger::E_LOG_LEVEL> getLogLevelMask() override
    {
        return system::ILogger::DefaultLogMask() | system::ILogger::ELL_INFO;
    }

    const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override;

private:
    [[noreturn]] void failExit(const char* msg, ...);

    bool initTestCases();
    bool pickModelPath(system::path& outPath);
    bool loadTestList(const system::path& jsonPath);
    bool isRowViewActive() const;

    static std::string normalizeExtension(const system::path& path);
    bool isWriteExtensionSupported(const std::string& ext) const;
    system::path resolveSavePath(const system::path& modelPath) const;

    bool startCase(size_t index);
    bool advanceToNextCase();
    void reloadInteractive();
    bool addRowViewCase();
    bool addRowViewCaseFromPath(const system::path& picked);
    bool reloadFromTestList();
    void resetRowViewScene();

    bool loadModel(const system::path& modelPath, bool updateCamera, bool storeCamera);
    bool loadRowView(RowViewReloadMode mode);
    bool writeAssetRoot(smart_refctd_ptr<const IAsset> asset, const std::string& savePath);

    void setupCameraFromAABB(const hlsl::shapes::AABB<3, double>& bound);

    void storeCameraState();
    void applyCameraState(const CameraState& state);

    static bool isValidAABB(const hlsl::shapes::AABB<3, double>& aabb);
    hlsl::shapes::AABB<3, double> getGeometryAABB(const ICPUPolygonGeometry* geometry) const;

    system::ILogger* getAssetLoadLogger() const;
    IAssetLoader::SAssetLoadParams makeLoadParams() const;
    bool loadAssetCallFromPath(const system::path& modelPath, const IAssetLoader::SAssetLoadParams& params, AssetLoadCallResult& out);
    bool initLoaderPerfLogger(const system::path& logPath);

    std::string makeUniqueCaseName(const system::path& path);
    static double toMs(const std::chrono::high_resolution_clock::duration& d);
    std::string makeCacheKey(const system::path& path) const;

    void logRowViewPerf(const RowViewPerfStats& stats) const;
    void logRowViewAssetLoad(const system::path& path, double ms, bool cached) const;
    void logRowViewLoadTotal(double ms, size_t hits, size_t misses) const;

    bool validateWrittenAsset(const system::path& path);
    bool captureScreenshot(const system::path& path, core::smart_refctd_ptr<asset::ICPUImageView>& outImage);
    bool compareImages(
        const asset::ICPUImageView* a,
        const asset::ICPUImageView* b,
        uint64_t& diffCodeUnitCount,
        uint32_t& maxDiffCodeUnitValue);

    void advanceCase();
    bool shouldKeepRunning() const override;

    constexpr static inline uint32_t MaxFramesInFlight = 3u;
    constexpr static inline uint32_t CiFramesBeforeCapture = 10u;
    constexpr static inline uint32_t NonCiFramesPerCase = 120u;
    constexpr static inline uint32_t RowViewFramesBeforeCapture = 10u;
    constexpr static inline uint64_t MaxImageDiffCodeUnits = 16u;
    constexpr static inline uint32_t MaxImageDiffCodeUnitValue = 1u;

    RenderState m_render;
    RuntimeState m_runtime;
    OutputState m_output;
    RowViewState m_rowView;

    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    Camera camera = Camera(
        core::vectorSIMDf(0, 0, 0),
        core::vectorSIMDf(0, 0, -1),
        nbl::hlsl::math::linalg::diagonal<nbl::hlsl::float32_t4x4>(1.0f));

    std::string m_modelPath;
    std::string m_caseName;

    DrawBoundingBoxMode m_drawBBMode = DBBM_AABB;
#ifdef NBL_BUILD_DEBUG_DRAW
    smart_refctd_ptr<ext::debug_draw::DrawAABB> m_drawAABB;
    std::vector<ext::debug_draw::InstanceData> m_aabbInstances;
    std::vector<ext::debug_draw::InstanceData> m_obbInstances;
#endif

    smart_refctd_ptr<system::ILogger> m_assetLoadLogger;
    smart_refctd_ptr<system::ILogger> m_loaderPerfLogger;
    asset::SFileIOPolicy::SRuntimeTuning::Mode m_runtimeTuningMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Heuristic;
    bool m_forceLoaderContentHashes = true;
    bool m_updateGeometryHashReferences = false;

    std::optional<CameraState> m_referenceCamera;
};

#endif
