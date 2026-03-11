#ifndef _NBL_EXAMPLES_12_MESHLOADERS_APP_H_INCLUDED_
#define _NBL_EXAMPLES_12_MESHLOADERS_APP_H_INCLUDED_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "nbl/examples/common/MonoWindowApplication.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
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
        CaptureOriginalPending,
        WrittenAssetPending,
        RenderWritten,
        CaptureWrittenPending
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

    struct PendingScreenshotCapture
    {
        nbl::system::path path;
        core::smart_refctd_ptr<const IGPUImageView> sourceView;
        core::smart_refctd_ptr<IGPUCommandBuffer> commandBuffer;
        core::smart_refctd_ptr<IGPUBuffer> texelBuffer;
        core::smart_refctd_ptr<ISemaphore> completionSemaphore;
        asset::IImage::SCreationParams imageParams = {};
        asset::IImage::SSubresourceRange subresourceRange = {};
        asset::E_FORMAT viewFormat = asset::EF_UNKNOWN;
        uint64_t completionValue = 0u;

        inline bool active() const
        {
            return static_cast<bool>(completionSemaphore);
        }
    };

public:
    struct LoadStageMetrics
    {
        double getAssetMs = 0.0;
        double extractMs = 0.0;
        double totalMs = 0.0;
        double nonLoaderMs = 0.0;
        uintmax_t inputSize = 0u;
        bool valid = false;
    };

    struct WriteStageMetrics
    {
        double openMs = 0.0;
        double writeMs = 0.0;
        double statMs = 0.0;
        double totalMs = 0.0;
        double nonWriterMs = 0.0;
        uintmax_t outputSize = 0u;
        bool usedMemoryTransport = false;
        bool usedDiskFallback = false;
        bool persistedDiskArtifact = false;
        bool valid = false;
    };

    struct CasePerformanceMetrics
    {
        std::string caseName;
        nbl::system::path inputPath;
        LoadStageMetrics originalLoad = {};
        WriteStageMetrics write = {};
        LoadStageMetrics writtenLoad = {};
    };

private:
    struct PerformanceOptions
    {
        std::optional<nbl::system::path> dumpDir;
        std::optional<nbl::system::path> referenceDir;
        std::optional<std::string> profileOverride;
        bool strict = false;
    };

    struct WrittenAssetRequest
    {
        core::smart_refctd_ptr<const IAsset> asset;
        nbl::system::path path;
        IAssetLoader::SAssetLoadParams loadParams = {};
        bool useMemoryTransport = false;
        bool allowDiskFallback = false;
        bool persistDiskArtifact = false;
    };

    struct WrittenAssetResult
    {
        bool success = false;
        std::string error;
        nbl::system::path path;
        std::string extension;
        double openMs = 0.0;
        double writeMs = 0.0;
        double statMs = 0.0;
        double totalWriteMs = 0.0;
        double nonWriterMs = 0.0;
        uintmax_t outputSize = 0u;
        bool usedMemoryTransport = false;
        bool usedDiskFallback = false;
        bool persistedDiskArtifact = false;
        AssetLoadCallResult loadResult = {};
    };

    struct BackgroundAssetWorker
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::thread thread;
        std::optional<WrittenAssetRequest> request;
        std::optional<WrittenAssetResult> result;
        bool busy = false;
        bool stop = false;
    };

    struct PreparedAssetLoad
    {
        size_t caseIndex = ~size_t(0u);
        bool success = false;
        std::string error;
        nbl::system::path path;
        AssetLoadCallResult loadResult = {};
    };

    struct BackgroundLoadWorker
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::thread thread;
        std::optional<size_t> requestCaseIndex;
        nbl::system::path requestPath;
        IAssetLoader::SAssetLoadParams requestParams = {};
        std::optional<PreparedAssetLoad> result;
        bool busy = false;
        bool stop = false;
    };

    struct PerformanceState
    {
        PerformanceOptions options = {};
        bool enabled = false;
        bool finalized = false;
        std::chrono::steady_clock::time_point runStart = {};
        size_t currentCaseIndex = ~size_t(0u);
        std::string profileId;
        std::string workloadId;
        nbl::system::path dumpPath = {};
        nbl::system::path referencePath = {};
        bool referenceMatched = false;
        core::vector<std::string> comparisonFailures = {};
        core::vector<CasePerformanceMetrics> completedCases = {};
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
        PendingScreenshotCapture pendingScreenshot;
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
    asset::writer_flags_t getWriterFlagsForPath(const IAsset* asset, const system::path& path) const;
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
    bool loadModel(const system::path& modelPath, bool updateCamera, bool storeCamera, LoadStageMetrics* perfMetrics);
    bool loadPreparedModel(const system::path& modelPath, AssetLoadCallResult&& loadResult, bool updateCamera, bool storeCamera);
    bool loadPreparedModel(const system::path& modelPath, AssetLoadCallResult&& loadResult, bool updateCamera, bool storeCamera, LoadStageMetrics* perfMetrics);
    bool loadRowView(RowViewReloadMode mode);
    bool writeAssetRoot(smart_refctd_ptr<const IAsset> asset, const std::string& savePath);
    bool writeAssetRoot(smart_refctd_ptr<const IAsset> asset, const std::string& savePath, WriteStageMetrics* perfMetrics);

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
    static bool validateWrittenBundle(const asset::SAssetBundle& bundle);
    bool requestScreenshotCapture(const system::path& path);
    bool finalizeScreenshotCapture(core::smart_refctd_ptr<asset::ICPUImageView>& outImage, bool& ready, bool waitForCompletion=false);
    bool startWrittenAssetWork(smart_refctd_ptr<const IAsset> asset, const system::path& path);
    bool finalizeWrittenAssetWork(WrittenAssetResult& result, bool& ready, bool waitForCompletion=false);
    void logWrittenAssetWork(const WrittenAssetResult& result) const;
    bool startBackgroundAssetWorker();
    void stopBackgroundAssetWorker();
    void backgroundAssetWorkerMain();
    bool startBackgroundLoadWorker();
    void stopBackgroundLoadWorker();
    void backgroundLoadWorkerMain();
    bool startPreparedAssetLoad(size_t caseIndex, const system::path& path);
    bool finalizePreparedAssetLoad(PreparedAssetLoad& result, bool& ready, bool waitForCompletion=false);
    bool performanceEnabled() const;
    void beginPerformanceRun();
    void beginPerformanceCase(const TestCase& testCase);
    void recordOriginalLoadMetrics(const LoadStageMetrics& metrics);
    void recordWrittenLoadMetrics(const LoadStageMetrics& metrics);
    void recordWriteMetrics(const WriteStageMetrics& metrics);
    void recordWriteMetrics(const WrittenAssetResult& result);
    void endPerformanceCase();
    void finalizePerformanceRun();
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
    BackgroundAssetWorker m_backgroundAssetWorker;
    BackgroundLoadWorker m_backgroundLoadWorker;
    PerformanceState m_perf;

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
