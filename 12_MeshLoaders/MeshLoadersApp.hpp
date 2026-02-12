#ifndef _NBL_EXAMPLES_12_MESHLOADERS_APP_H_INCLUDED_
#define _NBL_EXAMPLES_12_MESHLOADERS_APP_H_INCLUDED_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#ifdef NBL_BUILD_DEBUG_DRAW
#include "nbl/ext/DebugDraw/CDrawAABB.h"
#endif

class MeshLoadersApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
    using device_base_t = MonoWindowApplication;
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
        smart_refctd_ptr<const ICPUPolygonGeometry> cpu;
        video::asset_cached_t<asset::ICPUPolygonGeometry> gpu;
        hlsl::shapes::AABB<3, double> aabb = hlsl::shapes::AABB<3, double>::create();
        bool hasAabb = false;
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
        core::vectorSIMDf position;
        core::vectorSIMDf target;
        nbl::hlsl::float32_t4x4 projection;
        float moveSpeed = 1.0f;
    };

    struct AssetLoadCallResult
    {
        asset::SAssetBundle bundle = {};
        double getAssetMs = 0.0;
        uintmax_t inputSize = 0u;
    };

public:
    MeshLoadersApp(const path& localInputCWD, const path& localOutputCWD, const path& sharedInputCWD, const path& sharedOutputCWD);

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override;
    IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override;
    bool onAppTerminated() override;
    bool keepRunning() override;

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

    static std::string sanitizeCaseNameForFilename(std::string name);
    system::path getGeometryHashReferencePath(const std::string& caseName) const;
    static std::string geometryHashToHex(const core::blake3_hash_t& hash);
    static bool tryParseNibble(char c, uint8_t& out);
    static bool tryParseGeometryHashHex(std::string hex, core::blake3_hash_t& outHash);
    bool readGeometryHashReference(const system::path& refPath, core::blake3_hash_t& outHash) const;
    bool writeGeometryHashReference(const system::path& refPath, const core::blake3_hash_t& hash) const;

    bool startCase(size_t index);
    bool advanceToNextCase();
    void reloadInteractive();
    bool addRowViewCase();
    bool addRowViewCaseFromPath(const system::path& picked);
    bool reloadFromTestList();

    bool loadModel(const system::path& modelPath, bool updateCamera, bool storeCamera);
    bool loadRowView(RowViewReloadMode mode);
    bool writeGeometry(smart_refctd_ptr<const ICPUPolygonGeometry> geometry, const std::string& savePath);

    void setupCameraFromAABB(const hlsl::shapes::AABB<3, double>& bound);
    static hlsl::shapes::AABB<3, double> translateAABB(const hlsl::shapes::AABB<3, double>& aabb, const hlsl::float64_t3& translation);
    static hlsl::shapes::AABB<3, double> scaleAABB(const hlsl::shapes::AABB<3, double>& aabb, double scale);

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

    core::blake3_hash_t hashGeometry(const ICPUPolygonGeometry* geo);
    bool validateWrittenAsset(const system::path& path);
    bool captureScreenshot(const system::path& path, core::smart_refctd_ptr<asset::ICPUImageView>& outImage);
    bool appendGeometriesFromBundle(const asset::SAssetBundle& bundle, core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>>& out) const;
    bool compareImages(const asset::ICPUImageView* a, const asset::ICPUImageView* b, uint64_t& diffCount, uint8_t& maxDiff);

    void advanceCase();

    constexpr static inline uint32_t MaxFramesInFlight = 3u;
    constexpr static inline uint32_t CiFramesBeforeCapture = 10u;
    constexpr static inline uint32_t NonCiFramesPerCase = 120u;
    constexpr static inline uint32_t RowViewFramesBeforeCapture = 10u;
    constexpr static inline uint64_t MaxImageDiffBytes = 16u;
    constexpr static inline uint8_t MaxImageDiffValue = 1u;

    smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
    smart_refctd_ptr<ISemaphore> m_semaphore;
    uint64_t m_realFrameIx = 0;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;

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

    bool m_nonInteractiveTest = false;
    bool m_rowViewEnabled = true;
    bool m_rowViewScreenshotCaptured = false;
    bool m_fileDialogOpen = false;

    bool m_saveGeom = true;
    std::optional<const std::string> m_specifiedGeomSavePath;
    nbl::system::path m_saveGeomPrefixPath;
    nbl::system::path m_screenshotPrefixPath;
    nbl::system::path m_rowViewScreenshotPath;
    nbl::system::path m_testListPath;
    nbl::system::path m_geometryHashReferenceDir;
    nbl::system::path m_caseGeometryHashReferencePath;
    std::optional<nbl::system::path> m_loaderPerfLogPath;
    std::optional<nbl::system::path> m_rowAddPath;
    uint32_t m_rowDuplicateCount = 0u;
    smart_refctd_ptr<system::ILogger> m_assetLoadLogger;
    smart_refctd_ptr<system::ILogger> m_loaderPerfLogger;
    bool m_updateGeometryHashReferences = false;
    bool m_forceLoaderContentHashes = true;
    asset::SFileIOPolicy::SRuntimeTuning::Mode m_runtimeTuningMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Heuristic;

    RunMode m_runMode = RunMode::Batch;
    Phase m_phase = Phase::RenderOriginal;
    uint32_t m_phaseFrameCounter = 0u;
    size_t m_caseIndex = 0u;
    core::vector<TestCase> m_cases;
    std::unordered_map<std::string, uint32_t> m_caseNameCounts;
    std::unordered_map<std::string, CachedGeometryEntry> m_rowViewCache;
    bool m_shouldQuit = false;

    nbl::system::path m_writtenPath;
    nbl::system::path m_loadedScreenshotPath;
    nbl::system::path m_writtenScreenshotPath;

    core::smart_refctd_ptr<const ICPUPolygonGeometry> m_currentCpuGeom;
    core::blake3_hash_t m_referenceGeometryHash = {};
    bool m_hasReferenceGeometryHash = false;

    core::smart_refctd_ptr<asset::ICPUImageView> m_loadedScreenshot;
    core::smart_refctd_ptr<asset::ICPUImageView> m_writtenScreenshot;

    std::optional<CameraState> m_referenceCamera;
};

#endif
