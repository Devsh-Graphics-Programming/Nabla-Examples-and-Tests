// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "argparse/argparse.hpp"
#include "portable-file-dialogs/portable-file-dialogs.h"
#include "nlohmann/json.hpp"
#include "App.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#ifdef NBL_BUILD_MITSUBA_LOADER
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#endif

#include "nbl/system/CFileLogger.h"

namespace
{

void setupMeshLoadersArgumentParser(argparse::ArgumentParser& parser)
{
    parser.add_argument("--savegeometry")
        .help("Save the mesh on exit or reload")
        .flag();

    parser.add_argument("--savepath")
        .nargs(1)
        .help("Specify the file to which the mesh will be saved");
    parser.add_argument("--ci")
        .help("Run in CI mode: load test list, write .ply, capture screenshots, compare data, and exit.")
        .flag();
    parser.add_argument("--interactive")
        .help("Use file dialog to select a single model.")
        .flag();
    parser.add_argument("--testlist")
        .nargs(1)
        .help("JSON file with test cases. Relative JSON path resolves against local input CWD. Relative case paths inside the JSON resolve against the JSON file directory.");
    parser.add_argument("--row-add")
        .nargs(1)
        .help("Add a model path to row view on startup without using a dialog.");
    parser.add_argument("--row-duplicate")
        .nargs(1)
        .help("Duplicate the last case N times on startup.");
    parser.add_argument("--loader-perf-log")
        .nargs(1)
        .help("Write loader diagnostics to a file instead of stdout.");
    parser.add_argument("--perf-dump-dir")
        .nargs(1)
        .help("Write structured performance run artifacts to this directory.");
    parser.add_argument("--perf-ref-dir")
        .nargs(1)
        .help("Lookup directory for structured performance reference artifacts.");
    parser.add_argument("--perf-strict")
        .help("Fail the run if a matching structured performance reference exists and comparison exceeds thresholds.")
        .flag();
    parser.add_argument("--perf-profile-override")
        .nargs(1)
        .help("Override the automatically generated performance profile id.");
    parser.add_argument("--perf-update-reference")
        .help("Write the current structured performance run to the matching reference path.")
        .flag();
    parser.add_argument("--loader-content-hashes")
        .help("Keep loader content hashes enabled. This is already the default for this example.")
        .flag();
    parser.add_argument("--runtime-tuning")
        .nargs(1)
        .help("Runtime tuning mode for loaders: sequential|heuristic|hybrid. Default: heuristic.");
}

std::optional<uint32_t> parseUInt32Argument(const std::string_view value)
{
    uint32_t parsed = 0u;
    const auto parseResult = std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
    if (parseResult.ec != std::errc() || parseResult.ptr != value.data() + value.size())
        return std::nullopt;
    return parsed;
}

bool parseRuntimeTuningMode(const std::string_view modeRaw, asset::SFileIOPolicy::SRuntimeTuning::Mode& outMode)
{
    std::string mode(modeRaw);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (mode == "sequential" || mode == "none")
    {
        outMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Sequential;
        return true;
    }
    if (mode == "heuristic")
    {
        outMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Heuristic;
        return true;
    }
    if (mode == "hybrid")
    {
        outMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Hybrid;
        return true;
    }
    return false;
}

struct ParsedCommandLineOptions
{
    bool saveGeom = true;
    bool interactive = false;
    bool ci = false;
    bool forceRowViewForCurrentTestList = false;
    bool forceLoaderContentHashes = true;
    system::path saveGeomPrefixPath;
    system::path screenshotPrefixPath;
    system::path testListPath;
    std::optional<std::string> specifiedGeomSavePath;
    std::optional<system::path> loaderPerfLogPath;
    std::optional<system::path> perfDumpDir;
    std::optional<system::path> perfReferenceDir;
    std::optional<std::string> perfProfileOverride;
    bool perfStrict = false;
    bool perfUpdateReference = false;
    std::optional<system::path> rowAddPath;
    uint32_t rowDuplicateCount = 0u;
    asset::SFileIOPolicy::SRuntimeTuning::Mode runtimeTuningMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Heuristic;
};

struct CaseArtifacts
{
    std::string caseName;
    system::path writtenPath;
    system::path loadedScreenshotPath;
    system::path writtenScreenshotPath;
};

system::path resolveRuntimeCWD(const system::path& preferred)
{
    if (preferred.empty() || preferred == path("/") || preferred == path("\\"))
        return path(std::filesystem::current_path());
    return preferred;
}

system::path makeShortRuntimePath(const system::path& inputPath)
{
    if (inputPath.empty())
        return inputPath;

    std::filesystem::path pathValue(inputPath.string());
    pathValue = pathValue.lexically_normal();
    if (!pathValue.is_absolute())
        return system::path(pathValue.generic_string());

    std::error_code ec;
    const auto cwd = std::filesystem::current_path(ec);
    if (!ec)
    {
        const auto relativePath = std::filesystem::relative(pathValue, cwd, ec);
        if (!ec && !relativePath.empty() && !relativePath.is_absolute() && relativePath.generic_string().size() < pathValue.generic_string().size())
            return system::path(relativePath.lexically_normal().generic_string());
    }
    return system::path(pathValue.generic_string());
}

system::path resolveDefaultTestListPath(const system::path& effectiveInputCWD, const core::vector<std::string>& argv)
{
    const auto tryExisting = [](std::filesystem::path candidate) -> std::optional<system::path>
    {
        std::error_code ec;
        candidate = candidate.lexically_normal();
        if (!candidate.empty() && std::filesystem::exists(candidate, ec) && !ec)
            return system::path(candidate.generic_string());
        return std::nullopt;
    };

    if (auto resolved = tryExisting(effectiveInputCWD / "inputs.json"); resolved.has_value())
        return *resolved;

    if (!argv.empty() && !argv[0].empty())
    {
        std::error_code ec;
        auto exePath = std::filesystem::absolute(std::filesystem::path(argv[0]), ec);
        if (!ec)
        {
            if (auto resolved = tryExisting(exePath.parent_path() / ".." / "inputs.json"); resolved.has_value())
                return *resolved;
            if (auto resolved = tryExisting(exePath.parent_path() / "inputs.json"); resolved.has_value())
                return *resolved;
        }
    }

    return (effectiveInputCWD / "inputs.json").lexically_normal();
}

std::string makeCaptionModelPath(const std::string& modelPath, const core::vector<std::string>& argv)
{
    if (modelPath.empty())
        return {};

    std::error_code ec;
    if (modelPath.find('/') == std::string::npos && modelPath.find('\\') == std::string::npos)
    {
        if (!std::filesystem::exists(std::filesystem::path(modelPath), ec))
        {
            ec.clear();
            return modelPath;
        }
        ec.clear();
    }
    std::filesystem::path targetPath(modelPath);
    targetPath = targetPath.lexically_normal();
    const auto canonicalTarget = std::filesystem::weakly_canonical(targetPath, ec);
    if (!ec)
        targetPath = canonicalTarget;
    else
        ec.clear();

    if (!targetPath.is_absolute())
    {
        const auto absoluteTarget = std::filesystem::absolute(targetPath, ec);
        if (!ec)
            targetPath = absoluteTarget.lexically_normal();
        else
            ec.clear();
    }
    if (!targetPath.is_absolute())
        return targetPath.generic_string();

    auto relativeFromBase = [&](const std::filesystem::path& basePath) -> std::string
    {
        if (basePath.empty())
            return {};
        auto canonicalBase = std::filesystem::weakly_canonical(basePath, ec);
        if (ec)
        {
            ec.clear();
            canonicalBase = std::filesystem::absolute(basePath, ec);
        }
        if (ec)
        {
            ec.clear();
            return {};
        }
        const auto relativePath = std::filesystem::relative(targetPath, canonicalBase, ec);
        if (ec || relativePath.empty() || relativePath.is_absolute())
        {
            ec.clear();
            return {};
        }
        return relativePath.lexically_normal().generic_string();
    };

    std::string bestRelativePath;
    if (!argv.empty() && !argv[0].empty())
    {
        const auto exePath = std::filesystem::absolute(std::filesystem::path(argv[0]), ec);
        if (!ec)
        {
            const auto relativeToExe = relativeFromBase(exePath.parent_path());
            if (!relativeToExe.empty())
                bestRelativePath = relativeToExe;
        }
        else
            ec.clear();
    }

    const auto cwd = std::filesystem::current_path(ec);
    if (!ec)
    {
        const auto relativeToCwd = relativeFromBase(cwd);
        if (!relativeToCwd.empty() && (bestRelativePath.empty() || relativeToCwd.size() < bestRelativePath.size()))
            bestRelativePath = relativeToCwd;
    }
    else
        ec.clear();

    if (!bestRelativePath.empty())
        return bestRelativePath;
    return targetPath.generic_string();
}

template<typename ResolveSavePathFn>
CaseArtifacts makeCaseArtifacts(
    const std::string& preferredName,
    const system::path& casePath,
    const system::path& screenshotPrefixPath,
    ResolveSavePathFn&& resolveSavePath)
{
    const auto caseName = preferredName.empty() ? casePath.stem().string() : preferredName;
    return {
        .caseName = caseName,
        .writtenPath = resolveSavePath(casePath),
        .loadedScreenshotPath = screenshotPrefixPath / ("meshloaders_" + caseName + "_loaded.png"),
        .writtenScreenshotPath = screenshotPrefixPath / ("meshloaders_" + caseName + "_written.png")
    };
}

template<typename AddCaseFn>
bool appendRowViewDuplicates(const uint32_t duplicateCount, const system::path& lastPath, AddCaseFn&& addCase)
{
    for (uint32_t i = 0u; i < duplicateCount; ++i)
        if (!addCase(lastPath))
            return false;
    return true;
}

bool parseMeshLoadersCommandLine(
    const core::vector<std::string>& argv,
    const system::path& effectiveInputCWD,
    const system::path& effectiveOutputCWD,
    const system::path& defaultBenchmarkTestListPath,
    ParsedCommandLineOptions& out,
    std::string& error)
{
    out.saveGeomPrefixPath = effectiveOutputCWD / "saved";
    out.screenshotPrefixPath = effectiveOutputCWD / "screenshots";
    out.testListPath = resolveDefaultTestListPath(effectiveInputCWD, argv);

    argparse::ArgumentParser parser("12_meshloaders");
    setupMeshLoadersArgumentParser(parser);

    try
    {
        parser.parse_args({ argv.data(), argv.data() + argv.size() });
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }

    if (parser["--savegeometry"] == true)
        out.saveGeom = true;
    if (parser["--interactive"] == true)
        out.interactive = true;
    if (parser["--ci"] == true)
        out.ci = true;
    const bool hasExplicitTestListArg = parser.present("--testlist").has_value();

    if (parser.present("--savepath"))
    {
        auto tmp = path(parser.get<std::string>("--savepath"));
        if (tmp.empty() || !tmp.has_filename())
        {
            error = "Invalid path has been specified in --savepath argument";
            return false;
        }
        if (!std::filesystem::exists(tmp.parent_path()))
        {
            error = "Path specified in --savepath argument doesn't exist";
            return false;
        }
        out.specifiedGeomSavePath.emplace(std::move(tmp.generic_string()));
    }

    if (hasExplicitTestListArg)
    {
        auto tmp = path(parser.get<std::string>("--testlist"));
        if (tmp.empty())
        {
            error = "Invalid path has been specified in --testlist argument";
            return false;
        }
        if (tmp.is_relative())
            tmp = effectiveInputCWD / tmp;
        out.testListPath = tmp;
    }
    else if (!out.interactive && !out.ci && !defaultBenchmarkTestListPath.empty())
    {
        std::error_code benchmarkPathEc;
        if (std::filesystem::exists(defaultBenchmarkTestListPath, benchmarkPathEc) && !benchmarkPathEc)
        {
            out.testListPath = defaultBenchmarkTestListPath;
            out.forceRowViewForCurrentTestList = true;
        }
    }

    if (parser.present("--row-add"))
    {
        auto tmp = path(parser.get<std::string>("--row-add"));
        if (tmp.is_relative())
            tmp = effectiveInputCWD / tmp;
        out.rowAddPath = tmp;
    }
    if (parser.present("--row-duplicate"))
    {
        const auto parsedCount = parseUInt32Argument(parser.get<std::string>("--row-duplicate"));
        if (!parsedCount.has_value())
        {
            error = "Invalid --row-duplicate value.";
            return false;
        }
        out.rowDuplicateCount = *parsedCount;
    }
    if (parser.present("--loader-perf-log"))
    {
        auto tmp = path(parser.get<std::string>("--loader-perf-log"));
        if (tmp.empty())
        {
            error = "Invalid --loader-perf-log value.";
            return false;
        }
        if (tmp.is_relative())
            tmp = effectiveOutputCWD / tmp;
        out.loaderPerfLogPath = tmp;
    }
    if (parser.present("--perf-dump-dir"))
    {
        auto tmp = path(parser.get<std::string>("--perf-dump-dir"));
        if (tmp.empty())
        {
            error = "Invalid --perf-dump-dir value.";
            return false;
        }
        if (tmp.is_relative())
            tmp = effectiveOutputCWD / tmp;
        out.perfDumpDir = makeShortRuntimePath(tmp);
    }
    if (parser.present("--perf-ref-dir"))
    {
        auto tmp = path(parser.get<std::string>("--perf-ref-dir"));
        if (tmp.empty())
        {
            error = "Invalid --perf-ref-dir value.";
            return false;
        }
        if (tmp.is_relative())
            tmp = effectiveOutputCWD / tmp;
        out.perfReferenceDir = makeShortRuntimePath(tmp);
    }
    if (parser["--perf-strict"] == true)
        out.perfStrict = true;
    if (parser.present("--perf-profile-override"))
    {
        const auto value = parser.get<std::string>("--perf-profile-override");
        if (value.empty())
        {
            error = "Invalid --perf-profile-override value.";
            return false;
        }
        out.perfProfileOverride = value;
    }
    if (parser["--perf-update-reference"] == true)
        out.perfUpdateReference = true;
    if (parser["--loader-content-hashes"] == true)
        out.forceLoaderContentHashes = true;
    if (parser.present("--runtime-tuning"))
    {
        if (!parseRuntimeTuningMode(parser.get<std::string>("--runtime-tuning"), out.runtimeTuningMode))
        {
            error = "Invalid --runtime-tuning value. Expected: sequential|heuristic|hybrid.";
            return false;
        }
    }
    if (out.perfStrict && out.perfUpdateReference)
    {
        error = "Use either --perf-strict or --perf-update-reference, not both.";
        return false;
    }
    if (out.perfUpdateReference && !out.perfReferenceDir.has_value())
    {
        error = "--perf-update-reference requires --perf-ref-dir.";
        return false;
    }

    return true;
}
}

MeshLoadersApp::MeshLoadersApp(
    const path& localInputCWD,
    const path& localOutputCWD,
    const path& sharedInputCWD,
    const path& sharedOutputCWD)
    : nbl::examples::MonoWindowApplication({1280, 720}, EF_D32_SFLOAT, localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD)
    , IApplicationFramework(localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD)
    , device_base_t({1280, 720}, EF_D32_SFLOAT, localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD)
{
}

bool MeshLoadersApp::onAppInitialized(smart_refctd_ptr<ISystem>&& system)
{
    if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
        return false;
#ifdef NBL_BUILD_MITSUBA_LOADER
    m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CSerializedLoader>());
#endif
    if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
        return false;

    const path effectiveInputCWD = ::resolveRuntimeCWD(localInputCWD);
    const path effectiveOutputCWD = ::resolveRuntimeCWD(localOutputCWD);
#if defined(NBL_MESHLOADERS_DEFAULT_BENCHMARK_TESTLIST_PATH)
    const path defaultBenchmarkTestListPath = path(NBL_MESHLOADERS_DEFAULT_BENCHMARK_TESTLIST_PATH);
#else
    const path defaultBenchmarkTestListPath;
#endif
    ParsedCommandLineOptions parsed = {};
    std::string parseError;
    if (!parseMeshLoadersCommandLine(argv, effectiveInputCWD, effectiveOutputCWD, defaultBenchmarkTestListPath, parsed, parseError))
        return logFail(parseError.c_str());
    auto applyParsedCommandLineOptions = [this](ParsedCommandLineOptions&& options) -> void
    {
        m_runtime.mode = options.interactive ? RunMode::Interactive : (options.ci ? RunMode::CI : RunMode::Batch);
        m_runtime.forceRowViewForCurrentTestList = options.forceRowViewForCurrentTestList;
        m_output.saveGeom = options.saveGeom;
        m_output.saveGeomPrefixPath = std::move(options.saveGeomPrefixPath);
        m_output.screenshotPrefixPath = std::move(options.screenshotPrefixPath);
        m_output.testListPath = std::move(options.testListPath);
        if (options.specifiedGeomSavePath)
            m_output.specifiedGeomSavePath.emplace(std::move(*options.specifiedGeomSavePath));
        m_output.loaderPerfLogPath = std::move(options.loaderPerfLogPath);
        m_output.rowAddPath = std::move(options.rowAddPath);
        m_output.rowDuplicateCount = options.rowDuplicateCount;
        m_perf.options.dumpDir = std::move(options.perfDumpDir);
        m_perf.options.referenceDir = std::move(options.perfReferenceDir);
        m_perf.options.profileOverride = std::move(options.perfProfileOverride);
        m_perf.options.strict = options.perfStrict;
        m_perf.options.updateReference = options.perfUpdateReference;
        m_perf.enabled = m_perf.options.dumpDir.has_value() || m_perf.options.referenceDir.has_value() || m_perf.options.strict || m_perf.options.updateReference;
        m_forceLoaderContentHashes = options.forceLoaderContentHashes;
        m_runtimeTuningMode = options.runtimeTuningMode;
    };
    applyParsedCommandLineOptions(std::move(parsed));

    if (m_runtime.forceRowViewForCurrentTestList)
        m_logger->log("Using benchmark test list for default batch startup: %s", ILogger::ELL_INFO, m_output.testListPath.string().c_str());

    if (m_output.saveGeom)
        std::filesystem::create_directories(m_output.saveGeomPrefixPath);
    std::filesystem::create_directories(m_output.screenshotPrefixPath);
    m_assetLoadLogger = m_logger;
    if (m_output.loaderPerfLogPath)
    {
        if (!initLoaderPerfLogger(*m_output.loaderPerfLogPath))
            return false;
        m_logger->log("Loader diagnostics will be written to %s", ILogger::ELL_INFO, m_output.loaderPerfLogPath->string().c_str());
    }

    m_render.semaphore = m_device->createSemaphore(m_render.realFrameIx);
    if (!m_render.semaphore)
        return logFail("Failed to Create a Semaphore!");

    auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
    for (auto i = 0u; i < MaxFramesInFlight; i++)
    {
        if (!pool)
            return logFail("Couldn't create Command Pool!");
        if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_render.cmdBufs.data() + i,1 }))
            return logFail("Couldn't create Command Buffer!");
    }

    auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
    m_render.renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), scRes->getRenderpass(), 0, {});
    if (!m_render.renderer)
        return logFail("Failed to create renderer!");
    if (!startBackgroundAssetWorker())
        return logFail("Failed to start background asset worker.");
    if (!startBackgroundLoadWorker())
        return logFail("Failed to start background load worker.");

#ifdef NBL_BUILD_DEBUG_DRAW
    {
        auto* renderpass = scRes->getRenderpass();
        ext::debug_draw::DrawAABB::SCreationParameters params = {};
        params.assetManager = m_assetMgr;
        params.transfer = getTransferUpQueue();
        params.drawMode = ext::debug_draw::DrawAABB::ADM_DRAW_BATCH;
        params.batchPipelineLayout = ext::debug_draw::DrawAABB::createDefaultPipelineLayout(m_device.get());
        params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
        params.utilities = m_utils;
        m_drawAABB = ext::debug_draw::DrawAABB::create(std::move(params));
    }
#endif

    if (!initTestCases())
        return false;
    if (performanceEnabled())
        beginPerformanceRun();

    auto runInitialContent = [&]() -> bool
    {
        if (isRowViewActive())
        {
            m_runtime.nonInteractiveTest = false;
            if (!loadRowView(RowViewReloadMode::Full))
                return false;
            if (m_output.rowAddPath)
                if (!addRowViewCaseFromPath(*m_output.rowAddPath))
                    return false;
            if (m_output.rowDuplicateCount > 0u && !m_runtime.cases.empty())
            {
                const auto lastPath = m_runtime.cases.back().path;
                if (!appendRowViewDuplicates(m_output.rowDuplicateCount, lastPath, [this](const system::path& path) {
                    return addRowViewCaseFromPath(path);
                    }))
                    return false;
            }
            return true;
        }

        if (m_runtime.mode != RunMode::Interactive)
            m_runtime.nonInteractiveTest = true;
        return startCase(0u);
    };
    if (!runInitialContent())
        return false;

    camera.mapKeysToArrows();

    onAppInitializedFinish();
    return true;
}

IQueue::SSubmitInfo::SSemaphoreInfo MeshLoadersApp::renderFrame(const std::chrono::microseconds nextPresentationTimestamp)
{
    m_inputSystem->getDefaultMouse(&mouse);
    m_inputSystem->getDefaultKeyboard(&keyboard);

    const auto resourceIx = m_render.realFrameIx % MaxFramesInFlight;

    auto* const cb = m_render.cmdBufs.data()[resourceIx].get();
    cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
    cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
    // clear to black for both things
    {
        // begin renderpass
        {
            auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
            auto* framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex);
            const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
            const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
            const VkRect2D currentRenderArea =
            {
                .offset = {0,0},
                .extent = {framebuffer->getCreationParameters().width,framebuffer->getCreationParameters().height}
            };
            const IGPUCommandBuffer::SRenderpassBeginInfo info =
            {
                .framebuffer = framebuffer,
                .colorClearValues = &clearValue,
                .depthStencilClearValues = &depthValue,
                .renderArea = currentRenderArea
            };
            cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

                const SViewport viewport = {
                    .x = static_cast<float>(currentRenderArea.offset.x),
                    .y = static_cast<float>(currentRenderArea.offset.y),
                    .width = static_cast<float>(currentRenderArea.extent.width),
                    .height = static_cast<float>(currentRenderArea.extent.height)
                };
                cb->setViewport(0u,1u,&viewport);
    
                cb->setScissor(0u,1u,&currentRenderArea);
            }
            // late latch input
            if (!m_runtime.nonInteractiveTest)
            {
                struct SPendingInputActions
                {
                    bool reloadInteractive = false;
                    bool reloadList = false;
                    bool addRowView = false;
                    bool clearRowView = false;
                } pending;
                camera.beginInputProcessing(nextPresentationTimestamp);
                mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, m_logger.get());
                keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
                    {
                        for (const auto& event : events)
                        {
                            if (event.action != SKeyboardEvent::ECA_RELEASED)
                                continue;
                            if (event.keyCode == E_KEY_CODE::EKC_R)
                            {
                                if (isRowViewActive())
                                    pending.reloadList = true;
                                else
                                    pending.reloadInteractive = true;
                            }
                            else if (event.keyCode == E_KEY_CODE::EKC_A)
                            {
                                if (isRowViewActive())
                                    pending.addRowView = true;
                            }
                            else if (event.keyCode == E_KEY_CODE::EKC_X)
                            {
                                if (isRowViewActive())
                                    pending.clearRowView = true;
                            }
                        }
                        camera.keyboardProcess(events);
                    },
                    m_logger.get()
                );
                camera.endInputProcessing(nextPresentationTimestamp);
                if (pending.clearRowView)
                    resetRowViewScene();
                if (pending.addRowView)
                    addRowViewCase();
                if (pending.reloadList)
                {
                    if (!reloadFromTestList())
                        failExit("Failed to reload test list.");
                }
                if (pending.reloadInteractive)
                    reloadInteractive();
            }
            // draw scene
            const auto& viewMatrix = camera.getViewMatrix();
            const auto& viewProjMatrix = camera.getConcatenatedMatrix();
            {
                     m_render.renderer->render(cb,CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix));
            }
#ifdef NBL_BUILD_DEBUG_DRAW
            {
                const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_render.semaphore.get(),.value = m_render.realFrameIx + 1u };
                ext::debug_draw::DrawAABB::DrawParameters drawParams;
                drawParams.commandBuffer = cb;
                drawParams.cameraMat = viewProjMatrix;
                m_drawAABB->render(drawParams, drawFinished, m_aabbInstances);
            }
#endif
            cb->endRenderPass();
        }
        cb->end();

    IQueue::SSubmitInfo::SSemaphoreInfo retval =
    {
        .semaphore = m_render.semaphore.get(),
        .value = ++m_render.realFrameIx,
        .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
    };
    const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
    {
        {.cmdbuf = cb }
    };
    const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
        {
            .semaphore = device_base_t::getCurrentAcquire().semaphore,
            .value = device_base_t::getCurrentAcquire().acquireCount,
            .stageMask = PIPELINE_STAGE_FLAGS::NONE
        }
    };
    const IQueue::SSubmitInfo infos[] =
    {
        {
            .waitSemaphores = acquired,
            .commandBuffers = commandBuffers,
            .signalSemaphores = {&retval,1}
        }
    };

    if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
    {
        retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
        m_render.realFrameIx--;
    }

    std::string caption = "[Nabla Engine] Mesh Loaders";
    {
        caption += ", displaying [";
        caption += ::makeCaptionModelPath(m_modelPath, argv);
        caption += "]";
        m_window->setCaption(caption);
    }
    const uint64_t rowViewCaptureRequestFrame = (RowViewFramesBeforeCapture > 1u) ? (RowViewFramesBeforeCapture - 1u) : RowViewFramesBeforeCapture;
    if (isRowViewActive() && !m_runtime.rowViewScreenshotCaptured && m_render.realFrameIx >= rowViewCaptureRequestFrame)
    {
        if (!m_render.pendingScreenshot.active())
        {
            if (!requestScreenshotCapture(m_output.rowViewScreenshotPath))
                failExit("Failed to request row view screenshot capture.");
        }
        else
        {
            bool ready = false;
            if (!finalizeScreenshotCapture(m_render.loadedScreenshot, ready))
                failExit("Failed to finalize row view screenshot.");
            if (ready)
                m_runtime.rowViewScreenshotCaptured = true;
        }
    }
    advanceCase();
    return retval;
}

bool MeshLoadersApp::onAppTerminated()
{
    if (performanceEnabled() && !m_perf.finalized)
    {
        endPerformanceCase();
        finalizePerformanceRun();
    }
    stopBackgroundLoadWorker();
    stopBackgroundAssetWorker();
    return device_base_t::onAppTerminated();
}

bool MeshLoadersApp::shouldKeepRunning() const
{
    return !m_runtime.shouldQuit;
}

const video::IGPURenderpass::SCreationParams::SSubpassDependency* MeshLoadersApp::getDefaultSubpassDependencies() const
{
    // Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
    const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
        // wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
        {
            .srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
            .dstSubpass = 0,
            .memoryBarrier = {
            // last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
            .srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
            // don't want any writes to be available, we'll clear 
            .srcAccessMask = ACCESS_FLAGS::NONE,
            // destination needs to wait as early as possible
            .dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
            // because depth and color get cleared first no read mask
            .dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
        }
        // leave view offsets and flags default
    },
        // color from ATTACHMENT_OPTIMAL to PRESENT_SRC
        {
            .srcSubpass = 0,
            .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
            .memoryBarrier = {
            // last place where the color can get modified, depth is implicitly earlier
            .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
            // only write ops, reads can't be made available
            .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
            // spec says nothing is needed when presentation is the destination
        }
        // leave view offsets and flags default
    },
    IGPURenderpass::SCreationParams::DependenciesEnd
    };
    return dependencies;
}

[[noreturn]] void MeshLoadersApp::failExit(const char* msg, ...)
{
    char formatted[4096] = {};
    va_list args;
    va_start(args, msg);
    vsnprintf(formatted, sizeof(formatted), msg, args);
    va_end(args);
    if (m_logger)
        m_logger->log("%s", ILogger::ELL_ERROR, formatted);
    std::exit(-1);
}

bool MeshLoadersApp::initTestCases()
{
    m_runtime.cases.clear();
    m_runtime.caseNameCounts.clear();
    if (m_runtime.mode == RunMode::Interactive)
    {
        system::path picked;
        if (!pickModelPath(picked))
            return logFail("No file selected.");
        m_runtime.cases.push_back({ makeUniqueCaseName(picked), picked });
        return true;
    }
    return loadTestList(m_output.testListPath);
}

bool MeshLoadersApp::pickModelPath(system::path& outPath)
{
    if (m_runtime.fileDialogOpen)
    {
        if (m_logger)
            m_logger->log("File dialog is already open. Ignoring request.", ILogger::ELL_WARNING);
        return false;
    }

    struct DialogGuard
    {
        bool& flag;
        ~DialogGuard() { flag = false; }
    };

    m_runtime.fileDialogOpen = true;
    DialogGuard guard{m_runtime.fileDialogOpen};

    pfd::open_file file(
        "Choose a supported Model File",
        sharedInputCWD.string(),
        {
            "All Supported Formats", "*.ply *.stl *.serialized *.obj",
            "Polygon File Format (.ply)", "*.ply",
            "Stereolithography (.stl)", "*.stl",
            "Mitsuba 0.6 Serialized (.serialized)", "*.serialized",
            "Wavefront Object (.obj)", "*.obj"
        },
        false);

    const auto selected = file.result();
    if (selected.empty())
        return false;
    outPath = selected[0];
    return true;
}

bool MeshLoadersApp::loadTestList(const system::path& jsonPath)
{
    if (!std::filesystem::exists(jsonPath))
        return logFail("Missing test list: %s", jsonPath.string().c_str());
    m_runtime.rowViewEnabled = true;

    std::ifstream stream(jsonPath);
    if (!stream.is_open())
        return logFail("Failed to open test list: %s", jsonPath.string().c_str());

    nlohmann::json doc;
    try
    {
        stream >> doc;
    }
    catch (const std::exception& e)
    {
        return logFail("Invalid JSON in test list: %s", e.what());
    }

    if (!doc.contains("cases") || !doc["cases"].is_array())
        return logFail("Test list JSON missing \"cases\" array.");

    m_runtime.caseNameCounts.clear();

    if (doc.contains("row_view"))
    {
        if (!doc["row_view"].is_boolean())
            return logFail("\"row_view\" must be a boolean.");
        m_runtime.rowViewEnabled = doc["row_view"].get<bool>();
    }
    if (m_runtime.forceRowViewForCurrentTestList && m_runtime.mode == RunMode::Batch)
        m_runtime.rowViewEnabled = true;

    const auto baseDir = jsonPath.parent_path();
    for (const auto& entry : doc["cases"])
    {
        std::string pathString;

        if (entry.is_string())
        {
            pathString = entry.get<std::string>();
        }
        else if (entry.is_object())
        {
            if (!entry.contains("path") || !entry["path"].is_string())
                return logFail("Test list entry missing \"path\".");
            pathString = entry["path"].get<std::string>();
        }
        else
            return logFail("Invalid test list entry.");

        system::path path = pathString;
        if (path.is_relative())
            path = baseDir / path;
        if (!std::filesystem::exists(path))
            return logFail("Missing test input: %s", path.string().c_str());

        m_runtime.cases.push_back({ makeUniqueCaseName(path), path });
    }

    if (m_runtime.cases.empty())
        return logFail("No test cases in test list.");

    return true;
}

bool MeshLoadersApp::isRowViewActive() const
{
    return m_runtime.rowViewEnabled && m_runtime.mode != RunMode::CI && m_runtime.mode != RunMode::Interactive;
}

std::string MeshLoadersApp::normalizeExtension(const system::path& path)
{
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

asset::writer_flags_t MeshLoadersApp::getWriterFlagsForPath(const IAsset* const asset, const system::path& path) const
{
    if (!asset)
        return asset::EWF_NONE;

    const auto extension = system::extension_wo_dot(path);
    auto flags = asset::writer_flags_t(asset::EWF_NONE);
    if (const auto writerInfo = m_assetMgr->getAssetWriterFlagInfo(asset->getAssetType(), extension); writerInfo.has_value())
    {
        flags = writerInfo->forced;
        const auto preferred = asset::writer_flags_t(asset::EWF_MESH_IS_RIGHT_HANDED | asset::EWF_BINARY);
        flags |= preferred & writerInfo->supported;
        return flags;
    }

    return asset::writer_flags_t(asset::EWF_MESH_IS_RIGHT_HANDED);
}

bool MeshLoadersApp::isWriteExtensionSupported(const std::string& ext) const
{
    if (ext == ".ply" || ext == ".stl")
        return true;
#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_
    if (ext == ".obj")
        return true;
#endif
    return false;
}

system::path MeshLoadersApp::resolveSavePath(const system::path& modelPath) const
{
    if (m_output.specifiedGeomSavePath)
        return path(*m_output.specifiedGeomSavePath);
    const auto stem = modelPath.stem().string();
    auto ext = normalizeExtension(modelPath);
    if (ext.empty())
        ext = ".ply";
    if (!isWriteExtensionSupported(ext))
    {
        if (m_logger)
            m_logger->log("No writer for %s, writing .ply instead.", ILogger::ELL_WARNING, ext.c_str());
        ext = ".ply";
    }
    return m_output.saveGeomPrefixPath / (stem + "_written" + ext);
}

bool MeshLoadersApp::startCase(const size_t index)
{
    if (index >= m_runtime.cases.size())
        return false;

    auto resetCasePresentationState = [&]() -> void
    {
        m_runtime.phase = Phase::RenderOriginal;
        m_runtime.phaseFrameCounter = 0u;
        m_render.loadedScreenshot = nullptr;
        m_render.writtenScreenshot = nullptr;
        m_render.pendingScreenshot = {};
        m_referenceCamera.reset();
    };

    m_runtime.caseIndex = index;
    resetCasePresentationState();

    const auto& testCase = m_runtime.cases[m_runtime.caseIndex];
    if (performanceEnabled())
        beginPerformanceCase(testCase);
    const auto artifacts = makeCaseArtifacts(
        testCase.name,
        testCase.path,
        m_output.screenshotPrefixPath,
        [this](const system::path& path) { return resolveSavePath(path); });
    m_caseName = artifacts.caseName;
    m_output.writtenPath = artifacts.writtenPath;
    m_output.loadedScreenshotPath = artifacts.loadedScreenshotPath;
    m_output.writtenScreenshotPath = artifacts.writtenScreenshotPath;

    bool loaded = false;
    LoadStageMetrics loadMetrics = {};
    if (m_runtime.mode == RunMode::CI)
    {
        PreparedAssetLoad preparedLoad = {};
        bool preparedReady = false;
        const bool loadWorkerStateValid = finalizePreparedAssetLoad(preparedLoad, preparedReady, true);
        if (loadWorkerStateValid && preparedReady && preparedLoad.success && preparedLoad.caseIndex == index && preparedLoad.path == testCase.path)
            loaded = loadPreparedModel(testCase.path, std::move(preparedLoad.loadResult), true, true, &loadMetrics);
        else
            loaded = loadModel(testCase.path, true, true, &loadMetrics);
    }
    else
        loaded = loadModel(testCase.path, true, true, &loadMetrics);
    if (!loaded)
        return false;
    if (performanceEnabled() && loadMetrics.valid)
        recordOriginalLoadMetrics(loadMetrics);

    if (m_runtime.mode != RunMode::Interactive && m_output.saveGeom && m_render.currentCpuAsset)
    {
        if (!startWrittenAssetWork(m_render.currentCpuAsset, m_output.writtenPath))
        {
            if (m_runtime.mode == RunMode::CI)
                return logFail("Background written-asset preparation did not start for %s.", m_caseName.c_str());
            m_logger->log("Background written-asset preparation did not start for %s. Falling back to synchronous flow.", ILogger::ELL_WARNING, m_caseName.c_str());
        }
    }
    if (m_runtime.mode == RunMode::CI)
    {
        const auto nextIndex = index + 1u;
        if (nextIndex < m_runtime.cases.size())
            startPreparedAssetLoad(nextIndex, m_runtime.cases[nextIndex].path);
    }

    return true;
}

bool MeshLoadersApp::advanceToNextCase()
{
    const auto nextIndex = m_runtime.caseIndex + 1u;
    if (nextIndex >= m_runtime.cases.size())
    {
        if (performanceEnabled())
        {
            endPerformanceCase();
            finalizePerformanceRun();
        }
        m_runtime.shouldQuit = true;
        return false;
    }
    if (performanceEnabled())
        endPerformanceCase();
    if (!startCase(nextIndex))
    {
        m_runtime.shouldQuit = true;
        return false;
    }
    return true;
}

void MeshLoadersApp::reloadInteractive()
{
    system::path picked;
    if (!pickModelPath(picked))
        failExit("No file selected.");
    if (!loadModel(picked, true, true))
        failExit("Failed to load asset %s.", picked.string().c_str());
    if (m_render.currentCpuAsset && m_output.saveGeom)
    {
        const auto savePath = resolveSavePath(picked);
        if (!writeAssetRoot(m_render.currentCpuAsset, savePath.string()))
            failExit("Geometry write failed.");
    }
}

bool MeshLoadersApp::addRowViewCase()
{
    system::path picked;
    if (!pickModelPath(picked))
        return false;
    return addRowViewCaseFromPath(picked);
}

bool MeshLoadersApp::addRowViewCaseFromPath(const system::path& picked)
{
    if (picked.empty())
        return false;
    m_runtime.cases.push_back({ makeUniqueCaseName(picked), picked });
    m_runtime.shouldQuit = false;
    return loadRowView(RowViewReloadMode::Incremental);
}

bool MeshLoadersApp::reloadFromTestList()
{
    m_runtime.cases.clear();
    m_render.pendingScreenshot = {};
    if (!loadTestList(m_output.testListPath))
        return false;
    m_runtime.shouldQuit = false;
    m_runtime.rowViewScreenshotCaptured = false;
    if (isRowViewActive())
    {
        m_runtime.nonInteractiveTest = false;
        return loadRowView(RowViewReloadMode::Full);
    }
    m_runtime.nonInteractiveTest = (m_runtime.mode != RunMode::Interactive);
    return startCase(0u);
}

void MeshLoadersApp::resetRowViewScene()
{
    if (!isRowViewActive())
        return;
    m_runtime.cases.clear();
    m_render.pendingScreenshot = {};
    m_rowView.cache.clear();
    m_render.renderer->m_instances.clear();
    m_render.renderer->clearGeometries({ .semaphore = m_render.semaphore.get(),.value = m_render.realFrameIx });
#ifdef NBL_BUILD_DEBUG_DRAW
    m_aabbInstances.clear();
    m_obbInstances.clear();
#endif
    m_modelPath = "Row view (empty)";
    m_runtime.rowViewScreenshotCaptured = false;
    m_runtime.shouldQuit = false;
    m_runtime.nonInteractiveTest = false;
    m_logger->log("Row view reset to empty. Press A to add a model.", ILogger::ELL_INFO);
}




