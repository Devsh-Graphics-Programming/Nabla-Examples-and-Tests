// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "argparse/argparse.hpp"
#include "portable-file-dialogs/portable-file-dialogs.h"
#include "nlohmann/json.hpp"
#include "MeshLoadersApp.hpp"

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
        .help("JSON file with test cases. Relative paths are resolved against local input CWD.");
    parser.add_argument("--row-add")
        .nargs(1)
        .help("Add a model path to row view on startup without using a dialog.");
    parser.add_argument("--row-duplicate")
        .nargs(1)
        .help("Duplicate the last case N times on startup.");
    parser.add_argument("--loader-perf-log")
        .nargs(1)
        .help("Write loader diagnostics to a file instead of stdout.");
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

}

system::path MeshLoadersApp::resolveRuntimeCWD(const system::path& preferred)
{
    if (preferred.empty() || preferred == path("/") || preferred == path("\\"))
        return path(std::filesystem::current_path());
    return preferred;
}

std::string MeshLoadersApp::makeCaptionModelPath() const
{
    const auto& modelPath = m_modelPath;
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

bool MeshLoadersApp::parseCommandLineOptions(const system::path& effectiveInputCWD, const system::path& effectiveOutputCWD, const system::path& defaultBenchmarkTestListPath)
{
    m_runtime.mode = RunMode::Batch;
    m_output.saveGeomPrefixPath = effectiveOutputCWD / "saved";
    m_output.screenshotPrefixPath = effectiveOutputCWD / "screenshots";
    m_output.testListPath = effectiveInputCWD / "inputs.json";
    m_runtime.forceRowViewForCurrentTestList = false;

    argparse::ArgumentParser parser("12_meshloaders");
    setupMeshLoadersArgumentParser(parser);

    try
    {
        parser.parse_args({ argv.data(), argv.data() + argv.size() });
    }
    catch (const std::exception& e)
    {
        return logFail(e.what());
    }

    if (parser["--savegeometry"] == true)
        m_output.saveGeom = true;
    if (parser["--interactive"] == true)
        m_runtime.mode = RunMode::Interactive;
    if (parser["--ci"] == true)
        m_runtime.mode = RunMode::CI;
    const bool hasExplicitTestListArg = parser.present("--testlist").has_value();

    if (parser.present("--savepath"))
    {
        auto tmp = path(parser.get<std::string>("--savepath"));
        if (tmp.empty() || !tmp.has_filename())
            return logFail("Invalid path has been specified in --savepath argument");
        if (!std::filesystem::exists(tmp.parent_path()))
            return logFail("Path specified in --savepath argument doesn't exist");
        m_output.specifiedGeomSavePath.emplace(std::move(tmp.generic_string()));
    }

    if (hasExplicitTestListArg)
    {
        auto tmp = path(parser.get<std::string>("--testlist"));
        if (tmp.empty())
            return logFail("Invalid path has been specified in --testlist argument");
        if (tmp.is_relative())
            tmp = effectiveInputCWD / tmp;
        m_output.testListPath = tmp;
    }
    else if (m_runtime.mode == RunMode::Batch && !defaultBenchmarkTestListPath.empty())
    {
        std::error_code benchmarkPathEc;
        if (std::filesystem::exists(defaultBenchmarkTestListPath, benchmarkPathEc) && !benchmarkPathEc)
        {
            m_output.testListPath = defaultBenchmarkTestListPath;
            m_runtime.forceRowViewForCurrentTestList = true;
            m_logger->log("Using benchmark test list for default batch startup: %s", ILogger::ELL_INFO, m_output.testListPath.string().c_str());
        }
    }

    if (parser.present("--row-add"))
    {
        auto tmp = path(parser.get<std::string>("--row-add"));
        if (tmp.is_relative())
            tmp = effectiveInputCWD / tmp;
        m_output.rowAddPath = tmp;
    }
    if (parser.present("--row-duplicate"))
    {
        auto countStr = parser.get<std::string>("--row-duplicate");
        const auto parsedCount = parseUInt32Argument(countStr);
        if (!parsedCount.has_value())
            return logFail("Invalid --row-duplicate value.");
        m_output.rowDuplicateCount = *parsedCount;
    }
    if (parser.present("--loader-perf-log"))
    {
        auto tmp = path(parser.get<std::string>("--loader-perf-log"));
        if (tmp.empty())
            return logFail("Invalid --loader-perf-log value.");
        if (tmp.is_relative())
            tmp = effectiveOutputCWD / tmp;
        m_output.loaderPerfLogPath = tmp;
    }
    if (parser.present("--runtime-tuning"))
    {
        auto mode = parser.get<std::string>("--runtime-tuning");
        if (!parseRuntimeTuningMode(mode, m_runtimeTuningMode))
            return logFail("Invalid --runtime-tuning value. Expected: sequential|heuristic|hybrid.");
    }

    return true;
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

    const path effectiveInputCWD = resolveRuntimeCWD(localInputCWD);
    const path effectiveOutputCWD = resolveRuntimeCWD(localOutputCWD);
#if defined(NBL_MESHLOADERS_DEFAULT_BENCHMARK_TESTLIST_PATH)
    const path defaultBenchmarkTestListPath = path(NBL_MESHLOADERS_DEFAULT_BENCHMARK_TESTLIST_PATH);
#else
    const path defaultBenchmarkTestListPath;
#endif
    if (!parseCommandLineOptions(effectiveInputCWD, effectiveOutputCWD, defaultBenchmarkTestListPath))
        return false;

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
            for (uint32_t i = 0u; i < m_output.rowDuplicateCount; ++i)
                if (!addRowViewCaseFromPath(lastPath))
                    return false;
        }
    }
    else
    {
        if (m_runtime.mode != RunMode::Interactive)
            m_runtime.nonInteractiveTest = true;
        if (!startCase(0u))
            return false;
    }

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
        caption += makeCaptionModelPath();
        caption += "]";
        m_window->setCaption(caption);
    }
    if (isRowViewActive() && !m_runtime.rowViewScreenshotCaptured && m_render.realFrameIx >= RowViewFramesBeforeCapture)
    {
        if (!captureScreenshot(m_output.rowViewScreenshotPath, m_render.loadedScreenshot))
            failExit("Failed to capture row view screenshot.");
        m_runtime.rowViewScreenshotCaptured = true;
    }
    advanceCase();
    return retval;
}

bool MeshLoadersApp::onAppTerminated()
{
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

    m_runtime.caseIndex = index;
    m_runtime.phase = Phase::RenderOriginal;
    m_runtime.phaseFrameCounter = 0u;
    m_render.loadedScreenshot = nullptr;
    m_render.writtenScreenshot = nullptr;
    m_referenceCamera.reset();

    const auto& testCase = m_runtime.cases[m_runtime.caseIndex];
    m_caseName = testCase.name.empty() ? testCase.path.stem().string() : testCase.name;
    m_output.writtenPath = resolveSavePath(testCase.path);
    m_output.loadedScreenshotPath = m_output.screenshotPrefixPath / ("meshloaders_" + m_caseName + "_loaded.png");
    m_output.writtenScreenshotPath = m_output.screenshotPrefixPath / ("meshloaders_" + m_caseName + "_written.png");

    if (!loadModel(testCase.path, true, true))
        return false;

    return true;
}

bool MeshLoadersApp::advanceToNextCase()
{
    const auto nextIndex = m_runtime.caseIndex + 1u;
    if (nextIndex >= m_runtime.cases.size())
    {
        m_runtime.shouldQuit = true;
        return false;
    }
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




