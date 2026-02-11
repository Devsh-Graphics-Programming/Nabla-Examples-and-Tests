// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "argparse/argparse.hpp"
#include "portable-file-dialogs/portable-file-dialogs.h"
#include "nlohmann/json.hpp"
#include "MeshLoadersApp.hpp"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#ifdef NBL_BUILD_MITSUBA_LOADER
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#endif

#include "nbl/system/CFileLogger.h"

MeshLoadersApp::MeshLoadersApp(
    const path& localInputCWD,
    const path& localOutputCWD,
    const path& sharedInputCWD,
    const path& sharedOutputCWD)
    : IApplicationFramework(localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD)
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

    m_runMode = RunMode::Batch;
    m_saveGeomPrefixPath = localOutputCWD / "saved";
    m_screenshotPrefixPath = localOutputCWD / "screenshots";
    m_testListPath = localInputCWD / "inputs.json";

    argparse::ArgumentParser parser("12_meshloaders");
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
    parser.add_argument("--loader-content-hashes")
        .help("Force loaders to compute CPU buffer content hashes before returning. Enabled by default.")
        .flag();
    parser.add_argument("--runtime-tuning")
        .nargs(1)
        .help("Runtime tuning mode for loaders: none|heuristic|hybrid. Default: heuristic.");
    parser.add_argument("--update-references")
        .help("Update or create geometry hash references for CI validation.")
        .flag();

    try
    {
        parser.parse_args({ argv.data(), argv.data() + argv.size() });
    }
    catch (const std::exception& e)
    {
        return logFail(e.what());
    }

    if (parser["--savegeometry"] == true)
        m_saveGeom = true;
    if (parser["--interactive"] == true)
        m_runMode = RunMode::Interactive;
    if (parser["--ci"] == true)
        m_runMode = RunMode::CI;

    if (parser.present("--savepath"))
    {
        auto tmp = path(parser.get<std::string>("--savepath"));

        if (tmp.empty() || !tmp.has_filename())
            return logFail("Invalid path has been specified in --savepath argument");

        if (!std::filesystem::exists(tmp.parent_path()))
            return logFail("Path specified in --savepath argument doesn't exist");

        m_specifiedGeomSavePath.emplace(std::move(tmp.generic_string()));
    }

    if (parser.present("--testlist"))
    {
        auto tmp = path(parser.get<std::string>("--testlist"));
        if (tmp.empty())
            return logFail("Invalid path has been specified in --testlist argument");
        if (tmp.is_relative())
            tmp = localInputCWD / tmp;
        m_testListPath = tmp;
    }
    if (parser.present("--row-add"))
    {
        auto tmp = path(parser.get<std::string>("--row-add"));
        if (tmp.is_relative())
            tmp = localInputCWD / tmp;
        m_rowAddPath = tmp;
    }
    if (parser.present("--row-duplicate"))
    {
        auto countStr = parser.get<std::string>("--row-duplicate");
        try
        {
            m_rowDuplicateCount = static_cast<uint32_t>(std::stoul(countStr));
        }
        catch (const std::exception&)
        {
            return logFail("Invalid --row-duplicate value.");
        }
    }
    if (parser.present("--loader-perf-log"))
    {
        auto tmp = path(parser.get<std::string>("--loader-perf-log"));
        if (tmp.empty())
            return logFail("Invalid --loader-perf-log value.");
        if (tmp.is_relative())
            tmp = localOutputCWD / tmp;
        m_loaderPerfLogPath = tmp;
    }
    if (parser["--update-references"] == true)
        m_updateGeometryHashReferences = true;
    if (parser["--loader-content-hashes"] == true)
        m_forceLoaderContentHashes = true;
    if (parser.present("--runtime-tuning"))
    {
        auto mode = parser.get<std::string>("--runtime-tuning");
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "none")
            m_runtimeTuningMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::None;
        else if (mode == "heuristic")
            m_runtimeTuningMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Heuristic;
        else if (mode == "hybrid")
            m_runtimeTuningMode = asset::SFileIOPolicy::SRuntimeTuning::Mode::Hybrid;
        else
            return logFail("Invalid --runtime-tuning value. Expected: none|heuristic|hybrid.");
    }

    const path inputReferencesDir = localInputCWD / "references";
    const path outputReferencesDir = localOutputCWD / "references";
    std::error_code referenceDirEc;
    const bool hasInputReferencesDir = std::filesystem::is_directory(inputReferencesDir, referenceDirEc) && !referenceDirEc;
    referenceDirEc.clear();
    const bool hasOutputReferencesDir = std::filesystem::is_directory(outputReferencesDir, referenceDirEc) && !referenceDirEc;
    m_geometryHashReferenceDir = hasOutputReferencesDir || !hasInputReferencesDir ? outputReferencesDir : inputReferencesDir;
    if (hasOutputReferencesDir && !hasInputReferencesDir)
        m_logger->log("Geometry hash references resolved to output directory: %s", system::ILogger::ELL_INFO, m_geometryHashReferenceDir.string().c_str());
    if (m_runMode == RunMode::CI || m_updateGeometryHashReferences)
    {
        std::error_code ec;
        std::filesystem::create_directories(m_geometryHashReferenceDir, ec);
        if (ec)
            return logFail("Failed to create geometry hash reference directory: %s", m_geometryHashReferenceDir.string().c_str());
    }

    if (m_saveGeom)
        std::filesystem::create_directories(m_saveGeomPrefixPath);
    std::filesystem::create_directories(m_screenshotPrefixPath);
    m_assetLoadLogger = m_logger;
    if (m_loaderPerfLogPath)
    {
        if (!initLoaderPerfLogger(*m_loaderPerfLogPath))
            return false;
        m_logger->log("Loader diagnostics will be written to %s", ILogger::ELL_INFO, m_loaderPerfLogPath->string().c_str());
    }

    m_semaphore = m_device->createSemaphore(m_realFrameIx);
    if (!m_semaphore)
        return logFail("Failed to Create a Semaphore!");

    auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
    for (auto i = 0u; i < MaxFramesInFlight; i++)
    {
        if (!pool)
            return logFail("Couldn't create Command Pool!");
        if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i,1 }))
            return logFail("Couldn't create Command Buffer!");
    }

    auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
    m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), scRes->getRenderpass(), 0, {});
    if (!m_renderer)
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
        m_nonInteractiveTest = false;
        if (!loadRowView(RowViewReloadMode::Full))
            return false;
        if (m_rowAddPath)
            if (!addRowViewCaseFromPath(*m_rowAddPath))
                return false;
        if (m_rowDuplicateCount > 0u && !m_cases.empty())
        {
            const auto lastPath = m_cases.back().path;
            for (uint32_t i = 0u; i < m_rowDuplicateCount; ++i)
                if (!addRowViewCaseFromPath(lastPath))
                    return false;
        }
    }
    else
    {
        if (m_runMode != RunMode::Interactive)
            m_nonInteractiveTest = true;
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

    const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

    auto* const cb = m_cmdBufs.data()[resourceIx].get();
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
            if (!m_nonInteractiveTest)
            {
                bool reloadInteractiveRequested = false;
                bool reloadListRequested = false;
                bool addRowViewRequested = false;
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
                                    reloadListRequested = true;
                                else
                                    reloadInteractiveRequested = true;
                            }
                            else if (event.keyCode == E_KEY_CODE::EKC_A)
                            {
                                if (isRowViewActive())
                                    addRowViewRequested = true;
                            }
                        }
                        camera.keyboardProcess(events);
                    },
                    m_logger.get()
                );
                camera.endInputProcessing(nextPresentationTimestamp);
                if (addRowViewRequested)
                    addRowViewCase();
                if (reloadListRequested)
                {
                    if (!reloadFromTestList())
                        failExit("Failed to reload test list.");
                }
                if (reloadInteractiveRequested)
                    reloadInteractive();
            }
            // draw scene
            const auto& viewMatrix = camera.getViewMatrix();
            const auto& viewProjMatrix = camera.getConcatenatedMatrix();
            {
                     m_renderer->render(cb,CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix));
            }
#ifdef NBL_BUILD_DEBUG_DRAW
            {
                const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_semaphore.get(),.value = m_realFrameIx + 1u };
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
        .semaphore = m_semaphore.get(),
        .value = ++m_realFrameIx,
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
        m_realFrameIx--;
    }

    std::string caption = "[Nabla Engine] Mesh Loaders";
    {
        caption += ", displaying [";
        caption += m_modelPath;
        caption += "]";
        m_window->setCaption(caption);
    }
    if (isRowViewActive() && !m_rowViewScreenshotCaptured && m_realFrameIx >= RowViewFramesBeforeCapture)
    {
        if (!captureScreenshot(m_rowViewScreenshotPath, m_loadedScreenshot))
            failExit("Failed to capture row view screenshot.");
        m_rowViewScreenshotCaptured = true;
    }
    advanceCase();
    return retval;
}

bool MeshLoadersApp::onAppTerminated()
{
    return device_base_t::onAppTerminated();
}

bool MeshLoadersApp::keepRunning()
{
    if (m_shouldQuit)
        return false;
    return device_base_t::keepRunning();
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
    m_cases.clear();
    m_caseNameCounts.clear();
    if (m_runMode == RunMode::Interactive)
    {
        system::path picked;
        if (!pickModelPath(picked))
            return logFail("No file selected.");
        m_cases.push_back({ makeUniqueCaseName(picked), picked });
        return true;
    }
    return loadTestList(m_testListPath);
}

bool MeshLoadersApp::pickModelPath(system::path& outPath)
{
    if (m_fileDialogOpen)
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

    m_fileDialogOpen = true;
    DialogGuard guard{m_fileDialogOpen};

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

    m_caseNameCounts.clear();

    if (doc.contains("row_view"))
    {
        if (!doc["row_view"].is_boolean())
            return logFail("\"row_view\" must be a boolean.");
        m_rowViewEnabled = doc["row_view"].get<bool>();
    }

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

        m_cases.push_back({ makeUniqueCaseName(path), path });
    }

    if (m_cases.empty())
        return logFail("No test cases in test list.");

    return true;
}

bool MeshLoadersApp::isRowViewActive() const
{
    return m_rowViewEnabled && m_runMode != RunMode::CI && m_runMode != RunMode::Interactive;
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
    if (m_specifiedGeomSavePath)
        return path(*m_specifiedGeomSavePath);
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
    return m_saveGeomPrefixPath / (stem + "_written" + ext);
}

std::string MeshLoadersApp::sanitizeCaseNameForFilename(std::string name)
{
    for (auto& ch : name)
    {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (!(std::isalnum(uch) || ch == '_' || ch == '-' || ch == '.'))
            ch = '_';
    }
    if (name.empty())
        name = "unnamed_case";
    return name;
}

system::path MeshLoadersApp::getGeometryHashReferencePath(const std::string& caseName) const
{
    return m_geometryHashReferenceDir / (sanitizeCaseNameForFilename(caseName) + ".geomhash");
}

std::string MeshLoadersApp::geometryHashToHex(const core::blake3_hash_t& hash)
{
    static constexpr char HexDigits[] = "0123456789abcdef";
    std::string out;
    out.resize(sizeof(hash.data) * 2ull);
    for (size_t i = 0ull; i < sizeof(hash.data); ++i)
    {
        const uint8_t v = hash.data[i];
        out[2ull * i + 0ull] = HexDigits[(v >> 4) & 0xfu];
        out[2ull * i + 1ull] = HexDigits[v & 0xfu];
    }
    return out;
}

bool MeshLoadersApp::tryParseNibble(const char c, uint8_t& out)
{
    if (c >= '0' && c <= '9')
    {
        out = static_cast<uint8_t>(c - '0');
        return true;
    }
    if (c >= 'a' && c <= 'f')
    {
        out = static_cast<uint8_t>(10 + c - 'a');
        return true;
    }
    if (c >= 'A' && c <= 'F')
    {
        out = static_cast<uint8_t>(10 + c - 'A');
        return true;
    }
    return false;
}

bool MeshLoadersApp::tryParseGeometryHashHex(std::string hex, core::blake3_hash_t& outHash)
{
    hex.erase(std::remove_if(hex.begin(), hex.end(), [](unsigned char c) { return std::isspace(c) != 0; }), hex.end());
    if (hex.size() != sizeof(outHash.data) * 2ull)
        return false;

    for (size_t i = 0ull; i < sizeof(outHash.data); ++i)
    {
        uint8_t hi = 0u;
        uint8_t lo = 0u;
        if (!tryParseNibble(hex[2ull * i + 0ull], hi) || !tryParseNibble(hex[2ull * i + 1ull], lo))
            return false;
        outHash.data[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

bool MeshLoadersApp::readGeometryHashReference(const system::path& refPath, core::blake3_hash_t& outHash) const
{
    std::ifstream in(refPath);
    if (!in.is_open())
        return false;
    std::string line;
    std::getline(in, line);
    return tryParseGeometryHashHex(std::move(line), outHash);
}

bool MeshLoadersApp::writeGeometryHashReference(const system::path& refPath, const core::blake3_hash_t& hash) const
{
    std::error_code ec;
    std::filesystem::create_directories(refPath.parent_path(), ec);
    if (ec)
        return false;
    std::ofstream out(refPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
        return false;
    out << geometryHashToHex(hash) << '\n';
    return out.good();
}

bool MeshLoadersApp::startCase(const size_t index)
{
    if (index >= m_cases.size())
        return false;

    m_caseIndex = index;
    m_phase = Phase::RenderOriginal;
    m_phaseFrameCounter = 0u;
    m_loadedScreenshot = nullptr;
    m_writtenScreenshot = nullptr;
    m_referenceCamera.reset();
    m_hasReferenceGeometryHash = false;
    m_caseGeometryHashReferencePath.clear();

    const auto& testCase = m_cases[m_caseIndex];
    m_caseName = testCase.name.empty() ? testCase.path.stem().string() : testCase.name;
    m_writtenPath = resolveSavePath(testCase.path);
    m_loadedScreenshotPath = m_screenshotPrefixPath / ("meshloaders_" + m_caseName + "_loaded.png");
    m_writtenScreenshotPath = m_screenshotPrefixPath / ("meshloaders_" + m_caseName + "_written.png");

    if (!loadModel(testCase.path, true, true))
        return false;

    if (m_currentCpuGeom)
    {
        const auto loadedGeometryHash = hashGeometry(m_currentCpuGeom.get());
        m_referenceGeometryHash = loadedGeometryHash;
        m_hasReferenceGeometryHash = true;
        m_caseGeometryHashReferencePath = getGeometryHashReferencePath(m_caseName);

        if (m_updateGeometryHashReferences)
        {
            const bool referenceExisted = std::filesystem::exists(m_caseGeometryHashReferencePath);
            if (!writeGeometryHashReference(m_caseGeometryHashReferencePath, loadedGeometryHash))
                return logFail("Failed to write geometry hash reference: %s", m_caseGeometryHashReferencePath.string().c_str());
            if (!referenceExisted)
                m_logger->log("Geometry hash reference did not exist for %s. Created new reference at %s", ILogger::ELL_WARNING, m_caseName.c_str(), m_caseGeometryHashReferencePath.string().c_str());
            else
                m_logger->log("Geometry hash reference updated for %s at %s", ILogger::ELL_INFO, m_caseName.c_str(), m_caseGeometryHashReferencePath.string().c_str());
        }
        else if (m_runMode == RunMode::CI)
        {
            if (!std::filesystem::exists(m_caseGeometryHashReferencePath))
                return logFail("Missing geometry hash reference for %s at %s. Run once with --update-references.", m_caseName.c_str(), m_caseGeometryHashReferencePath.string().c_str());

            core::blake3_hash_t onDiskHash = {};
            if (!readGeometryHashReference(m_caseGeometryHashReferencePath, onDiskHash))
                return logFail("Invalid geometry hash reference for %s at %s", m_caseName.c_str(), m_caseGeometryHashReferencePath.string().c_str());

            m_referenceGeometryHash = onDiskHash;
            m_hasReferenceGeometryHash = true;
            if (loadedGeometryHash != onDiskHash)
            {
                m_logger->log("Loaded geometry hash mismatch for %s. Current=%s Reference=%s", ILogger::ELL_ERROR, m_caseName.c_str(), geometryHashToHex(loadedGeometryHash).c_str(), geometryHashToHex(onDiskHash).c_str());
                return logFail("Loaded asset differs from stored geometry hash reference for %s.", m_caseName.c_str());
            }
        }
    }

    return true;
}

bool MeshLoadersApp::advanceToNextCase()
{
    const auto nextIndex = m_caseIndex + 1u;
    if (nextIndex >= m_cases.size())
    {
        m_shouldQuit = true;
        return false;
    }
    if (!startCase(nextIndex))
    {
        m_shouldQuit = true;
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
    if (m_currentCpuGeom && m_saveGeom)
    {
        const auto savePath = resolveSavePath(picked);
        if (!writeGeometry(m_currentCpuGeom, savePath.string()))
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
    m_cases.push_back({ makeUniqueCaseName(picked), picked });
    m_shouldQuit = false;
    return loadRowView(RowViewReloadMode::Incremental);
}

bool MeshLoadersApp::reloadFromTestList()
{
    m_cases.clear();
    if (!loadTestList(m_testListPath))
        return false;
    m_shouldQuit = false;
    m_rowViewScreenshotCaptured = false;
    if (isRowViewActive())
    {
        m_nonInteractiveTest = false;
        return loadRowView(RowViewReloadMode::Full);
    }
    m_nonInteractiveTest = (m_runMode != RunMode::Interactive);
    return startCase(0u);
}



