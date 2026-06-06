// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

#include "nbl/examples/examples.hpp"

#include "renderer/CRenderer.h"
#include "renderer/resolve/CBasicRWMCResolver.h"
#include "renderer/present/CWindowPresenter.h"

#include "gui/CUIManager.h"
#include "nbl/ui/ICursorControl.h"

#include "nbl/examples/cameras/CCamera.hpp"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "nlohmann/json.hpp"
#include "imgui.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>


using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;
using namespace nbl::this_example;

// Bench headers depend on the unqualified `float64_t` (nbl::hlsl::float64_t)
// inside BenchmarkTypes.h, so they must be included AFTER the using namespace
// directives above.
#include "nbl/examples/Benchmark/IBenchmark.h"
#include "benchmarks/CPathTracerBenchmark.h"

// TODO: move to argument parsing class
struct AppArguments
{
   bool headless; // set in onAppInitialized() for now
   bool benchmark; // --benchmark: run the Aggregator-driven bench then exit
};

// Minimal in-place parse for 40-PT-specific CLI flags. Bench-side flags
// (--baseline / --focus / etc.) are forwarded untouched to Aggregator::applyCli.
static inline AppArguments parsePTArgs(std::span<const std::string> argv)
{
   AppArguments out = {};
   for (const auto& a : argv)
   {
      if (a == "--benchmark")
         out.benchmark = true;
      else if (a == "--headless")
         out.headless = true;
   }
   // Bench mode is always headless: no presenter, no UI, no input.
   if (out.benchmark)
      out.headless = true;
   return out;
}


class PathTracingApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
   using device_base_t = SimpleWindowedApplication;
   using asset_base_t  = BuiltinResourcesApplication;

   // TODO: move to Nabla proper
   static inline void jsonizeGitInfo(nlohmann::json& target, const nbl::gtml::GitInfo& info)
   {
      target["isPopulated"] = info.isPopulated;
      if (info.hasUncommittedChanges.has_value())
         target["hasUncommittedChanges"] = info.hasUncommittedChanges.value();
      else
         target["hasUncommittedChanges"] = "UNKNOWN, BUILT WITHOUT DIRTY-CHANGES CAPTURE";

      target["commitAuthorName"]  = info.commitAuthorName;
      target["commitAuthorEmail"] = info.commitAuthorEmail;
      target["commitHash"]        = info.commitHash;
      target["commitShortHash"]   = info.commitShortHash;
      target["commitDate"]        = info.commitDate;
      target["commitSubject"]     = info.commitSubject;
      target["commitBody"]        = info.commitBody;
      target["describe"]          = info.describe;
      target["branchName"]        = info.branchName;
      target["latestTag"]         = info.latestTag;
      target["latestTagName"]     = info.latestTagName;
   }

   inline void printGitInfos() const
   {
      nlohmann::json j;

      auto& modules = j["modules"];
      jsonizeGitInfo(modules["nabla"], nbl::gtml::nabla_git_info);
      jsonizeGitInfo(modules["dxc"], nbl::gtml::dxc_git_info);

      m_logger->log("Build Info:\n%s", ILogger::ELL_INFO, j.dump(4).c_str());
   }


public:
   inline PathTracingApp(const path& _localInputCWD,
      const path&                    _localOutputCWD,
      const path&                    _sharedInputCWD,
      const path&                    _sharedOutputCWD)
      : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
   {
   }

   inline IAPIConnection::SFeatures getAPIFeaturesToEnable() override
   {
      auto retval = device_base_t::getAPIFeaturesToEnable();
      if (m_args.headless)
         retval.swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
      return retval;
   }

   inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
   {
      auto retval = device_base_t::getRequiredDeviceFeatures();
      return retval.unionWith(CRenderer::RequiredDeviceFeatures());
   }

   inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
   {
      auto retval = device_base_t::getPreferredDeviceFeatures();
      if (m_args.headless)
         retval.swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
      retval.pipelineExecutableInfo = true;
      return retval.unionWith(CRenderer::PreferredDeviceFeatures());
   }

   inline SPhysicalDeviceLimits getRequiredDeviceLimits() const override
   {
      auto retval = device_base_t::getRequiredDeviceLimits();
      // TODO: need union/superset so Renderer can slap it in
      retval.rayTracingInvocationReorder         = true;
      retval.rayTracingPositionFetch             = true;
      retval.shaderStorageImageReadWithoutFormat = true;
      return retval;
   }

   inline void filterDevices(nbl::core::set<IPhysicalDevice*>& physicalDevices) const override
   {
      device_base_t::filterDevices(physicalDevices);
      std::erase_if(physicalDevices,
         [&](const IPhysicalDevice* device) -> bool
         {
            const auto& props           = device->getMemoryProperties();
            uint64_t    largestVRAMHeap = 0;
            using heap_flags_e          = IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS;
            for (uint32_t h = 0; h < props.memoryHeapCount; h++)
               if (const auto& heap = props.memoryHeaps[h];
                  heap.flags.hasFlags(heap_flags_e::EMHF_DEVICE_LOCAL_BIT))
                  largestVRAMHeap = nbl::hlsl::max(largestVRAMHeap, heap.size);
            const auto typeBits = device->getDirectVRAMAccessMemoryTypeBits();
            for (uint32_t t = 0; t < props.memoryTypeCount; t++)
               if (((typeBits >> t) & 0x1u) &&
                  props.memoryHeaps[props.memoryTypes[t].heapIndex].size == largestVRAMHeap)
                  return false;
            m_logger->log("Filtering out Device %p (%s) due to lack of ReBAR",
               ILogger::ELL_WARNING,
               device,
               device->getProperties().deviceName);
            return true;
         });
   }

   inline nbl::core::vector<SPhysicalDeviceFilter::SurfaceCompatibility>
   getSurfaces() const override
   {
      if (m_args.headless)
         return {};

      if (!m_presenter)
      {
         const_cast<std::remove_reference_t<decltype(m_presenter)>&>(m_presenter) =
            CWindowPresenter::create(
               { { .assMan = m_assetMgr, .logger = smart_refctd_ptr(m_logger) },
                  { .winMgr = m_winMgr },
                  m_api,
                  make_smart_refctd_ptr<CEventCallback>(
                     smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger)),
                  "Path Tracer" });
      }

      if (m_presenter)
      {
         const auto* presenter = m_presenter.get();
         return { { presenter->getSurface() /*,EQF_NONE*/ } };
      }

      return {};
   }

   inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
   {
      m_args = parsePTArgs(this->argv);

      if (!m_args.headless)
         m_inputSystem =
            make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

      if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
         return false;

      if (m_args.headless)
      {
         if (!BasicMultiQueueApplication::onAppInitialized(smart_refctd_ptr(system)))
            return false;
      }
      else if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
         return false;

      printGitInfos();

      //
      if (!m_args.headless && !m_presenter)
         return logFail("Failed to create CWindowPresenter");

      //
      m_renderer = CRenderer::create({ { .graphicsQueue = getGraphicsQueue(),
                                          .computeQueue = getComputeQueue(),
                                          .uploadQueue  = getTransferUpQueue(),
                                          .utilities    = smart_refctd_ptr(m_utils) },
         m_assetMgr.get(),
         (sharedOutputCWD /
            nbl::examples::CCachedOwenScrambledSequence::SCreationParams::DefaultFilename)
            .string() });
      if (!m_renderer)
         return logFail("Failed to create CRenderer");

      //
      if (!m_args.headless && !m_presenter->init(m_renderer.get()))
         return logFail("Failed to initialize CWindowPresenter");

      //
      m_resolver = CBasicRWMCResolver::create({ {}, m_renderer.get() });
      if (!m_resolver)
         return logFail("Failed to create CBasicRWMCResolver");

      // set up the scene loader
      m_sceneLoader = CSceneLoader::create(
         { { .assMan = smart_refctd_ptr(m_assetMgr), .logger = smart_refctd_ptr(m_logger) } });

      // TODO: tmp code
      {
         m_api->startCapture();
         m_currentScenePath = (sharedInputCWD / "mitsuba/ditt/render_720p.xml").string();
         m_currentScene     = m_renderer->createScene(
            { .load = m_sceneLoader->load(
                 { .relPath = m_currentScenePath, .workingDirectory = localOutputCWD }),
                   .converter = nullptr });
         auto scene_daily_pt = m_currentScene;

         // the UI would have you load the zip first, then present a dropdown of what to load
         // but still need to support archive mount for cmdline load
#if 0 // this particular zip goes down an unsupported path in our zip loader
            auto scene_bedroom = m_sceneLoader->load({
               .relPath = sharedInputCWD / "mitsuba/bedroom.zip/scene.xml",
               .workingDirectory = localOutputCWD
            });
#endif
         m_api->endCapture();

         if (!scene_daily_pt)
            return logFail("Could not create scene");

         // quick test code
         nbl::core::vector<CSession::sensor_t> sensors(3, scene_daily_pt->getSensors().front());
         {
            sensors[1].mutableDefaults.cropWidth   = 640;
            sensors[1].mutableDefaults.cropHeight  = 360;
            sensors[1].mutableDefaults.cropOffsetX = 0;
            sensors[1].mutableDefaults.cropOffsetY = 0;
         }
         {
            sensors[2].mutableDefaults.cropWidth   = 5120;
            sensors[2].mutableDefaults.cropHeight  = 2880;
            sensors[2].mutableDefaults.cropOffsetX = 128;
            sensors[2].mutableDefaults.cropOffsetY = 128;
         }
         for (auto i = 1; i < 3; i++)
         {
            sensors[i].constants.width =
               sensors[i].mutableDefaults.cropWidth + 2 * sensors[i].mutableDefaults.cropOffsetX;
            sensors[i].constants.height =
               sensors[i].mutableDefaults.cropHeight + 2 * sensors[i].mutableDefaults.cropOffsetY;
         }
         //				sensors.erase(sensors.begin());
         for (const auto& sensor : sensors)
            m_sessionQueue.push(scene_daily_pt->createSession(
               { { .mode = CSession::RenderMode::Beauty }, &sensor }));
      }

      // Initialize UI Manager (non-headless only)
      if (!m_args.headless)
      {
         m_uiManager = gui::CUIManager::create({ .assetManager = smart_refctd_ptr(m_assetMgr),
            .utilities                                         = smart_refctd_ptr(m_utils),
            .transferQueue                                     = getGraphicsQueue(),
            .logger                                            = smart_refctd_ptr(m_logger) });
         if (!m_uiManager)
            return logFail("Failed to create CUIManager");

         gui::CUIManager::SInitParams uiInitParams = { .renderpass = m_presenter->getRenderpass(),
            .onSensorSelected =
               [this](size_t sensorIdx)
            {
               // Create a new session from the selected sensor (GUI mode)
               if (m_currentScene)
               {
                  const auto sensors = m_currentScene->getSensors();
                  if (sensorIdx < sensors.size())
                  {
                     const auto mode = m_pendingSession
                        ? m_pendingSession->getConstructionParams().mode
                        : (m_resolver->getActiveSession()
                                ? m_resolver->getActiveSession()->getConstructionParams().mode
                                : CSession::RenderMode::Beauty);
                     auto       newSession =
                        m_currentScene->createSession({ { .mode = mode }, &sensors[sensorIdx] });
                     if (newSession)
                     {
                        m_currentSensorIdx = sensorIdx;
                        initCameraFromSensor(sensorIdx);
                        m_pendingSession = std::move(newSession);
                     }
                  }
               }
            },
            .onLoadSceneRequested =
               [this](const std::string& path)
            {
               if (path.empty())
                  return;

               m_logger->log("Loading scene: %s", ILogger::ELL_INFO, path.c_str());

               // Load the scene
               auto                   loadResult =
                  m_sceneLoader->load({ .relPath = path, .workingDirectory = localOutputCWD });

               if (!loadResult)
               {
                  m_logger->log("Failed to load scene: %s", ILogger::ELL_ERROR, path.c_str());
                  return;
               }

               // Create the scene
               auto                   newScene =
                  m_renderer->createScene({ .load = std::move(loadResult), .converter = nullptr });

               if (!newScene)
               {
                  m_logger->log(
                     "Failed to create scene from: %s", ILogger::ELL_ERROR, path.c_str());
                  return;
               }

               // Update current scene
               m_currentScene                      = std::move(newScene);
               m_currentScenePath                  = path;

               // Update UI
               if (m_uiManager)
                  m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);

               // Build a fresh session so the renderer picks up the new scene's DS + light tree.
               // The previous CScene stays alive via the still-active session's smart_refctd_ptr until the swap below.
               m_currentSensorIdx                  = 0;
               initCameraFromSensor(m_currentSensorIdx);
               {
                  const auto& sensors    = m_currentScene->getSensors();
                  auto        newSession = m_currentScene->createSession(
                     { { .mode = CSession::RenderMode::Beauty }, &sensors.front() });
                  if (newSession)
                     m_pendingSession = std::move(newSession);
               }

               m_logger->log("Scene loaded successfully: %s", ILogger::ELL_INFO, path.c_str());
            },
            .onReloadSceneRequested =
               [this]()
            {
               if (m_currentScenePath.empty())
               {
                  m_logger->log("No scene to reload", ILogger::ELL_WARNING);
                  return;
               }

               m_logger->log("Reloading scene: %s", ILogger::ELL_INFO, m_currentScenePath.c_str());

               // Reload the scene
               auto                   loadResult   = m_sceneLoader->load(
                  { .relPath = m_currentScenePath, .workingDirectory = localOutputCWD });

               if (!loadResult)
               {
                  m_logger->log(
                     "Failed to reload scene: %s", ILogger::ELL_ERROR, m_currentScenePath.c_str());
                  return;
               }

               auto                   newScene =
                  m_renderer->createScene({ .load = std::move(loadResult), .converter = nullptr });

               if (!newScene)
               {
                  m_logger->log("Failed to create scene from: %s",
                     ILogger::ELL_ERROR,
                     m_currentScenePath.c_str());
                  return;
               }

               m_currentScene                      = std::move(newScene);

               if (m_uiManager)
                  m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);

               // Preserve sensor + mode across reload, clamping the sensor index in case
               // the reloaded scene has fewer sensors.
               {
                  const auto&  sensors   = m_currentScene->getSensors();
                  const size_t sensorIdx = std::min<size_t>(m_currentSensorIdx, sensors.size() - 1);
                  m_currentSensorIdx     = sensorIdx;
                  initCameraFromSensor(sensorIdx);
                  CSession::RenderMode mode = CSession::RenderMode::Beauty;
                  if (auto* active = m_resolver->getActiveSession())
                     mode = active->getConstructionParams().mode;
                  auto newSession =
                     m_currentScene->createSession({ { .mode = mode }, &sensors[sensorIdx] });
                  if (newSession)
                     m_pendingSession = std::move(newSession);
               }

               m_logger->log("Scene reloaded successfully", ILogger::ELL_INFO);
            },
            .onEmitterDensityChanged =
               [this](float density)
            {
               if (!m_currentScene || m_currentScenePath.empty())
                  return;
               m_renderer->setEmitterDensity(density);
               m_logger->log(
                  "Emitter density -> %.3f, rebuilding scene", ILogger::ELL_INFO, density);

               auto                   loadResult   = m_sceneLoader->load(
                  { .relPath = m_currentScenePath, .workingDirectory = localOutputCWD });
               if (!loadResult)
                  return;
               auto                   newScene =
                  m_renderer->createScene({ .load = std::move(loadResult), .converter = nullptr });
               if (!newScene)
                  return;
               m_currentScene                      = std::move(newScene);
               if (m_uiManager)
                  m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);
               const auto&            sensors      = m_currentScene->getSensors();
               const size_t           sensorIdx    = std::min<size_t>(m_currentSensorIdx, sensors.size() - 1);
               m_currentSensorIdx                  = sensorIdx;
               CSession::RenderMode   mode         = CSession::RenderMode::Beauty;
               if (auto* active = m_resolver->getActiveSession())
                  mode = active->getConstructionParams().mode;
               auto                   newSession =
                  m_currentScene->createSession({ { .mode = mode }, &sensors[sensorIdx] });
               if (newSession)
                  m_pendingSession = std::move(newSession);
            },
            .onUseAliasNEEChanged =
               [this](bool useAlias)
            {
               if (m_renderer)
                  m_renderer->setUseAliasNEE(useAlias);
            },
            .onMisModeChanged =
               [this](int misMode)
            {
               if (m_renderer)
               {
                  m_renderer->setMisMode(static_cast<CSession::MisMode>(misMode));
                  // The MIS modes converge to DIFFERENT images, so blending across a switch is wrong;
                  // force one fresh frame to clear accumulation (the alias toggle doesn't need this).
                  m_restartAccumOnNextFrame = true;
               }
            },
            .onCameraMoveSpeedChanged              = [this](float moveSpeed)
            { m_camera.setMoveSpeed(moveSpeed); },
            .onProbeChanged =
               [this](float px, float py, float pz, float nx, float ny, float nz)
            {
               if (m_renderer)
               {
                  m_renderer->setProbe({ px, py, pz }, { nx, ny, nz });
                  m_uiManager->getSceneWindow().setProbePdfSum(m_renderer->getProbePdfSum());
               }
            },
            // Session callbacks
            .onRenderModeChanged =
               [this](CSession::RenderMode mode, CSession* session)
            {
               if (!m_currentScene)
                  return;
               const auto             sensors      = m_currentScene->getSensors();
               if (m_currentSensorIdx >= sensors.size())
                  return;
               auto                   newSession =
                  m_currentScene->createSession({ { .mode = mode }, &sensors[m_currentSensorIdx] });
               if (newSession)
                  m_pendingSession = std::move(newSession);
            },
            .onResolutionChanged = [this](uint16_t w, uint16_t h)
            { m_logger->log("Resolution changed to %dx%d (TODO)", ILogger::ELL_INFO, w, h); },
            .onMutablesChanged =
               [this](const SSensorDynamics& dyn, CSession* session)
            {
               session->update(dyn);
               m_logger->log("Mutables changed (Reset TODO)", ILogger::ELL_INFO);
            },
            .onDynamicsChanged                     = [this](const SSensorDynamics& dyn, CSession* session)
            { session->update(dyn); },
            // App-driven (not committed via update()): applied into the dynamics each
            // frame so the change reads as a delta and restarts accumulation.
            .onMaxPathDepthChanged                 = [this](uint16_t maxPathDepth)
            { m_maxPathDepth                       = maxPathDepth; },
            .onBufferSelected =
               [this](int id)
            {
               if (!m_presenter)
                  return;
               using BufferType                    = gui::CSessionWindow::BufferType;
               using ImgIdx                        = SensorDSBindings::SampledImageIndex;
               ImgIdx                 imageIndex   = ImgIdx::Beauty;
               switch (static_cast<BufferType>(id))
               {
                  case BufferType::Beauty:
                     imageIndex = ImgIdx::Beauty;
                     break;
                  case BufferType::Albedo:
                     imageIndex = ImgIdx::Albedo;
                     break;
                  case BufferType::Normal:
                     imageIndex = ImgIdx::Normal;
                     break;
                  case BufferType::Motion:
                     imageIndex = ImgIdx::Motion;
                     break;
                  case BufferType::Mask:
                     imageIndex = ImgIdx::Mask;
                     break;
                  case BufferType::RWMCCascades:
                     imageIndex = ImgIdx::RWMCCascades;
                     break;
                  case BufferType::SampleCount:
                     imageIndex = ImgIdx::SampleCount;
                     break;
                  default:
                     return;
               }
               m_presenter->setSelectedImageIndex(static_cast<uint8_t>(imageIndex));
            },
            .onBenchmarkRequested                  = [this](CSession* session)
            { runBenchmarkOnce(session, benchSceneLabel(), m_currentSensorIdx); },
            .onDumpImageRequested =
               [this](CSession* session)
            {
               dumpBeautyToEXR(session);
            } };

         if (!m_uiManager->init(uiInitParams))
            return logFail("Failed to initialize CUIManager");


         // Set up UI with the initially loaded scene
         if (m_currentScene)
            m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);

         // Create initial session from first sensor so GUI has something to display
         if (m_currentScene && !m_currentScene->getSensors().empty())
         {
            const auto& sensors = m_currentScene->getSensors();
            initCameraFromSensor(m_currentSensorIdx);
            auto initialSession = m_currentScene->createSession(
               { { .mode = CSession::RenderMode::Beauty }, &sensors.front() });

            m_pendingSession = std::move(initialSession);
         }
      }

      // --benchmark: drive the first queued session through the Aggregator
      // and exit. Same code path as the GUI button (see runBenchmarkOnce); the
      // only difference is the post-bench `m_benchmarkDone = true` that lets
      // keepRunning() short-circuit.
      if (m_args.benchmark)
      {
         if (m_sessionQueue.empty())
            return logFail("--benchmark: no sessions queued, cannot bench");

         smart_refctd_ptr<CSession> session = std::move(m_sessionQueue.front());
         m_sessionQueue.pop();
         while (!m_sessionQueue.empty())
            m_sessionQueue.pop();

         CSession* const sessionPtr = session.get();
         m_utils->autoSubmit<SIntendedSubmitInfo>({ .queue = getGraphicsQueue() },
            [sessionPtr](SIntendedSubmitInfo& info) -> bool { return sessionPtr->init(info); });
         m_resolver->changeSession(std::move(session));

         runBenchmarkOnce(m_resolver->getActiveSession(), benchSceneLabel(), m_currentSensorIdx);
         m_benchmarkDone = true;
         return true;
      }

      return true;
   }

   // Whole-frame bench against `session`. Reused by --benchmark (CLI) and the
   // GUI "Benchmark Current Session" button. The GUI path runs this inline on
   // the UI thread, so the window freezes for ~targetBudgetMs.
   void runBenchmarkOnce(CSession* session, const std::string& sceneLabel, size_t sensorIdx)
   {
      if (!session)
      {
         m_logger->log("runBenchmarkOnce: null session, skipping", ILogger::ELL_WARNING);
         return;
      }

      // Per-run artifact dir under ./benchmarks/, used by the FLIP compare
      // script: bench.json + one beauty EXR per row land here together.
      const std::filesystem::path runDir = std::filesystem::current_path() / "benchmarks" /
         std::format("{}_s{}_{}", sceneLabel, sensorIdx, nowStamp());
      std::error_code ec;
      std::filesystem::create_directories(runDir, ec);

      Aggregator agg(smart_refctd_ptr(m_logger),
         smart_refctd_ptr(m_device),
         m_physicalDevice,
         getComputeQueue()->getFamilyIndex());
      agg.applyCli({
         .argv              = this->argv,
         .defaultOutputPath = (runDir / "bench.json").string(),
         .appName           = "40_PathTracer",
      });

      constexpr uint64_t targetBudgetMs =
         2000; // unused in equal-spp mode (fixedTimedFrames), kept for the time-budget path
      // Equal-spp comparison: run every row to the SAME samples-per-pixel so FLIP isolates
      // per-sample variance (selection quality), not throughput. Must be <= the renderOnce
      // maxSPPOverride cap and the 22-bit maxSPP field. Frame count is derived from the
      // per-frame spp below.
      constexpr uint64_t targetSamplesPerPixel =
         3000; // high-spp reference run (fp32 Beauty accumulation, RWMC bypassed)
      const auto     renderSize = session->getConstructionParams().uniforms.renderSize;
      const uint32_t depth =
         session->getConstructionParams().type != CSession::sensor_type_e::Env ? 1u : 6u;
      const uint64_t totalThreads =
         uint64_t(renderSize.x) * uint64_t(renderSize.y) * uint64_t(depth);

      // Query the RT pipeline's executable info up front so the banner shows
      // the actual subgroup size as `wg=<sgSize>x1x1`. Falls back to the 3D
      // launch-grid view when the driver doesn't report a subgroup size.
      const auto    mode = session->getConstructionParams().mode;
      PipelineStats earlyStats;
      if (auto* const scene = session->getConstructionParams().scene.get())
         if (auto* const pipeline = scene->getPipeline(mode))
            extractPipelineStats(pipeline, earlyStats);

      const uint64_t sppPerInvocation = uint64_t(m_renderer->getMaxSppPerDispatch());

      WorkloadShape shape      = {};
      shape.workgroupSize      = { 1u, 1u, 1u };
      shape.dispatchGroupCount = { uint32_t(renderSize.x), uint32_t(renderSize.y), depth };
      shape.samplesPerDispatch = totalThreads * sppPerInvocation;

      const RunContext ctx = {
         .shape          = shape,
         .targetBudgetMs = targetBudgetMs,
         .sectionLabel   = "Path Tracer Whole-Frame",
      };

      const auto modeStr = nbl::system::to_string(session->getConstructionParams().mode);

      // Rows across the MIS-mode axis (separate Beauty pipelines) and NEE-selection axis. The bench
      // applies each row's SetupData.useAliasNEE / .misMode to the renderer right before its own
      // warmup/timed frames, so both are correct when its frames execute (regardless of construction
      // order or interleaving by the aggregator). We save/restore the renderer's values around the run.
      const bool              savedToggle  = m_renderer->getUseAliasNEE();
      const CSession::MisMode savedMisMode = m_renderer->getMisMode();

      // misLabel is its own name segment (the MIS mode), techLabel is the last (the
      // selector compared within that mode). So group_key = everything up to and
      // including misLabel: rows are grouped per MIS mode, and each mode gets its own
      // converged reference (subfoldered by misLabel). EXRs dump to runDir/<misLabel>/
      // <techLabel>.exr so the per-mode references mirror the run layout exactly.
      auto makeBenchData =
         [&](const std::string& misLabel, const std::string& techLabel, int useAlias, int misMode)
      {
         CPathTracerBenchmark::SetupData data = {};
         data.name                            = { "PathTracer",
                                       modeStr,
                                       sceneLabel + "/sensor" + std::to_string(sensorIdx),
                                       misLabel,
                                       techLabel };
         data.shape                           = ctx.shape;
         data.targetBudgetMs                  = ctx.targetBudgetMs;
         data.warmupDispatches                = 5;
         // Same frame count for every row -> same total spp, so the slower selector
         // can't look cleaner just by banking more samples in a fixed time budget.
         //data.fixedTimedFrames                = uint32_t((targetSamplesPerPixel + sppPerInvocation - 1ull) / sppPerInvocation);
         data.renderer           = m_renderer.get();
         data.session            = session;
         data.useAliasNEE        = useAlias;
         data.misMode            = misMode;
         data.onAfterTimedFrames = [this, session, runDir, misLabel, techLabel]()
         {
            dumpBeautyToEXR(session, runDir / misLabel / (techLabel + ".exr"));
         };
         return data;
      };

      m_logger->log(
         "ps/sample is GPU time per samplesPerDispatch (= renderSize.x * renderSize.y * depth * maxSppPerDispatch = %u * %u * %u * %u = %llu samples/frame)",
         ILogger::ELL_INFO,
         uint32_t(renderSize.x),
         uint32_t(renderSize.y),
         depth,
         uint32_t(sppPerInvocation),
         static_cast<uint64_t>(shape.samplesPerDispatch));

      // Rows span the MIS-mode axis (separate Beauty pipelines) x the NEE-selection axis. For the
      // NEE-bearing modes (Both, NEEOnly) both alias and tree are meaningful; BxDFOnly has no NEE, so
      // selection is irrelevant and it gets a single anchor row. BxDFOnly is the unbiased reference:
      // BOTH's direct lighting must match it (partition-of-unity), the extra brightness being indirect.
      using MisMode = CSession::MisMode;
      // One span (contiguous vector) so every row shares a single banner/header and one report write,
      // instead of one table per row. reserve() avoids reallocation so the benches never move.
      std::vector<CPathTracerBenchmark> benches;
      benches.reserve(5);
      benches.emplace_back(agg, makeBenchData("Both", "nee-alias", 1, int(MisMode::Both)));
      benches.emplace_back(agg, makeBenchData("Both", "nee-tree", 0, int(MisMode::Both)));
      benches.emplace_back(agg, makeBenchData("NEEOnly", "nee-alias", 1, int(MisMode::NEEOnly)));
      benches.emplace_back(agg, makeBenchData("NEEOnly", "nee-tree", 0, int(MisMode::NEEOnly)));
      benches.emplace_back(agg, makeBenchData("BxDFOnly", "bxdf", -1, int(MisMode::BxDFOnly)));

      agg.runSessionAndReport(Aggregator::makeSpan(benches, ctx));

      m_renderer->setUseAliasNEE(savedToggle);
      m_renderer->setMisMode(savedMisMode);
   }

   // Which session image gets read back when dumping EXRs (both for the
   // interactive button and the benchmark per-row dump). Flip this and rebuild.
   //   Beauty       -> immutables.beauty   (EF_R32G32B32A32_SFLOAT)
   //   RWMCCascades -> immutables.rwmcCascades layer 0 (EF_R16G16B16A16_SFLOAT)
   enum class DumpTarget : uint8_t
   {
      Beauty,
      RWMCCascades
   };
   static constexpr DumpTarget kDumpTarget = DumpTarget::Beauty;

   static constexpr std::string_view dumpTargetTag(DumpTarget t)
   {
      switch (t)
      {
         case DumpTarget::Beauty:
            return "beauty";
         case DumpTarget::RWMCCascades:
            return "rwmc";
      }
      return "unknown";
   }

   // EXR readback of the session's chosen image (see kDumpTarget), strictly
   // for offline image comparisons (e.g. NVIDIA FLIP). Caller chooses the
   // destination path; we create parent directories on demand.
   bool dumpBeautyToEXR(CSession* session, const std::filesystem::path& out)
   {
      if (!session || !session->isInitialized())
      {
         m_logger->log("Dump: no initialized session", ILogger::ELL_ERROR);
         return false;
      }
      const auto& immutables  = session->getActiveResources().immutables;
      IGPUImage*  srcGPUImage = nullptr;
      E_FORMAT    viewFormat  = EF_UNKNOWN;
      switch (kDumpTarget)
      {
         case DumpTarget::Beauty:
            srcGPUImage = immutables.beauty.image.get();
            viewFormat  = EF_R32G32B32A32_SFLOAT;
            break;
         case DumpTarget::RWMCCascades:
            srcGPUImage = immutables.rwmcCascades.image.get();
            viewFormat  = EF_R16G16B16A16_SFLOAT;
            break;
      }
      if (!srcGPUImage)
      {
         m_logger->log("Dump: %s source image missing",
            ILogger::ELL_ERROR,
            std::string(dumpTargetTag(kDumpTarget)).c_str());
         return false;
      }

      // Build an ad-hoc single-layer 2D view for the readback. The pre-baked
      // session views use viewType=2D_ARRAY with layerCount = all-layers, but
      // ext::ScreenShot sizes the staging buffer for a single layer only and
      // would corrupt the readback for multi-layer images like rwmcCascades.
      auto dumpView = m_device->createImageView({
         .subUsages        = IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT,
         .image            = smart_refctd_ptr<IGPUImage>(srcGPUImage),
         .viewType         = IGPUImageView::E_TYPE::ET_2D,
         .format           = viewFormat,
         .subresourceRange = { IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT, 0u, 1u, 0u, 1u },
      });
      if (!dumpView)
      {
         m_logger->log("Dump: failed to build single-layer readback view", ILogger::ELL_ERROR);
         return false;
      }
      IGPUImageView* const view = dumpView.get();

      std::error_code ec;
      if (out.has_parent_path())
         std::filesystem::create_directories(out.parent_path(), ec);

      auto cpuView = ext::ScreenShot::createScreenShot(m_device.get(),
         getGraphicsQueue()->getUnderlyingQueue(),
         nullptr,
         view,
         ACCESS_FLAGS::SHADER_WRITE_BITS,
         IImage::LAYOUT::GENERAL);
      if (!cpuView)
      {
         m_logger->log(
            "Dump: GPU readback failed for \"%s\"", ILogger::ELL_ERROR, out.string().c_str());
         return false;
      }

      auto srcImage           = cpuView->getCreationParameters().image;
      auto cleanParams        = srcImage->getCreationParameters();
      cleanParams.flags       = IImage::E_CREATE_FLAGS::ECF_NONE;
      cleanParams.viewFormats = {};
      cleanParams.usage       = IImage::E_USAGE_FLAGS::EUF_NONE;
      cleanParams.arrayLayers = 1u;
      cleanParams.format      = viewFormat;
      auto cleanImage         = ICPUImage::create(std::move(cleanParams));
      if (!cleanImage ||
         !cleanImage->setBufferAndRegions(
            smart_refctd_ptr<ICPUBuffer>(const_cast<ICPUBuffer*>(srcImage->getBuffer())),
            srcImage->getRegionArray()))
      {
         m_logger->log("Dump: failed to rebuild clean CPU image", ILogger::ELL_ERROR);
         return false;
      }
      cleanImage->setContentHash(IPreHashed::INVALID_HASH);

      auto cleanViewParams                        = cpuView->getCreationParameters();
      cleanViewParams.image                       = std::move(cleanImage);
      cleanViewParams.format                      = viewFormat;
      cleanViewParams.subresourceRange.layerCount = 1u;
      auto cleanView = ICPUImageView::create(std::move(cleanViewParams));
      if (!cleanView)
      {
         m_logger->log("Dump: failed to rebuild clean CPU view", ILogger::ELL_ERROR);
         return false;
      }

      IAssetWriter::SAssetWriteParams writeParams(cleanView.get());
      const bool                      ok = m_assetMgr->writeAsset(out.string(), writeParams);
      if (!ok)
         m_logger->log("Dump: failed to write \"%s\"", ILogger::ELL_ERROR, out.string().c_str());
      else
         m_logger->log("Dump: wrote \"%s\"", ILogger::ELL_DEBUG, out.string().c_str());
      return ok;
   }

   // Local wall-clock stamp, YYYYMMDD_HHMMSS, for artifact names.
   static std::string nowStamp()
   {
      const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
      return std::format("{:%Y%m%d_%H%M%S}", std::chrono::current_zone()->to_local(now));
   }

   // Interactive UI button: timestamped per-click EXR in ./dumps/.
   void dumpBeautyToEXR(CSession* session)
   {
      const auto out = std::filesystem::current_path() / "dumps" /
         std::format("{}_{}_s{}_{}.exr",
            dumpTargetTag(kDumpTarget),
            benchSceneLabel(),
            m_currentSensorIdx,
            nowStamp());
      dumpBeautyToEXR(session, out);
   }

   // Basename of m_currentScenePath, sans extension, for use in bench row names.
   std::string benchSceneLabel() const
   {
      if (m_currentScenePath.empty())
         return "unknown";
      const auto stem = std::filesystem::path(m_currentScenePath).stem().string();
      return stem.empty() ? "unknown" : stem;
   }

   inline void workLoopBody() override
   {
      if (m_args.benchmark && m_benchmarkDone)
         return;
      if (m_args.headless)
      {
         CSession* session = m_resolver->getActiveSession();
         while (!session || session->getProgress() >= 1.f)
         {
            if (m_sessionQueue.empty())
               return;
            session = m_sessionQueue.front().get();
            // init
            m_utils->autoSubmit<SIntendedSubmitInfo>({ .queue = getGraphicsQueue() },
               [&session](SIntendedSubmitInfo& info) -> bool { return session->init(info); });
            m_resolver->changeSession(std::move(m_sessionQueue.front()));
            m_sessionQueue.pop();
         }

         // Headless rendering
         m_api->startCapture();
         IQueue::SSubmitInfo::SSemaphoreInfo rendered = {};
         {
            auto deferredSubmit = m_renderer->render(session);
            if (deferredSubmit)
            {
               IGPUCommandBuffer* const cb = deferredSubmit;
               if (session->getProgress() >= 1.f)
                  m_resolver->resolve(cb, nullptr);
               rendered = deferredSubmit({});
            }
         }
         m_api->endCapture();
      }
      else
      {
         // GUI mode: check for pending session from double-click
         if (m_pendingSession)
         {
            auto pendingSession = m_pendingSession.get();
            m_utils->autoSubmit<SIntendedSubmitInfo>({ .queue = getGraphicsQueue() },
               [pendingSession](SIntendedSubmitInfo& info) -> bool
               { return pendingSession->init(info); });
            m_resolver->changeSession(std::move(m_pendingSession));

            // New session may have a different scene-default path depth; re-seed.
            m_maxPathDepth = 0u;

            // Reposition UI windows after session change (window will resize)
            if (m_uiManager)
               m_uiManager->resetWindowPositions();
         }
         CSession* session = m_resolver->getActiveSession();

         // Push the camera's pose into the active session so the next render uses it.
         if (session)
         {
            SSensorDynamics dyn = session->getActiveResources().currentSensorState;
            // True inverse of an orthonormal-rotation+translation 3x4 affine: (R^T, -R^T t).
            // nbl::math::linalg::pseudoInverse3x4 leaves the 3x3 part as R rather than R^T (the
            // double-transpose cancels out), so the path tracer would read a wrong column-2 and
            // produce orientation-dependent roll. Roll it ourselves.
            const auto&  V = m_camera.getViewMatrix();
            float32_t3x4 invView;
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
                  invView[i][j] = V[j][i];
            for (int i = 0; i < 3; ++i)
               invView[i][3] =
                  -(invView[i][0] * V[0][3] + invView[i][1] * V[1][3] + invView[i][2] * V[2][3]);
            dyn.invView = invView;
            // Max path depth is app-driven like the camera: lazily seed from the
            // session, then apply our copy every frame so a GUI change shows up as a
            // prev->cur delta in update() and restarts accumulation.
            if (m_maxPathDepth == 0u)
               m_maxPathDepth = uint16_t(dyn.lastPathDepth) + 1u;
            dyn.lastPathDepth = m_maxPathDepth - 1u;
            session->update(dyn);
         }

         // Render session if we have one
         IQueue::SSubmitInfo::SSemaphoreInfo rendered = {};
         if (session)
         {
            m_api->startCapture();
            {
               // One fresh frame after a MIS-mode change clears the old estimator's samples; normal
               // frames accumulate (forceFreshFrame=false). render() overrides keepAccumulating=0 for
               // that frame, so the camera/depth dynamics restart path is untouched.
               CRenderer::STimingScope scope = {};
               scope.forceFreshFrame         = m_restartAccumOnNextFrame;
               m_restartAccumOnNextFrame     = false;
               auto deferredSubmit           = m_renderer->render(session, scope);
               if (deferredSubmit)
               {
                  IGPUCommandBuffer* const cb = deferredSubmit;
                  m_resolver->resolve(cb, nullptr);
                  rendered = deferredSubmit({});
               }
            }
            m_api->endCapture();
         }

         // Acquire swapchain image (may resize window based on session resolution)
         m_presenter->acquire(session);

         // Handle inputs AFTER acquire so ImGui viewport has correct size
         handleInputs();
         if (!keepRunning())
            return;

         if (m_uiManager)
         {
            m_uiManager->setSession(session);
            // Push current view+proj for the debug gizmo (column-major float[16]).
            pushGizmoCameraMatrices();
            m_uiManager->drawWindows();

            const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_presenter->getSemaphore(),
               .value = m_presenter->getPresentCount() + 1 };

            // Render ImGui
            auto* const cb = m_presenter->beginRenderpass();
            if (!m_uiManager->render(cb, drawFinished))
               m_logger->log("UI Render failed", ILogger::ELL_ERROR);
            m_presenter->endRenderpassAndPresent(rendered);
         }
      }
   }

   inline void handleInputs()
   {
      if (m_args.headless)
         return;

      m_inputSystem->getDefaultMouse(&m_mouse);
      m_inputSystem->getDefaultKeyboard(&m_keyboard);

      struct
      {
         std::vector<SMouseEvent>    mouse {};
         std::vector<SKeyboardEvent> keyboard {};
      } capturedEvents;

      const ImGuiIO& io                 = ImGui::GetIO();
      const bool     imguiTakesMouse    = io.WantCaptureMouse;
      const bool     imguiTakesKeyboard = io.WantCaptureKeyboard;

      const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
         std::chrono::steady_clock::now().time_since_epoch());
      m_camera.beginInputProcessing(now);

      static std::chrono::microseconds previousEventTimestamp {};
      m_mouse.consumeEvents(
         [&](const IMouseEventChannel::range_t& events) -> void
         {
            for (const auto& e : events)
            {
               if (e.timeStamp < previousEventTimestamp)
                  continue;
               previousEventTimestamp = e.timeStamp;
               capturedEvents.mouse.emplace_back(e);
            }
            if (!imguiTakesMouse)
               m_camera.mouseProcess(events);
         },
         m_logger.get());
      m_keyboard.consumeEvents(
         [&](const IKeyboardEventChannel::range_t& events) -> void
         {
            for (const auto& e : events)
            {
               if (e.timeStamp < previousEventTimestamp)
                  continue;
               previousEventTimestamp = e.timeStamp;
               capturedEvents.keyboard.emplace_back(e);
            }
            if (!imguiTakesKeyboard)
               m_camera.keyboardProcess(events);
         },
         m_logger.get());

      m_camera.endInputProcessing(now);

      if (m_uiManager)
      {
         const SRange<const SMouseEvent> mouseEvents(
            capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
         const SRange<const SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(),
            capturedEvents.keyboard.data() + capturedEvents.keyboard.size());

         auto*      window         = m_presenter->getWindow();
         const auto cursorPosition = window->getCursorControl()->getPosition();
         const auto mousePosition  = float32_t2(cursorPosition.x, cursorPosition.y) -
            float32_t2(window->getX(), window->getY());

         const nbl::ext::imgui::UI::SUpdateParameters params = { .mousePosition = mousePosition,
            .displaySize    = { window->getWidth(), window->getHeight() },
            .mouseEvents    = mouseEvents,
            .keyboardEvents = keyboardEvents };
         m_uiManager->update(params);
      }
   }

   inline bool keepRunning() override
   {
      if (m_args.benchmark && m_benchmarkDone)
         return false;
      if (m_args.headless)
      {
         if (auto* const currentSession = m_resolver->getActiveSession();
            m_sessionQueue.empty() && (!currentSession || currentSession->getProgress() >= 1.f))
            return false;
         return true;
      }
      else
         return !m_presenter->irrecoverable();
   }

   inline bool onAppTerminated() override { return device_base_t::onAppTerminated(); }

private:
   AppArguments m_args          = {};
   bool         m_benchmarkDone = false;
   //
   smart_refctd_ptr<InputSystem>                     m_inputSystem;
   InputSystem::ChannelReader<IMouseEventChannel>    m_mouse;
   InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;
   //
   smart_refctd_ptr<CWindowPresenter> m_presenter;
   //
   smart_refctd_ptr<CRenderer>          m_renderer;
   smart_refctd_ptr<CBasicRWMCResolver> m_resolver;
   //
   smart_refctd_ptr<CSceneLoader> m_sceneLoader;
   //
   nbl::core::queue<smart_refctd_ptr<CSession>> m_sessionQueue; // for headless mode
   smart_refctd_ptr<CSession> m_pendingSession; // for GUI mode (set by double-clicking sensor)
   //
   smart_refctd_ptr<CScene>          m_currentScene;
   std::string                       m_currentScenePath;
   size_t                            m_currentSensorIdx = 0;
   smart_refctd_ptr<gui::CUIManager> m_uiManager;

   // Free-fly camera. Reset on every scene/sensor change.
   Camera m_camera;

   // App-driven max path depth (number of bounces), applied into the dynamics every
   // frame like the camera. 0 = re-seed from the active session's scene default.
   uint16_t m_maxPathDepth = 0u;
   // Set by onMisModeChanged; makes the next interactive frame force a fresh accumulation start.
   bool m_restartAccumOnNextFrame = false;

   void pushGizmoCameraMatrices()
   {
      if (!m_uiManager)
         return;
      // ImGui's main viewport size matches the window size.
      const ImGuiViewport* vp     = ImGui::GetMainViewport();
      const float          aspect = (vp && vp->Size.y > 0.f) ? (vp->Size.x / vp->Size.y) : 1.6667f;
      const float          fovY   = 60.f * (3.14159265f / 180.f);
      const float          zN = 0.05f, zF = 1000.f;

      const float f        = 1.f / std::tan(fovY * 0.5f);
      float       proj[16] = {};
      proj[0]              = f / aspect;
      proj[5]              = f;
      proj[10]             = (zF + zN) / (zN - zF);
      proj[11]             = -1.f;
      proj[14]             = (2.f * zF * zN) / (zN - zF);

      // Right-handed look-at from the camera pose (right = forward x up), written column-major.
      using vec3     = nbl::hlsl::float32_t3;
      const vec3 pos = nbl::core::convertToHLSLVector(m_camera.getPosition()).xyz;
      const vec3 fwd =
         nbl::hlsl::normalize(nbl::core::convertToHLSLVector(m_camera.getTarget()).xyz - pos);
      const vec3 up0      = nbl::core::convertToHLSLVector(m_camera.getUpVector()).xyz;
      const vec3 right    = nbl::hlsl::normalize(nbl::hlsl::cross(fwd, up0));
      const vec3 up       = nbl::hlsl::cross(right, fwd);
      float      view[16] = {
         right.x,
         up.x,
         -fwd.x,
         0.f,
         right.y,
         up.y,
         -fwd.y,
         0.f,
         right.z,
         up.z,
         -fwd.z,
         0.f,
         -nbl::hlsl::dot(right, pos),
         -nbl::hlsl::dot(up, pos),
         nbl::hlsl::dot(fwd, pos),
         1.f,
      };
      m_uiManager->getSceneWindow().setGizmoCameraMatrices(view, proj);
      if (m_renderer)
         m_uiManager->getSceneWindow().setProbePdfSum(m_renderer->getProbePdfSum());
   }

   // Seed the camera from the current scene/sensor pose.
   void initCameraFromSensor(size_t sensorIdx)
   {
      if (!m_currentScene)
         return;
      const auto sensors = m_currentScene->getSensors();
      if (sensorIdx >= sensors.size())
         return;
      const auto&        s    = sensors[sensorIdx];
      const auto&        absT = s.mutableDefaults.absoluteTransform;
      const vectorSIMDf  pos(absT[0][3], absT[1][3], absT[2][3]);
      const vectorSIMDf  fwd(-absT[0][2], -absT[1][2], -absT[2][2]);
      const vectorSIMDf  target = pos + fwd;
      const vectorSIMDf  upHint(0.f, 1.f, 0.f);
      const float32_t4x4 proj      = math::linalg::diagonal<float32_t4x4>(1.f);
      const float        moveSpeed = s.dynamicDefaults.moveSpeed * 0.005f;
      const float        rotateSpeed =
         nbl::hlsl::isnan(s.dynamicDefaults.rotateSpeed) ? 1.f : s.dynamicDefaults.rotateSpeed;
      m_camera = Camera(pos, target, proj, moveSpeed, rotateSpeed, upHint, upHint);
      m_camera.mapKeysToWASD();
      if (m_uiManager)
         m_uiManager->getSceneWindow().setCameraMoveSpeed(moveSpeed);
   }
};
NBL_MAIN_FUNC(PathTracingApp)