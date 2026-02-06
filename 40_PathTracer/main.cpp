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

#include "nlohmann/json.hpp"

#include <unordered_map>


using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;
using namespace nbl::this_example;

// TODO: move to argument parsing class
struct AppArguments
{
	bool headless; // set in onAppInitialized() for now
};


class PathTracingApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;

	// TODO: move to Nabla proper
	static inline void jsonizeGitInfo(nlohmann::json& target, const nbl::gtml::GitInfo& info)
	{
		target["isPopulated"] = info.isPopulated;
		if (info.hasUncommittedChanges.has_value())
			target["hasUncommittedChanges"] = info.hasUncommittedChanges.value();
		else
			target["hasUncommittedChanges"] = "UNKNOWN, BUILT WITHOUT DIRTY-CHANGES CAPTURE";

		target["commitAuthorName"] = info.commitAuthorName;
		target["commitAuthorEmail"] = info.commitAuthorEmail;
		target["commitHash"] = info.commitHash;
		target["commitShortHash"] = info.commitShortHash;
		target["commitDate"] = info.commitDate;
		target["commitSubject"] = info.commitSubject;
		target["commitBody"] = info.commitBody;
		target["describe"] = info.describe;
		target["branchName"] = info.branchName;
		target["latestTag"] = info.latestTag;
		target["latestTagName"] = info.latestTagName;
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
	inline PathTracingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
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
		return retval.unionWith(CRenderer::PreferredDeviceFeatures());
	}

	inline SPhysicalDeviceLimits getRequiredDeviceLimits() const override
	{
		auto retval = device_base_t::getRequiredDeviceLimits();
		// TODO: need union/superset
		retval.shaderStorageImageReadWithoutFormat = true;
		return retval;
	}

	inline void filterDevices(nbl::core::set<IPhysicalDevice*>& physicalDevices) const override
	{
		device_base_t::filterDevices(physicalDevices);
		std::erase_if(physicalDevices, [&](const IPhysicalDevice* device)->bool
			{
				const auto& props = device->getMemoryProperties();
				uint64_t largestVRAMHeap = 0;
				using heap_flags_e = IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS;
				for (uint32_t h = 0; h < props.memoryHeapCount; h++)
					if (const auto& heap = props.memoryHeaps[h]; heap.flags.hasFlags(heap_flags_e::EMHF_DEVICE_LOCAL_BIT))
						largestVRAMHeap = nbl::hlsl::max(largestVRAMHeap, heap.size);
				const auto typeBits = device->getDirectVRAMAccessMemoryTypeBits();
				for (uint32_t t = 0; t < props.memoryTypeCount; t++)
					if (((typeBits >> t) & 0x1u) && props.memoryHeaps[props.memoryTypes[t].heapIndex].size == largestVRAMHeap)
						return false;
				m_logger->log("Filtering out Device %p (%s) due to lack of ReBAR", ILogger::ELL_WARNING, device, device->getProperties().deviceName);
				return true;
			}
		);
	}

	inline nbl::core::vector<SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (m_args.headless)
			return {};

		if (!m_presenter)
		{
			const_cast<std::remove_reference_t<decltype(m_presenter)>&>(m_presenter) = CWindowPresenter::create({
				{
					.assMan = m_assetMgr,
					.logger = smart_refctd_ptr(m_logger)
				},
				{
					.winMgr = m_winMgr
				},
				m_api,
				make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem),smart_refctd_ptr(m_logger)),
				"Path Tracer"
				});
		}

		if (m_presenter)
		{
			const auto* presenter = m_presenter.get();
			return { {presenter->getSurface()/*,EQF_NONE*/} };
		}

		return {};
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// TODO: parse the arguments
		m_args.headless = false;

		if (!m_args.headless)
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

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
		m_renderer = CRenderer::create({
			{
				.graphicsQueue = getGraphicsQueue(),
				.computeQueue = getComputeQueue(),
				.uploadQueue = getTransferUpQueue(),
				.utilities = smart_refctd_ptr(m_utils)
			},
			"TODO Sample sequence cache",
			m_assetMgr.get()
			});
		if (!m_renderer)
			return logFail("Failed to create CRenderer");

		//
		if (!m_args.headless && !m_presenter->init(m_renderer.get()))
			return logFail("Failed to initialize CWindowPresenter");

		//
		m_resolver = CBasicRWMCResolver::create({
			{},
			m_renderer.get()
			});
		if (!m_resolver)
			return logFail("Failed to create CBasicRWMCResolver");

		// set up the scene loader
		m_sceneLoader = CSceneLoader::create({
			{
				.assMan = smart_refctd_ptr(m_assetMgr),
				.logger = smart_refctd_ptr(m_logger)
			}
			});

		// TODO: tmp code
		{
			m_api->startCapture();
			m_currentScenePath = (sharedInputCWD / "mitsuba/daily_pt.xml").string();
			m_currentScene = m_renderer->createScene({
					.load = m_sceneLoader->load({
					.relPath = m_currentScenePath,
					.workingDirectory = localOutputCWD
				}),
				.converter = nullptr
				});
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
		}

		// Initialize UI Manager (non-headless only)
		if (!m_args.headless)
		{
			m_uiManager = gui::CUIManager::create({ .assetManager = smart_refctd_ptr(m_assetMgr),.utilities = smart_refctd_ptr(m_utils),.transferQueue = getGraphicsQueue(),.logger = smart_refctd_ptr(m_logger) });
			if (!m_uiManager)
				return logFail("Failed to create CUIManager");

			gui::CUIManager::SInitParams uiInitParams = {
				.renderpass = m_presenter->getRenderpass(),

				.onSensorSelected = [this](size_t sensorIdx) {
					// Create a new session from the selected sensor (GUI mode)
					if (m_currentScene)
					{
						const auto sensors = m_currentScene->getSensors();
						if (sensorIdx < sensors.size())
						{
							auto newSession = m_currentScene->createSession({
								{.mode = CSession::RenderMode::Debug},
								&sensors[sensorIdx]
							});
							if (newSession)
							{
								m_pendingSession = std::move(newSession);
							}
						}
					}
				},
				.onLoadSceneRequested = [this](const std::string& path) {
					if (path.empty())
						return;

					m_logger->log("Loading scene: %s", ILogger::ELL_INFO, path.c_str());

					// Load the scene
					auto loadResult = m_sceneLoader->load({
						.relPath = path,
						.workingDirectory = localOutputCWD
					});

					if (!loadResult)
					{
						m_logger->log("Failed to load scene: %s", ILogger::ELL_ERROR, path.c_str());
						return;
					}

					// Create the scene
					auto newScene = m_renderer->createScene({
						.load = std::move(loadResult),
						.converter = nullptr
					});

					if (!newScene)
					{
						m_logger->log("Failed to create scene from: %s", ILogger::ELL_ERROR, path.c_str());
						return;
					}

					// Update current scene
					m_currentScene = std::move(newScene);
					m_currentScenePath = path;

					// Update UI
					if (m_uiManager)
						m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);

					m_logger->log("Scene loaded successfully: %s", ILogger::ELL_INFO, path.c_str());
				},
				.onReloadSceneRequested = [this]() {
					if (m_currentScenePath.empty())
					{
						m_logger->log("No scene to reload", ILogger::ELL_WARNING);
						return;
					}

					m_logger->log("Reloading scene: %s", ILogger::ELL_INFO, m_currentScenePath.c_str());

					// Reload the scene
					auto loadResult = m_sceneLoader->load({
						.relPath = m_currentScenePath,
						.workingDirectory = localOutputCWD
					});

					if (!loadResult)
					{
						m_logger->log("Failed to reload scene: %s", ILogger::ELL_ERROR, m_currentScenePath.c_str());
						return;
					}

					auto newScene = m_renderer->createScene({
						.load = std::move(loadResult),
						.converter = nullptr
					});

					if (!newScene)
					{
						m_logger->log("Failed to create scene from: %s", ILogger::ELL_ERROR, m_currentScenePath.c_str());
						return;
					}

					m_currentScene = std::move(newScene);

					if (m_uiManager)
						m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);

					m_logger->log("Scene reloaded successfully", ILogger::ELL_INFO);
				},
				// Session callbacks
				.onRenderModeChanged = [this](CSession::RenderMode mode, CSession* session) {
					// Recreate session with new mode
					if (session)
					{
						const CSession::SConstructionParams& params = session->getConstructionParams();
						auto creationParams = params; // Copy params
						creationParams.mode = mode;

						// TODO: Actually recreate the session. For now just log.
						m_logger->log("Render mode changed to %d (Recreation TODO)", ILogger::ELL_INFO, mode);
					}
				},
				.onResolutionChanged = [this](uint16_t w, uint16_t h) {
					m_logger->log("Resolution changed to %dx%d (TODO)", ILogger::ELL_INFO, w, h);
				},
				.onMutablesChanged = [this](const SSensorDynamics& dyn, CSession* session) {
					session->update(dyn);
					m_logger->log("Mutables changed (Reset TODO)", ILogger::ELL_INFO);
				},
				.onDynamicsChanged = [this](const SSensorDynamics& dyn, CSession* session) {
					session->update(dyn);
				},
				.onBufferSelected = [this](int id) {
					m_logger->log("Buffer %d selected (TODO)", ILogger::ELL_INFO, id);
				}
			};

			if (!m_uiManager->init(uiInitParams))
				return logFail("Failed to initialize CUIManager");


			// Set up UI with the initially loaded scene
			if (m_currentScene)
				m_uiManager->setScene(m_currentScene.get(), m_currentScenePath);

			// Create initial session from first sensor so GUI has something to display
			if (m_currentScene && !m_currentScene->getSensors().empty())
			{
				const auto& sensors = m_currentScene->getSensors();
				auto initialSession = m_currentScene->createSession({
					{.mode = CSession::RenderMode::Debug},
					&sensors.front()
					});

				m_pendingSession = std::move(initialSession);
			}
		}

		return true;
	}

	inline void workLoopBody() override
	{
		if (m_args.headless)
		{
			CSession* session = m_resolver->getActiveSession();
			while (!session || session->getProgress() >= 1.f)
			{
				if (m_sessionQueue.empty())
					return;
				session = m_sessionQueue.front().get();
				// init
				m_utils->autoSubmit<SIntendedSubmitInfo>({ .queue = getGraphicsQueue() }, [&session](SIntendedSubmitInfo& info)->bool
					{
						return session->init(info.getCommandBufferForRecording()->cmdbuf);
					}
				);
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
				m_utils->autoSubmit<SIntendedSubmitInfo>({ .queue = getGraphicsQueue() }, [pendingSession](SIntendedSubmitInfo& info)->bool
					{
						return pendingSession->init(info.getCommandBufferForRecording()->cmdbuf);
					}
				);
				m_resolver->changeSession(std::move(m_pendingSession));

				// Reposition UI windows after session change (window will resize)
				if (m_uiManager)
					m_uiManager->resetWindowPositions();
			}
			CSession* session = m_resolver->getActiveSession();

			// Render session if we have one
			IQueue::SSubmitInfo::SSemaphoreInfo rendered = {};
			if (session)
			{
				m_api->startCapture();
				{
					auto deferredSubmit = m_renderer->render(session);
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
				m_uiManager->drawWindows();

				const ISemaphore::SWaitInfo drawFinished = {
					.semaphore = m_presenter->getSemaphore(),
					.value = m_presenter->getPresentCount() + 1
				};

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
			std::vector<SMouseEvent> mouse{};
			std::vector<SKeyboardEvent> keyboard{};
		} capturedEvents;

		static std::chrono::microseconds previousEventTimestamp{};
		m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				for (const auto& e : events) // here capture
				{
					if (e.timeStamp < previousEventTimestamp)
						continue;

					previousEventTimestamp = e.timeStamp;
					capturedEvents.mouse.emplace_back(e);

				}
			}, m_logger.get()
				);
		m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				for (const auto& e : events) // here capture
				{
					if (e.timeStamp < previousEventTimestamp)
						continue;

					previousEventTimestamp = e.timeStamp;
					capturedEvents.keyboard.emplace_back(e);
				}
			}, m_logger.get()
				);

		if (m_uiManager)
		{
			const SRange<const SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			const SRange<const SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());

			auto* window = m_presenter->getWindow();
			const auto cursorPosition = window->getCursorControl()->getPosition();
			const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(window->getX(), window->getY());

			const nbl::ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = mousePosition,
				.displaySize = { window->getWidth(), window->getHeight() },
				.mouseEvents = mouseEvents,
				.keyboardEvents = keyboardEvents
			};
			m_uiManager->update(params);
		}
	}

	inline bool keepRunning() override
	{
		if (m_args.headless)
		{
			if (auto* const currentSession = m_resolver->getActiveSession(); m_sessionQueue.empty() && (!currentSession || currentSession->getProgress() >= 1.f))
				return false;
			return true;
		}
		else
			return !m_presenter->irrecoverable();
	}

	inline bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

private:
	AppArguments m_args = {};
	//
	smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;
	// 
	smart_refctd_ptr<CWindowPresenter> m_presenter;
	//
	smart_refctd_ptr<CRenderer> m_renderer;
	smart_refctd_ptr<CBasicRWMCResolver> m_resolver;
	//
	smart_refctd_ptr<CSceneLoader> m_sceneLoader;
	//
	nbl::core::queue<smart_refctd_ptr<CSession>> m_sessionQueue; // for headless mode
	smart_refctd_ptr<CSession> m_pendingSession; // for GUI mode (set by double-clicking sensor)
	//
	smart_refctd_ptr<CScene> m_currentScene;
	std::string m_currentScenePath;
	smart_refctd_ptr<gui::CUIManager> m_uiManager;



};
NBL_MAIN_FUNC(PathTracingApp)