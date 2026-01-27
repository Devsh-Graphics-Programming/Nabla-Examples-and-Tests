// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

#include "nbl/examples/examples.hpp"

#include "renderer/CRenderer.h"
#include "renderer/resolve/CBasicRWMCResolver.h"
#include "renderer/present/CWindowPresenter.h"

#include "nlohmann/json.hpp"


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
	bool headless = false;
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
			jsonizeGitInfo(modules["nabla"],nbl::gtml::nabla_git_info);
			jsonizeGitInfo(modules["dxc"],nbl::gtml::dxc_git_info);

			m_logger->log("Build Info:\n%s",ILogger::ELL_INFO,j.dump(4).c_str());
		}


	public:
		inline PathTracingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)	{}

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
				auto scene_daily_pt = m_renderer->createScene({
						.load = m_sceneLoader->load({
						.relPath = sharedInputCWD/"mitsuba/daily_pt.xml",
						.workingDirectory = localOutputCWD 
					}),
					.converter = nullptr
				});
				// the UI would have you load the zip first, then present a dropdown of what to load
				// but still need to support archive mount for cmdline load
		#if 0 // this particular zip goes down an unsupported path in our zip loader
				auto scene_bedroom = m_sceneLoader->load({
					.relPath = sharedInputCWD/"mitsuba/bedroom.zip/scene.xml",
					.workingDirectory = localOutputCWD
				});
		#endif
				m_api->endCapture();

				// quick test code
				nbl::core::vector<CSession::sensor_t> sensors(3,scene_daily_pt->getSensors().front());
				{
					sensors[1].constants.width = 640;
					sensors[1].constants.height = 360;
					sensors[1].mutableDefaults.cropOffsetX = 0;
					sensors[1].mutableDefaults.cropOffsetY = 0;
					sensors[1].mutableDefaults.cropWidth = 0;
					sensors[1].mutableDefaults.cropHeight = 0;
				}
				{
					sensors[2].mutableDefaults.cropWidth = 5120;
					sensors[2].mutableDefaults.cropHeight = 2880;
					sensors[2].mutableDefaults.cropOffsetX = 128;
					sensors[2].mutableDefaults.cropOffsetY = 128;
					sensors[2].constants.width = sensors[2].mutableDefaults.cropWidth+2*sensors[2].mutableDefaults.cropOffsetX;
					sensors[2].constants.height = sensors[2].mutableDefaults.cropHeight+2*sensors[2].mutableDefaults.cropOffsetY;
				}
				for (const auto& sensor : sensors)
					m_sessionQueue.push(
						scene_daily_pt->createSession({
							{.mode=CSession::RenderMode::Debug},&sensor
						})
					);
			}

			return true;

#if 0 // ui
		// gui descriptor setup
		{
			using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
			{
				IGPUSampler::SParams params;
				params.AnisotropicFilter = 1u;
				params.TextureWrapU = ETC_REPEAT;
				params.TextureWrapV = ETC_REPEAT;
				params.TextureWrapW = ETC_REPEAT;

				m_ui.samplers.gui = m_device->createSampler(params);
				m_ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
			}

			std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
			for (auto& it : immutableSamplers)
				it = smart_refctd_ptr(m_ui.samplers.scene);

			immutableSamplers[nbl::ext::imgui::UI::FontAtlasTexId] = smart_refctd_ptr(m_ui.samplers.gui);

			nbl::ext::imgui::UI::SCreationParameters params;

			params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
			params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
			params.assetManager = m_assetMgr;
			params.pipelineCache = nullptr;
			params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
			params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
			params.streamingBuffer = nullptr;
			params.subpassIx = 0u;
			params.transfer = getGraphicsQueue();
			params.utilities = m_utils;
			{
				m_ui.manager = ext::imgui::UI::create(std::move(params));

				// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
				const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
				const auto& params = m_ui.manager->getCreationParameters();

				IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = MaxUITextureCount;
				descriptorPoolInfo.maxSets = 1u;
				descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

				m_guiDescriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
				assert(m_guiDescriptorSetPool);

				m_guiDescriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
				assert(m_ui.descriptorSet);
			}
		}

		m_ui.manager->registerListener(
			[this]() -> void {
				ImGuiIO& io = ImGui::GetIO();

				m_camera.setProjectionMatrix([&]()
					{
						static matrix4SIMD projection;

						projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
							core::radians(m_cameraSetting.fov),
							io.DisplaySize.x / io.DisplaySize.y,
							m_cameraSetting.zNear,
							m_cameraSetting.zFar);

						return projection;
					}());

				ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

				// create a window and insert the inspector
				ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
				ImGui::Begin("Controls");

				ImGui::SameLine();

				ImGui::Text("Camera");

				ImGui::SliderFloat("Move speed", &m_cameraSetting.moveSpeed, 0.1f, 10.f);
				ImGui::SliderFloat("Rotate speed", &m_cameraSetting.rotateSpeed, 0.1f, 10.f);
				ImGui::SliderFloat("Fov", &m_cameraSetting.fov, 20.f, 150.f);
				ImGui::SliderFloat("zNear", &m_cameraSetting.zNear, 0.1f, 100.f);
				ImGui::SliderFloat("zFar", &m_cameraSetting.zFar, 110.f, 10000.f);
				Light m_oldLight = m_light;
				int light_type = m_light.type;
				ImGui::ListBox("LightType", &light_type, s_lightTypeNames, ELT_COUNT);
				m_light.type = static_cast<E_LIGHT_TYPE>(light_type);
				if (m_light.type == ELT_DIRECTIONAL)
				{
					ImGui::SliderFloat3("Light Direction", &m_light.direction.x, -1.f, 1.f);
				}
				else if (m_light.type == ELT_POINT)
				{
					ImGui::SliderFloat3("Light Position", &m_light.position.x, -20.f, 20.f);
				}
				else if (m_light.type == ELT_SPOT)
				{
					ImGui::SliderFloat3("Light Direction", &m_light.direction.x, -1.f, 1.f);
					ImGui::SliderFloat3("Light Position", &m_light.position.x, -20.f, 20.f);

					float32_t dOuterCutoff = hlsl::degrees(acos(m_light.outerCutoff));
					if (ImGui::SliderFloat("Light Outer Cutoff", &dOuterCutoff, 0.0f, 45.0f))
					{
						m_light.outerCutoff = cos(hlsl::radians(dOuterCutoff));
					}
				}
				ImGui::Checkbox("Use Indirect Command", &m_useIndirectCommand);
				if (m_light != m_oldLight)
				{
					m_frameAccumulationCounter = 0;
				}

				ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);

				ImGui::End();
			}
		);
#endif
		}

#if 0 // gui
	bool updateGUIDescriptorSet()
	{
		// texture atlas, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
		static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
		static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

		descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

		for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
		{
			writes[i].dstSet = m_ui.descriptorSet.get();
			writes[i].binding = 0u;
			writes[i].arrayElement = i;
			writes[i].count = 1u;
		}
		writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;

		return m_device->updateDescriptorSets(writes, {});
	}
#endif

		inline void workLoopBody() override
		{
			CSession* session;
			volatile bool skip = true; // skip using the debugger
			for (session=m_resolver->getActiveSession(); !session || session->getProgress()>=1.f || skip;)
			{
				skip = false;
				if (m_sessionQueue.empty())
				{
					if (!m_args.headless)
						handleInputs();
					return;
				}
				session = m_sessionQueue.front().get();
				// init
				m_utils->autoSubmit<SIntendedSubmitInfo>({.queue=getGraphicsQueue()},[&session](SIntendedSubmitInfo& info)->bool
					{
						return session->init(info.getCommandBufferForRecording()->cmdbuf);
					}
				);
				m_resolver->changeSession(std::move(m_sessionQueue.front()));
				m_sessionQueue.pop();
			}

			m_api->startCapture();
			IQueue::SSubmitInfo::SSemaphoreInfo rendered = {};
			{
				auto deferredSubmit = m_renderer->render(session);
				if (deferredSubmit)
				{
					IGPUCommandBuffer* const cb = deferredSubmit;
					if (!m_args.headless || session->getProgress()>=1.f)
					{
						m_resolver->resolve(cb,nullptr);
					}
					rendered = deferredSubmit({});
				}
			}
			m_api->endCapture();

			if (m_args.headless)
				return;
			handleInputs();
			if (!keepRunning())
				return;

			m_presenter->acquire(session);
			auto* const cb = m_presenter->beginRenderpass();
			{
				// can do additional stuff like ImGUI work here
			}
			m_presenter->endRenderpassAndPresent(rendered);
#if 0 // gui

// ...
		const auto uiParams = m_ui.manager->getCreationParameters();
		auto* uiPipeline = m_ui.manager->getPipeline();
		cmdbuf->bindGraphicsPipeline(uiPipeline);
		cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
		ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };
		m_ui.manager->render(cmdbuf, waitInfo);

			{
				{

					updateGUIDescriptorSet();

				}
			}
#endif
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

//			const auto& io = ImGui::GetIO();
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
#if 0 // ui
			const SRange<const SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			const SRange<const SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
			const auto cursorPosition = m_window->getCursorControl()->getPosition();
			const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY());

			const nbl::ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = mousePosition,
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = mouseEvents,
				.keyboardEvents = keyboardEvents
			};
			m_ui.manager->update(params);
#endif
		}

		inline bool keepRunning() override
		{
			if (m_args.headless)
			{
				if (auto* const currentSession=m_resolver->getActiveSession(); m_sessionQueue.empty() && (!currentSession || currentSession->getProgress()>=1.f))
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
		nbl::core::queue<smart_refctd_ptr<CSession>> m_sessionQueue;

#if 0 // gui
	struct C_UI
	{
		nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

		struct
		{
			core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
		} samplers;

		core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
	} m_ui;
	core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;
#endif

};
NBL_MAIN_FUNC(PathTracingApp)