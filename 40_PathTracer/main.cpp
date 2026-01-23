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

		inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = device_base_t::getRequiredDeviceFeatures();
			return retval.unionWith(CRenderer::RequiredDeviceFeatures());
		}

		inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
		{
			auto retval = device_base_t::getPreferredDeviceFeatures();
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
						.winMgr = m_winMgr,
						.logger = smart_refctd_ptr(m_logger)
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
			m_args = {};

			if (!m_args.headless)
				m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
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

			auto session = scene_daily_pt->createSession({
				{.mode=CSession::RenderMode::Debug},
				scene_daily_pt->getSensors().data()
			});

			// temporary test
			m_presenter->acquire({},session.get());
			{
				auto cb = m_renderer->getConstructionParams().commandBuffers[0].get();
				cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				session->init(cb);
//				renderer->render(cb);
				m_resolver->resolve(cb,nullptr);
//				m_presenter->acquire({},session);
//				m_presenter->beginRenderpass(cb);
//				m_presenter->endRenderpassAndPresent(cb);

				// TODO: submit
			}
			session->deinit();
			scene_daily_pt = nullptr;

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
#if 0
		// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
		const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
		// We block for semaphores for 2 reasons here:
		  // A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
		  // B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
		if (m_realFrameIx >= framesInFlight)
		{
			const ISemaphore::SWaitInfo cbDonePending[] =
			{
			  {
				.semaphore = m_semaphore.get(),
				.value = m_realFrameIx + 1 - framesInFlight
			  }
			};
			if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
				return;
		}
		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		m_api->startCapture();

		update();

		auto queue = getGraphicsQueue();
		auto cmdbuf = m_cmdBufs[resourceIx].get();

		if (!keepRunning())
			return;

		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		const auto viewMatrix = m_camera.getViewMatrix();
		const auto projectionMatrix = m_camera.getProjectionMatrix();
		const auto viewProjectionMatrix = m_camera.getConcatenatedMatrix();

		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
		modelMatrix.setRotation(quaternion(0, 0, 0));

		core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
		if (m_cachedModelViewProjectionMatrix != modelViewProjectionMatrix)
		{
			m_frameAccumulationCounter = 0;
			m_cachedModelViewProjectionMatrix = modelViewProjectionMatrix;
		}
		core::matrix4SIMD invModelViewProjectionMatrix;
		modelViewProjectionMatrix.getInverseTransform(invModelViewProjectionMatrix);

		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
			imageBarriers[0].barrier = {
			   .dep = {
				 .srcStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, // previous frame read from framgent shader
				 .srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
				 .dstStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
				 .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
			  }
			};
			imageBarriers[0].image = m_hdrImage.get();
			imageBarriers[0].subresourceRange = {
			  .aspectMask = IImage::EAF_COLOR_BIT,
			  .baseMipLevel = 0u,
			  .levelCount = 1u,
			  .baseArrayLayer = 0u,
			  .layerCount = 1u
			};
			imageBarriers[0].oldLayout = m_frameAccumulationCounter == 0 ? IImage::LAYOUT::UNDEFINED : IImage::LAYOUT::READ_ONLY_OPTIMAL;
			imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
		}

		// Trace Rays Pass
		{
			SPushConstants pc;
			pc.light = m_light;
			pc.proceduralGeomInfoBuffer = m_proceduralGeomInfoBuffer->getDeviceAddress();
			pc.triangleGeomInfoBuffer = m_triangleGeomInfoBuffer->getDeviceAddress();
			pc.frameCounter = m_frameAccumulationCounter;
			const core::vector3df camPos = m_camera.getPosition().getAsVector3df();
			pc.camPos = { camPos.X, camPos.Y, camPos.Z };
			memcpy(&pc.invMVP, invModelViewProjectionMatrix.pointer(), sizeof(pc.invMVP));

			cmdbuf->bindRayTracingPipeline(m_rayTracingPipeline.get());
			cmdbuf->setRayTracingPipelineStackSize(m_rayTracingStackSize);
			cmdbuf->pushConstants(m_rayTracingPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING, 0, sizeof(SPushConstants), &pc);
			cmdbuf->bindDescriptorSets(EPBP_RAY_TRACING, m_rayTracingPipeline->getLayout(), 0, 1, &m_rayTracingDs.get());
			if (m_useIndirectCommand)
			{
				cmdbuf->traceRaysIndirect(
					SBufferBinding<const IGPUBuffer>{
					.offset = 0,
						.buffer = m_indirectBuffer,
				});
			}
			else
			{
				cmdbuf->traceRays(
					m_shaderBindingTable.raygenGroupRange,
					m_shaderBindingTable.missGroupsRange, m_shaderBindingTable.missGroupsStride,
					m_shaderBindingTable.hitGroupsRange, m_shaderBindingTable.hitGroupsStride,
					m_shaderBindingTable.callableGroupsRange, m_shaderBindingTable.callableGroupsStride,
					WIN_W, WIN_H, 1);
			}
		}

		// pipeline barrier
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
			imageBarriers[0].barrier = {
			  .dep = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			  }
			};
			imageBarriers[0].image = m_hdrImage.get();
			imageBarriers[0].subresourceRange = {
			  .aspectMask = IImage::EAF_COLOR_BIT,
			  .baseMipLevel = 0u,
			  .levelCount = 1u,
			  .baseArrayLayer = 0u,
			  .layerCount = 1u
			};
			imageBarriers[0].oldLayout = IImage::LAYOUT::GENERAL;
			imageBarriers[0].newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
		}

// ...
		const auto uiParams = m_ui.manager->getCreationParameters();
		auto* uiPipeline = m_ui.manager->getPipeline();
		cmdbuf->bindGraphicsPipeline(uiPipeline);
		cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
		ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };
		m_ui.manager->render(cmdbuf, waitInfo);
// ...


			{
			  {
				.semaphore = m_semaphore.get(),
				.value = ++m_realFrameIx,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			  }
			};
			{
				{

					updateGUIDescriptorSet();

				}
			}
			=
		}
		m_api->endCapture();
		m_frameAccumulationCounter++;
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
				return true; //  TODO: till renders are complete
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

	uint64_t m_realFrameIx = 0;
	uint32_t m_frameAccumulationCounter = 0;
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
	uint64_t m_rayTracingStackSize;

};
NBL_MAIN_FUNC(PathTracingApp)