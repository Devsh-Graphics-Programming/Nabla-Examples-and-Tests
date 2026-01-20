// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "common.hpp"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

#ifdef NBL_BUILD_MITSUBA_LOADER
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#endif

#include "nbl/ext/DebugDraw/CDrawAABB.h"
#include "nbl/ext/ImGui/ImGui.h"

class GeometryInspectorApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
		using device_base_t = MonoWindowApplication;
		using asset_base_t = BuiltinResourcesApplication;

		enum DrawBoundingBoxMode
		{
		  DBBM_NONE,
			DBBM_AABB,
			DBBM_OBB,
			DBBM_COUNT
		};

	public:
		inline GeometryInspectorApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
			device_base_t({1280,720}, EF_D32_SFLOAT, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
		#ifdef NBL_BUILD_MITSUBA_LOADER
			m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CSerializedLoader>());
		#endif
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			for (auto i=0u; i<MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
					return logFail("Couldn't create Command Buffer!");
			}
			

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(),scRes->getRenderpass(),0,{});
			if (!m_renderer)
				return logFail("Failed to create renderer!");

      auto* renderpass = scRes->getRenderpass();
			
			{
				ext::debug_draw::DrawAABB::SCreationParameters params = {};
				params.assetManager = m_assetMgr;
				params.transfer = getTransferUpQueue();
				params.drawMode = ext::debug_draw::DrawAABB::ADM_DRAW_BATCH;
				params.batchPipelineLayout = ext::debug_draw::DrawAABB::createDefaultPipelineLayout(m_device.get());
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
				params.utilities = m_utils;
				m_bbRenderer = ext::debug_draw::DrawAABB::create(std::move(params));
			}
			
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
						static hlsl::float32_t4x4 projection;

						projection = hlsl::math::thin_lens::rhPerspectiveFovMatrix(
							core::radians(m_cameraSetting.fov),
							io.DisplaySize.x / io.DisplaySize.y,
							m_cameraSetting.zNear,
							m_cameraSetting.zFar);

						return projection;
					}());

				ImGuizmo::SetOrthographic(false);
				ImGuizmo::BeginFrame();

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


				ImGui::Text("Inspector");
				ImGui::ListBox("Selected polygon", &m_selectedMesh, 
					[](void* userData, int index) -> const char* {
						auto* meshInstances = reinterpret_cast<MeshInstance*>(userData);
						return meshInstances[index].name.data();
					}, 
					m_meshInstances.data(),
					m_meshInstances.size());

				ImGui::Checkbox("Draw AABB", &m_shouldDrawAABB);
				ImGui::Checkbox("Draw OBB", &m_shouldDrawOBB);
				if (ImGuizmo::IsUsing())
				{
					ImGui::Text("Using gizmo");
				}
				else
				{
					ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
					ImGui::SameLine();
					ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
					ImGui::SameLine();
					ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
					ImGui::SameLine();
					ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
				}
				ImGui::Separator();

				static struct
				{
					hlsl::float32_t4x4 view, projection, model;
				} imguizmoM16InOut;

				ImGuizmo::SetID(0u);

				auto& selectedInstance = m_renderer->getInstance(m_selectedMesh);

				imguizmoM16InOut.view = hlsl::transpose(hlsl::math::linalg::promote_affine<4, 4, 3, 4>(m_camera.getViewMatrix()));
				imguizmoM16InOut.projection = hlsl::transpose(m_camera.getProjectionMatrix());
				imguizmoM16InOut.projection[1][1] *= -1.f; // Flip y coordinates. https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/
				imguizmoM16InOut.model = hlsl::transpose(hlsl::math::linalg::promote_affine<4, 4, 3, 4>(selectedInstance.world));
				{
					m_transformParams.enableViewManipulate = true;
					EditTransform(&imguizmoM16InOut.view[0][0], &imguizmoM16InOut.projection[0][0], &imguizmoM16InOut.model[0][0], m_transformParams);
				}
				selectedInstance.world = hlsl::float32_t3x4(hlsl::transpose(imguizmoM16InOut.model));

				ImGui::End();
			});
			//
			if (!reloadModel())
				return false;

			m_camera.mapKeysToArrows();

			onAppInitializedFinish();
			return true;
		}

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

    inline void update(const std::chrono::microseconds nextPresentationTimestamp)
    {
      m_camera.setMoveSpeed(m_cameraSetting.moveSpeed);
      m_camera.setRotateSpeed(m_cameraSetting.rotateSpeed);

      static std::chrono::microseconds previousEventTimestamp{};

      m_inputSystem->getDefaultMouse(&m_mouse);
      m_inputSystem->getDefaultKeyboard(&m_keyboard);

      struct
      {
        std::vector<SMouseEvent> mouse{};
        std::vector<SKeyboardEvent> keyboard{};
      } capturedEvents;

      m_camera.beginInputProcessing(nextPresentationTimestamp);
      {
        const auto& io = ImGui::GetIO();
        m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
          {
            if (!io.WantCaptureMouse)
              m_camera.mouseProcess(events); // don't capture the events, only let m_camera handle them with its impl

            for (const auto& e : events) // here capture
            {
              if (e.timeStamp < previousEventTimestamp)
                continue;

              previousEventTimestamp = e.timeStamp;
              capturedEvents.mouse.emplace_back(e);

            }
          }, m_logger.get());

				bool reload = false;
        m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
          {
            if (!io.WantCaptureKeyboard)
              m_camera.keyboardProcess(events); // don't capture the events, only let m_camera handle them with its impl

            for (const auto& e : events) // here capture
            {
              if (e.timeStamp < previousEventTimestamp)
                continue;
              if (e.keyCode == E_KEY_CODE::EKC_R && e.action == SKeyboardEvent::ECA_RELEASED)
                reload = true;

              previousEventTimestamp = e.timeStamp;
              capturedEvents.keyboard.emplace_back(e);
            }
          }, m_logger.get());
				if (reload) reloadModel();

      }
      m_camera.endInputProcessing(nextPresentationTimestamp);

      const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
      const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
      const auto cursorPosition = m_window->getCursorControl()->getPosition();
      const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY());

      const ext::imgui::UI::SUpdateParameters params =
      {
        .mousePosition = mousePosition,
        .displaySize = { m_window->getWidth(), m_window->getHeight() },
        .mouseEvents = mouseEvents,
        .keyboardEvents = keyboardEvents
      };

      m_ui.manager->update(params);
    }

		inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
		{
      update(nextPresentationTimestamp);

			//
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
					const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
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
					cb->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

					const SViewport viewport = {
						.x = static_cast<float>(currentRenderArea.offset.x),
						.y = static_cast<float>(currentRenderArea.offset.y),
						.width = static_cast<float>(currentRenderArea.extent.width),
						.height = static_cast<float>(currentRenderArea.extent.height)
					};
					cb->setViewport(0u,1u,&viewport);
		
					cb->setScissor(0u,1u,&currentRenderArea);
				}

				// draw scene
				float32_t3x4 viewMatrix = m_camera.getViewMatrix();
				float32_t4x4 viewProjMatrix = m_camera.getConcatenatedMatrix();

 				m_renderer->render(cb,CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix));

        const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_semaphore.get(),.value = m_realFrameIx + 1u };
				const auto& renderInstance = m_renderer->getInstance(m_selectedMesh);
				const auto& meshInstance = m_meshInstances[m_selectedMesh];
				core::vector<ext::debug_draw::InstanceData> debugDrawInstances;
				debugDrawInstances.reserve(2);
        const auto world4x4 = float32_t4x4{
          renderInstance.world[0],
          renderInstance.world[1],
          renderInstance.world[2],
          float32_t4(0, 0, 0, 1)
        };
				if (m_shouldDrawAABB)
				{
					const auto aabbTransform = ext::debug_draw::DrawAABB::getTransformFromAABB(meshInstance.aabb);
					debugDrawInstances.push_back(ext::debug_draw::InstanceData{ .transform = math::linalg::promoted_mul(world4x4, aabbTransform), .color = float32_t4(1, 1, 1, 1)});
				}
				if (m_shouldDrawOBB)
				{
					debugDrawInstances.push_back(ext::debug_draw::InstanceData{ .transform = math::linalg::promoted_mul(world4x4, meshInstance.obb.transform), .color = float32_t4(0, 0, 1, 1)});
				}
				m_bbRenderer->render({ cb, viewProjMatrix }, drawFinished, debugDrawInstances);

        cb->beginDebugMarker("Render ImGui");
        const auto uiParams = m_ui.manager->getCreationParameters();
        auto* uiPipeline = m_ui.manager->getPipeline();
        cb->bindGraphicsPipeline(uiPipeline);
        cb->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
        if (!m_ui.manager->render(cb, drawFinished))
        {
          m_logger->log("TODO: need to present acquired image before bailing because its already acquired.",ILogger::ELL_ERROR);
          return {};
        }
        cb->endDebugMarker();

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

			std::string caption = "[Nabla Engine] Geometry Inspector";
			{
				caption += ", displaying [";
				caption += m_modelPath;
				caption += "]";
				m_window->setCaption(caption);
			}

			updateGUIDescriptorSet();
			return retval;
		}

	protected:
		const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
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
						// TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
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

	private:
		// TODO: standardise this across examples, and take from `argv`
		bool m_nonInteractiveTest = false;

		bool reloadModel()
		{
			if (m_nonInteractiveTest) // TODO: maybe also take from argv and argc
				m_modelPath = (sharedInputCWD/"ply/Spanner-ply.ply").string();
			else
			{
				pfd::open_file file("Choose a supported Model File", sharedInputCWD.string(),
					{
						"All Supported Formats", "*.ply *.stl *.serialized *.obj",
						"TODO (.ply)", "*.ply",
						"TODO (.stl)", "*.stl",
						"Mitsuba 0.6 Serialized (.serialized)", "*.serialized",
						"Wavefront Object (.obj)", "*.obj"
					},
					false
				);
				if (file.result().empty())
					return false;
				m_modelPath = file.result()[0];
			}

			// free up
			m_renderer->m_instances.clear();
			m_renderer->clearGeometries({.semaphore=m_semaphore.get(),.value=m_realFrameIx});
			m_assetMgr->clearAllAssetCache();

			//! load the geometry
			IAssetLoader::SAssetLoadParams params = {};
			params.logger = m_logger.get();
			auto bundle = m_assetMgr->getAsset(m_modelPath,params);
			if (bundle.getContents().empty())
				return false;

			// 
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
			switch (bundle.getAssetType())
			{
				case IAsset::E_TYPE::ET_GEOMETRY:
					for (const auto& item : bundle.getContents())
					if (auto polyGeo=IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
						geometries.push_back(polyGeo);
					break;
				default:
					m_logger->log("Asset loaded but not a supported type (ET_GEOMETRY,ET_GEOMETRY_COLLECTION)",ILogger::ELL_ERROR);
					break;
			}
			if (geometries.empty())
				return false;

			using aabb_t = hlsl::shapes::AABB<3,float32_t>;
			auto printAABB = [&](const aabb_t& aabb, const char* extraMsg="")->void
			{
				m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)",ILogger::ELL_INFO,extraMsg,aabb.minVx.x,aabb.minVx.y,aabb.minVx.z,aabb.maxVx.x,aabb.maxVx.y,aabb.maxVx.z);
			};
			auto bound = aabb_t::create();
			// convert the geometries
			{
				smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({.device=m_device.get()});

				const auto transferFamily = getTransferUpQueue()->getFamilyIndex();

				struct SInputs : CAssetConverter::SInputs
				{
					virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const CAssetConverter::patch_t<asset::ICPUBuffer>& patch) const
					{
						return sharedBufferOwnership;
					}

					core::vector<uint32_t> sharedBufferOwnership;
				} inputs = {};
				core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(),CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
				{
					inputs.logger = m_logger.get();
					std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = {&geometries.front().get(),geometries.size()};
					std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
					// set up shared ownership so we don't have to 
					core::unordered_set<uint32_t> families;
					families.insert(transferFamily);
					families.insert(getGraphicsQueue()->getFamilyIndex());
					if (families.size()>1)
					for (const auto fam : families)
						inputs.sharedBufferOwnership.push_back(fam);
				}
				
				// reserve
				auto reservation = converter->reserve(inputs);
				if (!reservation)
				{
					m_logger->log("Failed to reserve GPU objects for CPU->GPU conversion!",ILogger::ELL_ERROR);
					return false;
				}

				// convert
				{
					auto semaphore = m_device->createSemaphore(0u);

					constexpr auto MultiBuffering = 2;
					std::array<smart_refctd_ptr<IGPUCommandBuffer>,MultiBuffering> commandBuffers = {};
					{
						auto pool = m_device->createCommandPool(transferFamily,IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
						pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,commandBuffers,smart_refctd_ptr(m_logger));
					}
					commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

					std::array<IQueue::SSubmitInfo::SCommandBufferInfo,MultiBuffering> commandBufferSubmits;
					for (auto i=0; i<MultiBuffering; i++)
						commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

					SIntendedSubmitInfo transfer = {};
					transfer.queue = getTransferUpQueue();
					transfer.scratchCommandBuffers = commandBufferSubmits;
					transfer.scratchSemaphore = {
						.semaphore = semaphore.get(),
						.value = 0u,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
					};

					CAssetConverter::SConvertParams cpar = {};
					cpar.utilities = m_utils.get();
					cpar.transfer = &transfer;

					// basically it records all data uploads and submits them right away
					auto future = reservation.convert(cpar);
					if (future.copy()!=IQueue::RESULT::SUCCESS)
					{
						m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
						return false;
					}
				}

				auto tmp = hlsl::float32_t4x3(
					hlsl::float32_t3(1,0,0),
					hlsl::float32_t3(0,1,0),
					hlsl::float32_t3(0,0,1),
					hlsl::float32_t3(0,0,0)
				);
				const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
				core::vector<float32_t3x4> meshWorlds;
				for (uint32_t i = 0; i < converted.size(); i++)
				{
					const auto& geom = converted[i];
					const auto aabb = geom.value->getAABB<aabb_t>();
					printAABB(aabb,"Geometry");
					tmp[3].x += aabb.getExtent().x;
					meshWorlds.emplace_back(hlsl::transpose(tmp));
					const auto transformed = hlsl::shapes::util::transform(meshWorlds.back(), aabb);
					bound = hlsl::shapes::util::union_(transformed,bound);

					const auto& cpuGeom = geometries[i].get();
					const auto obb = CPolygonGeometryManipulator::calculateOBB(
						cpuGeom->getPositionView().getElementCount(),
            [geo = cpuGeom](size_t vertex_i) {
							hlsl::float32_t3 pt;
							geo->getPositionView().decodeElement(vertex_i, pt);
							return pt;
						});

					m_meshInstances.push_back({ .name = std::format("Mesh {}", i), .aabb = aabb, .obb =  obb });
				}

				printAABB(bound,"Total");
				if (!m_renderer->addGeometries({ &converted.front().get(),converted.size() }))
					return false;

				for (auto geom_i = 0u; geom_i < m_renderer->getGeometries().size(); geom_i++)
					m_renderer->m_instances.push_back({
						.world = meshWorlds[geom_i],
						.packedGeo = &m_renderer->getGeometry(geom_i)
					});
			}

			// get scene bounds and reset m_camera
			{
				const float32_t distance = 0.05;
				const auto diagonal = bound.getExtent();
				{
					const auto measure = hlsl::length(diagonal);
					const auto aspectRatio = float(m_window->getWidth())/float(m_window->getHeight());
					m_camera.setProjectionMatrix(hlsl::math::thin_lens::rhPerspectiveFovMatrix(1.2f,aspectRatio,distance*measure*0.1f,measure*4.0f));
					m_camera.setMoveSpeed(measure*0.04);
				}
				const auto pos = bound.maxVx+diagonal*distance;
				m_camera.setPosition(vectorSIMDf(pos.x,pos.y,pos.z));
				const auto center = (bound.minVx+bound.maxVx)*0.5f;
				m_camera.setTarget(vectorSIMDf(center.x,center.y,center.z));
			}

			// TODO: write out the geometry

			return true;
		}

		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;
    constexpr static inline uint8_t MaxUITextureCount = 1u;
		//
		smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;

		struct MeshInstance
		{
			std::string name;
			hlsl::shapes::AABB<3, float32_t> aabb;
			hlsl::shapes::OBB<3, float32_t> obb;
		};
		core::vector<MeshInstance> m_meshInstances;
		int m_selectedMesh = 0;
		//
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
		//
		InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;
		//
    struct CameraSetting
    {
      float fov = 60.f;
      float zNear = 0.1f;
      float zFar = 10000.f;
      float moveSpeed = 1.f;
      float rotateSpeed = 1.f;
      float viewWidth = 10.f;
      float camYAngle = 165.f / 180.f * 3.14159f;
      float camXAngle = 32.f / 180.f * 3.14159f;

    } m_cameraSetting;
		Camera m_camera = Camera(core::vectorSIMDf(0,0,0), core::vectorSIMDf(0,0,0), hlsl::float32_t4x4());
		// mutables
		std::string m_modelPath;

		smart_refctd_ptr<ext::debug_draw::DrawAABB> m_bbRenderer;
		bool m_shouldDrawAABB;
		bool m_shouldDrawOBB;

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

    TransformRequestParams m_transformParams;
  };

NBL_MAIN_FUNC(GeometryInspectorApp)
