// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

#include "common.hpp"

#ifdef NBL_BUILD_DEBUG_DRAW
#include "nbl/ext/DebugDraw/CDrawAABB.h"
#endif

class GeometryCreatorApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
	using device_base_t = MonoWindowApplication;
	using asset_base_t = BuiltinResourcesApplication;

	public:
		GeometryCreatorApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
			device_base_t({1280,720}, EF_D16_UNORM, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			for (auto i = 0u; i < MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			const uint32_t addtionalBufferOwnershipFamilies[] = {getGraphicsQueue()->getFamilyIndex()};
			m_scene = CGeometryCreatorScene::create(
				{
					.transferQueue = getTransferUpQueue(),
					.utilities = m_utils.get(),
					.logger = m_logger.get(),
					.addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies
				},
				CSimpleDebugRenderer::DefaultPolygonGeometryPatch // we want to use the vertex data through UTBs
			);

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const auto& geometries = m_scene->getInitParams().geometries;
			m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(),scRes->getRenderpass(),0,{&geometries.front().get(),geometries.size()});
			if (!m_renderer || m_renderer->getGeometries().size() != geometries.size())
				return logFail("Could not create Renderer!");
			// special case
			{
				const auto& pipelines = m_renderer->getInitParams().pipelines;
				auto ix = 0u;
				for (const auto& name : m_scene->getInitParams().geometryNames)
				{
					if (name=="Cone")
						m_renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
					ix++;
				}
			}
			m_renderer->m_instances.resize(1);
			m_renderer->m_instances[0].world = float32_t3x4(
				float32_t4(1,0,0,0),
				float32_t4(0,1,0,0),
				float32_t4(0,0,1,0)
			);

#ifdef NBL_BUILD_DEBUG_DRAW
			{
				ext::debug_draw::DrawAABB::SCreationParameters params = {};
				params.assetManager = m_assetMgr;
				params.transfer = getTransferUpQueue();
				params.drawMode = ext::debug_draw::DrawAABB::ADM_DRAW_BATCH;
				params.batchPipelineLayout = ext::debug_draw::DrawAABB::createDefaultPipelineLayout(m_device.get());
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(scRes->getRenderpass());
				params.depthCompareOp = asset::ECO_GREATER;
				params.utilities = m_utils;
				m_drawAABB = ext::debug_draw::DrawAABB::create(std::move(params));
				if (!m_drawAABB)
					return logFail("Could not create the axis renderer!");

				const auto addBox = [&](const float32_t3& minVx, const float32_t3& maxVx, const float32_t4& color) -> void
				{
					const auto transform = ext::debug_draw::DrawAABB::getTransformFromAABB(hlsl::shapes::AABB<3,float>{minVx, maxVx});
					auto& instance = m_axisInstances.emplace_back();
					instance.transform = float32_t4x4(transform[0], transform[1], transform[2], float32_t4(0.f, 0.f, 0.f, 1.f));
					instance.color = color;
				};
				const auto addAxis = [&](const float32_t3& axis, const float32_t4& color) -> void
				{
					constexpr float AxisLength = 3.f;
					constexpr float ShaftHalfWidth = 0.04f;
					constexpr float HeadLength = 0.4f;
					constexpr float HeadHalfWidth = 0.14f;
					const auto across = float32_t3(1.f) - axis;
					addBox(-across * ShaftHalfWidth, axis * AxisLength + across * ShaftHalfWidth, color);
					addBox(axis * (AxisLength - HeadLength) - across * HeadHalfWidth, axis * AxisLength + across * HeadHalfWidth, color);
				};
				addAxis(float32_t3(1.f, 0.f, 0.f), float32_t4(1.f, 0.f, 0.f, 1.f));
				addAxis(float32_t3(0.f, 1.f, 0.f), float32_t4(0.f, 1.f, 0.f, 1.f));
				addAxis(float32_t3(0.f, 0.f, 1.f), float32_t4(0.f, 0.f, 1.f, 1.f));
			}
#endif

			if (!setupCamera(0u, ECameraRig::FPS))
				return logFail("Could not initialize camera orientation!");

			onAppInitializedFinish();
			return true;
		}

		inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
		{
			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->beginDebugMarker("GeometryCreatorApp Frame");
			{
				std::vector<ui::SMouseEvent> mouseEvents;
				std::vector<ui::SKeyboardEvent> keyboardEvents;

				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { mouseEvents.insert(mouseEvents.end(), events.begin(), events.end()); }, m_logger.get());
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { keyboardEvents.insert(keyboardEvents.end(), events.begin(), events.end()); }, m_logger.get());

				mouseProcess({ mouseEvents.data(), mouseEvents.size() });

				auto worldCoordinateSystemIndex = m_worldCoordinateSystemIndex;
				auto rig = m_rig;
				for (const auto& ev : keyboardEvents)
				{
					if (ev.action != ui::SKeyboardEvent::ECA_RELEASED)
						continue;
					if (ev.keyCode == ui::EKC_H)
						worldCoordinateSystemIndex = (worldCoordinateSystemIndex + 1u) % WorldCoordinateSystemCount;
					if (ev.keyCode == ui::EKC_C)
						rig = rig == ECameraRig::FPS ? ECameraRig::Orbit : ECameraRig::FPS;
				}
				if (worldCoordinateSystemIndex != m_worldCoordinateSystemIndex || rig != m_rig)
				{
					cameraController.reset();
					setupCamera(worldCoordinateSystemIndex, rig);
				}

				const auto controls = cameraController.collect(nextPresentationTimestamp, keyboardEvents, mouseEvents);
				camera->manipulate(controls);
			}


			asset::SViewport viewport;
			{
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = m_window->getWidth();
				viewport.height = m_window->getHeight();
			}
			cb->setViewport(0u, 1u, &viewport);

			VkRect2D scissor =
			{
				.offset = { 0, 0 },
				.extent = { m_window->getWidth(), m_window->getHeight() },
			};
			cb->setScissor(0u, 1u, &scissor);

			{
				const VkRect2D currentRenderArea =
				{
					.offset = {0,0},
					.extent = {m_window->getWidth(),m_window->getHeight()}
				};

				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
				const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo info =
				{
					.framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &depthValue,
					.renderArea = currentRenderArea
				};

				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			}

			const auto viewMatrix = hlsl::float32_t3x4(camera->getGimbal().getViewMatrix());
			const auto viewProjMatrix = hlsl::math::linalg::promoted_mul(cameraProjection, viewMatrix);
			const auto viewParams = CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix);

			// tear down scene every frame
			m_renderer->m_instances[0].packedGeo = m_renderer->getGeometries().data()+gcIndex;
 			m_renderer->render(cb,viewParams);

#ifdef NBL_BUILD_DEBUG_DRAW
			{
				const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };
				ext::debug_draw::DrawAABB::DrawParameters drawParams;
				drawParams.commandBuffer = cb;
				drawParams.cameraMat = viewProjMatrix;
				m_drawAABB->render(drawParams, drawFinished, m_axisInstances);
			}
#endif

			cb->endRenderPass();
			cb->endDebugMarker();
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

			std::string caption = "[Nabla Engine] Geometry Creator";
			{
				caption += ", displaying [";
				caption += m_scene->getInitParams().geometryNames[gcIndex];
				caption += "] | H: ";
				caption += WorldCoordinateSystems[m_worldCoordinateSystemIndex].name;
				caption += " world, expect ";
				caption += WorldCoordinateSystems[m_worldCoordinateSystemIndex].expectation;
				caption += " | C: ";
				caption += m_rig == ECameraRig::FPS ? "FPS camera (WASD, mouse)" : "orbit camera (mouse, Q/E distance)";
				m_window->setCaption(caption);
			}
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
		//
		smart_refctd_ptr<CGeometryCreatorScene> m_scene;
		smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
		//
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,device_base_t::MaxFramesInFlight> m_cmdBufs;
		//
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		//
		core::smart_refctd_ptr<ext::cameras::ICamera> camera;
		ext::cameras::CCameraMouseKeyboardController cameraController;
		hlsl::float32_t4x4 cameraProjection = hlsl::float32_t4x4(1.0f);

		struct SWorldCoordinateSystem
		{
			const char* name;
			const char* expectation;
			ext::cameras::SWorldFrame frame;
		};
		static inline const SWorldCoordinateSystem WorldCoordinateSystems[] = {
			{ "Y-up LH", "red +X right, green +Y up, blue +Z away from you", ext::cameras::SWorldFrame::yUpLeftHanded() },
			{ "Y-up RH", "red +X right, green +Y up, blue +Z toward you", ext::cameras::SWorldFrame::yUpRightHanded() },
			{ "Z-up RH", "red +X right, blue +Z up, green +Y away from you", ext::cameras::SWorldFrame::zUpRightHanded() },
		};
		static inline constexpr uint32_t WorldCoordinateSystemCount = sizeof(WorldCoordinateSystems) / sizeof(WorldCoordinateSystems[0]);
		enum class ECameraRig : uint8_t { FPS, Orbit };
		uint32_t m_worldCoordinateSystemIndex = 0u;
		ECameraRig m_rig = ECameraRig::FPS;

#ifdef NBL_BUILD_DEBUG_DRAW
		smart_refctd_ptr<ext::debug_draw::DrawAABB> m_drawAABB;
		std::vector<ext::debug_draw::InstanceData> m_axisInstances;
#endif

		uint16_t gcIndex = {};

		static void getTestPose(const ext::cameras::SWorldFrame& frame, hlsl::float64_t3& outPosition, hlsl::float64_t3& outTarget)
		{
			outPosition = frame.forward * -7.0 + frame.right * -3.0 + frame.up * 3.0;
			outTarget = hlsl::float64_t3(0.0);
		}

		bool setupCamera(const uint32_t worldCoordinateSystemIndex, const ECameraRig rig)
		{
			using namespace ext::cameras;
			m_worldCoordinateSystemIndex = worldCoordinateSystemIndex;
			m_rig = rig;
			const auto& frame = WorldCoordinateSystems[worldCoordinateSystemIndex].frame;

			hlsl::float64_t3 cameraPosition, cameraTarget;
			getTestPose(frame, cameraPosition, cameraTarget);
			const auto aspectRatio = float(m_initialResolution.x) / m_initialResolution.y;
			if (frame.handedness == ECameraHandedness::LeftHanded)
				cameraProjection = hlsl::math::thin_lens::lhPerspectiveFovMatrix<float>(core::radians(60.0f), aspectRatio, 10000.0f, 0.1f);
			else
				cameraProjection = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(core::radians(60.0f), aspectRatio, 10000.0f, 0.1f);

			if (rig == ECameraRig::FPS)
			{
				camera = core::make_smart_refctd_ptr<CFPSCamera>(cameraPosition, hlsl::math::quaternion<hlsl::float64_t>::identity(), frame);
				if (!camera->setLookAt(cameraPosition, cameraTarget, frame.up))
					return false;
			}
			else
				camera = core::make_smart_refctd_ptr<COrbitCamera>(cameraPosition, cameraTarget, frame);

			auto& binding = cameraController.binding;
			binding = CCameraMouseKeyboardPresets::makeDefaultBinding(*camera);
			if (rig == ECameraRig::Orbit)
			{
				const auto gain = binding[ECameraControlAxis::RotateX].mouseMovementGain.x;
				binding[ECameraControlAxis::RotateY].mouseMovementGain = hlsl::float64_t2(gain, 0.0);
				binding[ECameraControlAxis::RotateX].mouseMovementGain = hlsl::float64_t2(0.0, gain);
				binding[ECameraControlAxis::Distance].mouseScrollGain = hlsl::float64_t2(0.0);
			}
			binding.scaleSensitivity(ECameraControlAxis::Translate, 1.069);
			binding.scaleSensitivity(ECameraControlAxis::Rotate, 0.4);
			binding.setMouseMovementGate(ECameraControlAxis::Rotate, ui::EMB_LEFT_BUTTON);

			const auto basis = camera->getBasis();
			m_logger->log("%s %s: right = (% .3f, % .3f, % .3f) up = (% .3f, % .3f, % .3f) forward = (% .3f, % .3f, % .3f)",
				system::ILogger::ELL_INFO, WorldCoordinateSystems[worldCoordinateSystemIndex].name, camera->getIdentifier().data(),
				basis.right.x, basis.right.y, basis.right.z,
				basis.up.x, basis.up.y, basis.up.z,
				basis.forward.x, basis.forward.y, basis.forward.z);
			return true;
		}

		void mouseProcess(std::span<const nbl::ui::SMouseEvent> events)
		{
			for (const auto& ev : events)
			{
				if (ev.type==nbl::ui::SMouseEvent::EET_SCROLL && m_renderer)
				{
					gcIndex += int16_t(core::sign(ev.scrollEvent.verticalScroll));
					gcIndex = core::clamp(gcIndex,0ull,m_renderer->getGeometries().size()-1);
				}
			}
		}
};

NBL_MAIN_FUNC(GeometryCreatorApp)
