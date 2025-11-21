// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl>
#include <nbl/builtin/hlsl/projection/projection.hlsl>

#include "common.hpp"

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

			// camera
			{
				hlsl::float32_t3 cameraPosition(-5.81655884, 2.58630896, -4.23974705);
				hlsl::float32_t3 cameraTarget(-0.349590302, -0.213266611, 0.317821503);
				float32_t4x4 projectionMatrix = hlsl::buildProjectionMatrixPerspectiveFovLH<float>(core::radians(60.0f), float(m_initialResolution.x) / m_initialResolution.y, 0.1f, 10000.0f);
				camera = Camera(core::constructVecorSIMDFromHLSLVector(cameraPosition), core::constructVecorSIMDFromHLSLVector(cameraTarget), projectionMatrix, 1.069f, 0.4f);
			}

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
				camera.beginInputProcessing(nextPresentationTimestamp);
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
				camera.endInputProcessing(nextPresentationTimestamp);
			}


			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
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

			float32_t3x4 viewMatrix = camera.getViewMatrix();
			float32_t4x4 viewProjMatrix = camera.getConcatenatedMatrix();
			const auto viewParams = CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix);

			// tear down scene every frame
			m_renderer->m_instances[0].packedGeo = m_renderer->getGeometries().data()+gcIndex;
 			m_renderer->render(cb,viewParams);

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
				caption += "]";
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
		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), hlsl::float32_t4x4());

		uint16_t gcIndex = {};

		void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
		{
			for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
			{
				auto ev = *eventIt;

				if (ev.type==nbl::ui::SMouseEvent::EET_SCROLL && m_renderer)
				{
					gcIndex += int16_t(core::sign(ev.scrollEvent.verticalScroll));
					gcIndex = core::clamp(gcIndex,0ull,m_renderer->getGeometries().size()-1);
				}
			}
		}
};

NBL_MAIN_FUNC(GeometryCreatorApp)