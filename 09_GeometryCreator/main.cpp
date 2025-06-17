// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"

class GeometryCreatorApp final : public MonoWindowApplication
{
		using base_t = MonoWindowApplication;

	public:
		GeometryCreatorApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: base_t({1280,720}, EF_D16_UNORM, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = base_t::getRequiredDeviceFeatures();
			retval.geometryShader = true;
			return retval;
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(smart_refctd_ptr(system)))
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
			// we want to use the vertex data through UTBs
			using usage_f = IGPUBuffer::E_USAGE_FLAGS;
			CAssetConverter::patch_t<asset::ICPUPolygonGeometry> patch = {};
			patch.positionBufferUsages = usage_f::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			patch.indexBufferUsages = usage_f::EUF_INDEX_BUFFER_BIT;
			patch.otherBufferUsages = usage_f::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			auto scene = CGeometryCreatorScene::create(
				{
					.transferQueue = getTransferUpQueue(),
					.utilities = m_utils.get(),
					.logger = m_logger.get(),
					.addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies
				},patch
			);
			
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			auto renderer = CSimpleDebugRenderer::create(scRes->getRenderpass(),0,scene.get());

			// camera
			{
				core::vectorSIMDf cameraPosition(-5.81655884, 2.58630896, -4.23974705);
				core::vectorSIMDf cameraTarget(-0.349590302, -0.213266611, 0.317821503);
				matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(m_initialResolution.x)/float(m_initialResolution.y), 0.1, 10000);
				camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);
			}

			onAppInitializedFinish();
			return true;
		}

		inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
		{
			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			const auto resourceIx = m_realFrameIx % base_t::MaxFramesInFlight;

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->beginDebugMarker("GeometryCreatorApp Frame");
			{
				camera.beginInputProcessing(nextPresentationTimestamp);
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
				camera.endInputProcessing(nextPresentationTimestamp);
#if 0
				const auto type = static_cast<ObjectType>(gcIndex);
				const auto& [gpu, meta] = resources.objects[type];

				object.meta.type = type;
				object.meta.name = meta.name;
#endif
			}

			const auto viewMatrix = camera.getViewMatrix();
			const auto viewProjectionMatrix = camera.getConcatenatedMatrix();
#if 0
			SBasicViewParameters uboData;
			memcpy(uboData.MVP, modelViewProjectionMatrix.pointer(), sizeof(uboData.MVP));
			memcpy(uboData.MV, modelViewMatrix.pointer(), sizeof(uboData.MV));
			memcpy(uboData.NormalMat, normalMatrix.pointer(), sizeof(uboData.NormalMat));
			{
				SBufferRange<IGPUBuffer> range;
				range.buffer = core::smart_refctd_ptr(resources.ubo.buffer);
				range.size = resources.ubo.buffer->getSize();

				cb->updateBuffer(range, &uboData);
			}
#endif
			auto* queue = getGraphicsQueue();

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
					.framebuffer = scRes->getFramebuffer(base_t::getCurrentAcquire().imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &depthValue,
					.renderArea = currentRenderArea
				};

				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			}
#if 0
			const auto& [hook, meta] = resources.objects[object.meta.type];
			auto* rawPipeline = hook.pipeline.get();

			SBufferBinding<const IGPUBuffer> vertex = hook.bindings.vertex, index = hook.bindings.index;

			cb->bindGraphicsPipeline(rawPipeline);
			cb->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &resources.descriptorSet.get());
			cb->bindVertexBuffers(0, 1, &vertex);

			if (index.buffer && hook.indexType != EIT_UNKNOWN)
			{
				cb->bindIndexBuffer(index, hook.indexType);
				cb->drawIndexed(hook.indexCount, 1, 0, 0, 0);
			}
			else
				cb->draw(hook.indexCount, 1, 0, 0);
#endif
			cb->endRenderPass();
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
				{.semaphore = base_t::getCurrentAcquire().semaphore, .value = base_t::getCurrentAcquire().acquireCount, .stageMask = PIPELINE_STAGE_FLAGS::NONE}
			};
			const IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = acquired,
					.commandBuffers = commandBuffers,
					.signalSemaphores = {&retval,1}
				}
			};

			if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
			{
				retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
				m_realFrameIx--;
			}

			std::string caption = "[Nabla Engine] Geometry Creator";
			{
//					caption += ", displaying [" + std::string(object.meta.name.data()) + "]";
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
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,base_t::MaxFramesInFlight> m_cmdBufs;

		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

//		ResourcesBundle resources;
//		ObjectDrawHookCpu object;
		uint16_t gcIndex = {};

		void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
		{
			for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
			{
				auto ev = *eventIt;

				if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
					gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(ev.scrollEvent.verticalScroll)), int64_t(0), int64_t(CGeometryCreatorScene::OT_COUNT - (uint8_t)1u));
			}
		}
};

NBL_MAIN_FUNC(GeometryCreatorApp)