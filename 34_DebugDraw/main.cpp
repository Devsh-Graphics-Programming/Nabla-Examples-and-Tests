// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "app_resources/simple_common.hlsl"

class DebugDrawSampleApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280, WIN_H = 720;

public:
	inline DebugDrawSampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<examples::CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
				params.width = WIN_W;
				params.height = WIN_H;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "DebugDrawSampleApp";
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
		}

		if (m_surface)
			return { {m_surface->getSurface()/*,EQF_NONE*/} };

		return {};
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

	    {
	        core::vectorSIMDf cameraPosition(14, 8, 12);
		    core::vectorSIMDf cameraTarget(0, 0, 0);
		    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, zNear, zFar);
		    camera = Camera(cameraPosition, cameraTarget, projectionMatrix, moveSpeed, rotateSpeed);
	    }

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
		if (!swapchainParams.deduceFormat(m_physicalDevice))
			return logFail("Could not choose a Surface Format for the Swapchain!");

		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = 
		{
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = 
				{
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier = 
				{
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
		};

		auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
		auto* renderpass = scResources->getRenderpass();
		
		if (!renderpass)
			return logFail("Failed to create Renderpass!");

		auto gQueue = getGraphicsQueue();
		if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
			return logFail("Could not create Window & Surface or initialize the Surface!");

		m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!m_cmdPool)
				return logFail("Couldn't create Command Pool!");
			if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
		m_surface->recreateSwapchain();

	    {
			ext::drawdebug::DrawAABB::SCreationParameters params;
			params.pushConstantRange = {
			    .stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX,
			    .offset = 0,
			    .size = sizeof(SPushConstants)
			};
            drawAABB = ext::drawdebug::DrawAABB::create(std::move(params));
	    }
		{
			auto vertices = ext::drawdebug::DrawAABB::getVerticesFromAABB(testAABB);
			IGPUBuffer::SCreationParams params;
			params.size = sizeof(float32_t3) * vertices.size();
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			
			m_utils->createFilledDeviceLocalBufferOnDedMem(
				SIntendedSubmitInfo{ .queue = getTransferUpQueue() },
				std::move(params),
				vertices.data()
			).move_into(verticesBuffer);
		}

		auto compileShader = [&](const std::string& filePath) -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lparams = {};
				lparams.logger = m_logger.get();
				lparams.workingDirectory = localInputCWD;
				auto bundle = m_assetMgr->getAsset(filePath, lparams);
				if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
				{
					m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath.c_str());
					exit(-1);
				}

				const auto assets = bundle.getContents();
				assert(assets.size() == 1);
				smart_refctd_ptr<IShader> shaderSrc = IAsset::castDown<IShader>(assets[0]);
				if (!shaderSrc)
					return nullptr;

				return m_device->compileShader({ shaderSrc.get() });
			};
		auto vertexShader = compileShader("app_resources/simple.vertex.hlsl");
		auto fragmentShader = compileShader("app_resources/simple.fragment.hlsl");

		const auto pipelineLayout = ext::drawdebug::DrawAABB::createDefaultPipelineLayout(m_device.get(), drawAABB->getCreationParameters().pushConstantRange);

		IGPUGraphicsPipeline::SShaderSpecInfo vs = { .shader = vertexShader.get(), .entryPoint = "main" };
		IGPUGraphicsPipeline::SShaderSpecInfo fs = { .shader = fragmentShader.get(), .entryPoint = "main" };
		if (!ext::drawdebug::DrawAABB::createDefaultPipeline(&m_pipeline, m_device.get(), pipelineLayout.get(), renderpass, vs, fs))
			return logFail("Graphics pipeline creation failed");

		m_window->setCaption("[Nabla Engine] Debug Draw App Test Demo");
		m_winMgr->show(m_window.get());
		oracle.reportBeginFrameRecord();

		return true;
	}

	inline void workLoopBody() override
	{
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

		m_inputSystem->getDefaultMouse(&mouse);
		m_inputSystem->getDefaultKeyboard(&keyboard);

		auto updatePresentationTimestamp = [&]()
		{
			m_currentImageAcquire = m_surface->acquireNextImage();

			oracle.reportEndFrameRecord();
			const auto timestamp = oracle.getNextPresentationTimeStamp();
			oracle.reportBeginFrameRecord();

			return timestamp;
		};

		const auto nextPresentationTimestamp = updatePresentationTimestamp();

		if (!m_currentImageAcquire)
			return;

		// render whole scene to offline frame buffer & submit

		auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdbuf->beginDebugMarker("DebugDrawSampleApp IMGUI Frame");

		{
			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, m_logger.get());
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
			camera.endInputProcessing(nextPresentationTimestamp);
		}

		core::matrix4SIMD modelViewProjectionMatrix;
	    {
	        const auto viewMatrix = camera.getViewMatrix();
		    const auto projectionMatrix = camera.getProjectionMatrix();
		    const auto viewProjectionMatrix = camera.getConcatenatedMatrix();

		    core::matrix3x4SIMD modelMatrix;
		    modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
		    modelMatrix.setRotation(quaternion(0, 0, 0));
			modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
	    }

		auto* queue = getGraphicsQueue();

		asset::SViewport viewport;
		{
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = WIN_W;
			viewport.height = WIN_H;
		}
		cmdbuf->setViewport(0u, 1u, &viewport);

		VkRect2D scissor{
			.offset = { 0, 0 },
			.extent = { m_window->getWidth(), m_window->getHeight() }
		};
		cmdbuf->setScissor(0u, 1u, &scissor);

		const VkRect2D currentRenderArea =
		{
			.offset = {0,0},
			.extent = {m_window->getWidth(),m_window->getHeight()}
		};

		{
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
			const IGPUCommandBuffer::SRenderpassBeginInfo beginInfo = 
			{
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};

			SPushConstants pc;
			memcpy(pc.MVP, modelViewProjectionMatrix.pointer(), sizeof(pc.MVP));
			pc.pVertices = verticesBuffer->getDeviceAddress();

			cmdbuf->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			cmdbuf->bindGraphicsPipeline(m_pipeline.get());
			cmdbuf->pushConstants(m_pipeline->getLayout(), ESS_VERTEX, 0, sizeof(SPushConstants), &pc);
			drawAABB->renderSingle(cmdbuf);

			cmdbuf->endRenderPass();
		}
		cmdbuf->end();

		{
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = 
			{ 
				{
					.semaphore = m_semaphore.get(),
					.value = ++m_realFrameIx,
					.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
				} 
			};

			{
				{
					const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
					{
						{.cmdbuf = cmdbuf }
					};

					const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = 
					{ 
						{
							.semaphore = m_currentImageAcquire.semaphore,
							.value = m_currentImageAcquire.acquireCount,
							.stageMask = PIPELINE_STAGE_FLAGS::NONE
						} 
					};

					const IQueue::SSubmitInfo infos[] = 
					{ 
						{
							.waitSemaphores = acquired,
							.commandBuffers = commandBuffers,
							.signalSemaphores = rendered
						} 
					};

					if (queue->submit(infos) == IQueue::RESULT::SUCCESS)
					{
						const nbl::video::ISemaphore::SWaitInfo waitInfos[] =
						{ {
							.semaphore = m_semaphore.get(),
							.value = m_realFrameIx
						} };

						m_device->blockForSemaphores(waitInfos); // this is not solution, quick wa to not throw validation errors
					}
					else
						--m_realFrameIx;
				}
			}

			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}
	}

	inline bool keepRunning() override
	{
		if (m_surface->irrecoverable())
			return false;

		return true;
	}

	inline bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

private:
	std::array<float32_t3, 24> getVerticesFromAABB(core::aabbox3d<float>& aabb)
	{
		const auto& pMin = aabb.MinEdge;
		const auto& pMax = aabb.MaxEdge;

		std::array<float32_t3, 24> vertices;
		vertices[0] = float32_t3(pMin.X, pMin.Y, pMin.Z);
		vertices[1] = float32_t3(pMax.X, pMin.Y, pMin.Z);
		vertices[2] = float32_t3(pMin.X, pMin.Y, pMin.Z);
		vertices[3] = float32_t3(pMin.X, pMin.Y, pMax.Z);

		vertices[4] = float32_t3(pMax.X, pMin.Y, pMax.Z);
		vertices[5] = float32_t3(pMax.X, pMin.Y, pMin.Z);
		vertices[6] = float32_t3(pMax.X, pMin.Y, pMax.Z);
		vertices[7] = float32_t3(pMin.X, pMin.Y, pMax.Z);

		vertices[8] = float32_t3(pMin.X, pMax.Y, pMin.Z);
		vertices[9] = float32_t3(pMax.X, pMax.Y, pMin.Z);
		vertices[10] = float32_t3(pMin.X, pMax.Y, pMin.Z);
		vertices[11] = float32_t3(pMin.X, pMax.Y, pMax.Z);

		vertices[12] = float32_t3(pMax.X, pMax.Y, pMax.Z);
		vertices[13] = float32_t3(pMax.X, pMax.Y, pMin.Z);
		vertices[14] = float32_t3(pMax.X, pMax.Y, pMax.Z);
		vertices[15] = float32_t3(pMin.X, pMax.Y, pMax.Z);

		vertices[16] = float32_t3(pMin.X, pMin.Y, pMin.Z);
		vertices[17] = float32_t3(pMin.X, pMax.Y, pMin.Z);
		vertices[18] = float32_t3(pMax.X, pMin.Y, pMin.Z);
		vertices[19] = float32_t3(pMax.X, pMax.Y, pMin.Z);

		vertices[20] = float32_t3(pMin.X, pMin.Y, pMax.Z);
		vertices[21] = float32_t3(pMin.X, pMax.Y, pMax.Z);
		vertices[22] = float32_t3(pMax.X, pMin.Y, pMax.Z);
		vertices[23] = float32_t3(pMax.X, pMax.Y, pMax.Z);

		return vertices;
	}

	// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
	constexpr static inline uint32_t MaxFramesInFlight = 3u;

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	core::smart_refctd_ptr<InputSystem> m_inputSystem;
    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::smart_refctd_ptr<IDescriptorPool> m_descriptorSetPool;

	Camera camera;
	video::CDumbPresentationOracle oracle;

	uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

	float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;

	smart_refctd_ptr<ext::drawdebug::DrawAABB> drawAABB;
	core::aabbox3d<float> testAABB = core::aabbox3d<float>({ 0, 0, 0 }, { 10, 10, -10 });
	smart_refctd_ptr<IGPUBuffer> verticesBuffer;
};

NBL_MAIN_FUNC(DebugDrawSampleApp)