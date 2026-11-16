// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>
#include "app_resources/common.hlsl"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"


class oitApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
		using device_base_t = SimpleWindowedApplication;
		using asset_base_t = BuiltinResourcesApplication;
		using surface_t = CSimpleResizeSurface<CDefaultSwapchainFramebuffers>;

		constexpr static inline uint32_t WIN_W = 1280;
		constexpr static inline uint32_t WIN_H = 720;
		constexpr static inline uint8_t MaxFramesInFlight = 3;

		constexpr static inline E_FORMAT DepthFormat = EF_D32_SFLOAT;
		
		inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = device_base_t::getRequiredDeviceFeatures();
			retval.fragmentShaderPixelInterlock = true;
			return retval;
		}

	public:
		inline oitApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		inline core::vector<SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem),smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.width = WIN_W;
					params.height = WIN_H;
					params.x = 32;
					params.y = 32;
					params.flags = IWindow::ECF_HIDDEN|IWindow::ECF_RESIZABLE|IWindow::ECF_CAN_MINIMIZE;
					params.windowCaption = "OIT";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = surface_t::create(std::move(surface));
			}

			if (m_surface)
				return {{m_surface->getSurface()}};

			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!MonoSystemMonoLoggerApplication::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			if (!createSwapchain())
				return false;
			if (!createVertexBuffer() || !createIndexBuffer())
				return false;
	

			uint32_t hostVisibleBits = m_physicalDevice->getHostVisibleMemoryTypeBits();
			uint32_t deviceLocalBits = m_physicalDevice->getDeviceLocalMemoryTypeBits();
			uint32_t hostCachedBits = m_physicalDevice->getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT);
			uint32_t hostVisibleDeviceLocalBits = hostVisibleBits & deviceLocalBits & ~hostCachedBits;

			IGPUBuffer::SCreationParams params;
			params.size = BUFFER_SIZE;
			params.usage = IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
			outBuffer = m_device->createBuffer(std::move(params));
			if (!outBuffer)
				return logFail("Failed to create GPU buffer of size %d!\n", BUFFER_SIZE);

			outBuffer->setObjectDebugName("MLABBuffer");

			auto reqs = outBuffer->getMemoryReqs();
			reqs.memoryTypeBits &= hostVisibleDeviceLocalBits;

			auto outAllocation = m_device->allocate(reqs, { outBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT });
			if (!outAllocation.isValid())
				return logFail("Failed to allocate Device Memory!\n");

			auto outMappedPtr = outAllocation.memory->map({ 0ull, outAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_WRITE);
			if (!outMappedPtr)
				return logFail("Failed to map Device Memory!\n");

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to create a Semaphore!");

			m_cmdPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!m_cmdPool || !m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,m_cmdBufs))
				return logFail("Failed to create Command Buffers!");

			auto loadPrecompiledShader = [&]<core::StringLiteral ShaderKey>()->smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "app_resources";

				auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_physicalDevice->getLimits(), m_physicalDevice->getFeatures());
				m_logger->log("Loading shader with key: %s", ILogger::ELL_INFO, key.data());

				auto assetBundle = m_assetMgr->getAsset(key.data(), lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
				{
					m_logger->log("Asset bundle is empty for key: %s", ILogger::ELL_ERROR, key.data());
					return smart_refctd_ptr<IShader>(nullptr);
				}

				m_logger->log("Asset count: %u, asset type: %u", ILogger::ELL_INFO, assets.size(), (uint32_t)assets[0]->getAssetType());

				auto shader = IAsset::castDown<IShader>(assets[0]);
				return shader;
			};

			auto vertexShader = loadPrecompiledShader.operator()<"triangle_vert">();
			auto fragmentShader = loadPrecompiledShader.operator()<"triangle_frag">();
			if (!vertexShader || !fragmentShader)
				return logFail("Failed to load the shaders!");

			const SPushConstantRange pcRange = {
				.stageFlags = hlsl::ShaderStage::ESS_VERTEX | hlsl::ShaderStage::ESS_FRAGMENT,
				.offset = 0,
				.size = sizeof(nbl::hlsl::this_example::SPushConstants)
			};
			m_pipelineLayout = m_device->createPipelineLayout({&pcRange,1});
			if (!m_pipelineLayout)
				return logFail("Failed to create the Pipeline Layout!");

			{
				IGPUGraphicsPipeline::SCreationParams params[1] = {};
				params[0].layout = m_pipelineLayout.get();
				params[0].vertexShader = {.shader=vertexShader.get(),.entryPoint="VSMain"};
				params[0].fragmentShader = {.shader=fragmentShader.get(),.entryPoint="PSMain"};
				params[0].renderpass = getSwapchainFramebuffers()->getRenderpass();
				params[0].cached.subpassIx = 0u;
				{
					auto& vertexInput = params[0].cached.vertexInput;
					vertexInput.bindings[0].stride = sizeof(SVertex);
					vertexInput.bindings[0].inputRate = SVertexInputBindingParams::EVIR_PER_VERTEX;
					vertexInput.enabledBindingFlags = 0b1u;
					vertexInput.attributes[0] = {.binding = 0, .format = EF_R32G32_SFLOAT, .relativeOffset = offsetof(SVertex, position)};
					vertexInput.attributes[1] = { .binding = 0, .format = EF_R32G32B32A32_SFLOAT, .relativeOffset = offsetof(SVertex, color) };
					vertexInput.enabledAttribFlags = 0b11u;
				}
				params[0].cached.primitiveAssembly.primitiveType = EPT_TRIANGLE_LIST;
				params[0].cached.rasterization.faceCullingMode = EFCM_NONE;
				params[0].cached.rasterization.depthWriteEnable = true;
				params[0].cached.rasterization.depthTestEnable();
				params[0].cached.rasterization.depthCompareOp = ECO_ALWAYS;
				
				//params[0].cached.blend.blendParams->blendEnabled();
				//params[0].cached.blend.blendParams->colorWriteMask = 0xF;
				//params[0].cached.blend.blendParams->srcColorFactor = EBF_SRC_ALPHA;
				//params[0].cached.blend.blendParams->colorBlendOp = EBO_ADD;
				//params[0].cached.blend.blendParams->dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
				//params[0].cached.blend.blendParams->alphaBlendOp = EBO_ADD;
				//params[0].cached.blend.blendParams->srcAlphaFactor = EBF_ONE;
				//params[0].cached.blend.blendParams->dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
			

				if (!m_device->createGraphicsPipelines(nullptr,params,&m_pipeline) || !m_pipeline)
					return logFail("Failed to create the Graphics Pipeline!");
			}

			{
				ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
				if (!fsTriProtoPPln)
					return logFail("Failed to create the Full Screen Triangle proto pipeline!");

				auto mlabShader = loadPrecompiledShader.operator() < "mlab_frag" > ();
				if (!mlabShader)
					return logFail("Failed to load the resolve shader!");

				const IGPUPipelineBase::SShaderSpecInfo fragSpec = { .shader = mlabShader.get(), .entryPoint = "PSMain" };
				SBlendParams blend = {};
				
				blend.blendParams->blendEnabled();
				blend.blendParams->colorWriteMask = 0xF;
				blend.blendParams->srcColorFactor = EBF_SRC_ALPHA;
				blend.blendParams->colorBlendOp = EBO_ADD;
				blend.blendParams->dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
				blend.blendParams->alphaBlendOp = EBO_ADD;
				blend.blendParams->srcAlphaFactor = EBF_ONE;
				blend.blendParams->dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
				m_mlabPipeline = fsTriProtoPPln.createPipeline(fragSpec, m_pipelineLayout.get(), getSwapchainFramebuffers()->getRenderpass(),0u,blend);
				if (!m_mlabPipeline)
					return logFail("Failed to create the resolve pipeline!");
			}

			{
				const auto cameraPosition = hlsl::float64_t3(0.0, 0.0, -3.0);
				const auto cameraTarget = hlsl::float64_t3(0.0, 0.0, 0.0);
				m_cameraProjection = hlsl::math::thin_lens::lhPerspectiveFovMatrix<float>(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1f, 100.0f);

				hlsl::math::quaternion<hlsl::float64_t> cameraOrientation;
				if (!ext::cameras::CCameraMathUtilities::tryCreateQuaternionFromLookAt(cameraPosition, cameraTarget, hlsl::float64_t3(0.0, 1.0, 0.0), ext::cameras::SWorldFrame::yUpLeftHanded(), cameraOrientation))
					return logFail("Could not initialize camera orientation!");
				m_camera = core::make_smart_refctd_ptr<ext::cameras::CFPSCamera>(cameraPosition, cameraOrientation);

				// WASD moves, the mouse looks while the left button is held
				{
					using namespace ext::cameras;
					auto& binding = m_cameraController.binding;
					binding = CCameraMouseKeyboardPresets::makeDefaultBinding(ICamera::CameraKind::FPS);
					binding.scaleSensitivity(ECameraControlAxis::Translate, 1.069);
					binding.scaleSensitivity(ECameraControlAxis::Rotate, 0.4);
					binding.setMouseMovementGate(ECameraControlAxis::Rotate, ui::EMB_LEFT_BUTTON);
				}
			}


			m_oracle.reportBeginFrameRecord();
			m_winMgr->show(m_window.get());
			return true;
		}

		inline bool keepRunning() override
		{
			return m_surface->isWindowOpen() && !m_surface->irrecoverable();
		}

		inline void workLoopBody() override
		{
			const uint32_t framesInFlight = core::min<uint32_t>(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
			if (m_realFrameIx >= framesInFlight)
			{
				const ISemaphore::SWaitInfo framesDone[] = { {.semaphore = m_semaphore.get(),.value = m_realFrameIx + 1 - framesInFlight} };
				if (m_device->blockForSemaphores(framesDone) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			const auto acquire = m_surface->acquireNextImage();
			if (!acquire)
				return;

			m_oracle.reportEndFrameRecord();
			const auto nextPresentationTimestamp = m_oracle.getNextPresentationTimeStamp();
			m_oracle.reportBeginFrameRecord();

			m_inputSystem->getDefaultMouse(&m_mouse);
			m_inputSystem->getDefaultKeyboard(&m_keyboard);
			{
				std::vector<ui::SMouseEvent> mouseEvents;
				std::vector<ui::SKeyboardEvent> keyboardEvents;

				m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { mouseEvents.insert(mouseEvents.end(), events.begin(), events.end()); }, m_logger.get());
				m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { keyboardEvents.insert(keyboardEvents.end(), events.begin(), events.end()); }, m_logger.get());

				const auto controls = m_cameraController.collect(nextPresentationTimestamp, keyboardEvents, mouseEvents);
				m_camera->manipulate(controls);
			}

			nbl::hlsl::this_example::SPushConstants pc;
			{
				const auto viewMatrix = hlsl::float32_t3x4(m_camera->getGimbal().getViewMatrix());
				pc.viewProj = hlsl::math::linalg::promoted_mul(m_cameraProjection, viewMatrix);
				pc.bdaAddress = outBuffer.get()->getDeviceAddress();
				pc.width = WIN_W;
			}

			auto* const framebuffer = getSwapchainFramebuffers()->getFramebuffer(acquire.imageIndex);
			const auto& fbParams = framebuffer->getCreationParameters();
			const VkRect2D renderArea = { .offset = {0,0},.extent = {fbParams.width,fbParams.height} };

			auto* const cmdbuf = m_cmdBufs[m_realFrameIx % MaxFramesInFlight].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			cmdbuf->fillBuffer({ .offset = 0, .size = outBuffer->getSize() , .buffer = outBuffer }, hlsl::numeric_limits<float>::max);

			const SViewport viewport = {
				.x = 0.f,
				.y = 0.f,
				.width = float(fbParams.width),
				.height = float(fbParams.height),
				.minDepth = 0.f,
				.maxDepth = 1.f
			};
			cmdbuf->setViewport({ &viewport,1 });
			cmdbuf->setScissor({ &renderArea,1 });

			const IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.1f,0.1f,0.1f,1.f} };
			const IGPUCommandBuffer::SClearDepthStencilValue clearDepth = { .depth = 0.f };
			const IGPUCommandBuffer::SRenderpassBeginInfo beginInfo = {
				.framebuffer = framebuffer,
				.colorClearValues = &clearColor,
				.depthStencilClearValues = &clearDepth,
				.renderArea = renderArea
			};
		{
			cmdbuf->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

			cmdbuf->bindGraphicsPipeline(m_pipeline.get());
			{
				const nbl::asset::SBufferBinding<const nbl::video::IGPUBuffer> binding = {
					.offset = 0,
					.buffer = smart_refctd_ptr<const nbl::video::IGPUBuffer>(m_vertexBuffer)
				};
				cmdbuf->bindVertexBuffers(0, 1, &binding);
			}
			{
				const nbl::asset::SBufferBinding<const nbl::video::IGPUBuffer> binding = {
					.offset = 0, // in bytes, must be a multiple of the index size
					.buffer = smart_refctd_ptr<const nbl::video::IGPUBuffer>(m_indexBuffer)
				};
				if (!cmdbuf->bindIndexBuffer(binding, nbl::asset::EIT_16BIT))
				{
					m_logger->log("Failed to bind the index buffer!", nbl::system::ILogger::ELL_ERROR);
					return;
				}
			}

			cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_VERTEX | hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
			cmdbuf->drawIndexed(6, 3, 0, 0, 0);
			cmdbuf->endRenderPass();
		}

		{
			SMemoryBarrier memBarrier;
			memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
			memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_INPUT_BITS;
			memBarrier.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
		}
		//pass 2
		{
			cmdbuf->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			cmdbuf->bindGraphicsPipeline(m_mlabPipeline.get());
			cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_VERTEX | hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
			ext::FullScreenTriangle::recordDrawCall(cmdbuf);
			cmdbuf->endRenderPass();
		}
			cmdbuf->end();

			const IQueue::SSubmitInfo::SSemaphoreInfo waits[] = {{
				.semaphore = acquire.semaphore,
				.value = acquire.acquireCount,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
			}};
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = {{
				.semaphore = m_semaphore.get(),
				.value = m_realFrameIx+1,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
			}};
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = {{.cmdbuf=cmdbuf}};
			const IQueue::SSubmitInfo submits[] = {{
				.waitSemaphores = waits,
				.commandBuffers = commandBuffers,
				.signalSemaphores = rendered
			}};
			if (getGraphicsQueue()->submit(submits)!=IQueue::RESULT::SUCCESS)
			{
				logFail("Failed to submit the Command Buffer!");
				return;
			}
			m_realFrameIx++;

			m_surface->present(acquire.imageIndex,rendered);
		}

		inline bool onAppTerminated() override
		{
			if (m_device)
				m_device->waitIdle();
			return device_base_t::onAppTerminated();
		}

	private:

		
		smart_refctd_ptr<InputSystem> m_inputSystem;
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<surface_t> m_surface;

		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;

		smart_refctd_ptr<IGPUPipelineLayout> m_pipelineLayout;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_mlabPipeline;

		InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;
		video::CDumbPresentationOracle m_oracle;

		core::smart_refctd_ptr<ext::cameras::CFPSCamera> m_camera;
		ext::cameras::CCameraMouseKeyboardController m_cameraController;
		hlsl::float32_t4x4 m_cameraProjection = hlsl::float32_t4x4(1.0f);

		static constexpr uint32_t QuadCount = 3;


		smart_refctd_ptr<nbl::video::IGPUBuffer> m_vertexBuffer;
		smart_refctd_ptr<nbl::video::IGPUBuffer> m_indexBuffer;

		uint32_t BUFFER_SIZE = WIN_H * WIN_W * sizeof(Smlab) * 4;
		smart_refctd_ptr<IGPUBuffer> outBuffer;


		inline CDefaultSwapchainFramebuffers* getSwapchainFramebuffers()
		{
			return static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
		}

		static const IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies()
		{
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {

				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
						.dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				},
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};
			return dependencies;
		}

		inline bool createSwapchain()
		{
			ISwapchain::SCreationParams swapchainParams = {.surface=smart_refctd_ptr<ISurface>(m_surface->getSurface())};
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(),swapchainParams.surfaceFormat.format,getDefaultSubpassDependencies());
			if (!scResources->getRenderpass())
				return logFail("Failed to create the Renderpass!");

			if (!m_surface->init(getGraphicsQueue(),std::move(scResources),swapchainParams.sharedParams))
				return logFail("Failed to initialize the Surface!");

			m_winMgr->setWindowSize(m_window.get(),WIN_W,WIN_H);
			if (!m_surface->recreateSwapchain())
				return logFail("Failed to create the Swapchain!");
			return true;
		}
		bool createVertexBuffer()
		{
			using buffer_t = nbl::video::IGPUBuffer;

			const SVertex vertices[4] = {
				{.position = { -0.5f,0.5f}}, 
				{.position = { 0.5f, 0.5f}}, 
				{.position = {-0.5f, -0.5f}},
				{.position = {0.5f, -0.5f}}
			};

			buffer_t::SCreationParams params = {};
			params.size = sizeof(vertices);

			params.usage = buffer_t::EUF_VERTEX_BUFFER_BIT | buffer_t::EUF_TRANSFER_DST_BIT;

	
			m_utils->createFilledDeviceLocalBufferOnDedMem(
				nbl::video::SIntendedSubmitInfo{ .queue = getGraphicsQueue() }, std::move(params), vertices
			).move_into(m_vertexBuffer);
			if (!m_vertexBuffer)
			{
				m_logger->log("Failed to create the vertex buffer!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}
			return true;
		}

		bool createIndexBuffer()
		{
			using buffer_t = nbl::video::IGPUBuffer;

			const uint16_t indices[6] = { 0, 1, 2, 1,3,2 };

			buffer_t::SCreationParams params = {};
			params.size = sizeof(indices);

			params.usage = buffer_t::EUF_INDEX_BUFFER_BIT | buffer_t::EUF_TRANSFER_DST_BIT;

			m_utils->createFilledDeviceLocalBufferOnDedMem(
				nbl::video::SIntendedSubmitInfo{ .queue = getGraphicsQueue() }, std::move(params), indices
			).move_into(m_indexBuffer);
			if (!m_indexBuffer)
			{
				m_logger->log("Failed to create the index buffer!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}
			return true;
		}

};

NBL_MAIN_FUNC(oitApp)
