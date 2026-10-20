// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/examples/examples.hpp"

#include "nbl/this_example/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

class CSwapchainImageViews final : public ISimpleManagedSurface::ISwapchainResources
{
	public:
		inline CSwapchainImageViews(ILogicalDevice* device) : m_device(device) {}

		inline IGPUImageView* getImageView(const uint8_t imageIx)
		{
			if (imageIx<m_imageViews.size())
				return m_imageViews[imageIx].get();
			return nullptr;
		}

	protected:
		inline void invalidate_impl() override
		{
			std::fill(m_imageViews.begin(),m_imageViews.end(),nullptr);
		}

		inline bool onCreateSwapchain_impl(const uint8_t qFam) override
		{
			const auto count = getSwapchain()->getImageCount();
			for (uint32_t i=0u; i<count; i++)
			{
				m_imageViews[i] = m_device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = smart_refctd_ptr<IGPUImage>(getImage(i)),
					.viewType = IGPUImageView::ET_2D,
					.format = getImage(i)->getCreationParameters().format
				});
				if (!m_imageViews[i])
					return false;
			}
			return true;
		}

		ILogicalDevice* const m_device;
		std::array<smart_refctd_ptr<IGPUImageView>,ISwapchain::MaxImages> m_imageViews;
};

class DynamicRenderingApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
		using device_base_t = SimpleWindowedApplication;
		using asset_base_t = BuiltinResourcesApplication;
		using surface_t = CSimpleResizeSurface<CSwapchainImageViews>;
		using clock_t = std::chrono::steady_clock;

		constexpr static inline uint32_t WIN_W = 1280;
		constexpr static inline uint32_t WIN_H = 720;
		constexpr static inline uint8_t MaxFramesInFlight = 3;

	public:
		inline DynamicRenderingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
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
					params.windowCaption = "DynamicRendering";
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

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to create a Semaphore!");

			m_cmdPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!m_cmdPool || !m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,m_cmdBufs,smart_refctd_ptr(m_logger)))
				return logFail("Failed to create Command Buffers!");

			auto loadPrecompiledShader = [&]<core::StringLiteral ShaderKey>()->smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "app_resources";

				auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_physicalDevice->getLimits(),m_physicalDevice->getFeatures());
				auto assetBundle = m_assetMgr->getAsset(key.data(),lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return nullptr;
				return IAsset::castDown<IShader>(assets[0]);
			};

			auto vertexShader = loadPrecompiledShader.operator()<"triangle_vert">();
			auto fragmentShader = loadPrecompiledShader.operator()<"triangle_frag">();
			if (!vertexShader || !fragmentShader)
				return logFail("Failed to load the triangle shaders!");

			m_pipelineLayout = m_device->createPipelineLayout();
			if (!m_pipelineLayout)
				return logFail("Failed to create the Pipeline Layout!");

			{
				IGPUGraphicsPipeline::SCreationParams params[1] = {};
				params[0].layout = m_pipelineLayout.get();
				params[0].vertexShader = {.shader=vertexShader.get(),.entryPoint="VSMain"};
				params[0].fragmentShader = {.shader=fragmentShader.get(),.entryPoint="PSMain"};
				params[0].renderpass = nullptr;
				params[0].cached.dynamicRendering.colorAttachmentFormats[0] = m_swapchainFormat;
				params[0].cached.vertexInput = {};
				params[0].cached.primitiveAssembly.primitiveType = EPT_TRIANGLE_LIST;
				params[0].cached.rasterization.faceCullingMode = EFCM_NONE;
				params[0].cached.rasterization.depthWriteEnable = false;
				params[0].cached.rasterization.depthCompareOp = ECO_ALWAYS;

				if (!m_device->createGraphicsPipelines(nullptr,params,&m_pipeline) || !m_pipeline)
					return logFail("Failed to create the Graphics Pipeline!");
			}

			if (argv.size()>=3 && argv[1]=="-timeout_seconds")
				m_timeout = std::chrono::seconds(std::atoi(argv[2].c_str()));
			m_start = clock_t::now();

			m_winMgr->show(m_window.get());
			return true;
		}

		inline bool keepRunning() override
		{
			if (duration_cast<decltype(m_timeout)>(clock_t::now()-m_start)>m_timeout)
				return false;
			return m_surface->isWindowOpen() && !m_surface->irrecoverable();
		}

		inline void workLoopBody() override
		{
			const uint32_t framesInFlight = core::min<uint32_t>(MaxFramesInFlight,m_surface->getMaxAcquiresInFlight());
			if (m_realFrameIx>=framesInFlight)
			{
				const ISemaphore::SWaitInfo framesDone[] = {{.semaphore=m_semaphore.get(),.value=m_realFrameIx+1-framesInFlight}};
				if (m_device->blockForSemaphores(framesDone)!=ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			const auto acquire = m_surface->acquireNextImage();
			if (!acquire)
				return;

			auto* const swapchainResources = static_cast<CSwapchainImageViews*>(m_surface->getSwapchainResources());
			auto* const imageView = swapchainResources->getImageView(acquire.imageIndex);
			auto* const image = swapchainResources->getImage(acquire.imageIndex);
			const auto& imageParams = image->getCreationParameters();
			const VkRect2D renderArea = {.offset={0,0},.extent={imageParams.extent.width,imageParams.extent.height}};

			auto* const cmdbuf = m_cmdBufs[m_realFrameIx%MaxFramesInFlight].get();
			bool success = cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			success &= cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			const IGPUImage::SSubresourceRange colorRange = {
				.aspectMask = IGPUImage::EAF_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			};
			using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
			{
				const image_barrier_t barriers[] = {{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						}
					},
					.image = image,
					.subresourceRange = colorRange,
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
					.newLayout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL
				}};
				success &= cmdbuf->pipelineBarrier(EDF_NONE,{.imgBarriers=barriers});
			}

			const SViewport viewport = {
				.x = 0.f,
				.y = 0.f,
				.width = float(renderArea.extent.width),
				.height = float(renderArea.extent.height),
				.minDepth = 0.f,
				.maxDepth = 1.f
			};
			success &= cmdbuf->setViewport({&viewport,1});
			success &= cmdbuf->setScissor({&renderArea,1});

			const IGPUCommandBuffer::SColorAttachmentInfo colorAttachments[] = {{
				{.imageView=imageView},
				/*.imageLayout = */IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL,
				/*.resolveImageLayout = */IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL,
				/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
				/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
				/*.clearValue = */{.float32={0.1f,0.1f,0.1f,1.f}}
			}};
			const IGPUCommandBuffer::SRenderingInfo renderingInfo = {
				.renderArea = renderArea,
				.layerCount = 1,
				.colorAttachments = colorAttachments
			};
			success &= cmdbuf->beginRendering(renderingInfo);
			success &= cmdbuf->bindGraphicsPipeline(m_pipeline.get());
			success &= cmdbuf->draw(3,1,0,0);
			success &= cmdbuf->endRendering();

			{
				const image_barrier_t barriers[] = {{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
							.dstStageMask = PIPELINE_STAGE_FLAGS::NONE,
							.dstAccessMask = ACCESS_FLAGS::NONE
						}
					},
					.image = image,
					.subresourceRange = colorRange,
					.oldLayout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL,
					.newLayout = IGPUImage::LAYOUT::PRESENT_SRC
				}};
				success &= cmdbuf->pipelineBarrier(EDF_NONE,{.imgBarriers=barriers});
			}
			success &= cmdbuf->end();
			if (!success)
			{
				logFail("Failed to record the Command Buffer!");
				return;
			}

			const IQueue::SSubmitInfo::SSemaphoreInfo waits[] = {{
				.semaphore = acquire.semaphore,
				.value = acquire.acquireCount,
				.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
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
			m_logger->log("Rendered %llu frames with dynamic rendering",ILogger::ELL_INFO,m_realFrameIx);
			return device_base_t::onAppTerminated();
		}

	private:
		inline bool createSwapchain()
		{
			ISwapchain::SCreationParams swapchainParams = {.surface=smart_refctd_ptr<ISurface>(m_surface->getSurface())};
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");
			m_swapchainFormat = swapchainParams.surfaceFormat.format;

			if (!m_surface->init(getGraphicsQueue(),std::make_unique<CSwapchainImageViews>(m_device.get()),swapchainParams.sharedParams))
				return logFail("Failed to initialize the Surface!");

			m_winMgr->setWindowSize(m_window.get(),WIN_W,WIN_H);
			if (!m_surface->recreateSwapchain())
				return logFail("Failed to create the Swapchain!");
			return true;
		}

		smart_refctd_ptr<InputSystem> m_inputSystem;
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<surface_t> m_surface;
		E_FORMAT m_swapchainFormat = EF_UNKNOWN;

		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;

		smart_refctd_ptr<IGPUPipelineLayout> m_pipelineLayout;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;

		std::chrono::seconds m_timeout = std::chrono::seconds(0x7fffFFFFu);
		clock_t::time_point m_start;
};

NBL_MAIN_FUNC(DynamicRenderingApp)
