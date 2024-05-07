// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "../common/SimpleWindowedApplication.hpp"

// TODO: move back to Nabla
#include "CAssetConverter.h"

//
#include "nbl/video/surface/CSurfaceVulkan.h"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

#include "app_resources/push_constants.hlsl"

// Why not have multiple Frame In Flight depth buffers? Drawing two frames in parallel **on the GPU** is wasteful and could actually increase latency.
class CSwapchainFramebuffersAndDepth final : public nbl::video::CDefaultSwapchainFramebuffers
{
		using base_t = CDefaultSwapchainFramebuffers;

	public:
		template<typename... Args>
		inline CSwapchainFramebuffersAndDepth(ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args) : CDefaultSwapchainFramebuffers(device,std::forward<Args>(args)...)
		{
			const IPhysicalDevice::SImageFormatPromotionRequest req = {
				.originalFormat = _desiredDepthFormat,
				.usages = {IGPUImage::EUF_RENDER_ATTACHMENT_BIT}
			};
			m_depthFormat = m_device->getPhysicalDevice()->promoteImageFormat(req,IGPUImage::TILING::OPTIMAL);
			
			const static IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
				{{
					{
						.format = m_depthFormat,
						.samples = IGPUImage::ESCF_1_BIT,
						.mayAlias = false
					},
					/*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
					/*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
					/*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED}, // because we clear we don't care about contents
					/*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} // transition to presentation right away so we can skip a barrier
				}},
				IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
			};
			m_params.depthStencilAttachments = depthAttachments;
			
			static IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
				m_params.subpasses[0],
				IGPURenderpass::SCreationParams::SubpassesEnd
			};
			subpasses[0].depthStencilAttachment.render = {.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL};
			m_params.subpasses = subpasses;
		}

	protected:
		inline bool onCreateSwapchain_impl(const uint8_t qFam) override
		{
			auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

			const auto depthFormat = m_renderpass->getCreationParameters().depthStencilAttachments[0].format;
			const auto& sharedParams = getSwapchain()->getCreationParameters().sharedParams;
			auto image = device->createImage({IImage::SCreationParams{
				.type = IGPUImage::ET_2D,
				.samples = IGPUImage::ESCF_1_BIT,
				.format = depthFormat,
				.extent = {sharedParams.width,sharedParams.height,1},
				.mipLevels = 1,
				.arrayLayers = 1,
				.depthUsage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT
			}});

			device->allocate(image->getMemoryReqs(),image.get());

			m_depthBuffer = device->createImageView({
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
				.image = std::move(image),
				.viewType = IGPUImageView::ET_2D,
				.format = depthFormat,
				.subresourceRange = {IGPUImage::EAF_DEPTH_BIT,0,1,0,1}
			});

			const auto retval = base_t::onCreateSwapchain_impl(qFam);
			m_depthBuffer = nullptr;
			return retval;
		}

		inline smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params) override
		{
			params.depthStencilAttachments = &m_depthBuffer.get();
			return m_device->createFramebuffer(std::move(params));
		}

		E_FORMAT m_depthFormat;
		// only used to pass a parameter from `onCreateSwapchain_impl` to `createFramebuffer`
		smart_refctd_ptr<IGPUImageView> m_depthBuffer;
};

class DepthBufferAndCameraSampleApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		inline DepthBufferAndCameraSampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}
		
		// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			// So let's create our Window and Surface then!
			if (!m_surface)
			{
				{
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = 1280;
					params.height = 720;
					params.x = 32;
					params.y = 32;
					// Don't want to have a window lingering about before we're ready so create it hidden.
					params.flags = ui::IWindow::ECF_HIDDEN|IWindow::ECF_BORDERLESS;
					params.windowCaption = "DepthBufferAndCameraSampleApp";
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}
				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
			}
			if (m_surface)
				return {{m_surface->getSurface()/*,EQF_NONE*/}};
			return {};
		}
		
		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			// Test Asset Converter


#if 0
			// Load Custom Shader
			auto loadCompileAndCreateShader = [&](const std::string& relPath) -> smart_refctd_ptr<IGPUShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset(relPath,lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return nullptr;

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto source = IAsset::castDown<ICPUShader>(assets[0]);
				if (!source)
					return nullptr;

				return m_device->createShader(source.get());
			};
			auto fragmentShader = loadCompileAndCreateShader("app_resources/present.frag.hlsl");
			if (!fragmentShader)
				return logFail("Failed to Load and Compile Fragment Shader!");

			// create the descriptor sets layout
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				auto defaultSampler = m_device->createSampler({
					.AnisotropicFilter = 0
				});

				const IGPUDescriptorSetLayout::SBinding bindings[1] = {{
					.binding = 0,
					.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::ESS_FRAGMENT,
					.count = 1,
					.samplers = &defaultSampler
				}};
				dsLayout = m_device->createDescriptorSetLayout(bindings);
				if (!dsLayout)
					return logFail("Failed to Create Descriptor Layout");
			}

			// Now create the pipeline
			{
				const asset::SPushConstantRange range = {
					.stageFlags = IShader::ESS_FRAGMENT,
					.offset = 0,
					.size = sizeof(push_constants_t)
				};
				auto layout = m_device->createPipelineLayout({&range,1},nullptr,nullptr,nullptr,core::smart_refctd_ptr(dsLayout));
				const IGPUShader::SSpecInfo fragSpec = {
					.entryPoint = "main",
					.shader = fragmentShader.get()
				};
				m_pipeline = fsTriProtoPPln.createPipeline(fragSpec,layout.get(),scResources->getRenderpass()/*,default is subpass 0*/);
				if (!m_pipeline)
					return logFail("Could not create Graphics Pipeline!");
			}
#endif 
			ISwapchain::SCreationParams swapchainParams = {.surface=m_surface->getSurface()};
			// Need to choose a surface format
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			// Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition of Color to ATTACHMENT_OPTIMAL
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
						// last place where the depth can get modified in previous frame
						.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
						// only write ops, reads can't be made available
						.srcAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						// destination needs to wait as early as possible
						.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
						// because of depth test needing a read and a write
						.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT
					}
					// leave view offsets and flags default
				},
				// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						// last place where the depth can get modified
						.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						// only write ops, reads can't be made available
						.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						// spec says nothing is needed when presentation is the destination
					}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};
			auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(),EF_D16_UNORM,swapchainParams.surfaceFormat.format,dependencies);
			if (!scResources->getRenderpass())
				return logFail("Failed to create Renderpass!");

			m_semaphore = m_device->createSemaphore(m_submitIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			auto gfxQueue = getGraphicsQueue();
			// Let's just use the same queue since there's no need for async present
			if (!m_surface || !m_surface->init(gfxQueue,std::move(scResources),swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");
			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();

#if 0
			// create the descriptor sets, 1 per FIF and with enough room for one image sampler
			{
				const uint32_t setCount = m_maxFramesInFlight;
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE,{&dsLayout.get(),1},&setCount);
				if (!pool)
					return logFail("Failed to Create Descriptor Pool");

				for (auto i=0u; i<m_maxFramesInFlight; i++)
				{
					m_descriptorSets[i] = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
					if (!m_descriptorSets[i])
						return logFail("Could not create Descriptor Set!");
				}
			}
#endif
#if 0
			// load the image view
			system::path filename, extension;
			smart_refctd_ptr<ICPUImageView> cpuImgView;
			{
				m_logger->log("Loading image from path %s", ILogger::ELL_INFO, m_nextPath.c_str());

				constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
				const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get(), m_loadCWD);
				auto bundle = m_assetMgr->getAsset(m_nextPath, loadParams);
				auto contents = bundle.getContents();
				if (contents.empty())
				{
					m_logger->log("Failed to load image with path %s, skipping!", ILogger::ELL_ERROR, (m_loadCWD / m_nextPath).c_str());
					return;
				}

				core::splitFilename(m_nextPath.c_str(), nullptr, &filename, &extension);
			}
#endif
			
			// create the commandbuffers
			for (auto i=0u; i<m_maxFramesInFlight; i++)
			{
				m_cmdPools[i] = m_device->createCommandPool(gfxQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::NONE);
				if (!m_cmdPools[i])
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
					return logFail("Couldn't create Command Buffer!");
			}

			// Now show the window
			return m_winMgr->show(m_window.get());
		}

		// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
		inline void workLoopBody() override
		{
			// Can't reset a cmdbuffer before the previous use of commandbuffer is finished!
			if (m_submitIx>=m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[] = {
					{ 
						.semaphore = m_semaphore.get(),
						.value = m_submitIx+1-m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cmdbufDonePending)!=ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}
			const auto resourceIx = m_submitIx%m_maxFramesInFlight;

			if (!m_cmdPools[resourceIx]->reset())
				return;
			auto cmdbuf = m_cmdBufs[resourceIx].get();
			
			// Acquire
			auto acquire = m_surface->acquireNextImage();
			if (!acquire)
				return;

			// Render to the Image
			{
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				// begin the renderpass
				{
					const IGPUCommandBuffer::SClearColorValue colorValue = { .float32 = {1.f,0.f,1.f,1.f} };
					// we encourage Reverse-Z by default
					const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
					auto fb = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources())->getFramebuffer(acquire.imageIndex);
					const IGPUCommandBuffer::SRenderpassBeginInfo info = {
						.framebuffer = fb,
						.colorClearValues = &colorValue,
						.depthStencilClearValues = &depthValue,
						.renderArea = {
							.offset = {0,0},
							.extent = {fb->getCreationParameters().width,fb->getCreationParameters().height}
						}
					};
					cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::SECONDARY_COMMAND_BUFFERS);
				}

#if 0
				const push_constants_t pc = { .mvp = () };
				cmdbuf->pushConstants(m_pipeline->getLayout(), IGPUShader::ESS_VERTEX, 0, sizeof(pc), &pc);

				// TODO: secondary commandbuffer
				const VkRect2D currentRenderArea =
				{
					.offset = {0,0},
					.extent = {newWindowResolution.width,newWindowResolution.height}
				};
				// set viewport
				{
					const asset::SViewport viewport =
					{
						.width = float(newWindowResolution.width),
						.height = float(newWindowResolution.height)
					};
					cmdbuf->setViewport({&viewport,1});
				}
				cmdbuf->setScissor({&currentRenderArea,1});
				cmdbuf->bindGraphicsPipeline(m_pipeline.get());
				cmdbuf->pushConstants(m_pipeline->getLayout(),IGPUShader::ESS_FRAGMENT,0,sizeof(push_constants_t),&pc);
				cmdbuf->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS,m_pipeline->getLayout(),3,1,&ds);
				ext::FullScreenTriangle::recordDrawCall(cmdbuf);
#endif
				cmdbuf->endRenderPass();
				cmdbuf->end();
			}
			
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
				.semaphore = m_semaphore.get(),
				.value = ++m_submitIx,
				// just as we've outputted all pixels, signal
				.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
			}};

			auto queue = getGraphicsQueue();
			queue->startCapture();
			// submit
			{
				{
					const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{
						.cmdbuf = cmdbuf
					}};
					// we don't need to wait for the transfer semaphore, because we submit everything to the same queue
					const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {{
						.semaphore = acquire.semaphore,
						.value = acquire.acquireCount,
						.stageMask = PIPELINE_STAGE_FLAGS::NONE
					}};
					const IQueue::SSubmitInfo infos[1] = {{
						.waitSemaphores = acquired,
						.commandBuffers = commandBuffers,
						.signalSemaphores = rendered
					}};
					// we won't signal the sema if no success
					if (queue->submit(infos)!=IQueue::RESULT::SUCCESS)
						m_submitIx--;
				}
			}

			// Present
			m_surface->present(acquire.imageIndex,rendered);
			queue->endCapture();
		}

		inline bool keepRunning() override
		{
			// Keep arunning as long as we have a surface to present to (usually this means, as long as the window is open)
			return dynamic_cast<ISimpleManagedSurface::ICallback*>(m_window->getEventCallback())->isWindowOpen() && !m_surface->irrecoverable();
		}

		inline bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	protected:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Use a separate counter to cycle through our resources for clarity
		uint64_t m_submitIx : 59 = 0;
		// Maximum frames which can be simultaneously rendered
		uint64_t m_maxFramesInFlight : 5;
		// pipeline for the draw
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		// Use the One Pool Per Frame Paradigm to have most efficient pool usage
		std::array<smart_refctd_ptr<IGPUCommandPool>,ISwapchain::MaxImages> m_cmdPools;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;
};

NBL_MAIN_FUNC(DepthBufferAndCameraSampleApp)