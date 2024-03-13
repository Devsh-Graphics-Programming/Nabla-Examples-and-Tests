// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "../common/SimpleWindowedApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

//
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/CVulkanSwapchain.h"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// Just a class to hold framebuffers derived from swapchain images
// WARNING: It assumes the format won't change between swapchain recreates!
class CDefaultSwapchainFramebuffers : public ISimpleManagedSurface::ISwapchainResources
{
	public:
		inline CDefaultSwapchainFramebuffers(core::smart_refctd_ptr<IGPURenderpass>&& _renderpass) : m_renderpass(std::move(_renderpass)) {}

	protected:
		core::smart_refctd_ptr<IGPURenderpass> m_renderpass;
};

class ColorSpaceTestSampleApp final : public examples::SimpleWindowedApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;
		using clock_t = std::chrono::steady_clock;

		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
		constexpr static inline std::string_view DefaultImagePathsFile = "../imagesTestList.txt";

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		inline ColorSpaceTestSampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
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
					params.width = 256;
					params.height = 256;
					params.x = 32;
					params.y = 32;
					// Don't want to have a window lingering about before we're ready so create it hidden.
					// Only programmatic resize, not regular.
					params.flags = ui::IWindow::ECF_HIDDEN|IWindow::ECF_BORDERLESS|IWindow::ECF_RESIZABLE;
					params.windowCaption = "ColorSpaceTestSampleApp";
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}
				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<CDefaultSwapchainFramebuffers>::create(std::move(surface));
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
			
			// get list of files to test
			system::path m_loadCWD = DefaultImagePathsFile;
			if (IApplicationFramework::argv.size()>=2)
			{
				m_testPathsFile = std::ifstream(argv[1]);
				if (m_testPathsFile.is_open())
					m_loadCWD = argv[1];
				else
					m_logger->log("Couldn't open test file given by argument 1 = %s, falling back to default!",ILogger::ELL_ERROR,argv[1].c_str());
			}

			if (!m_testPathsFile.is_open())
				m_testPathsFile = std::ifstream(m_loadCWD);

			if (!m_testPathsFile.is_open())
				return logFail("Could not open the test paths file");
			m_loadCWD = m_loadCWD.parent_path();
			
			// Now surface indep resources
			m_semaphore = m_device->createSemaphore(m_submitIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			// TODO: Use the widest gamut possible
			const auto format = asset::EF_R8G8B8A8_SRGB;

			// Create our Swapchain Resources and Initialize the Managed Surface
			{
				smart_refctd_ptr<IGPURenderpass> renderpass;
				// Create the renderpass
				{
					//
					const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
						{{
							.format = format,
							.samples = IGPUImage::ESCF_1_BIT,
							.mayAlias = false,
							.loadOp = IGPURenderpass::LOAD_OP::CLEAR,
							.storeOp = IGPURenderpass::STORE_OP::STORE,
							.initialLayout = IGPUImage::LAYOUT::UNDEFINED, // because we clear we don't care about contents
							.finalLayout = IGPUImage::LAYOUT::PRESENT_SRC // transition to presentation right away so we can skip a barrier
						}},
						IGPURenderpass::SCreationParams::ColorAttachmentsEnd
					};
					IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
						{},
						IGPURenderpass::SCreationParams::SubpassesEnd
					};
					subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
					// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
					const IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
						// wipe-transition to ATTACHMENT_OPTIMAL
						{
							.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
							.dstSubpass = 0,
							.memoryBarrier = {
								// since we're uploading the image data we're about to draw 
								.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
								.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
								.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								// because we clear and don't blend
								.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
							}
							// leave view offsets and flags default
						},
						// ATTACHMENT_OPTIMAL to PRESENT_SRC
						{
							.srcSubpass = 0,
							.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
							.memoryBarrier = {
								.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
								// we can have NONE as the Destinations because the spec says so about presents
							}
							// leave view offsets and flags default
						},
						IGPURenderpass::SCreationParams::DependenciesEnd
					};

					IGPURenderpass::SCreationParams params = {};
					params.colorAttachments = colorAttachments;
					params.subpasses = subpasses;
					params.dependencies = dependencies;
					renderpass = m_device->createRenderpass(params);
					if (!renderpass)
						return logFail("Failed to Create a Renderpass!");
				}

				// Let's just use the same queue since there's no need for async present
				if (!m_surface || !m_surface->init(getGraphicsQueue(),std::make_unique<CDefaultSwapchainFramebuffers>(std::move(renderpass)),{}))
					return logFail("Could not create Window & Surface or initialize the Surface!");
			}

			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();

			// create the descriptor sets, 1 per FIF and with enough room for one image sampler
			{
				auto defaultSampler = m_device->createSampler({
					.AnisotropicFilter = 0
				});

				const IGPUDescriptorSetLayout::SBinding bindings[1] = {{
					.binding = 0,
					.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::ESS_COMPUTE,
					.count = 1,
					.samplers = &defaultSampler
				}};
				auto layout = m_device->createDescriptorSetLayout(bindings);

				const uint32_t setCount = m_maxFramesInFlight;
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE,{&layout.get(),1},&setCount);
				if (!pool)
					return logFail("Failed to Create Descriptor Pool");


			}

			// create the commandbuffers and pools, this time properly 1 pool per FIF
			for (auto i=0u; i<m_maxFramesInFlight; i++)
			{
				return false;
			}

			return true;
		}

		// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
		inline void workLoopBody() override
		{
			// load the image view
			system::path filename, extension;
			smart_refctd_ptr<ICPUImageView> cpuImgView;
			{
				m_logger->log("Loading image from path %s",ILogger::ELL_INFO,m_nextPath.c_str());

				constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
				const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get(), m_loadCWD);
				auto bundle = m_assetMgr->getAsset(m_nextPath,loadParams);
				auto contents = bundle.getContents();
				if (contents.empty())
				{
					m_logger->log("Failed to load image with path %s, skipping!",ILogger::ELL_ERROR,(m_loadCWD/m_nextPath).c_str());
					return;
				}

				core::splitFilename(m_nextPath.c_str(),nullptr,&filename,&extension);

				const auto& asset = contents[0];
				switch (asset->getAssetType())
				{
					case IAsset::ET_IMAGE:
					{
						auto image = smart_refctd_ptr_static_cast<ICPUImage>(asset);
						const auto format = image->getCreationParameters().format;

						ICPUImageView::SCreationParams viewParams = {
							.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
							.image = std::move(image),
							.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
							.format = format,
							.subresourceRange = {
								.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = 1u
							}
						};

						cpuImgView = ICPUImageView::create(std::move(viewParams));
					} break;

					case IAsset::ET_IMAGE_VIEW:
						cpuImgView = smart_refctd_ptr_static_cast<ICPUImageView>(asset);
						break;
					default:
						m_logger->log("Failed to load ICPUImage or ICPUImageView got some other Asset Type, skipping!",ILogger::ELL_ERROR);
						return;
				}
			}
			
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

			std::this_thread::sleep_until(m_lastImageEnqueued+DisplayImageDuration);
			m_lastImageEnqueued = clock_t::now();

			const auto newWindowResolution = cpuImgView->getCreationParameters().image->getCreationParameters().extent;
			if (newWindowResolution.width!=m_window->getWidth() || newWindowResolution.height!=m_window->getHeight())
			{
				// Resize the window
				m_winMgr->setWindowSize(m_window.get(),newWindowResolution.width,newWindowResolution.height);
				// The swapchain will recreate automatically during acquire
			}
			// Now show the window (ideally should happen just after present, but don't want to mess with acquire/recreation)
			m_winMgr->show(m_window.get());

			// Acquire
			auto imageIx = m_surface->acquireNextImage();
			if (imageIx==ISwapchain::MaxImages)
				return;

			// Render to the Image
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
				.semaphore = m_semaphore.get(),
				.value = ++m_submitIx,
				// just as we've outputted all pixels, signal
				.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
			}};
			{
				const auto resourceIx = m_submitIx%m_maxFramesInFlight;

				// submit
				{
					const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {{
						.semaphore = m_surface->getAcquireSemaphore(),
						.value = m_surface->getAcquireCount(),
						.stageMask = PIPELINE_STAGE_FLAGS::NONE
					}};
					const IQueue::SSubmitInfo infos[1] = {{
						.waitSemaphores = acquired,
						.commandBuffers = {},
						.signalSemaphores = rendered
					}};
					// we won't signal the sema if no success
					if (getGraphicsQueue()->submit(infos)!=IQueue::RESULT::SUCCESS)
						m_submitIx--;
				}
			}

			// Present
			m_surface->present(imageIx,rendered);
			
			// Set the Caption
			std::string viewTypeStr;
			switch (cpuImgView->getCreationParameters().viewType)
			{
				case IImageView<video::IGPUImage>::ET_2D:
					viewTypeStr = "ET_2D";
				case IImageView<video::IGPUImage>::ET_2D_ARRAY:
					viewTypeStr = "ET_2D_ARRAY";
					break;
				case IImageView<video::IGPUImage>::ET_CUBE_MAP:
					viewTypeStr = "ET_CUBE_MAP";
					break;
				default:
					assert(false);
					break;
			};
			m_window->setCaption("[Nabla Engine] Color Space Test Demo - CURRENT IMAGE: " + filename.string() + " - VIEW TYPE: " + viewTypeStr + " - EXTENSION: " + extension.string());

			// Now do a write to disk in the meantime
			{
				const std::string assetPath = "imageAsset_" + filename.string() + extension.string();

				auto tryToWrite = [&](IAsset* asset)->bool
				{
					IAssetWriter::SAssetWriteParams wparams(asset);
					wparams.workingDirectory = localOutputCWD;
					return m_assetMgr->writeAsset(assetPath,wparams);
				};

				// try write as an image, else try as image view
				if (!tryToWrite(cpuImgView->getCreationParameters().image.get()))
					if (!tryToWrite(cpuImgView.get()))
						m_logger->log("Failed to write %s to disk!",ILogger::ELL_ERROR,assetPath.c_str());
			}

			// Block so we can reuse the resources without frames in flight (cmon we do like one swap every 900ms)
		}

		inline bool keepRunning() override
		{
			// Keep arunning as long as we have a surface to present to (usually this means, as long as the window is open)
			if (m_surface->irrecoverable())
				return false;

			while (std::getline(m_testPathsFile,m_nextPath))
			if (m_nextPath!="" && m_nextPath[0]!=';')
				return true;
			// no more inputs in the file
			return false;
		}

		inline bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	protected:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
		//
		std::ifstream m_testPathsFile;
		system::path m_loadCWD;
		//
		std::string m_nextPath;
		clock_t::time_point m_lastImageEnqueued = {};
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Use a separate counter to cycle through our resources for clarity
		uint64_t m_submitIx : 59 = 0;
		// Maximum frames which can be simultaneously rendered
		uint64_t m_maxFramesInFlight : 5;
		// Simple Renderpass
		smart_refctd_ptr<IGPURenderpass> m_renderpass;
};
#if 0
class ColorSpaceTestSampleApp : public ApplicationBase
{
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;

	core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUFramebuffer>> fbos;
	std::array<std::array<core::smart_refctd_ptr<video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<system::ISystem> system;
	core::smart_refctd_ptr<asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	video::ISwapchain::SCreationParams m_swapchainCreationParams;

public:
	void onAppInitialized_impl() override
	{

		video::IGPUObjectFromAssetConverter cpu2gpu;

		auto createDescriptorPool = [&](const uint32_t textureCount)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				video::IDescriptorPool::SCreateInfo createInfo;
				createInfo.maxSets = maxItemCount;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)] = textureCount;
				return logicalDevice->createDescriptorPool(std::move(createInfo));
			}
		};

		asset::ISampler::SParams samplerParams = { asset::ISampler::ETC_CLAMP_TO_EDGE, asset::ISampler::ETC_CLAMP_TO_EDGE, asset::ISampler::ETC_CLAMP_TO_EDGE, asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK, asset::ISampler::ETF_LINEAR, asset::ISampler::ETF_LINEAR, asset::ISampler::ESMM_LINEAR, 0u, false, asset::ECO_ALWAYS };
		auto immutableSampler = logicalDevice->createSampler(samplerParams);

		video::IGPUDescriptorSetLayout::SBinding binding{ 0u, asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, video::IGPUShader::ESS_FRAGMENT, 1u, &immutableSampler };
		auto gpuDescriptorSetLayout3 = logicalDevice->createDescriptorSetLayout(&binding, &binding + 1u);
		auto gpuDescriptorPool = createDescriptorPool(1u); // per single texture
		auto fstProtoPipeline = ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams, 0u);

		auto createGPUPipeline = [&](asset::IImageView<asset::ICPUImage>::E_TYPE typeOfImage) -> core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>
		{
			auto getPathToFragmentShader = [&]() -> std::string
			{
				switch (typeOfImage)
				{
				case asset::IImageView<asset::ICPUImage>::ET_2D:
					return "../present2D.frag";
				case asset::IImageView<asset::ICPUImage>::ET_2D_ARRAY:
					return "../present2DArray.frag";
				case asset::IImageView<asset::ICPUImage>::ET_CUBE_MAP:
					return "../presentCubemap.frag";
				default:
					assert(false);
					return "";
				}
			};

			auto fs_bundle = assetManager->getAsset(getPathToFragmentShader(), {});
			auto fs_contents = fs_bundle.getContents();
			if (fs_contents.empty())
				assert(false);

			asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<asset::ICPUSpecializedShader*>(fs_contents.begin()->get());

			core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
			{
				cpu2gpuParams.beginCommandBuffers();
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
				cpu2gpuParams.waitForCreationToComplete(false);

				if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				gpuFragmentShader = (*gpu_array)[0];
			}

			auto constants = std::get<asset::SPushConstantRange>(fstProtoPipeline);
			auto gpuPipelineLayout = logicalDevice->createPipelineLayout(&constants, &constants + 1, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));
			return ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));
		};

		auto gpuPipelineFor2D = createGPUPipeline(asset::IImageView<asset::ICPUImage>::E_TYPE::ET_2D);
		auto gpuPipelineFor2DArrays = createGPUPipeline(asset::IImageView<asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
		auto gpuPipelineForCubemaps = createGPUPipeline(asset::IImageView<asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);




		core::vector<core::smart_refctd_ptr<asset::ICPUImageView>> cpuImageViews;
		
		// we clone because we need these cpuimages later for directly using upload utilitiy
		core::vector<core::smart_refctd_ptr<asset::ICPUImageView>> clonedCpuImageViews(cpuImageViews.size());
		for(uint32_t i = 0; i < cpuImageViews.size(); ++i)
			clonedCpuImageViews[i] = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(cpuImageViews[i]->clone());
		
		// Allocate and Leave 8MB for image uploads, to test image copy with small memory remaining 
		{
			uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
			uint32_t maxFreeBlock = utilities->getDefaultUpStreamingBuffer()->max_size();
			const uint32_t allocationAlignment = 64u;
			const uint32_t allocationSize = maxFreeBlock - (0x00F0000u * 4u);
			utilities->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
		}

		cpu2gpuParams.beginCommandBuffers();
		auto gpuImageViews = cpu2gpu.getGPUObjectsFromAssets(clonedCpuImageViews.data(), clonedCpuImageViews.data() + clonedCpuImageViews.size(), cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);
		
		if (!gpuImageViews || gpuImageViews->size() < cpuImageViews.size())
			assert(false);

		// Creates GPUImageViews from Loaded CPUImageViews but this time use IUtilities::updateImageViaStagingBuffer directly and only copy sub-regions for testing purposes.
		core::vector<core::smart_refctd_ptr<video::IGPUImageView>> weirdGPUImages;
		{
			// Create GPU Images based on cpuImageViews
			for(uint32_t i = 0; i < cpuImageViews.size(); ++i)
			{
				auto& cpuImageView = cpuImageViews[i];
				auto imageviewCreateParams = cpuImageView->getCreationParameters();
				auto& cpuImage = imageviewCreateParams.image;
				auto imageCreateParams = cpuImage->getCreationParameters();
				
				auto regions = cpuImage->getRegions();
				std::vector<asset::IImage::SBufferCopy> newRegions;
				// Make new regions weird
				for(uint32_t r = 0; r < regions.size(); ++r)
				{
					auto & region = regions[0];
					
					const auto quarterWidth = core::max(region.imageExtent.width / 4, 1u);
					const auto quarterHeight = core::max(region.imageExtent.height / 4, 1u);
					const auto texelBlockInfo = asset::TexelBlockInfo(imageCreateParams.format);
					const auto imageExtentsInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth));

					// Pattern we're trying to achieve (Copy only the regions marked by X)
					// +----+----+
					// |  xx|    |
					// +----+----+
					// |    |xx  |
					// +----+----+
					{
						asset::IImage::SBufferCopy newRegion = region;
						newRegion.imageExtent.width = quarterWidth;
						newRegion.imageExtent.height = quarterHeight;
						newRegion.imageExtent.depth = region.imageExtent.depth;
						newRegion.imageOffset.x = quarterWidth;
						newRegion.imageOffset.y = quarterHeight;
						newRegion.imageOffset.z = 0u;
						auto offsetInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(newRegion.imageOffset.x, newRegion.imageOffset.y, newRegion.imageOffset.z));
						newRegion.bufferOffset =  (offsetInBlocks.y * imageExtentsInBlocks.x + offsetInBlocks.x) * texelBlockInfo.getBlockByteSize();
						newRegion.bufferRowLength = region.imageExtent.width;
						newRegion.bufferImageHeight = region.imageExtent.height;
						newRegions.push_back(newRegion);
					}
					{
						asset::IImage::SBufferCopy newRegion = region;
						newRegion.imageExtent.width = quarterWidth;
						newRegion.imageExtent.height = quarterHeight;
						newRegion.imageExtent.depth = 1u;
						newRegion.imageOffset.x = quarterWidth * 2;
						newRegion.imageOffset.y = quarterHeight * 2;
						newRegion.imageOffset.z = 0u;
						auto offsetInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(newRegion.imageOffset.x, newRegion.imageOffset.y, newRegion.imageOffset.z));
						newRegion.bufferOffset =  (offsetInBlocks.y * imageExtentsInBlocks.x + offsetInBlocks.x) * texelBlockInfo.getBlockByteSize();
						newRegion.bufferRowLength = region.imageExtent.width;
						newRegion.bufferImageHeight = region.imageExtent.height;
						newRegions.push_back(newRegion);
					}
				}
				
				video::IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
				promotionRequest.originalFormat = imageCreateParams.format;
				promotionRequest.usages = imageCreateParams.usage | asset::IImage::EUF_TRANSFER_DST_BIT;
				auto newFormat = physicalDevice->promoteImageFormat(promotionRequest, video::IGPUImage::ET_OPTIMAL);

				video::IGPUImage::SCreationParams gpuImageCreateInfo = {};
				gpuImageCreateInfo.flags = imageCreateParams.flags;
				gpuImageCreateInfo.type = imageCreateParams.type;
				gpuImageCreateInfo.format = newFormat;
				gpuImageCreateInfo.extent = imageCreateParams.extent;
				gpuImageCreateInfo.mipLevels = imageCreateParams.mipLevels;
				gpuImageCreateInfo.arrayLayers = imageCreateParams.arrayLayers;
				gpuImageCreateInfo.samples = imageCreateParams.samples;
				gpuImageCreateInfo.tiling = video::IGPUImage::ET_OPTIMAL;
				gpuImageCreateInfo.usage = imageCreateParams.usage | asset::IImage::EUF_TRANSFER_DST_BIT;
				gpuImageCreateInfo.queueFamilyIndexCount = 0u;
				gpuImageCreateInfo.queueFamilyIndices = nullptr;
				gpuImageCreateInfo.initialLayout = asset::IImage::EL_UNDEFINED;
				auto gpuImage = logicalDevice->createImage(std::move(gpuImageCreateInfo));
				
				auto gpuImageMemReqs = gpuImage->getMemoryReqs();
				gpuImageMemReqs.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
				logicalDevice->allocate(gpuImageMemReqs, gpuImage.get(), video::IDeviceMemoryAllocation::EMAF_NONE);

				video::IGPUImageView::SCreationParams gpuImageViewCreateInfo = {};
				gpuImageViewCreateInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(imageviewCreateParams.flags);
				gpuImageViewCreateInfo.image = gpuImage;
				gpuImageViewCreateInfo.viewType = static_cast<video::IGPUImageView::E_TYPE>(imageviewCreateParams.viewType);
				gpuImageViewCreateInfo.format = gpuImageCreateInfo.format;
				memcpy(&gpuImageViewCreateInfo.components, &imageviewCreateParams.components, sizeof(imageviewCreateParams.components));
				gpuImageViewCreateInfo.subresourceRange = imageviewCreateParams.subresourceRange;
				gpuImageViewCreateInfo.subresourceRange.levelCount = imageCreateParams.mipLevels - imageviewCreateParams.subresourceRange.baseMipLevel;
				
				auto gpuImageView = logicalDevice->createImageView(std::move(gpuImageViewCreateInfo));

				weirdGPUImages.push_back(gpuImageView);

				auto& transferCommandPools = commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP];
				auto transferQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
				core::smart_refctd_ptr<video::IGPUCommandBuffer> transferCmd;
				logicalDevice->createCommandBuffers(transferCommandPools[0u].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmd);
				
				auto transferFence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

				transferCmd->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
				
				video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransition = {};
				layoutTransition.barrier.srcAccessMask = asset::EAF_NONE;
				layoutTransition.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				layoutTransition.oldLayout = asset::IImage::EL_UNDEFINED;
				layoutTransition.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
				layoutTransition.srcQueueFamilyIndex = ~0u;
				layoutTransition.dstQueueFamilyIndex = ~0u;
				layoutTransition.image = gpuImageView->getCreationParameters().image;
				layoutTransition.subresourceRange = gpuImageView->getCreationParameters().subresourceRange;
				transferCmd->pipelineBarrier(asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &layoutTransition);
				
				video::IGPUQueue::SSubmitInfo submit = {};
				submit.commandBufferCount = 1u;
				submit.commandBuffers = &transferCmd.get();
				submit.waitSemaphoreCount = 0u;
				submit.pWaitSemaphores = nullptr;
				submit.pWaitDstStageMask = nullptr;
				core::SRange<const asset::IImage::SBufferCopy> copyRegions(newRegions.data(), newRegions.data() + newRegions.size());
				
				utilities->updateImageViaStagingBufferAutoSubmit( cpuImage->getBuffer(), imageCreateParams.format, gpuImage.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, copyRegions, transferQueue, transferFence.get(), submit);
				// transferCmd->end();

				logicalDevice->blockForFences(1u, &transferFence.get());
			}
		}

		auto getCurrentGPURenderpassIndependentPipeline = [&](video::IGPUImageView* gpuImageView)
		{
			switch (gpuImageView->getCreationParameters().viewType)
			{
			case asset::IImageView<video::IGPUImage>::ET_2D:
			{
				return gpuPipelineFor2D;
			}

			case asset::IImageView<video::IGPUImage>::ET_2D_ARRAY:
			{
				return gpuPipelineFor2DArrays;
			}

			case asset::IImageView<video::IGPUImage>::ET_CUBE_MAP:
			{
				return gpuPipelineForCubemaps;
			}

			default:
				assert(false);
			}
		};

		auto ds = gpuDescriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		auto presentImageOnTheScreen = [&](core::smart_refctd_ptr<video::IGPUImageView> gpuImageView, const NBL_CAPTION_DATA_TO_DISPLAY& captionData)
		{
			// can't just use windowExtent as the actual window size may have been capped by windows
			VkExtent3D imgExtents = { window->getWidth(), window->getHeight(), 1 };

			if (didResize)
			{
				CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, imgExtents.width, imgExtents.height, swapchain);
				assert(swapchain);
				fbos = CommonAPI::createFBOWithSwapchainImages(
					swapchain->getImageCount(), imgExtents.width, imgExtents.height,
					logicalDevice, swapchain, renderpass,
					asset::EF_UNKNOWN
				);

				lastWidth = imgExtents.width;
				lastHeight = imgExtents.height;
			}

			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuImageView;
				info.info.image.sampler = nullptr;
				info.info.image.imageLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
			}

			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = ds.get();
			write.binding = 0u;
			write.arrayElement = 0u;
			write.count = 1u;
			write.descriptorType = asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
			write.info = &info;

			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);

			auto currentGpuRenderpassIndependentPipeline = getCurrentGPURenderpassIndependentPipeline(gpuImageView.get());
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
			{
				video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams = {};
				graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(currentGpuRenderpassIndependentPipeline.get()));
				graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

				gpuGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
			}


			core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

			core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

			const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				logicalDevice->createCommandBuffers(graphicsCommandPools[i].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers+i);
				imageAcquire[i] = logicalDevice->createSemaphore();
				renderFinished[i] = logicalDevice->createSemaphore();
			}

			auto startPoint = std::chrono::high_resolution_clock::now();

			uint32_t imgnum = 0u;
			int32_t resourceIx = -1;
			for (;;)
			{
				resourceIx++;
				if (resourceIx >= FRAMES_IN_FLIGHT)
					resourceIx = 0;

				auto& cb = commandBuffers[resourceIx];
				auto& fence = frameComplete[resourceIx];
				if (fence)
				{
					while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
					{
					}
					logicalDevice->resetFences(1u, &fence.get());
				}
				else
					fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

				auto aPoint = std::chrono::high_resolution_clock::now();
				if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
					break;

				// acquire image 
				swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &imgnum);

				cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);  // TODO: Reset Frame's CommandPool


				video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransition = {};
				layoutTransition.barrier.srcAccessMask = asset::EAF_NONE;
				layoutTransition.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
				layoutTransition.oldLayout = asset::IImage::EL_UNDEFINED;
				layoutTransition.newLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
				layoutTransition.srcQueueFamilyIndex = ~0u;
				layoutTransition.dstQueueFamilyIndex = ~0u;
				layoutTransition.image = gpuImageView->getCreationParameters().image;
				layoutTransition.subresourceRange = gpuImageView->getCreationParameters().subresourceRange;

				cb->pipelineBarrier(asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &layoutTransition);

				asset::SViewport viewport;
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = imgExtents.width;
				viewport.height = imgExtents.height;
				cb->setViewport(0u, 1u, &viewport);

				VkRect2D scissor;
				scissor.offset = { 0, 0 };
				scissor.extent = { imgExtents.width, imgExtents.height };

				cb->setScissor(0u, 1u, &scissor);

				video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
				{
					VkRect2D area;
					area.offset = { 0,0 };
					area.extent = { imgExtents.width, imgExtents.height };
					asset::SClearValue clear;
					clear.color.float32[0] = 1.f;
					clear.color.float32[1] = 1.f;
					clear.color.float32[2] = 1.f;
					clear.color.float32[3] = 1.f;
					beginInfo.clearValueCount = 1u;
					beginInfo.framebuffer = fbos->begin()[imgnum];
					beginInfo.renderpass = renderpass;
					beginInfo.renderArea = area;
					beginInfo.clearValues = &clear;
				}

				cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);
				cb->bindGraphicsPipeline(gpuGraphicsPipeline.get());
				cb->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 3, 1, &ds.get());
				ext::FullScreenTriangle::recordDrawCalls(gpuGraphicsPipeline, 0u, swapchain->getPreTransform(), cb.get());
				cb->endRenderPass();
				cb->end();

				// submit
				// present
			}

			logicalDevice->waitIdle();

			const auto& fboCreationParams = fbos->begin()[imgnum]->getCreationParameters();
			auto gpuSourceImageView = fboCreationParams.attachments[0];

			const std::string writePath = "screenShot_" + captionData.name + ".png";

			return ext::ScreenShot::createScreenShot(
				logicalDevice.get(),
				queues[decltype(initOutput)::EQT_TRANSFER_UP],
				nullptr,
				gpuSourceImageView.get(),
				assetManager.get(),
				writePath,
				asset::IImage::EL_PRESENT_SRC,
				asset::EAF_NONE);
		};

		for (size_t i = 0; i < gpuImageViews->size(); ++i)
		{
			auto gpuImageView = (*gpuImageViews)[i];
			if (gpuImageView)
			{
				auto& captionData = captionTexturesData[i];
		
				bool status = presentImageOnTheScreen(core::smart_refctd_ptr(gpuImageView), captionData);
				assert(status);
			}
		}

		// Now present weird images (sub-region copies)
		for (size_t i = 0; i < weirdGPUImages.size(); ++i)
		{
			auto gpuImageView = weirdGPUImages[i];
			if (gpuImageView)
			{
				NBL_CAPTION_DATA_TO_DISPLAY captionData = {};
				captionData.name = "Weird Region";
				bool status = presentImageOnTheScreen(core::smart_refctd_ptr(gpuImageView), captionData);
				assert(status);
			}
		}
	}
};
#endif

NBL_MAIN_FUNC(ColorSpaceTestSampleApp)