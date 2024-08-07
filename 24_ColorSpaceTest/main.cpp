﻿// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "../common/SimpleWindowedApplication.hpp"

//
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"


using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// defines for sampler tests can be found in the file below
#include "app_resources/push_constants.hlsl"


class ColorSpaceTestSampleApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
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
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
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

			// Load FSTri Shader
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(),m_device.get(),m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

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
			
			// Now surface indep resources
			m_semaphore = m_device->createSemaphore(m_submitIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			/* 
			* We'll be using a combined image sampler for this example, which lets us assign both a sampled image and a sampler to the same binding. 
			* In this example we provide a sampler at descriptor set creation time, via the SBinding struct below. This specifies that the sampler for this binding is immutable,
			* as evidenced by the name of the field in the SBinding. 
			* Samplers for combined image samplers can also be mutable, which for a binding of a descriptor set is specified also at creation time by leaving the immutableSamplers
			* field set to its default (nullptr). 
			*/
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				auto defaultSampler = m_device->createSampler({
					.AnisotropicFilter = 0
				});

				const IGPUDescriptorSetLayout::SBinding bindings[1] = { {
					.binding = 0,
					.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1,
					.immutableSamplers = &defaultSampler
				}
				};
				dsLayout = m_device->createDescriptorSetLayout(bindings);
				if (!dsLayout)
					return logFail("Failed to Create Descriptor Layout");

			}

			ISwapchain::SCreationParams swapchainParams = {.surface=m_surface->getSurface()};
			// Need to choose a surface format
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");
			// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
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
			auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(),swapchainParams.surfaceFormat.format,dependencies);
			if (!scResources->getRenderpass())
				return logFail("Failed to create Renderpass!");

			// Now create the pipeline
			{
				const asset::SPushConstantRange range = {
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
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

			auto queue = getGraphicsQueue();
			// Let's just use the same queue since there's no need for async present
			if (!m_surface || !m_surface->init(queue,std::move(scResources),swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");
			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();

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
			
			// need resetttable commandbuffers for the upload utility
			m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			// create the commandbuffers
			for (auto i=0u; i<m_maxFramesInFlight; i++)
			{
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
					return logFail("Couldn't create Command Buffer!");
			}

			// things for IUtilities
			m_scratchSemaphore = m_device->createSemaphore(0);
			if (!m_scratchSemaphore)
				return logFail("Could not create Scratch Semaphore");
			m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
			// we don't want to overcomplicate the example with multi-queue
			m_intendedSubmit.queue = queue;
			// wait for nothing before upload
			m_intendedSubmit.waitSemaphores = {};
			// fill later
			m_intendedSubmit.commandBuffers = {};
			m_intendedSubmit.scratchSemaphore = {
				.semaphore = m_scratchSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};

			// Allocate and Leave 1/4 for image uploads, to test image copy with small memory remaining 
			{
				uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
				uint32_t maxFreeBlock = m_utils->getDefaultUpStreamingBuffer()->max_size();
				const uint32_t allocationAlignment = 64u;
				const uint32_t allocationSize = (maxFreeBlock/4)*3;
				m_utils->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
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
								.levelCount = ICPUImageView::remaining_mip_levels,
								.baseArrayLayer = 0u,
								.layerCount = ICPUImageView::remaining_array_layers
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
			const auto resourceIx = m_submitIx%m_maxFramesInFlight;

			// we don't want to overcomplicate the example with multi-queue
			auto queue = getGraphicsQueue();
			auto cmdbuf = m_cmdBufs[resourceIx].get();
			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = {cmdbuf};
			m_intendedSubmit.commandBuffers = {&cmdbufInfo,1};
			
			// there's no previous operation to wait for
			const SMemoryBarrier toTransferBarrier = {
				.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
				.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
			};

			// upload image and write to descriptor set
			queue->startCapture();
			smart_refctd_ptr<IGPUImage> gpuImg;
			auto ds = m_descriptorSets[resourceIx].get();
			{
				/*
				* Since we're using a combined image sampler with an immutable sampler, we only need to update the sampled image at the binding. Do note however that had we chosen
				* to use a mutable sampler instead, we'd need to write to it at least once, via the SDescriptorInfo info.info.combinedImageSampler.sampler field
				* WARNING: With an immutable sampler on a combined image sampler, trying to write to it is valid according to Vulkan spec, although the sampler is ignored and only
				* the image is updated. Please note that this is NOT the case in Nabla: if you try to write to a combined image sampler, then
				* info.info.combinedImageSampler.sampler MUST be nullptr
				*/
				IGPUDescriptorSet::SDescriptorInfo info = {};
				info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
				{
					const auto& origParams = cpuImgView->getCreationParameters();
					const auto origImage = origParams.image;

					// create matching size image
					IGPUImage::SCreationParams imageParams = {};
					imageParams = origImage->getCreationParameters();
					imageParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT|IGPUImage::EUF_SAMPLED_BIT|IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT;
					// promote format because RGB8 and friends don't actually exist in HW
					{
						const IPhysicalDevice::SImageFormatPromotionRequest request = {
							.originalFormat = imageParams.format,
							.usages = IPhysicalDevice::SFormatImageUsages::SUsage(imageParams.usage)
						};
						imageParams.format = m_physicalDevice->promoteImageFormat(request,imageParams.tiling);
					}
					if (imageParams.type==IGPUImage::ET_3D)
						imageParams.flags |= IGPUImage::ECF_2D_ARRAY_COMPATIBLE_BIT;
					gpuImg = m_device->createImage(std::move(imageParams));
					if (!gpuImg || !m_device->allocate(gpuImg->getMemoryReqs(),gpuImg.get()).isValid())
						return;
					gpuImg->setObjectDebugName(m_nextPath.c_str());

					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					// change the layout of the image
					const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {{
						.barrier = {
							.dep = toTransferBarrier
							// no ownership transfers
						},
						.image = gpuImg.get(),
						// transition the whole view
						.subresourceRange = origParams.subresourceRange,
						// a wiping transition
						.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
					}};
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.imgBarriers=imgBarriers});
					// upload contents and submit right away
					m_utils->updateImageViaStagingBufferAutoSubmit(m_intendedSubmit,origImage->getBuffer(),origImage->getCreationParameters().format,gpuImg.get(),IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,origImage->getRegions());

					IGPUImageView::SCreationParams viewParams = {
						.image = gpuImg,
						.viewType = IGPUImageView::ET_2D_ARRAY,
						.format = gpuImg->getCreationParameters().format
					};
					info.desc = m_device->createImageView(std::move(viewParams));
				}
				const IGPUDescriptorSet::SWriteDescriptorSet writes[] = {{
					.dstSet = ds,
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = &info
				}};
				m_device->updateDescriptorSets(writes,{});
			}

			// now we can sleep till we're ready for next render
			std::this_thread::sleep_until(m_lastImageEnqueued+DisplayImageDuration);
			m_lastImageEnqueued = clock_t::now();

			const auto& params = gpuImg->getCreationParameters();
			const auto imageExtent = params.extent;
			push_constants_t pc;
			{
				const float realLayers = core::max(params.arrayLayers,imageExtent.depth);
				pc.grid.x = ceil(sqrt(realLayers));
				pc.grid.y = ceil(realLayers/float(pc.grid.x));
			}
			const VkExtent2D newWindowResolution = {imageExtent.width*pc.grid.x,imageExtent.height*pc.grid.y};
			if (newWindowResolution.width!=m_window->getWidth() || newWindowResolution.height!=m_window->getHeight())
			{
				// Resize the window
				m_winMgr->setWindowSize(m_window.get(),newWindowResolution.width,newWindowResolution.height);
				// Don't want to rely on the Swapchain OUT_OF_DATE causing an implicit re-create in the `acquireNextImage` because the
				// swapchain may report OUT_OF_DATE after the next VBlank after the resize, not getting the message right away.
				m_surface->recreateSwapchain();
			}
			// Now show the window (ideally should happen just after present, but don't want to mess with acquire/recreation)
			m_winMgr->show(m_window.get());

			// Acquire
			auto acquire = m_surface->acquireNextImage();
			if (!acquire)
				return;

			// Render to the Image
			{
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				// need a pipeline barrier to transition layout
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {{
					.barrier = {
						.dep = toTransferBarrier.nextBarrier(PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,ACCESS_FLAGS::SAMPLED_READ_BIT)
					},
					.image = gpuImg.get(),
					.subresourceRange = cpuImgView->getCreationParameters().subresourceRange,
					.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					.newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
				}};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.imgBarriers=imgBarriers});

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

				// begin the renderpass
				{
					const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					const IGPUCommandBuffer::SRenderpassBeginInfo info = {
						.framebuffer = scRes->getFramebuffer(acquire.imageIndex),
						.colorClearValues = &clearValue,
						.depthStencilClearValues = nullptr,
						.renderArea = currentRenderArea
					};
					cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				}
				cmdbuf->bindGraphicsPipeline(m_pipeline.get());
				cmdbuf->pushConstants(m_pipeline->getLayout(),IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT,0,sizeof(push_constants_t),&pc);
				cmdbuf->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS,m_pipeline->getLayout(),3,1,&ds);
				ext::FullScreenTriangle::recordDrawCall(cmdbuf);
				cmdbuf->endRenderPass();

				cmdbuf->end();
			}

			// submit
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
				.semaphore = m_semaphore.get(),
				.value = ++m_submitIx,
				// just as we've outputted all pixels, signal
				.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
			}};
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

			// Present
			m_surface->present(acquire.imageIndex, rendered);
			getGraphicsQueue()->endCapture();

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

			// TODO: reuse the resources without frames in flight (cmon we block and do like one swap every 900ms)
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
		//
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		// for image uploads
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;
		// Use a separate counter to cycle through our resources for clarity
		uint64_t m_submitIx : 59 = 0;
		// Maximum frames which can be simultaneously rendered
		uint64_t m_maxFramesInFlight : 5;
		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUDescriptorSet>,ISwapchain::MaxImages> m_descriptorSets;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;
};


#if 0 // old test code to check region copies
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
#endif

NBL_MAIN_FUNC(ColorSpaceTestSampleApp)