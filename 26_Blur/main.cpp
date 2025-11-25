// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"

#include <bit>
#include <limits>

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

#include "app_resources/common.hlsl"



class BlurApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
		using device_base_t = SimpleWindowedApplication;
		using asset_base_t = BuiltinResourcesApplication;
		using clock_t = std::chrono::steady_clock;

	public:
		inline BlurApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool isComputeOnly() const override { return false; }

		// tired of packing and unpacking from float16_t, let some Junior do device traits / manual pack
		virtual video::SPhysicalDeviceLimits getRequiredDeviceLimits() const override
		{
			auto retval = device_base_t::getRequiredDeviceLimits();
			retval.shaderSubgroupArithmetic = true;
			retval.shaderFloat16 = true;
			return retval;
		}

		inline core::vector<SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					// We resize the window later
					params.width = 0;
					params.height = 0;
					params.x = 32;
					params.y = 32;
					params.flags = IWindow::ECF_BORDERLESS | IWindow::ECF_HIDDEN | IWindow::ECF_CAN_RESIZE;
					params.windowCaption = "BlurApp";
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}
				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
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
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;
			
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				const IGPUDescriptorSetLayout::SBinding bindings[2] = {
					{
						.binding = 0,
						.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1
					},
					{
						.binding = 1,
						.type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1
					}
				};
				dsLayout = m_device->createDescriptorSetLayout(bindings);
				if (!dsLayout)
					return logFail("Failed to Create Descriptor Layout");
			}

			core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			auto queue = getGraphicsQueue();
			{
				core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags = static_cast<IGPUCommandPool::CREATE_FLAGS>(IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT | IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), flags);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmdbuf))
					return false;
			}

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			auto converter = CAssetConverter::create({ .device = m_device.get() });

			IQueue::SSubmitInfo::SCommandBufferInfo commandBufferInfo;
			commandBufferInfo.cmdbuf = cmdbuf.get();

			core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
			imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");
			// scratch command buffers for asset converter transfer commands
			SIntendedSubmitInfo transfer = {
				.queue = queue,
				.waitSemaphores = {},
				.prevCommandBuffers = {},
				.scratchCommandBuffers = { &commandBufferInfo, 1 },
				.scratchSemaphore = {
					.semaphore = imgFillSemaphore.get(),
					.value = 0,
					// because of layout transitions
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				}
			};
			CAssetConverter::SConvertParams params = {};
			params.transfer = &transfer;
			params.utilities = m_utils.get();

			IAssetLoader::SAssetLoadParams lp;
			SAssetBundle bundle = m_assetMgr->getAsset("../../media/color_space_test/R8G8B8_2.jpg", lp);
			if (bundle.getContents().empty())
				logFail("Couldn't load an asset.", ILogger::ELL_ERROR);

			auto cpu_image = IAsset::castDown<ICPUImage>(bundle.getContents()[0]);
			cpu_image->addImageUsageFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);
			if (!cpu_image)
				logFail("Failed to load image", ILogger::ELL_ERROR);
			struct SInputs final : CAssetConverter::SInputs
			{
				// we also need to override this to have concurrent sharing
				inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					if (familyIndices.size() > 1)
						return familyIndices;
					return {};
				}

				inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return image->getCreationParameters().mipLevels;
				}
				inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return 0b0u;
				}

				std::vector<uint32_t> familyIndices;
			} inputs = {};
			CAssetConverter::patch_t<ICPUImage> patch(cpu_image.get());
			patch.mutableFormat = true;
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { &cpu_image.get(),1 };
			std::get<CAssetConverter::SInputs::patch_span_t<ICPUImage>>(inputs.patches) = { &patch, 1 };
			inputs.readCache = converter.get();
			inputs.logger = m_logger.get();
			{
				const core::set<uint32_t> uniqueFamilyIndices = { getTransferUpQueue()->getFamilyIndex(), getGraphicsQueue()->getFamilyIndex(), getComputeQueue()->getFamilyIndex() };
				inputs.familyIndices = { uniqueFamilyIndices.begin(),uniqueFamilyIndices.end() };
			}
			// assert that we don't need to provide patches
			assert(cpu_image->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
			auto reservation = converter->reserve(inputs);
			// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
			m_inputImg = reservation.getGPUObjects<ICPUImage>().front().value;
			if (!m_inputImg)
				logFail("Failed to convert image into an IGPUImage handle", ILogger::ELL_ERROR);

			// debug log about overflows
			transfer.overflowCallback = [&](const ISemaphore::SWaitInfo&)->void
				{
					m_logger->log("Overflown when uploading image!\n", ILogger::ELL_PERFORMANCE);
				};
			// and launch the conversions
			auto result = reservation.convert(params);
			if (!result.blocking() && result.copy() != IQueue::RESULT::SUCCESS)
				logFail("Failed to record or submit conversions");

			auto image_params = m_inputImg->getCreationParameters();

			m_horzImg = m_device->createImage({
				{
					.type = IGPUImage::E_TYPE::ET_2D,
					.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
					.format = E_FORMAT::EF_R8G8B8A8_UNORM,
					.extent = image_params.extent,
					.mipLevels = 1,
					.arrayLayers = 1,
					.usage = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT
				}
				});
			// make sure we're always allocating from VRAM
			auto reqs = m_horzImg->getMemoryReqs();
			reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			if (!m_horzImg || !m_device->allocate(reqs, m_horzImg.get()).isValid())
				return logFail("Could not create HDR Image");

			m_vertImg = m_device->createImage({
				{
					.type = IGPUImage::E_TYPE::ET_2D,
					.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
					.format = E_FORMAT::EF_R8G8B8A8_UNORM,
					.extent = image_params.extent,
					.mipLevels = 1,
					.arrayLayers = 1,
					.usage = IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT
				}
				});
			reqs = m_horzImg->getMemoryReqs();
			reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			if (!m_vertImg || !m_device->allocate(reqs, m_vertImg.get()).isValid())
				return logFail("Could not create HDR Image");

			smart_refctd_ptr<IShader> shader;
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl", lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return logFail("Failed to load shader from disk");

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto sourceRaw = IAsset::castDown<IShader>(assets[0]);
				if (!sourceRaw)
					return logFail("Failed to load shader from disk");
				smart_refctd_ptr<IShader> source = CHLSLCompiler::createOverridenCopy(
					sourceRaw.get(),
					"static const uint16_t WORKGROUP_SIZE = %d;\n"
					"static const uint16_t MAX_SCANLINE_SIZE = %d;\n"
					"static const uint16_t MAX_SUBGROUP_SIZE = %d;\n"
					"static const uint16_t CHANNELS = %d;\n",
					m_physicalDevice->getLimits().maxOptimallyResidentWorkgroupInvocations,
					max(image_params.extent.width, image_params.extent.height),
					m_physicalDevice->getLimits().maxSubgroupSize,
					asset::getFormatChannelCount(image_params.format)
				);

#ifndef _NBL_DEBUG
				const ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses[] = {
					ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO,
					ISPIRVOptimizer::EOP_INLINE,
					ISPIRVOptimizer::EOP_DEAD_BRANCH_ELIM,
					ISPIRVOptimizer::EOP_DEAD_INSERT_ELIM,
					ISPIRVOptimizer::EOP_IF_CONVERSION,
					ISPIRVOptimizer::EOP_LOOP_INVARIANT_CODE_MOTION,
					ISPIRVOptimizer::EOP_LOCAL_MULTI_STORE_ELIM
				};
				auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(optPasses);
				shader = m_device->compileShader({ source.get(),opt.get() });
#else
				shader = m_device->compileShader({ source.get() });
#endif
				if (!shader)
					return false;
			}

			{
				const asset::SPushConstantRange ranges[] = { {
					.stageFlags = hlsl::ShaderStage::ESS_COMPUTE,
					.offset = 0,
					.size = sizeof(PushConstants)
				} };
				auto layout = m_device->createPipelineLayout(ranges, smart_refctd_ptr(dsLayout));

				IGPUComputePipeline::SCreationParams params = {};
				params.layout = layout.get();
				params.shader.shader = shader.get();
				params.shader.entryPoint = "main";
				params.cached.requireFullSubgroups = true;
				if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &m_ppln))
					return logFail("Failed to create Pipeline");
			}

			smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(0);
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {
				.semaphore = progress.get(),
				.value = 1,
				// wait for the Copy Image to Buffer to finish before we signal
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT
			} };

			IQueue::SSubmitInfo submitInfos[1];
			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfos[1] = { cmdbuf.get() };
			submitInfos[0].commandBuffers = cmdbufInfos;
			submitInfos[0].signalSemaphores = signals;

			const IGPUImage::SSubresourceRange whole2DColorImage =
			{
				.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			};

			using image_memory_barrier_t = IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>;
			image_memory_barrier_t imgBarriers[] = {
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
							.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS
						}
					},
					.image = m_horzImg.get(),
					.subresourceRange = whole2DColorImage,
					.oldLayout = IImage::LAYOUT::UNDEFINED,
					.newLayout = IImage::LAYOUT::GENERAL
				},
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
							.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS
						}
					},
					.image = m_vertImg.get(),
					.subresourceRange = whole2DColorImage,
					.oldLayout = IImage::LAYOUT::UNDEFINED,
					.newLayout = IImage::LAYOUT::GENERAL
				},
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
							.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS
						}
					},
					.image = m_inputImg.get(),
					.subresourceRange = whole2DColorImage,
					.oldLayout = IImage::LAYOUT::UNDEFINED,
					.newLayout = IImage::LAYOUT::GENERAL
				}
			};

			// clear the image
			cmdbuf->reset({});
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {},.bufBarriers = {},.imgBarriers = {imgBarriers,3} });
			cmdbuf->end();
			queue->submit(submitInfos);
			const ISemaphore::SWaitInfo waitInfos[1] = { {
					.semaphore = progress.get(),
					.value = 1
			} };
			m_device->blockForSemaphores(waitInfos);

			{
				IGPUDescriptorSetLayout* const dsLayouts[] = {dsLayout.get(), dsLayout.get()};
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, { dsLayouts, 2 });
				if (!pool)
					return logFail("Could not create Descriptor Pool");

				m_ds0 = pool->createDescriptorSet(smart_refctd_ptr(dsLayout));
				if (!m_ds0)
					return logFail("Could not create Descriptor Set");
				m_ds1 = pool->createDescriptorSet(std::move(dsLayout));
				if (!m_ds1)
					return logFail("Could not create Descriptor Set");

				auto image_params = m_inputImg->getCreationParameters();
				IGPUDescriptorSet::SDescriptorInfo inputImgSampledInfo = {};
				{
					inputImgSampledInfo.desc = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
						.image = m_inputImg,
						.viewType = IGPUImageView::E_TYPE::ET_2D,
						.format = image_params.format,
						});
					if (!inputImgSampledInfo.desc)
						return logFail("Failed to create image view");
					inputImgSampledInfo.info.combinedImageSampler.imageLayout = IGPUImage::LAYOUT::GENERAL;
					inputImgSampledInfo.info.combinedImageSampler.sampler = m_device->createSampler({});
				}
				IGPUDescriptorSet::SDescriptorInfo horzImgStorageInfo = {};
				{
					horzImgStorageInfo.desc = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
						.image = m_horzImg,
						.viewType = IGPUImageView::E_TYPE::ET_2D,
						.format = E_FORMAT::EF_R8G8B8A8_UNORM
						});
					if (!horzImgStorageInfo.desc)
						return logFail("Failed to create image view");
					horzImgStorageInfo.info.image.imageLayout = IGPUImage::LAYOUT::GENERAL;
				}
				IGPUDescriptorSet::SDescriptorInfo horzImgSampledInfo = {};
				{
					horzImgSampledInfo.desc = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
						.image = m_horzImg,
						.viewType = IGPUImageView::E_TYPE::ET_2D,
						.format = E_FORMAT::EF_R8G8B8A8_UNORM
						});
					if (!horzImgSampledInfo.desc)
						return logFail("Failed to create image view");
					horzImgSampledInfo.info.combinedImageSampler.imageLayout = IGPUImage::LAYOUT::GENERAL;
					horzImgSampledInfo.info.combinedImageSampler.sampler = m_device->createSampler({});
				}
				IGPUDescriptorSet::SDescriptorInfo vertImgStorageInfo = {};
				{
					vertImgStorageInfo.desc = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
						.image = m_vertImg,
						.viewType = IGPUImageView::E_TYPE::ET_2D,
						.format = E_FORMAT::EF_R8G8B8A8_UNORM
						});
					if (!vertImgStorageInfo.desc)
						return logFail("Failed to create image view");
					vertImgStorageInfo.info.image.imageLayout = IGPUImage::LAYOUT::GENERAL;
				}

				const IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
					{
						.dstSet = m_ds0.get(),
						.binding = 0,
						.arrayElement = 0,
						.count = 1,
						.info = &inputImgSampledInfo
					},
					{
						.dstSet = m_ds0.get(),
						.binding = 1,
						.arrayElement = 0,
						.count = 1,
						.info = &horzImgStorageInfo
					},
					{
						.dstSet = m_ds1.get(),
						.binding = 0,
						.arrayElement = 0,
						.count = 1,
						.info = &horzImgSampledInfo
					},
					{
						.dstSet = m_ds1.get(),
						.binding = 1,
						.arrayElement = 0,
						.count = 1,
						.info = &vertImgStorageInfo
					}
				};
				if (!m_device->updateDescriptorSets(writes,{}))
					return logFail("Failed to write descriptor set");
			}

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			auto gQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(gQueue, std::make_unique<ISimpleManagedSurface::ISwapchainResources>(), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			auto pool = m_device->createCommandPool(gQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			for (auto i=0u; i<MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
					return logFail("Couldn't create Command Buffer!");
			}

			m_winMgr->setWindowSize(m_window.get(), image_params.extent.width, image_params.extent.height);
			m_surface->recreateSwapchain();

			auto assetManager = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(system));

			m_winMgr->show(m_window.get());

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

			auto image_params = m_inputImg->getCreationParameters();
			m_inputSystem->getDefaultMouse(&mouse);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void {
				for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
				{
					auto ev = *eventIt;
					if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
					{
						blurRadius = std::clamp<hlsl::float32_t>(blurRadius + 5 * core::sign(ev.scrollEvent.verticalScroll),
																std::numeric_limits<hlsl::float32_t>::min(),
																max(image_params.extent.width, image_params.extent.height));
						m_logger->log("Radius changed: %f", ILogger::ELL_DEBUG, blurRadius);
					}
					if (ev.type == nbl::ui::SMouseEvent::EET_CLICK && ev.clickEvent.mouseButton == nbl::ui::EMB_LEFT_BUTTON)
					{
						if (ev.clickEvent.action == nbl::ui::SMouseEvent::SClickEvent::EA_RELEASED)
						{
							blurEdgeWrapMode = (blurEdgeWrapMode + 1) % ISampler::E_TEXTURE_CLAMP::ETC_COUNT;
							m_logger->log("Edge wrapping mode changed: %d", ILogger::ELL_DEBUG, blurEdgeWrapMode);
						}
					}
				}
			}, m_logger.get());

			m_currentImageAcquire = m_surface->acquireNextImage();
			if (!m_currentImageAcquire)
				return;

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			
			const IGPUImage::SSubresourceRange whole2DColorImage =
			{
				.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			};

			using image_memory_barrier_t = IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>;
			image_memory_barrier_t vertImgBarrier = {
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
						.srcAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
						.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				},
				.image = m_vertImg.get(),
				.subresourceRange = whole2DColorImage,
				.oldLayout = IImage::LAYOUT::GENERAL,
				.newLayout = IImage::LAYOUT::GENERAL
			};

			// clear the image
			cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers={},.imgBarriers={&vertImgBarrier,1}});
			{
				const IGPUCommandBuffer::SClearColorValue color = {
					.float32 = {0,0,0,1}
				};
				const IGPUImage::SSubresourceRange range = {
					.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				};
				cb->clearColorImage(m_vertImg.get(),IGPUImage::LAYOUT::GENERAL,&color,1,&range);
				// now we stay in same layout for remainder of the frame
				vertImgBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			}

			const SMemoryBarrier computeToBlit = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT|ACCESS_FLAGS::STORAGE_READ_BIT,
				.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
				.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
			};

			auto& imgDep = vertImgBarrier.barrier.dep;
			// use the "generate a barrier between the one before and after" API
			imgDep = imgDep.nextBarrier(computeToBlit);
			cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers={},.imgBarriers={&vertImgBarrier,1}});

			// write the image
			{
				cb->bindComputePipeline(m_ppln.get());
				auto* layout = m_ppln->getLayout();

				cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {}, .bufBarriers = {},.imgBarriers = {&vertImgBarrier,1} });
				cb->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout, 0, 1, &m_ds0.get());
				PushConstants pc = { .radius = blurRadius, .activeAxis = 0, .edgeWrapMode = blurEdgeWrapMode };
				cb->pushConstants(layout,  hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(pc), &pc);
				cb->dispatch(image_params.extent.height, 1, 1);

				image_memory_barrier_t horzImgBarrier = {
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.srcAccessMask = ACCESS_FLAGS::STORAGE_READ_BIT | ACCESS_FLAGS::STORAGE_WRITE_BIT,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::STORAGE_READ_BIT | ACCESS_FLAGS::STORAGE_WRITE_BIT,
						}
					},
					.image = m_horzImg.get(),
					.subresourceRange = whole2DColorImage,
					.oldLayout = IImage::LAYOUT::GENERAL,
					.newLayout = IImage::LAYOUT::GENERAL
				};
				cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {}, .bufBarriers = {},.imgBarriers = {&horzImgBarrier,1} });
				cb->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout, 0, 1, &m_ds1.get());
				pc.activeAxis = 1;
				cb->pushConstants(layout, hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(pc), &pc);
				cb->dispatch(image_params.extent.width, 1, 1);
			}

			{
				auto swapImg = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);
				auto swapImgParams = swapImg->getCreationParameters();
				imgDep = computeToBlit;
				// special case, the swapchain is a NONE stage with NONE accesses
				image_memory_barrier_t imgBarriers[] = {
					vertImgBarrier,
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
								.srcAccessMask = ACCESS_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
								.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
							}
						},
						.image = swapImg,
						.subresourceRange = whole2DColorImage,
						.oldLayout = IImage::LAYOUT::UNDEFINED, // don't care about old contents
						.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL
					}
				};
				cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers={},.imgBarriers=imgBarriers});

				const IGPUCommandBuffer::SImageBlit regions[] = {{
					.srcMinCoord = {0,0,0},
					.srcMaxCoord = {image_params.extent.width, image_params.extent.height,1},
					.dstMinCoord = {0,0,0},
					.dstMaxCoord = {swapImgParams.extent.width, swapImgParams.extent.height,1},
					.layerCount = 1,
					.srcBaseLayer = 0,
					.dstBaseLayer = 0,
					.srcMipLevel = 0,
					.dstMipLevel = 0,
					.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT
				}};
				cb->blitImage(m_vertImg.get(),IGPUImage::LAYOUT::GENERAL,swapImg,IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,regions,IGPUSampler::ETF_NEAREST);

				auto& swapImageBarrier = imgBarriers[1];
				swapImageBarrier.barrier.dep = swapImageBarrier.barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::NONE,ACCESS_FLAGS::NONE);
				swapImageBarrier.oldLayout = imgBarriers[1].newLayout;
				swapImageBarrier.newLayout = IGPUImage::LAYOUT::PRESENT_SRC;
				cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers={},.imgBarriers={&swapImageBarrier,1}});
			}

			cb->end();

			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS // because of the layout transition of the swapchain image
					}
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = {{.cmdbuf = cb }};

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

						if (getGraphicsQueue()->submit(infos) == IQueue::RESULT::SUCCESS)
						{
							const ISemaphore::SWaitInfo waitInfos[] = {{ .semaphore = m_semaphore.get(), .value = m_realFrameIx }};
							m_device->blockForSemaphores(waitInfos); // this is not solution, quick wa to not throw validation errors
						}
						else
							--m_realFrameIx;
					}
				}

				m_surface->present(m_currentImageAcquire.imageIndex,rendered);
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
		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
		
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;

		hlsl::float32_t blurRadius = 6;
		uint16_t blurEdgeWrapMode = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;

		smart_refctd_ptr<IGPUComputePipeline> m_ppln;
		smart_refctd_ptr<IGPUDescriptorSet> m_ds0;
		smart_refctd_ptr<IGPUDescriptorSet> m_ds1;
		smart_refctd_ptr<IGPUImage> m_inputImg;
		smart_refctd_ptr<IGPUImage> m_horzImg;
		smart_refctd_ptr<IGPUImage> m_vertImg;
		smart_refctd_ptr<ISemaphore> m_semaphore;

		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
};

NBL_MAIN_FUNC(BlurApp)