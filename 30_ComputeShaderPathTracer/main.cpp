// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/this_example/common.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// TODO: Add a QueryPool for timestamping once its ready
class ComputeShaderPathtracer final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;

	enum E_LIGHT_GEOMETRY : uint8_t
	{
		ELG_SPHERE,
		ELG_TRIANGLE,
		ELG_RECTANGLE,
		ELG_COUNT
	};

	struct SBasicViewParametersAligned
	{
		SBasicViewParameters uboData;
	};

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t2 WindowDimensions = { 1280, 720 };
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t FramesInFlight = 5;
	_NBL_STATIC_INLINE_CONSTEXPR clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
	_NBL_STATIC_INLINE_CONSTEXPR E_LIGHT_GEOMETRY LightGeom = E_LIGHT_GEOMETRY::ELG_SPHERE;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t DefaultWorkGroupSize = 16u;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxDescriptorCount = 256u;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxDepthLog2 = 4u; // 5
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxSamplesLog2 = 10u; // 18
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxBufferDimensions = 3u << MaxDepthLog2;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxBufferSamples = 1u << MaxSamplesLog2;
	_NBL_STATIC_INLINE_CONSTEXPR uint8_t MaxUITextureCount = 2u;
	_NBL_STATIC_INLINE_CONSTEXPR uint8_t SceneTextureIndex = 1u;
	_NBL_STATIC_INLINE std::string DefaultImagePathsFile = "../../media/envmap/envmap_0.exr";
	_NBL_STATIC_INLINE std::array<std::string, 3> ShaderPaths = { "app_resources/litBySphere.comp", "app_resources/litByTriangle.comp", "app_resources/litByRectangle.comp" };

	public:
		inline ComputeShaderPathtracer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
			const auto cameraPos = core::vectorSIMDf(0, 5, -10);
			matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
				core::radians(fov),
				static_cast<float32_t>(WindowDimensions.x) / static_cast<float32_t>(WindowDimensions.y),
				zNear,
				zFar
			);

			m_camera = Camera(cameraPos, core::vectorSIMDf(0, 0, 0), proj);
		}

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WindowDimensions.x;
					params.height = WindowDimensions.y;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "ComputeShaderPathtracer";
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
			// Init systems
			{
				m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

				// Remember to call the base class initialization!
				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;
				if (!asset_base_t::onAppInitialized(std::move(system)))
					return false;

				m_uiSemaphore = m_device->createSemaphore(m_realFrameIx);
				m_renderSemaphore = m_device->createSemaphore(m_realFrameIx);
				if (!m_uiSemaphore || !m_renderSemaphore)
					return logFail("Failed to Create semaphores!");
			}

			// Create renderpass and init surface
			nbl::video::IGPURenderpass* renderpass;
			{
				ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
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
				renderpass = scResources->getRenderpass();

				if (!renderpass)
					return logFail("Failed to create Renderpass!");

				auto gQueue = getGraphicsQueue();
				if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
					return logFail("Could not create Window & Surface or initialize the Surface!");
			}

			// Compute no of frames in flight
			{
				m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
				if (FramesInFlight < m_maxFramesInFlight)
				{
					m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
					m_maxFramesInFlight = FramesInFlight;
				}
			}

			// image upload utils
			{
				m_scratchSemaphore = m_device->createSemaphore(0);
				if (!m_scratchSemaphore)
					return logFail("Could not create Scratch Semaphore");
				m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
				// we don't want to overcomplicate the example with multi-queue
				m_intendedSubmit.queue = getGraphicsQueue();
				// wait for nothing before upload
				m_intendedSubmit.waitSemaphores = {};
				m_intendedSubmit.waitSemaphores = {};
				// fill later
				m_intendedSubmit.commandBuffers = {};
				m_intendedSubmit.scratchSemaphore = {
					.semaphore = m_scratchSemaphore.get(),
					.value = 0,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};
			}

			// Create command pool and buffers
			{
				auto gQueue = getGraphicsQueue();
				m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");

				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(), 2 * m_maxFramesInFlight }))
					return logFail("Couldn't create Command Buffer!");
			}

			// Create descriptor layouts and pipeline for the pathtracer
			smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDescriptorSetLayout0, gpuDescriptorSetLayout1, gpuDescriptorSetLayout2;
			{
				IGPUDescriptorSetLayout::SBinding descriptorSet0Bindings[] = {
					{
						.binding = 0u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					}
				};
				IGPUDescriptorSetLayout::SBinding uboBindings[] = {
					{
						.binding = 0u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					}
				};
				IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
					{
						.binding = 0u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					},
					{
						.binding = 1u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					},
					{
						.binding = 2u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					},
				};

				gpuDescriptorSetLayout0 = m_device->createDescriptorSetLayout({ descriptorSet0Bindings, 1 });
				gpuDescriptorSetLayout1 = m_device->createDescriptorSetLayout({ uboBindings, 1 });
				gpuDescriptorSetLayout2 = m_device->createDescriptorSetLayout({ descriptorSet3Bindings, 3 });

				if (!gpuDescriptorSetLayout0 || !gpuDescriptorSetLayout1 || !gpuDescriptorSetLayout2) {
					return logFail("Failed to create descriptor set layouts!\n");
				}

				auto createGpuResources = [&](std::string pathToShader, smart_refctd_ptr<IGPUComputePipeline>&& pipeline) -> bool
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.logger = m_logger.get();
					lp.workingDirectory = ""; // virtual root
					auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						return logFail("Could not load shader!");
					}

					auto source = IAsset::castDown<ICPUShader>(assets[0]);
					// The down-cast should not fail!
					assert(source);

					// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
					auto shader = m_device->createShader(source.get());
					if (!shader)
					{
						return logFail("Shader creationed failed: %s!", pathToShader);
					}

					auto gpuPipelineLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), core::smart_refctd_ptr(gpuDescriptorSetLayout2), nullptr);
					if (!gpuPipelineLayout) {
						return logFail("Failed to create pipeline layout");
					}

					IGPUComputePipeline::SCreationParams params = {};
					params.layout = gpuPipelineLayout.get();
					params.shader.shader = shader.get();
					params.shader.entryPoint = "main";
					params.shader.entries = nullptr;
					params.shader.requireFullSubgroups = true;
					params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(5);
					if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline)) {
						return logFail("Failed to create compute pipeline!\n");
					}

					return true;
				};

				if (!createGpuResources(ShaderPaths[LightGeom], std::move(m_pipeline))) {
					return logFail("Pipeline creation failed!");
				}
			}

			// load image
			smart_refctd_ptr<ICPUImageView> cpuImgView;
			{
				IAssetLoader::SAssetLoadParams params;
				auto imageBundle = m_assetMgr->getAsset(DefaultImagePathsFile.data(), params);
				auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().begin()[0]);
				auto format = cpuImg->getCreationParameters().format;

				ICPUImageView::SCreationParams viewParams = {
					.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
					.image = std::move(cpuImg),
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
			}

			// create views for textures
			{
				auto createHDRIImageView = [&](const asset::E_FORMAT colorFormat, const uint32_t width, const uint32_t height) ->smart_refctd_ptr<IGPUImageView>
				{
					smart_refctd_ptr<IGPUImageView> view;
					{
						IGPUImage::SCreationParams imgInfo;
						imgInfo.format = colorFormat;
						imgInfo.type = IGPUImage::ET_2D;
						imgInfo.extent.width = width;
						imgInfo.extent.height = height;
						imgInfo.extent.depth = 1u;
						imgInfo.mipLevels = 1u;
						imgInfo.arrayLayers = 1u;
						imgInfo.samples = IGPUImage::ESCF_1_BIT;
						imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
						imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT;

						auto image = m_device->createImage(std::move(imgInfo));
						auto imageMemReqs = image->getMemoryReqs();
						imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
						m_device->allocate(imageMemReqs, image.get());

						IGPUImageView::SCreationParams imgViewInfo;
						imgViewInfo.image = std::move(image);
						imgViewInfo.format = colorFormat;
						imgViewInfo.viewType = IGPUImageView::ET_2D;
						imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
						imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
						imgViewInfo.subresourceRange.baseArrayLayer = 0u;
						imgViewInfo.subresourceRange.baseMipLevel = 0u;
						imgViewInfo.subresourceRange.layerCount = 1u;
						imgViewInfo.subresourceRange.levelCount = 1u;

						view = m_device->createImageView(std::move(imgViewInfo));
					}

					return view;
				};

				auto params = cpuImgView->getCreationParameters();
				auto extent = params.image->getCreationParameters().extent;
				m_envMapView = createHDRIImageView(params.format, extent.width, extent.height);
				m_scrambleView = createHDRIImageView(asset::E_FORMAT::EF_R32G32_UINT, extent.width, extent.height);
				m_outImgView = createHDRIImageView(asset::E_FORMAT::EF_R16G16B16A16_SFLOAT, WindowDimensions.x, WindowDimensions.y);
			}

			// create ubo and sequence buffer view
			{
				{
					IGPUBuffer::SCreationParams params = {};
					params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
					params.size = sizeof(SBasicViewParametersAligned);

					m_ubo = m_device->createBuffer(std::move(params));
					auto memReqs = m_ubo->getMemoryReqs();
					memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					m_device->allocate(memReqs, m_ubo.get());
				}

				{
					auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * MaxBufferDimensions * MaxBufferSamples);

					core::OwenSampler sampler(MaxBufferDimensions, 0xdeadbeefu);
					//core::SobolSampler sampler(MaxBufferDimensions);

					auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
					for (auto dim = 0u; dim < MaxBufferDimensions; dim++)
						for (uint32_t i = 0; i < MaxBufferSamples; i++)
						{
							out[i * MaxBufferDimensions + dim] = sampler.sample(dim, i);
						}

					IGPUBuffer::SCreationParams params = {};
					params.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT;
					params.size = sampleSequence->getSize();

					// we don't want to overcomplicate the example with multi-queue
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
					m_intendedSubmit.commandBuffers = { &cmdbufInfo, 1 };

					queue->startCapture();

					auto bufferFuture = m_utils->createFilledDeviceLocalBufferOnDedMem(
						m_intendedSubmit,
						std::move(params),
						sampleSequence->getPointer()
					);
					bufferFuture.wait();
					auto buffer = bufferFuture.get();

					queue->endCapture();

					m_sequenceBufferView = m_device->createBufferView({ 0u, buffer->get()->getSize(), *buffer }, asset::E_FORMAT::EF_R32G32B32_UINT);
				}
			}

			// upload data
			{
				// upload env map
				{
					auto& gpuImg = m_envMapView->getCreationParameters().image;
					auto& cpuImg = cpuImgView->getCreationParameters().image;

					// we don't want to overcomplicate the example with multi-queue
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
					m_intendedSubmit.commandBuffers = { &cmdbufInfo, 1 };

					// there's no previous operation to wait for
					const SMemoryBarrier transferBarriers[] = {
						{
							.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
							.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
						},
						{
							.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
							.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						}
					};

					// upload image and write to descriptor set
					queue->startCapture();

					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					// change the layout of the image
					const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers1[] = {
						{
							.barrier = {
								.dep = transferBarriers[0]
								// no ownership transfers
							},
							.image = gpuImg.get(),
						// transition the whole view
						.subresourceRange = cpuImgView->getCreationParameters().subresourceRange,
						// a wiping transition
						.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
					}
					};
					const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers2[] = {
						{
							.barrier = {
								.dep = transferBarriers[1]
								// no ownership transfers
							},
							.image = gpuImg.get(),
						// transition the whole view
						.subresourceRange = cpuImgView->getCreationParameters().subresourceRange,
						// a wiping transition
						.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
						.newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
					}
					};
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers1 });
					// upload contents
					m_utils->updateImageViaStagingBuffer(
						m_intendedSubmit,
						cpuImg->getBuffer()->getPointer(),
						cpuImg->getCreationParameters().format,
						gpuImg.get(),
						IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
						cpuImg->getRegions()
					);
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers2 });
					m_utils->autoSubmit(m_intendedSubmit, [&](SIntendedSubmitInfo& nextSubmit) -> bool { return true; });

					queue->endCapture();
				}

				// upload scramble data
				{
					auto extent = cpuImgView->getCreationParameters().image->getCreationParameters().extent;

					IGPUImage::SBufferCopy region = {};
					region.bufferOffset = 0u;
					region.bufferRowLength = 0u;
					region.bufferImageHeight = 0u;
					region.imageExtent = extent;
					region.imageOffset = { 0u,0u,0u };
					region.imageSubresource.layerCount = 1u;
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;

					constexpr auto ScrambleStateChannels = 2u;
					const auto renderPixelCount = extent.width * extent.height;
					core::vector<uint32_t> random(renderPixelCount * ScrambleStateChannels);
					{
						core::RandomSampler rng(0xbadc0ffeu);
						for (auto& pixel : random)
							pixel = rng.nextSample();
					}

					const std::span<const asset::IImage::SBufferCopy> regions = { &region, 1 };

					// we don't want to overcomplicate the example with multi-queue
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
					m_intendedSubmit.commandBuffers = { &cmdbufInfo, 1 };

					queue->startCapture();

					m_utils->updateImageViaStagingBufferAutoSubmit(
						m_intendedSubmit,
						random.data(),
						asset::E_FORMAT::EF_R32G32_UINT,
						m_scrambleView->getCreationParameters().image.get(),
						IGPUImage::LAYOUT::UNDEFINED,
						regions
					);

					queue->endCapture();
				}
			}

			// create pathtracer descriptors
			{
				nbl::video::IDescriptorPool::SCreateInfo createInfo;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = MaxDescriptorCount * 1;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = MaxDescriptorCount * 8;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)] = MaxDescriptorCount * 2;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)] = MaxDescriptorCount * 1;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] = MaxDescriptorCount * 1;
				createInfo.maxSets = MaxDescriptorCount;

				auto descriptorPool = m_device->createDescriptorPool(std::move(createInfo));

				m_descriptorSet0 = descriptorPool->createDescriptorSet(gpuDescriptorSetLayout0);
				IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet = {};
				writeDescriptorSet.dstSet = m_descriptorSet0.get();
				writeDescriptorSet.binding = 0;
				writeDescriptorSet.count = 1u;
				writeDescriptorSet.arrayElement = 0u;
				video::IGPUDescriptorSet::SDescriptorInfo info;
				{
					info.desc = m_outImgView;
					info.info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
				}
				writeDescriptorSet.info = &info;

				m_device->updateDescriptorSets(1, &writeDescriptorSet, 0u, nullptr);

				m_uboDescriptorSet1 = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
				{
					video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
					uboWriteDescriptorSet.dstSet = m_uboDescriptorSet1.get();
					uboWriteDescriptorSet.binding = 0;
					uboWriteDescriptorSet.count = 1u;
					uboWriteDescriptorSet.arrayElement = 0u;
					video::IGPUDescriptorSet::SDescriptorInfo info;
					{
						info.desc = m_ubo;
						info.info.buffer.offset = 0ull;
						info.info.buffer.size = sizeof(SBasicViewParametersAligned);
					}
					uboWriteDescriptorSet.info = &info;
					m_device->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
				}

				ISampler::SParams samplerParams0 = {
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETBC_FLOAT_OPAQUE_BLACK,
					ISampler::ETF_LINEAR,
					ISampler::ETF_LINEAR,
					ISampler::ESMM_LINEAR,
					0u,
					false,
					ECO_ALWAYS
				};
				auto sampler0 = m_device->createSampler(samplerParams0);
				ISampler::SParams samplerParams1 = {
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETBC_INT_OPAQUE_BLACK,
					ISampler::ETF_NEAREST,
					ISampler::ETF_NEAREST,
					ISampler::ESMM_NEAREST,
					0u,
					false,
					ECO_ALWAYS
				};
				auto sampler1 = m_device->createSampler(samplerParams1);

				m_descriptorSet2 = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout2));
				{
					constexpr auto kDescriptorCount = 3;
					IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet[kDescriptorCount];
					IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo[kDescriptorCount];
					for (auto i = 0; i < kDescriptorCount; i++)
					{
						samplerWriteDescriptorSet[i].dstSet = m_descriptorSet2.get();
						samplerWriteDescriptorSet[i].binding = i;
						samplerWriteDescriptorSet[i].arrayElement = 0u;
						samplerWriteDescriptorSet[i].count = 1u;
						samplerWriteDescriptorSet[i].info = samplerDescriptorInfo + i;
					}

					samplerDescriptorInfo[0].desc = m_envMapView;
					{
						// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
						samplerDescriptorInfo[0].info.combinedImageSampler.sampler = sampler0;
						samplerDescriptorInfo[0].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
					}
					samplerDescriptorInfo[1].desc = m_sequenceBufferView;
					samplerDescriptorInfo[2].desc = m_scrambleView;
					{
						// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
						samplerDescriptorInfo[2].info.combinedImageSampler.sampler = sampler1;
						samplerDescriptorInfo[2].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
					}

					m_device->updateDescriptorSets(kDescriptorCount, samplerWriteDescriptorSet, 0u, nullptr);
				}
			}

			// Create ui descriptors
			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				{
					IGPUSampler::SParams params;
					params.AnisotropicFilter = 1u;
					params.TextureWrapU = ISampler::ETC_REPEAT;
					params.TextureWrapV = ISampler::ETC_REPEAT;
					params.TextureWrapW = ISampler::ETC_REPEAT;

					m_ui.samplers.gui = m_device->createSampler(params);
					m_ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
				}

				{
					IGPUSampler::SParams params;
					params.MinLod = 0.f;
					params.MaxLod = 0.f;
					params.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;

					m_ui.samplers.scene = m_device->createSampler(params);
					m_ui.samplers.scene->setObjectDebugName("Nabla IMGUI Scene Sampler");
				}

				std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
				for (auto& it : immutableSamplers)
					it = smart_refctd_ptr(m_ui.samplers.scene);

				immutableSamplers[nbl::ext::imgui::UI::FontAtlasTexId] = smart_refctd_ptr(m_ui.samplers.gui);

				nbl::ext::imgui::UI::SCreationParameters params;

				params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
				params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
				params.assetManager = m_assetMgr;
				params.pipelineCache = nullptr;
				params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils.get(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
				params.streamingBuffer = nullptr;
				params.subpassIx = 0u;
				params.transfer = getTransferUpQueue();
				params.utilities = m_utils;
				{
					m_ui.manager = core::make_smart_refctd_ptr<nbl::ext::imgui::UI>(std::move(params));

					// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
					const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
					const auto& params = m_ui.manager->getCreationParameters();

					IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
					descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
					descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = MaxUITextureCount;
					descriptorPoolInfo.maxSets = 1u;
					descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

					m_guiDescriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
					assert(m_guiDescriptorSetPool);

					m_guiDescriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
					assert(m_ui.descriptorSet);
				}
			}
			m_ui.manager->registerListener(
				[this]() -> void {
					ImGuiIO& io = ImGui::GetIO();

					m_camera.setProjectionMatrix([&]()
					{
						static matrix4SIMD projection;

						projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);

						return projection;
					}());

					ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

					// create a window and insert the inspector
					ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
					ImGui::Begin("Editor");

					ImGui::SameLine();

					ImGui::Text("Camera");

					ImGui::Checkbox("Enable camera movement", &move);
					ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);

					ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);

					ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
					ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

					ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);

					ImGui::Image(SceneTextureIndex, ImGui::GetContentRegionAvail());

					// Nabla Imgui backend MDI buffer info
					{
						auto* streamingBuffer = m_ui.manager->getStreamingBuffer();
						const size_t totalAllocatedSize = streamingBuffer->get_total_size();
						const size_t isUse = ((nbl::ext::imgui::UI::SMdiBuffer::compose_t*)streamingBuffer)->max_size();

						float freePercentage = 100.0f * (float)(totalAllocatedSize - isUse) / (float)totalAllocatedSize;
						float allocatedPercentage = 1.0f - (float)(totalAllocatedSize - isUse) / (float)totalAllocatedSize;

						ImVec2 barSize = ImVec2(400, 30);
						float windowPadding = 10.0f;
						float verticalPadding = ImGui::GetStyle().FramePadding.y;

						ImGui::SetNextWindowSize(ImVec2(barSize.x + 2 * windowPadding, 110 + verticalPadding), ImGuiCond_Always);
						ImGui::Begin("Nabla Imgui MDI Buffer Info", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);

						ImGui::Text("Total Allocated Size: %zu bytes", totalAllocatedSize);
						ImGui::Text("In use: %zu bytes", isUse);
						ImGui::Text("Buffer Usage:");

						ImGui::SetCursorPosX(windowPadding);

						if (freePercentage > 70.0f)
							ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 1.0f, 0.0f, 0.4f));
						else if (freePercentage > 30.0f)
							ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 1.0f, 0.0f, 0.4f));
						else
							ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 0.4f));

						ImGui::ProgressBar(allocatedPercentage, barSize, "");

						ImGui::PopStyleColor();

						ImDrawList* drawList = ImGui::GetWindowDrawList();

						ImVec2 progressBarPos = ImGui::GetItemRectMin();
						ImVec2 progressBarSize = ImGui::GetItemRectSize();

						const char* text = "%.2f%% free";
						char textBuffer[64];
						snprintf(textBuffer, sizeof(textBuffer), text, freePercentage);

						ImVec2 textSize = ImGui::CalcTextSize(textBuffer);
						ImVec2 textPos = ImVec2
						(
							progressBarPos.x + (progressBarSize.x - textSize.x) * 0.5f,
							progressBarPos.y + (progressBarSize.y - textSize.y) * 0.5f
						);

						ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
						drawList->AddRectFilled
						(
							ImVec2(textPos.x - 5, textPos.y - 2),
							ImVec2(textPos.x + textSize.x + 5, textPos.y + textSize.y + 2),
							ImGui::GetColorU32(bgColor)
						);

						ImGui::SetCursorScreenPos(textPos);
						ImGui::Text("%s", textBuffer);

						ImGui::Dummy(ImVec2(0.0f, verticalPadding));

						ImGui::End();
					}

					ImGui::End();
				}
			);

			m_winMgr->setWindowSize(m_window.get(), WindowDimensions.x, WindowDimensions.y);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());
			m_oracle.reportBeginFrameRecord();
			m_camera.mapKeysToArrows();

			return true;
		}

		bool updateGUIDescriptorSet()
		{
			// texture atlas + our scene texture, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

			descriptorInfo[SceneTextureIndex].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

			descriptorInfo[SceneTextureIndex].desc = m_outImgView;

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = m_ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;
			writes[SceneTextureIndex].info = descriptorInfo.data() + SceneTextureIndex;

			return m_device->updateDescriptorSets(writes, {});
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

			if (m_realFrameIx >= m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] = 
				{
					{
						.semaphore = m_uiSemaphore.get(),
						.value = m_realFrameIx + 1 - m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			// CPU events
			update();

			// render whole scene to offline frame buffer & submit
			{
				auto queue = getGraphicsQueue();
				auto& cmdbuf = m_cmdBufs[resourceIx];
				cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
				const auto viewMatrix = m_camera.getViewMatrix();
				const auto viewProjectionMatrix = matrix4SIMD();
				/*
				* Temporarily use identity matrix (Desktop only)
					matrix4SIMD::concatenateBFollowedByAPrecisely(
						video::ISurface::getSurfaceTransformationMatrix(swapchain->getPreTransform()),
						m_camera.getConcatenatedMatrix()
					);
				*/

				queue->startCapture();

				// safe to proceed
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				{
					auto mv = viewMatrix;
					auto mvp = viewProjectionMatrix;
					core::matrix3x4SIMD normalMat;
					mv.getSub3x3InverseTranspose(normalMat);

					SBasicViewParametersAligned viewParams;
					memcpy(viewParams.uboData.MV, mv.pointer(), sizeof(mv));
					memcpy(viewParams.uboData.MVP, mvp.pointer(), sizeof(mvp));
					memcpy(viewParams.uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));

					asset::SBufferRange<video::IGPUBuffer> range;
					range.buffer = m_ubo;
					range.offset = 0ull;
					range.size = sizeof(viewParams);
					
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf.get() };
					m_intendedSubmit.commandBuffers = { &cmdbufInfo, 1 };
					
					m_utils->updateBufferRangeViaStagingBuffer(m_intendedSubmit, range, &viewParams);
					m_utils->autoSubmit(m_intendedSubmit, [&](SIntendedSubmitInfo& nextSubmit) -> bool { return true; });
				}

				// TRANSITION m_outImgView to GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
				{
					constexpr SMemoryBarrier barriers[] = {
						{
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
						},
						{
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
						},
						{
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
						}
					};

					const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
						{
							.barrier = {
								.dep = barriers[0]
							},
							.image = m_outImgView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = 1u
							},
							.oldLayout = IImage::LAYOUT::UNDEFINED,
							.newLayout = IImage::LAYOUT::GENERAL
						},
						{
							.barrier = {
								.dep = barriers[1]
							},
							.image = m_scrambleView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = 1u
							},
							.oldLayout = IImage::LAYOUT::UNDEFINED,
							.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL
						},
						{
							.barrier = {
								.dep = barriers[2]
							},
							.image = m_envMapView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = m_envMapView->getCreationParameters().subresourceRange.levelCount,
								.baseArrayLayer = 0u,
								.layerCount = m_envMapView->getCreationParameters().subresourceRange.layerCount
							},
							.oldLayout = IImage::LAYOUT::UNDEFINED,
							.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL
						}
					};
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
				}

				// cube envmap handle
				{
					cmdbuf->bindComputePipeline(m_pipeline.get());
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, &m_descriptorSet0.get());
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 1u, 1u, &m_uboDescriptorSet1.get());
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 2u, 1u, &m_descriptorSet2.get());
					cmdbuf->dispatch(1 + (WindowDimensions.x - 1) / DefaultWorkGroupSize, 1 + (WindowDimensions.y - 1) / DefaultWorkGroupSize, 1u);
				}
				cmdbuf->end();
				// TODO: tone mapping and stuff

				// submit
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = { {
					.semaphore = m_renderSemaphore.get(),
					.value = m_realFrameIx + 1u,
					// just as we've outputted all pixels, signal
					.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				} };
				const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
					.cmdbuf = cmdbuf.get()
				} };
				const IQueue::SSubmitInfo infos[1] = { {
					.waitSemaphores = {},
					.commandBuffers =  commandBuffers,
					.signalSemaphores = rendered
				}};

				queue->submit({ infos, 1 });

				queue->endCapture();
			}

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->beginDebugMarker("ComputeShaderPathtracer IMGUI Frame");

			auto* queue = getGraphicsQueue();

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WindowDimensions.x;
				viewport.height = WindowDimensions.y;
			}
			cb->setViewport(0u, 1u, &viewport);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			// UI render pass
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo info = 
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearColor,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				ISemaphore::SWaitInfo waitInfo = { .semaphore = m_uiSemaphore.get(), .value = m_realFrameIx + 1u };

				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				m_ui.manager->render(cb, waitInfo);
				cb->endRenderPass();
			}
			cb->end();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = 
				{ 
					{
						.semaphore = m_uiSemaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
					} 
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = 
						{ 
							{ .cmdbuf = cb } 
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

						const nbl::video::ISemaphore::SWaitInfo waitInfos[] = 
						{ {
							.semaphore = m_renderSemaphore.get(),
							.value = m_realFrameIx
						} };
						
						m_device->blockForSemaphores(waitInfos);

						updateGUIDescriptorSet();

						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_realFrameIx--;
					}
				}

				m_window->setCaption("[Nabla Engine] UI App Test Demo");
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

		inline void update()
		{
			m_camera.setMoveSpeed(moveSpeed);
			m_camera.setRotateSpeed(rotateSpeed);

			static std::chrono::microseconds previousEventTimestamp{};

			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				m_oracle.reportEndFrameRecord();
				const auto timestamp = m_oracle.getNextPresentationTimeStamp();
				m_oracle.reportBeginFrameRecord();

				return timestamp;
			};

			const auto nextPresentationTimestamp = updatePresentationTimestamp();

			struct
			{
				std::vector<SMouseEvent> mouse{};
				std::vector<SKeyboardEvent> keyboard{};
			} capturedEvents;

			if (move) m_camera.beginInputProcessing(nextPresentationTimestamp);
			{
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (move)
						m_camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.mouse.emplace_back(e);

						if (e.type == nbl::ui::SMouseEvent::EET_SCROLL)
							gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(e.scrollEvent.verticalScroll)), int64_t(0), int64_t(ELG_COUNT - (uint8_t)1u));
					}
				}, m_logger.get());

			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (move)
						m_camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.keyboard.emplace_back(e);
					}
				}, m_logger.get());
			}
			if (move) m_camera.endInputProcessing(nextPresentationTimestamp);

			const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			const ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = mouseEvents,
				.keyboardEvents = keyboardEvents
			};

			m_ui.manager->update(params);
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

		// gpu resources
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet0, m_uboDescriptorSet1, m_descriptorSet2;

		core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

		// system resources
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		// pathtracer resources
		smart_refctd_ptr<IGPUImageView> m_envMapView, m_scrambleView;
		smart_refctd_ptr<IGPUBufferView> m_sequenceBufferView;
		smart_refctd_ptr<IGPUBuffer> m_ubo;
		smart_refctd_ptr<IGPUImageView> m_outImgView;

		// sync
		smart_refctd_ptr<ISemaphore> m_uiSemaphore, m_renderSemaphore;

		// image upload resources
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;

		struct C_UI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		} m_ui;

		Camera m_camera;
		video::CDumbPresentationOracle m_oracle;

		uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

		bool move = false;
		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

		bool m_firstFrame = true;
		IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
};

NBL_MAIN_FUNC(ComputeShaderPathtracer)

#if 0
int main()
{
	uint32_t resourceIx = 0;
	while (windowCb->isWindowOpen())
	{
		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
			while (device->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
			{
			} else
				fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

			// Copy HDR Image to SwapChain
			auto srcImgViewCreationParams = outHDRImageViews[imgnum]->getCreationParameters();
			auto dstImgViewCreationParams = fbo->begin()[imgnum]->getCreationParameters().attachments[0]->getCreationParameters();

			// Getting Ready for Blit
			// TRANSITION outHDRImageViews[imgnum] to EIL_TRANSFER_SRC_OPTIMAL
			// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_TRANSFER_DST_OPTIMAL
			{
				IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[2u] = {};
				imageBarriers[0].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				imageBarriers[0].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[0].newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
				imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].image = srcImgViewCreationParams.image;
				imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[0].subresourceRange.baseMipLevel = 0u;
				imageBarriers[0].subresourceRange.levelCount = 1;
				imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[0].subresourceRange.layerCount = 1;

				imageBarriers[1].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[1].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				imageBarriers[1].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[1].newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
				imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[1].image = dstImgViewCreationParams.image;
				imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[1].subresourceRange.baseMipLevel = 0u;
				imageBarriers[1].subresourceRange.levelCount = 1;
				imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[1].subresourceRange.layerCount = 1;
				cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, imageBarriers);
			}

			// Blit Image
			{
				SImageBlit blit = {};
				blit.srcOffsets[0] = { 0, 0, 0 };
				blit.srcOffsets[1] = { WIN_W, WIN_H, 1 };

				blit.srcSubresource.aspectMask = srcImgViewCreationParams.subresourceRange.aspectMask;
				blit.srcSubresource.mipLevel = srcImgViewCreationParams.subresourceRange.baseMipLevel;
				blit.srcSubresource.baseArrayLayer = srcImgViewCreationParams.subresourceRange.baseArrayLayer;
				blit.srcSubresource.layerCount = srcImgViewCreationParams.subresourceRange.layerCount;
				blit.dstOffsets[0] = { 0, 0, 0 };
				blit.dstOffsets[1] = { WIN_W, WIN_H, 1 };
				blit.dstSubresource.aspectMask = dstImgViewCreationParams.subresourceRange.aspectMask;
				blit.dstSubresource.mipLevel = dstImgViewCreationParams.subresourceRange.baseMipLevel;
				blit.dstSubresource.baseArrayLayer = dstImgViewCreationParams.subresourceRange.baseArrayLayer;
				blit.dstSubresource.layerCount = dstImgViewCreationParams.subresourceRange.layerCount;

				auto srcImg = srcImgViewCreationParams.image;
				auto dstImg = dstImgViewCreationParams.image;

				cb->blitImage(srcImg.get(), asset::IImage::EL_TRANSFER_SRC_OPTIMAL, dstImg.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, 1u, &blit, ISampler::ETF_NEAREST);
			}

			// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_PRESENT
			{
				IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
				imageBarriers[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				imageBarriers[0].barrier.dstAccessMask = asset::EAF_NONE;
				imageBarriers[0].oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
				imageBarriers[0].newLayout = asset::IImage::EL_PRESENT_SRC;
				imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].image = dstImgViewCreationParams.image;
				imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[0].subresourceRange.baseMipLevel = 0u;
				imageBarriers[0].subresourceRange.levelCount = 1;
				imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[0].subresourceRange.layerCount = 1;
				cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TOP_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
			}

			cb->end();
			device->resetFences(1, &fence.get());
			CommonAPI::Submit(device.get(), cb.get(), graphicsQueue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
			CommonAPI::Present(device.get(), swapchain.get(), graphicsQueue, renderFinished[resourceIx].get(), imgnum);

			if (LOG_TIMESTAMP)
			{
				std::array<uint64_t, 4> timestamps{};
				auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT) | video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT | video::IQueryPool::EQRF_64_BIT;
				device->getQueryPoolResults(timestampQueryPool.get(), 0u, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint64_t) * 2ull, queryResultFlags);
				const float timePassed = (timestamps[2] - timestamps[0]) * device->getPhysicalDevice()->getLimits().timestampPeriodInNanoSeconds;
				logger->log("Time Passed (Seconds) = %f", system::ILogger::ELL_INFO, (timePassed * 1e-9));
				logger->log("Timestamps availablity: %d, %d", system::ILogger::ELL_INFO, timestamps[1], timestamps[3]);
			}
	}

	const auto& fboCreationParams = fbo->begin()[0]->getCreationParameters();
	auto gpuSourceImageView = fboCreationParams.attachments[0];

	device->waitIdle();

	// bool status = ext::ScreenShot::createScreenShot(device.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], renderFinished[0].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
	// assert(status);

	return 0;
}

#endif