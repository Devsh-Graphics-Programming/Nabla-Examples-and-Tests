// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "../common/SimpleWindowedApplication.hpp"

#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "app_resources/common.hlsl"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class AutoexposureApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline std::string_view DefaultImagePathsFile = "../../media/noises/spp_benchmark_4k_512.exr";
	constexpr static inline uint32_t2 Dimensions = { 1280, 720 };
	constexpr static inline float32_t2 MeteringWindowScale = { 0.5f, 0.5f };
	constexpr static inline float32_t2 MeteringWindowOffset = { 0.25f, 0.25f };
	constexpr static inline float32_t2 LumaMinMax = { 1.0f / 4096.0f, 32768.0f };

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	inline AutoexposureApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		// So let's create our Window and Surface then!
		if (!m_surface)
		{
			{
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
				params.width = Dimensions[0];
				params.height = Dimensions[1];
				params.x = 32;
				params.y = 32;
				// Don't want to have a window lingering about before we're ready so create it hidden.
				// Only programmatic resize, not regular.
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "AutoexposureApp";
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
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		/*
			* We'll be using a combined image sampler for this example, which lets us assign both a sampled image and a sampler to the same binding.
			* In this example we provide a sampler at descriptor set creation time, via the SBinding struct below. This specifies that the sampler for this binding is immutable,
			* as evidenced by the name of the field in the SBinding.
			* Samplers for combined image samplers can also be mutable, which for a binding of a descriptor set is specified also at creation time by leaving the immutableSamplers
			* field set to its default (nullptr).
			*/
		std::array<smart_refctd_ptr<IGPUDescriptorSetLayout>, 3> dsLayouts;
		{
			auto defaultSampler = m_device->createSampler(
				{
					.AnisotropicFilter = 0
				}
			);

			const IGPUDescriptorSetLayout::SBinding imgBindings[3][1] = {
				{
					{
						.binding = 0,
						.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1,
						.immutableSamplers = &defaultSampler
					}
				},
				{
					{
						.binding = 0,
						.type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1,
						.immutableSamplers = nullptr
					}
				},
				{
					{
						.binding = 0,
						.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.count = 1,
						.immutableSamplers = &defaultSampler
					}
				}
			};

			bool dsLayoutCreation = true;
			for (uint32_t index = 0; index < dsLayouts.size(); index++) {
				dsLayouts[index] = m_device->createDescriptorSetLayout(imgBindings[index]);
				dsLayoutCreation = dsLayoutCreation && dsLayouts[index];
			}

			if (!dsLayoutCreation)
				return logFail("Failed to Create Descriptor Layouts");
		}

		// Create semaphores
		m_meterSemaphore = m_device->createSemaphore(m_submitIx);
		m_gatherSemaphore = m_device->createSemaphore(m_submitIx);
		m_presentSemaphore = m_device->createSemaphore(m_submitIx);

		// create the descriptor sets and with enough room
		{
			std::array<core::smart_refctd_ptr<IDescriptorPool>, 3> dsPools;
			bool dsPoolCreation = true;
			{
				const video::IGPUDescriptorSetLayout* const layouts[] = { dsLayouts[0].get() };
				const uint32_t setCounts[] = { 1u };
				dsPools[0] = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				dsPoolCreation = dsPoolCreation && dsPools[0];
			}
			{
				const video::IGPUDescriptorSetLayout* const layouts[] = { dsLayouts[1].get() };
				const uint32_t setCounts[] = { 1u };
				dsPools[1] = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				dsPoolCreation = dsPoolCreation && dsPools[1];
			}
			{
				const video::IGPUDescriptorSetLayout* const layouts[] = { dsLayouts[2].get() };
				const uint32_t setCounts[] = { 1u };
				dsPools[2] = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				dsPoolCreation = dsPoolCreation && dsPools[2];
			}

			if (!dsPoolCreation)
				return logFail("Failed to Create Descriptor Pools");

			bool dsCreation = true;
			{
				m_ds[0] = dsPools[0]->createDescriptorSet(dsLayouts[0]);
				dsCreation = dsCreation && m_ds[0];
			}
			{
				m_ds[1] = dsPools[1]->createDescriptorSet(dsLayouts[1]);
				dsCreation = dsCreation && m_ds[1];
			}
			{
				m_ds[2] = dsPools[2]->createDescriptorSet(dsLayouts[2]);
				dsCreation = dsCreation && m_ds[2];
			}

			if (!dsCreation)
				return logFail("Could not create Descriptor Sets!");
		}

		auto graphicsQueue = getGraphicsQueue();
		auto computeQueue = getComputeQueue();

		// Gather swapchain resources
		std::unique_ptr<CDefaultSwapchainFramebuffers> scResources;
		ISwapchain::SCreationParams swapchainParams;
		{
			swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
			// Need to choose a surface format
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");
			// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
			constexpr IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
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
			scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
			if (!scResources->getRenderpass())
				return logFail("Failed to create Renderpass!");
		}

		// Load the shaders and create the pipelines
		{
			auto loadCompileAndCreateShader = [&](const std::string& relPath) -> smart_refctd_ptr<IGPUShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset(relPath, lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return nullptr;

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto source = IAsset::castDown<ICPUShader>(assets[0]);
				if (!source)
					return nullptr;
				const uint32_t workgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
				const uint32_t subgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
				auto overriddenSource = CHLSLCompiler::createOverridenCopy(
					source.get(),
					"#define WorkgroupSize %d\n#define DeviceSubgroupSize %d\n",
					workgroupSize,
					subgroupSize
				);

				return m_device->createShader(overriddenSource.get());
			};

			auto createComputePipeline = [&](smart_refctd_ptr<IGPUShader>& shader, smart_refctd_ptr<IGPUComputePipeline>& pipeline, smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout) -> bool
			{
				{
					IGPUComputePipeline::SCreationParams params = {};
					params.layout = pipelineLayout.get();
					params.shader.shader = shader.get();
					params.shader.entryPoint = "main";
					params.shader.entries = nullptr;
					params.shader.requireFullSubgroups = true;
					params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(5);
					if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
						return logFail("Failed to create compute pipeline!\n");
				}

				return true;
			};

			const nbl::asset::SPushConstantRange pcRange = {
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.offset = 0,
					.size = sizeof(AutoexposurePushData)
			};

			// Luma Meter
			auto meterShader = loadCompileAndCreateShader("app_resources/luma_meter.comp.hlsl");
			if (!meterShader)
				return logFail("Failed to Load and Compile Compute Shader: meterShader!");
			auto meterLayout = m_device->createPipelineLayout(
				{ &pcRange, 1 },
				core::smart_refctd_ptr(dsLayouts[0]),
				nullptr,
				nullptr,
				nullptr
			);
			if (!createComputePipeline(meterShader, m_meterPipeline, meterLayout))
				return logFail("Could not create Luma Meter Pipeline!");

			// Luma Gather
			auto gatherShader = loadCompileAndCreateShader("app_resources/luma_gather.comp.hlsl");
			if (!gatherShader)
				return logFail("Failed to Load and Compile Compute Shader: gatherShader!");
			auto gatherLayout = m_device->createPipelineLayout(
				{ &pcRange, 1 },
				core::smart_refctd_ptr(dsLayouts[0]),
				nullptr,
				nullptr,
				core::smart_refctd_ptr(dsLayouts[1])
			);
			if (!createComputePipeline(gatherShader, m_gatherPipeline, gatherLayout))
				return logFail("Could not create Luma Gather Pipeline!");

			// Load FSTri Shader
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			// Load Fragment Shader
			auto fragmentShader = loadCompileAndCreateShader("app_resources/present.frag.hlsl");;
			if (!fragmentShader)
				return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

			const IGPUShader::SSpecInfo fragSpec = {
				.entryPoint = "main",
				.shader = fragmentShader.get()
			};
			auto presentLayout = m_device->createPipelineLayout(
				{ &pcRange, 1 },
				nullptr,
				nullptr,
				nullptr,
				core::smart_refctd_ptr(dsLayouts[2])
			);
			m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scResources->getRenderpass());
			if (!m_presentPipeline)
				return logFail("Could not create Graphics Pipeline!");
		}

		// Init the surface and create the swapchain
		if (!m_surface || !m_surface->init(graphicsQueue, std::move(scResources), swapchainParams.sharedParams))
			return logFail("Could not create Window & Surface or initialize the Surface!");

		// need resetttable commandbuffers for the upload utility
		{
			m_graphicsCmdPool = m_device->createCommandPool(graphicsQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			m_computeCmdPool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

			// create the commandbuffers
			if (!m_graphicsCmdPool || !m_computeCmdPool)
				return logFail("Couldn't create Command Pools!");

			if (
				!m_graphicsCmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_graphicsCmdBufs.data(), 1 }) ||
				!m_computeCmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_computeCmdBufs.data(), 2 })
			)
				return logFail("Couldn't create Command Buffers!");
		}

		// things for IUtilities
		{
			m_scratchSemaphore = m_device->createSemaphore(0);
			if (!m_scratchSemaphore)
				return logFail("Could not create Scratch Semaphore");
			m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
			// we don't want to overcomplicate the example with multi-queue
			m_intendedSubmit.queue = graphicsQueue;
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

		// Allocate and create buffer for Luma Gather
		{
			// Allocate memory
			m_gatherAllocation = {};
			smart_refctd_ptr<IGPUBuffer> buffer;
			{
				auto build_buffer = [this](
					smart_refctd_ptr<ILogicalDevice> m_device,
					nbl::video::IDeviceMemoryAllocator::SAllocation* allocation,
					smart_refctd_ptr<IGPUBuffer>& buffer,
					size_t buffer_size,
					const char* label)
				{
					IGPUBuffer::SCreationParams params;
					params.size = buffer_size;
					params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
					buffer = m_device->createBuffer(std::move(params));
					if (!buffer)
						return logFail("Failed to create GPU buffer of size %d!\n", buffer_size);

					buffer->setObjectDebugName(label);

					auto reqs = buffer->getMemoryReqs();
					reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

					*allocation = m_device->allocate(reqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
					if (!allocation->isValid())
						return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

					assert(allocation->memory.get() == buffer->getBoundMemory().memory);
				};

				build_buffer(m_device, &m_gatherAllocation, buffer, m_physicalDevice->getLimits().maxSubgroupSize, "Luma Gather Buffer");
			}
			m_gatherBDA = buffer->getDeviceAddress();

			auto mapped_memory = m_gatherAllocation.memory->map({ 0ull, m_gatherAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ);
			if (!mapped_memory)
				return logFail("Failed to map the Device Memory!\n");
		}

		// Allocate and Leave 1/4 for image uploads, to test image copy with small memory remaining
		{
			uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
			uint32_t maxFreeBlock = m_utils->getDefaultUpStreamingBuffer()->max_size();
			const uint32_t allocationAlignment = 64u;
			const uint32_t allocationSize = (maxFreeBlock / 4) * 3;
			m_utils->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
		}

		// Load exr file into gpu
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

			const auto cpuImgView = ICPUImageView::create(std::move(viewParams));
			const auto& cpuImgParams = cpuImgView->getCreationParameters();

			// create matching size image upto dimensions
			IGPUImage::SCreationParams imageParams = {};
			imageParams = cpuImgParams.image->getCreationParameters();
			imageParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT | IGPUImage::EUF_SAMPLED_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT;
			// promote format because RGB8 and friends don't actually exist in HW
			{
				const IPhysicalDevice::SImageFormatPromotionRequest request = {
					.originalFormat = imageParams.format,
					.usages = IPhysicalDevice::SFormatImageUsages::SUsage(imageParams.usage)
				};
				imageParams.format = m_physicalDevice->promoteImageFormat(request, imageParams.tiling);
			}
			if (imageParams.type == IGPUImage::ET_3D)
				imageParams.flags |= IGPUImage::ECF_2D_ARRAY_COMPATIBLE_BIT;
			m_gpuImg = m_device->createImage(std::move(imageParams));
			if (!m_gpuImg || !m_device->allocate(m_gpuImg->getMemoryReqs(), m_gpuImg.get()).isValid())
				return false;
			m_gpuImg->setObjectDebugName("Autoexposure Image");

			imageParams = m_gpuImg->getCreationParameters();
			imageParams.usage = IGPUImage::EUF_SAMPLED_BIT | IGPUImage::EUF_STORAGE_BIT;
			m_tonemappedImg = m_device->createImage(std::move(imageParams));
			if (!m_tonemappedImg || !m_device->allocate(m_tonemappedImg->getMemoryReqs(), m_tonemappedImg.get()).isValid())
				return false;
			m_tonemappedImg->setObjectDebugName("Tonemapped Image");

			// Now show the window
			m_winMgr->show(m_window.get());

			// we don't want to overcomplicate the example with multi-queue
			auto queue = getGraphicsQueue();
			auto cmdbuf = m_graphicsCmdBufs[0].get();
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
					.image = m_gpuImg.get(),
					// transition the whole view
					.subresourceRange = cpuImgParams.subresourceRange,
					// a wiping transition
					.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
				},
				{
					.image = m_tonemappedImg.get(),
					.subresourceRange = cpuImgParams.subresourceRange,
					.newLayout = IGPUImage::LAYOUT::GENERAL
				}
			};
			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers2[] = {
				{
					.barrier = {
						.dep = transferBarriers[1]
						// no ownership transfers
					},
					.image = m_gpuImg.get(),
					// transition the whole view
					.subresourceRange = cpuImgParams.subresourceRange,
					// a wiping transition
					.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					.newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
				}
			};
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers1 });
			// upload contents
			m_utils->updateImageViaStagingBuffer(
				m_intendedSubmit,
				cpuImgParams.image->getBuffer(),
				cpuImgParams.image->getCreationParameters().format,
				m_gpuImg.get(),
				IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				cpuImgParams.image->getRegions()
			);
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers2 });
			m_utils->autoSubmit(m_intendedSubmit, [&](SIntendedSubmitInfo& nextSubmit) -> bool { return true; });

			IGPUImageView::SCreationParams gpuImgViewParams = {
				.image = m_gpuImg,
				.viewType = IGPUImageView::ET_2D,
				.format = m_gpuImg->getCreationParameters().format,
			};
			IGPUImageView::SCreationParams tonemappedImgViewParams = {
				.image = m_tonemappedImg,
				.viewType = IGPUImageView::ET_2D,
				.format = m_tonemappedImg->getCreationParameters().format
			};

			m_gpuImgView = m_device->createImageView(std::move(gpuImgViewParams));
			m_tonemappedImgView = m_device->createImageView(std::move(tonemappedImgViewParams));

			IGPUDescriptorSet::SDescriptorInfo infos[3];
			infos[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			infos[0].desc = m_gpuImgView;
			infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
			infos[1].desc = m_tonemappedImgView;
			infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
			infos[2].desc = m_tonemappedImgView;


			IGPUDescriptorSet::SWriteDescriptorSet writeDescriptors[] = {
				{
					.dstSet = m_ds[0].get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = infos
				},
				{
					.dstSet = m_ds[1].get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = infos + 1
				},
				{
					.dstSet = m_ds[2].get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = infos + 2
				}
			};

			m_device->updateDescriptorSets(3, writeDescriptors, 0, nullptr);

			queue->endCapture();
		}

		return true;
	}

	// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
	inline void workLoopBody() override
	{
		const uint32_t SubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;

		uint32_t2 viewportSize = { m_gpuImg->getCreationParameters().extent.width, m_gpuImg->getCreationParameters().extent.height };
		float32_t sampleCount = (viewportSize.x * viewportSize.y) / 4;
		uint32_t workgroupSize = SubgroupSize * SubgroupSize;
		sampleCount = workgroupSize * (1 + (sampleCount - 1) / workgroupSize);

		// Luma Meter
		{
			auto queue = getComputeQueue();
			auto cmdbuf = m_computeCmdBufs[0].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
			auto ds = m_ds[0].get();

			auto pc = AutoexposurePushData
			{
				.window = nbl::hlsl::luma_meter::MeteringWindow::create(MeteringWindowScale, MeteringWindowOffset),
				.lumaMinMax = LumaMinMax,
				.sampleCount = sampleCount,
				.viewportSize = viewportSize,
				.lumaMeterBDA = m_gatherBDA
			};

			const uint32_t2 dispatchSize = {
				1 + ((viewportSize.x / 2) - 1) / SubgroupSize,
				1 + ((viewportSize.y / 2) - 1) / SubgroupSize
			};

			queue->startCapture();

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->bindComputePipeline(m_meterPipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_meterPipeline->getLayout(), 0, 1, &ds); // also if you created DS Set with 3th index you need to respect it here - firstSet tells you the index of set and count tells you what range from this index it should update, useful if you had 2 DS with lets say set index 2,3, then you can bind both with single call setting firstSet to 2, count to 2 and last argument would be pointet to your DS pointers
			cmdbuf->pushConstants(m_meterPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(pc), &pc);
			cmdbuf->dispatch(dispatchSize.x, dispatchSize.y);
			cmdbuf->end();

			{
				IQueue::SSubmitInfo submit_infos[1];
				IQueue::SSubmitInfo::SCommandBufferInfo cmdBufs[] = {
					{
						.cmdbuf = cmdbuf
					}
				};
				submit_infos[0].commandBuffers = cmdBufs;
				IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
					{
						.semaphore = m_meterSemaphore.get(),
						.value = m_submitIx + 1,
						.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
					}
				};
				submit_infos[0].signalSemaphores = signals;

				queue->submit(submit_infos);
				queue->endCapture();
			}

			const ISemaphore::SWaitInfo wait_infos[] = {
				{
					.semaphore = m_meterSemaphore.get(),
					.value = m_submitIx + 1
				}
			};
			m_device->blockForSemaphores(wait_infos);
		}

		// Luma Gather and Tonemapping
		{
			auto queue = getComputeQueue();
			auto cmdbuf = m_computeCmdBufs[1].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
			auto ds1 = m_ds[0].get();
			auto ds2 = m_ds[1].get();

			auto pc = AutoexposurePushData
			{
				.window = nbl::hlsl::luma_meter::MeteringWindow::create(MeteringWindowScale, MeteringWindowOffset),
				.lumaMinMax = LumaMinMax,
				.sampleCount = sampleCount,
				.viewportSize = viewportSize,
				.lumaMeterBDA = m_gatherBDA
			};

			const uint32_t2 dispatchSize = {
				1 + ((viewportSize.x) - 1) / SubgroupSize,
				1 + ((viewportSize.y) - 1) / SubgroupSize
			};

			queue->startCapture();

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->bindComputePipeline(m_gatherPipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_gatherPipeline->getLayout(), 0, 1, &ds1); // also if you created DS Set with 3th index you need to respect it here - firstSet tells you the index of set and count tells you what range from this index it should update, useful if you had 2 DS with lets say set index 2,3, then you can bind both with single call setting firstSet to 2, count to 2 and last argument would be pointet to your DS pointers
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_gatherPipeline->getLayout(), 3, 1, &ds2);
			cmdbuf->pushConstants(m_gatherPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(pc), &pc);
			cmdbuf->dispatch(dispatchSize.x, dispatchSize.y);
			cmdbuf->end();

			{
				IQueue::SSubmitInfo submit_infos[1];
				IQueue::SSubmitInfo::SCommandBufferInfo cmdBufs[] = {
					{
						.cmdbuf = cmdbuf
					}
				};
				submit_infos[0].commandBuffers = cmdBufs;
				IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
					{
						.semaphore = m_gatherSemaphore.get(),
						.value = m_submitIx + 1,
						.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
					}
				};
				submit_infos[0].signalSemaphores = signals;

				queue->submit(submit_infos);
				queue->endCapture();
			}

			const ISemaphore::SWaitInfo wait_infos[] = {
				{
					.semaphore = m_gatherSemaphore.get(),
					.value = m_submitIx + 1
				}
			};
			m_device->blockForSemaphores(wait_infos);
		}

		// Render to swapchain
		{
			// Acquire
			auto acquire = m_surface->acquireNextImage();
			if (!acquire)
				return;

			auto queue = getGraphicsQueue();
			auto cmdbuf = m_graphicsCmdBufs[0].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
			auto ds = m_ds[2].get();

			auto pc = AutoexposurePushData
			{
				.window = nbl::hlsl::luma_meter::MeteringWindow::create(MeteringWindowScale, MeteringWindowOffset),
				.lumaMinMax = LumaMinMax,
				.sampleCount = sampleCount,
				.viewportSize = viewportSize,
				.lumaMeterBDA = m_gatherBDA
			};

			queue->startCapture();

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = { m_window->getWidth(), m_window->getHeight() }
			};
			// set viewport
			{
				const asset::SViewport viewport =
				{
					.width = float32_t(m_window->getWidth()),
					.height = float32_t(m_window->getHeight())
				};
				cmdbuf->setViewport({ &viewport, 1 });
			}
			cmdbuf->setScissor({ &currentRenderArea, 1 });

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
				cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			}

			cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, m_presentPipeline->getLayout(), 3, 1, &ds);
			ext::FullScreenTriangle::recordDrawCall(cmdbuf);
			cmdbuf->endRenderPass();

			cmdbuf->end();

			// submit
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = { {
				.semaphore = m_presentSemaphore.get(),
				.value = m_submitIx + 1,
				// just as we've outputted all pixels, signal
				.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
			} };
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
					.cmdbuf = cmdbuf
				} };
				// we don't need to wait for the transfer semaphore, because we submit everything to the same queue
				const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = { {
					.semaphore = acquire.semaphore,
					.value = acquire.acquireCount,
					.stageMask = PIPELINE_STAGE_FLAGS::NONE
				} };
				const IQueue::SSubmitInfo infos[1] = { {
					.waitSemaphores = acquired,
					.commandBuffers = commandBuffers,
					.signalSemaphores = rendered
				} };

				queue->submit(infos);
			}

			// Present
			m_surface->present(acquire.imageIndex, rendered);
			queue->endCapture();

			// Wait for completion
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[] = {
					{
						.semaphore = m_presentSemaphore.get(),
						.value = m_submitIx
					}
				};
				if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}
		}

		m_submitIx++;
	}

	inline bool keepRunning() override
	{
		// Keep arunning as long as we have a surface to present to (usually this means, as long as the window is open)
		if (m_surface->irrecoverable())
			return false;

		return true;
	}

	inline bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

protected:
	nbl::video::IDeviceMemoryAllocator::SAllocation m_gatherAllocation;
	uint64_t m_gatherBDA;
	smart_refctd_ptr<IGPUImage> m_gpuImg, m_tonemappedImg;
	smart_refctd_ptr<IGPUImageView> m_gpuImgView, m_tonemappedImgView;

	// for image uploads
	smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
	SIntendedSubmitInfo m_intendedSubmit;

	// Pipelines
	smart_refctd_ptr<IGPUComputePipeline> m_meterPipeline, m_gatherPipeline;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;

	// Descriptor Sets
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, 3> m_ds;

	// Command Buffers
	smart_refctd_ptr<IGPUCommandPool> m_graphicsCmdPool, m_computeCmdPool;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, 2> m_graphicsCmdBufs, m_computeCmdBufs;

	// Semaphores
	smart_refctd_ptr<ISemaphore> m_meterSemaphore, m_gatherSemaphore, m_presentSemaphore;
	uint64_t m_submitIx = 0;

	// window
	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
};

NBL_MAIN_FUNC(AutoexposureApp)

#if 0

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include <iostream>
#include <cstdio>


#include "nbl/ext/ToneMapper/CToneMapper.h"

#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;


int main()
{
	nbl::SIrrlichtCreationParameters deviceParams;
	deviceParams.Bits = 24; //may have to set to 32bit for some platforms
	deviceParams.ZBufferBits = 24; //we'd like 32bit here
	deviceParams.DriverType = EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	deviceParams.WindowSize = dimension2d<uint32_t>(1280, 720);
	deviceParams.Fullscreen = false;
	deviceParams.Vsync = true; //! If supported by target platform
	deviceParams.Doublebuffer = true;
	deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

	auto device = createDeviceEx(deviceParams);
	if (!device)
		return 1; // could not create selected driver.

	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	constexpr auto MeterMode = LumaMeterClass::EMM_MEDIAN;
	const float minLuma = 1.f/2048.f;
	const float maxLuma = 65536.f;

	auto cpuLumaMeasureSpecializedShader = LumaMeterClass::createShader(glslCompiler,inputColorSpace,MeterMode,minLuma,maxLuma);
	auto gpuLumaMeasureShader = driver->createShader(smart_refctd_ptr<const ICPUShader>(cpuLumaMeasureSpecializedShader->getUnspecialized()));
	auto gpuLumaMeasureSpecializedShader = driver->createSpecializedShader(gpuLumaMeasureShader.get(), cpuLumaMeasureSpecializedShader->getSpecializationInfo());

	const float meteringMinUV[2] = { 0.1f,0.1f };
	const float meteringMaxUV[2] = { 0.9f,0.9f };
	LumaMeterClass::Uniforms_t<MeterMode> uniforms;
	auto lumaDispatchInfo = LumaMeterClass::buildParameters(uniforms, outImg->getCreationParameters().extent, meteringMinUV, meteringMaxUV);

	auto uniformBuffer = driver->createFilledDeviceLocalBufferOnDedMem(sizeof(uniforms),&uniforms);


	using ToneMapperClass = ext::ToneMapper::CToneMapper;
	constexpr auto TMO = ToneMapperClass::EO_ACES;
	constexpr bool usingLumaMeter = MeterMode<LumaMeterClass::EMM_COUNT;
	constexpr bool usingTemporalAdapatation = true;

	auto cpuTonemappingSpecializedShader = ToneMapperClass::createShader(am->getGLSLCompiler(),
		inputColorSpace,
		std::make_tuple(outFormat,ECP_SRGB,OETF_sRGB),
		TMO,usingLumaMeter,MeterMode,minLuma,maxLuma,usingTemporalAdapatation
	);
	auto gpuTonemappingShader = driver->createShader(smart_refctd_ptr<const ICPUShader>(cpuTonemappingSpecializedShader->getUnspecialized()));
	auto gpuTonemappingSpecializedShader = driver->createSpecializedShader(gpuTonemappingShader.get(),cpuTonemappingSpecializedShader->getSpecializationInfo());

	auto outImgStorage = ToneMapperClass::createViewForImage(driver,false,core::smart_refctd_ptr(outImg),{static_cast<IImage::E_ASPECT_FLAGS>(0u),0,1,0,1});

	auto parameterBuffer = driver->createDeviceLocalGPUBufferOnDedMem(ToneMapperClass::getParameterBufferSize<TMO,MeterMode>());
	constexpr float Exposure = 0.f;
	constexpr float Key = 0.18;
	auto params = ToneMapperClass::Params_t<TMO>(Exposure, Key, 0.85f);
	{
		params.setAdaptationFactorFromFrameDelta(0.f);
		driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(),0u,sizeof(params),&params);
	}

	auto commonPipelineLayout = ToneMapperClass::getDefaultPipelineLayout(driver,usingLumaMeter);

	auto lumaMeteringPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuLumaMeasureSpecializedShader));
	auto toneMappingPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuTonemappingSpecializedShader));

	auto commonDescriptorSet = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(commonPipelineLayout->getDescriptorSetLayout(0u)));
	ToneMapperClass::updateDescriptorSet<TMO,MeterMode>(driver,commonDescriptorSet.get(),parameterBuffer,imgToTonemapView,outImgStorage,1u,2u,usingLumaMeter ? 3u:0u,uniformBuffer,0u,usingTemporalAdapatation);


	constexpr auto dynOffsetArrayLen = usingLumaMeter ? 2u : 1u;

	auto lumaDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);
	lumaDynamicOffsetArray->back() = sizeof(ToneMapperClass::Params_t<TMO>);

	auto toneDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);


	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->bindComputePipeline(lumaMeteringPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&lumaDynamicOffsetArray);
		driver->pushConstants(commonPipelineLayout.get(),IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(outBufferIx),&outBufferIx); outBufferIx ^= 0x1u;
		LumaMeterClass::dispatchHelper(driver,lumaDispatchInfo,true);

		driver->bindComputePipeline(toneMappingPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&toneDynamicOffsetArray);
		ToneMapperClass::dispatchHelper(driver,outImgStorage.get(),true);

		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
		if (usingTemporalAdapatation)
		{
			auto thisPresentStamp = std::chrono::high_resolution_clock::now();
			auto microsecondsElapsedBetweenPresents = std::chrono::duration_cast<std::chrono::microseconds>(thisPresentStamp-lastPresentStamp);
			lastPresentStamp = thisPresentStamp;

			params.setAdaptationFactorFromFrameDelta(float(microsecondsElapsedBetweenPresents.count())/1000000.f);
			// dont override shader output
			constexpr auto offsetPastLumaHistory = offsetof(decltype(params),lastFrameExtraEVAsHalf)+sizeof(decltype(params)::lastFrameExtraEVAsHalf);
			auto* paramPtr = reinterpret_cast<const uint8_t*>(&params);
			driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(), offsetPastLumaHistory, sizeof(params)-offsetPastLumaHistory, paramPtr+offsetPastLumaHistory);
		}
	}

	return 0;
}

#endif