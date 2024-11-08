// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"

struct SPushConstants
{
	uint64_t vertexBufferAddress;
	uint64_t indexBufferAddress;
};

struct SCameraParameters
{
	float camPos[3];

	float MVP[4 * 4];
	float invMVP[4 * 4];
	float V[3 * 4];
	float P[3 * 4];
};

class CSwapchainFramebuffersAndDepth final : public nbl::video::CDefaultSwapchainFramebuffers
{
	using base_t = CDefaultSwapchainFramebuffers;

public:
	template<typename... Args>
	inline CSwapchainFramebuffersAndDepth(ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args) : CDefaultSwapchainFramebuffers(device, std::forward<Args>(args)...)
	{
		const IPhysicalDevice::SImageFormatPromotionRequest req = {
			.originalFormat = _desiredDepthFormat,
			.usages = {IGPUImage::EUF_RENDER_ATTACHMENT_BIT}
		};
		m_depthFormat = m_device->getPhysicalDevice()->promoteImageFormat(req, IGPUImage::TILING::OPTIMAL);

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
		subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL };
		m_params.subpasses = subpasses;
	}

protected:
	inline bool onCreateSwapchain_impl(const uint8_t qFam) override
	{
		auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

		const auto depthFormat = m_renderpass->getCreationParameters().depthStencilAttachments[0].format;
		const auto& sharedParams = getSwapchain()->getCreationParameters().sharedParams;
		auto image = device->createImage({ IImage::SCreationParams{
			.type = IGPUImage::ET_2D,
			.samples = IGPUImage::ESCF_1_BIT,
			.format = depthFormat,
			.extent = {sharedParams.width,sharedParams.height,1},
			.mipLevels = 1,
			.arrayLayers = 1,
			.depthUsage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT
		} });

		device->allocate(image->getMemoryReqs(), image.get());

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

class RayQueryGeometryApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
		using clock_t = std::chrono::steady_clock;

		constexpr static inline uint32_t WIN_W = 1280, WIN_H = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
		static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

	public:
		inline RayQueryGeometryApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = device_base_t::getRequiredDeviceFeatures();
			retval.accelerationStructure = true;
			retval.rayQuery = true;
			return retval;
		}

		// TODO: Implement the feature intersection operator in the JSON generator and fix the TODO in `MonoDeviceApplication::getPreferredDeviceFeatures`!
		inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
		{
			auto retval = device_base_t::getPreferredDeviceFeatures();
			retval.accelerationStructureHostCommands = true;
			return retval;
		}

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WIN_W;
					params.height = WIN_H;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "RayQueryGeometryApp";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}

		// so that we can use the same queue for asset converter and rendering
		inline core::vector<queue_req_t> getQueueRequirements() const override
		{
			auto reqs = device_base_t::getQueueRequirements();
			reqs.front().requiredFlags |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
			return reqs;
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
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
					.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT
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

			// TODO: promote the depth format if D16 not supported, or quote the spec if there's guaranteed support for it
			auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), EF_D16_UNORM, swapchainParams.surfaceFormat.format, dependencies);

			auto* renderpass = scResources->getRenderpass();

			if (!renderpass)
				return logFail("Failed to create Renderpass!");

			auto gQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
			if (FRAMES_IN_FLIGHT < m_maxFramesInFlight)
			{
				m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
				m_maxFramesInFlight = FRAMES_IN_FLIGHT;
			}

			auto pool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

			for (auto i = 0u; i < m_maxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
			m_surface->recreateSwapchain();

			// create output images
			for (uint32_t i = 0; i < m_maxFramesInFlight; i++)
			{
				IGPUImage::SCreationParams imgInfo;
				imgInfo.format = asset::EF_R16G16B16A16_SFLOAT;
				imgInfo.type = IGPUImage::ET_2D;
				imgInfo.extent.width = WIN_W;
				imgInfo.extent.height = WIN_H;
				imgInfo.extent.depth = 1u;
				imgInfo.mipLevels = 1u;
				imgInfo.arrayLayers = 1u;
				imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
				imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
				imgInfo.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT;

				auto image = m_device->createImage(std::move(imgInfo));
				auto imageReqs = image->getMemoryReqs();
				imageReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto imageMem = m_device->allocate(imageReqs, image.get());

				IGPUImageView::SCreationParams imgViewInfo;
				imgViewInfo.image = std::move(image);
				imgViewInfo.format = asset::EF_R16G16B16A16_SFLOAT;
				imgViewInfo.viewType = IGPUImageView::ET_2D;
				imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
				imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				imgViewInfo.subresourceRange.baseArrayLayer = 0u;
				imgViewInfo.subresourceRange.baseMipLevel = 0u;
				imgViewInfo.subresourceRange.layerCount = 1u;
				imgViewInfo.subresourceRange.levelCount = 1u;

				outHDRImageViews[i] = m_device->createImageView(std::move(imgViewInfo));
			}

			{
				IGPUBuffer::SCreationParams params;
				params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
				params.size = sizeof(SCameraParameters);
				cameraBuffer = createBuffer(params);
			}

			auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
			auto* geometryCreator = assetManager->getGeometryCreator();

			auto cQueue = getComputeQueue();
			smart_refctd_ptr<nbl::video::IGPUCommandPool> singleUsePool = m_device->createCommandPool(cQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);

			// create geometry objects
			createGeometries(singleUsePool, geometryCreator);

			// create blas/tlas
			createAccelerationStructures(singleUsePool);

			// create pipelines
			{
				// shader
				const std::string shaderPath = "app_resources/render.comp.hlsl";
				IAssetLoader::SAssetLoadParams lparams = {};
				lparams.logger = m_logger.get();
				lparams.workingDirectory = "";
				auto bundle = m_assetMgr->getAsset(shaderPath, lparams);
				if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
				{
					m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, shaderPath);
					exit(-1);
				}

				const auto assets = bundle.getContents();
				assert(assets.size() == 1);
				smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);
				shaderSrc->setShaderStage(IShader::E_SHADER_STAGE::ESS_COMPUTE);
				auto shader = m_device->createShader(shaderSrc.get());

				// descriptors
				IGPUDescriptorSetLayout::SBinding bindings[] = {
					{
						.binding = 0,
						.type = asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1,
					},
					{
						.binding = 1,
						.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1,
					},
					{
						.binding = 2,
						.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1,
					}
				};
				auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

				const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayoutPtrs = { descriptorSetLayout.get() };
				const uint32_t setCounts[1u] = { m_maxFramesInFlight };
				renderPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs.begin(), dsLayoutPtrs.end()), setCounts);
				for (uint32_t i = 0; i < m_maxFramesInFlight; i++)
					renderDs[i] = renderPool->createDescriptorSet(descriptorSetLayout);

				SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0u, .size = sizeof(SPushConstants)};
				auto pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

				IGPUComputePipeline::SCreationParams params = {};
				params.layout = pipelineLayout.get();
				params.shader.shader = shader.get();
				m_device->createComputePipelines(nullptr, { &params, 1 }, &renderPipeline);
			}

			// write descriptors
			for (uint32_t i = 0; i < m_maxFramesInFlight; i++)
			{
				IGPUDescriptorSet::SDescriptorInfo infos[3];
				infos[0].desc = gpuTlas;
				infos[1].desc = smart_refctd_ptr(cameraBuffer);
				infos[1].info.buffer.size = cameraBuffer->getSize();
				infos[2].desc = outHDRImageViews[i];
				infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
					{.dstSet = renderDs[i].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
					{.dstSet = renderDs[i].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
					{.dstSet = renderDs[i].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
				};
				m_device->updateDescriptorSets(std::span(writes, 3), {});
			}

			// camera
			{
				core::vectorSIMDf cameraPosition(-5.81655884, 2.58630896, -4.23974705);
				core::vectorSIMDf cameraTarget(-0.349590302, -0.213266611, 0.317821503);
				matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 10000);
				camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);
			}

			m_winMgr->show(m_window.get());
			oracle.reportBeginFrameRecord();

			return true;
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

			if (m_realFrameIx >= m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

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

			auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->beginDebugMarker("RayQueryGeometryApp Frame");
			{
				camera.beginInputProcessing(nextPresentationTimestamp);
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
				camera.endInputProcessing(nextPresentationTimestamp);

				const auto type = static_cast<ObjectType>(gcIndex);
				// const auto& [gpu, meta] = resources.objects[type];

				//object.meta.type = type;
				//object.meta.name = meta.name;

				// TODO: hard code test one object first
				object.meta.type = OT_SPHERE;
				object.meta.name = objectsGpu[OT_SPHERE].meta.name;
			}

			const auto viewMatrix = camera.getViewMatrix();
			const auto projectionMatrix = camera.getProjectionMatrix();
			const auto viewProjectionMatrix = camera.getConcatenatedMatrix();

			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
			modelMatrix.setRotation(quaternion(0, 0, 0));

			core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
			core::matrix4SIMD invModelViewProjectionMatrix;
			modelViewProjectionMatrix.getInverseTransform(invModelViewProjectionMatrix);

			SCameraParameters uboData;
			const core::vector3df camPos = camera.getPosition().getAsVector3df();
			camPos.getAs3Values(uboData.camPos);
			memcpy(uboData.MVP, modelViewProjectionMatrix.pointer(), sizeof(uboData.MVP));
			memcpy(uboData.invMVP, invModelViewProjectionMatrix.pointer(), sizeof(uboData.invMVP));
			memcpy(uboData.V, viewMatrix.pointer(), sizeof(uboData.V));
			memcpy(uboData.P, projectionMatrix.pointer(), sizeof(uboData.P));
			{
				SBufferRange<IGPUBuffer> range;
				range.buffer = smart_refctd_ptr(cameraBuffer);
				range.size = cameraBuffer->getSize();

				cmdbuf->updateBuffer(range, &uboData);
			}

			auto* queue = getGraphicsQueue();

			// TODO: transition to render
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
				imageBarriers[0].barrier = {
				   .dep = {
					   .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
					   .srcAccessMask = ACCESS_FLAGS::NONE,
					   .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
					   .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
					}
				};
				imageBarriers[0].image = outHDRImageViews[m_currentImageAcquire.imageIndex]->getCreationParameters().image.get();
				imageBarriers[0].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
				imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
			}

			// do ray query
			auto obj = objectsGpu[OT_SPHERE];
			auto vertex = obj.bindings.vertex;
			auto index = obj.bindings.index;

			const SPushConstants pc = {
				.vertexBufferAddress = vertex.buffer->getDeviceAddress(),
				.indexBufferAddress = obj.useIndex() ? index.buffer->getDeviceAddress() : vertex.buffer->getDeviceAddress()
			};

			cmdbuf->bindComputePipeline(renderPipeline.get());
			cmdbuf->pushConstants(renderPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(SPushConstants), &pc);
			cmdbuf->bindDescriptorSets(EPBP_COMPUTE, renderPipeline->getLayout(), 0, 1, &renderDs[m_currentImageAcquire.imageIndex].get());
			cmdbuf->dispatch(getWorkgroupCount(WIN_W, WorkgroupSize), getWorkgroupCount(WIN_H, WorkgroupSize), 1);
			
			// blit
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[2];
				imageBarriers[0].barrier = {
				   .dep = {
					   .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
					   .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
					   .dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
					   .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				};
				imageBarriers[0].image = outHDRImageViews[m_currentImageAcquire.imageIndex]->getCreationParameters().image.get();
				imageBarriers[0].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
				imageBarriers[0].newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;

				imageBarriers[1].barrier = {
				   .dep = {
					   .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
					   .srcAccessMask = ACCESS_FLAGS::NONE,
					   .dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
					   .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				};
				imageBarriers[1].image = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);
				imageBarriers[1].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[1].oldLayout = IImage::LAYOUT::UNDEFINED;
				imageBarriers[1].newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL;

				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
			}

			{
				IGPUCommandBuffer::SImageBlit regions[] = {{
					.srcMinCoord = {0,0,0},
					.srcMaxCoord = {WIN_W,WIN_H,1},
					.dstMinCoord = {0,0,0},
					.dstMaxCoord = {WIN_W,WIN_H,1},
					.layerCount = 1,
					.srcBaseLayer = 0,
					.dstBaseLayer = 0,
					.srcMipLevel = 0,
					.dstMipLevel = 0,
					.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT
				}};
				
				auto srcImg = outHDRImageViews[m_currentImageAcquire.imageIndex]->getCreationParameters().image.get();
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				auto dstImg = scRes->getImage(m_currentImageAcquire.imageIndex);

				cmdbuf->blitImage(srcImg, IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, dstImg, IImage::LAYOUT::TRANSFER_DST_OPTIMAL, regions, ISampler::ETF_NEAREST);
			}

			// TODO: transition to present
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
				imageBarriers[0].barrier = {
				   .dep = {
					   .srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
					   .srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					   .dstStageMask = PIPELINE_STAGE_FLAGS::NONE,
					   .dstAccessMask = ACCESS_FLAGS::NONE
					}
				};
				imageBarriers[0].image = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);
				imageBarriers[0].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[0].oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
				imageBarriers[0].newLayout = IImage::LAYOUT::PRESENT_SRC;

				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
			}

			cmdbuf->endDebugMarker();
			cmdbuf->end();

			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
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

				std::string caption = "[Nabla Engine] Geometry Creator";
				{
					caption += ", displaying [" + std::string(object.meta.name.data()) + "]";
					m_window->setCaption(caption);
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
		uint32_t getWorkgroupCount(uint32_t dim, uint32_t size)
		{
			return (dim + size - 1) / size;
		}

		smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams& params)
		{
			smart_refctd_ptr<IGPUBuffer> buffer;
			buffer = m_device->createBuffer(std::move(params));
			auto bufReqs = buffer->getMemoryReqs();
			bufReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			m_device->allocate(bufReqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

			return buffer;
		}

		smart_refctd_ptr<IGPUCommandBuffer> getSingleUseCommandBufferAndBegin(smart_refctd_ptr<IGPUCommandPool> pool)
		{
			smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
				return nullptr;

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			return cmdbuf;
		}

		void cmdbufSubmitAndWait(smart_refctd_ptr<IGPUCommandBuffer> cmdbuf, CThreadSafeQueueAdapter* queue, uint64_t startValue)
		{
			cmdbuf->end();

			uint64_t finishedValue = startValue + 1;

			// submit builds
			{
				auto completed = m_device->createSemaphore(startValue);

				std::array<IQueue::SSubmitInfo::SSemaphoreInfo, 1u> signals;
				{
					auto& signal = signals.front();
					signal.value = finishedValue;
					signal.stageMask = bitflag(PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);
					signal.semaphore = completed.get();
				}

				const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
					.cmdbuf = cmdbuf.get()
				} };

				const IQueue::SSubmitInfo infos[] =
				{
					{
						.waitSemaphores = {},
						.commandBuffers = commandBuffers,
						.signalSemaphores = signals
					}
				};

				if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
				{
					m_logger->log("Failed to submit geometry transfer upload operations!", ILogger::ELL_ERROR);
					return;
				}

				const ISemaphore::SWaitInfo info[] =
				{ {
					.semaphore = completed.get(),
					.value = finishedValue
				} };

				m_device->blockForSemaphores(info);
			}
		}

		bool createGeometries(smart_refctd_ptr<IGPUCommandPool> pool, const IGeometryCreator* gc)
		{
			std::array<ReferenceObjectCpu, OT_COUNT> objectsCpu;
			objectsCpu[OT_CUBE] = ReferenceObjectCpu{ .meta = {.type = OT_CUBE, .name = "Cube Mesh" }, .shadersType = GP_BASIC, .data = gc->createCubeMesh(nbl::core::vector3df(1.f, 1.f, 1.f)) };
			objectsCpu[OT_SPHERE] = ReferenceObjectCpu{ .meta = {.type = OT_SPHERE, .name = "Sphere Mesh" }, .shadersType = GP_BASIC, .data = gc->createSphereMesh(2, 16, 16) };
			objectsCpu[OT_CYLINDER] = ReferenceObjectCpu{ .meta = {.type = OT_CYLINDER, .name = "Cylinder Mesh" }, .shadersType = GP_BASIC, .data = gc->createCylinderMesh(2, 2, 20) };
			objectsCpu[OT_RECTANGLE] = ReferenceObjectCpu{ .meta = {.type = OT_RECTANGLE, .name = "Rectangle Mesh" }, .shadersType = GP_BASIC, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) };
			objectsCpu[OT_DISK] = ReferenceObjectCpu{ .meta = {.type = OT_DISK, .name = "Disk Mesh" }, .shadersType = GP_BASIC, .data = gc->createDiskMesh(2, 30) };
			objectsCpu[OT_ARROW] = ReferenceObjectCpu{ .meta = {.type = OT_ARROW, .name = "Arrow Mesh" }, .shadersType = GP_BASIC, .data = gc->createArrowMesh() };
			objectsCpu[OT_CONE] = ReferenceObjectCpu{ .meta = {.type = OT_CONE, .name = "Cone Mesh" }, .shadersType = GP_CONE, .data = gc->createConeMesh(2, 3, 10) };
			objectsCpu[OT_ICOSPHERE] = ReferenceObjectCpu{ .meta = {.type = OT_ICOSPHERE, .name = "Icosphere Mesh" }, .shadersType = GP_ICO, .data = gc->createIcoSphere(1, 3, true) };

			auto cmdbuf = getSingleUseCommandBufferAndBegin(pool);

			for (uint32_t i = 0; i < objectsCpu.size(); i++)
			{
				const auto& geom = objectsCpu[i];
				auto& obj = objectsGpu[i];

				obj.meta.name = geom.meta.name;
				obj.meta.type = geom.meta.type;

				obj.indexCount = geom.data.indexCount;
				obj.indexType = geom.data.indexType;

// TODO: use asset converter to convert all buffers to IGPUBuffers (that part 100% works)

				auto vBuffer = smart_refctd_ptr(geom.data.bindings[0].buffer); // no offset
				auto vUsage = bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | 
					asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				obj.bindings.vertex.offset = 0u;
				auto vertexBuffer = m_device->createBuffer(IGPUBuffer::SCreationParams({.size = vBuffer->getSize(), .usage = vUsage}));

				//if (!vertexBuffer)
				//	return false;

				auto iBuffer = smart_refctd_ptr(geom.data.indexBuffer.buffer); // no offset
				auto iUsage = bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF |
					asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				obj.bindings.index.offset = 0u;
				auto indexBuffer = iBuffer ? m_device->createBuffer(IGPUBuffer::SCreationParams({ .size = iBuffer->getSize(), .usage = iUsage })) : nullptr;

				//if (geom.data.indexType != EIT_UNKNOWN)
				//	if (!indexBuffer)
				//		return false;

				for (auto buf : { vertexBuffer, indexBuffer })
				{
					if (buf)
					{
						auto reqs = buf->getMemoryReqs();
						reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
						m_device->allocate(reqs, buf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
					}
				}

				obj.bindings.vertex = { .offset = 0u, .buffer = std::move(vertexBuffer) };
				SBufferRange<IGPUBuffer> vRange = { .offset = obj.bindings.vertex.offset, .size = obj.bindings.vertex.buffer->getSize(), .buffer = obj.bindings.vertex.buffer };
				cmdbuf->updateBuffer(vRange, vBuffer->getPointer());

				if (iBuffer)
				{
					obj.bindings.index = { .offset = 0u, .buffer = std::move(indexBuffer) };
					SBufferRange<IGPUBuffer> iRange = { .offset = obj.bindings.index.offset, .size = obj.bindings.index.buffer->getSize(), .buffer = obj.bindings.index.buffer };
					cmdbuf->updateBuffer(iRange, iBuffer->getPointer());
				}
			}

			cmdbufSubmitAndWait(cmdbuf, getComputeQueue(), 24);

			return true;
		}

		bool createAccelerationStructures(smart_refctd_ptr<IGPUCommandPool> pool)
		{
			IQueryPool::SCreationParams qParams{ .queryCount = 1, .queryType = IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE };
			smart_refctd_ptr<IQueryPool> queryPool = m_device->createQueryPool(std::move(qParams));

			auto cmdbufBlas = getSingleUseCommandBufferAndBegin(pool);

			// build bottom level ASes
			{
				const auto& obj = objectsGpu[OT_CUBE];

				const uint32_t vertexStride = 12 * sizeof(float32_t);	// TODO: vary by object type, this is standard triangles for sphere etc.
				const uint32_t numVertices = obj.bindings.vertex.buffer->getSize() / vertexStride;
				uint32_t trisCount;
				if (obj.useIndex())
					trisCount = obj.indexCount / 3;
				else
					trisCount = numVertices / 3;

				IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer> triangles;
				triangles.vertexData[0] = obj.bindings.vertex;
				triangles.indexData = obj.bindings.index;
				triangles.maxVertex = numVertices - 1;
				triangles.vertexStride = vertexStride;
				triangles.vertexFormat = EF_R32G32B32_SFLOAT;
				triangles.indexType = obj.indexType;

				const auto blasFlags = bitflag(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT) | IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_COMPACTION_BIT;

				IGPUBottomLevelAccelerationStructure::DeviceBuildInfo blasBuildInfo;
				blasBuildInfo.buildFlags = blasFlags;
				blasBuildInfo.geometryCount = 1;	// only 1 geometry object per blas
				blasBuildInfo.srcAS = nullptr;
				blasBuildInfo.dstAS = nullptr;
				blasBuildInfo.triangles = &triangles;
				blasBuildInfo.scratch = {};

				ILogicalDevice::AccelerationStructureBuildSizes buildSizes;
				{
					const uint32_t maxPrimCount[1] = { trisCount };
					buildSizes = m_device->getAccelerationStructureBuildSizes(blasFlags, false, std::span{ &triangles, 1 }, maxPrimCount);
				}

				{
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
					params.size = buildSizes.accelerationStructureSize;
					smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

					IGPUBottomLevelAccelerationStructure::SCreationParams blasParams;
					blasParams.bufferRange.buffer = asBuffer;
					blasParams.bufferRange.offset = 0u;
					blasParams.bufferRange.size = buildSizes.accelerationStructureSize;
					blasParams.flags = IGPUBottomLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
					gpuBlas = m_device->createBottomLevelAccelerationStructure(std::move(blasParams));
				}

				smart_refctd_ptr<IGPUBuffer> scratchBuffer;
				{
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
					params.size = buildSizes.buildScratchSize;
					scratchBuffer = createBuffer(params);
				}

				blasBuildInfo.dstAS = gpuBlas.get();
				blasBuildInfo.scratch.buffer = scratchBuffer;
				blasBuildInfo.scratch.offset = 0u;

				IGPUBottomLevelAccelerationStructure::BuildRangeInfo buildRangeInfos[1u];
				buildRangeInfos[0].primitiveCount = trisCount;
				buildRangeInfos[0].primitiveByteOffset = 0u;
				buildRangeInfos[0].firstVertex = 0u;
				buildRangeInfos[0].transformByteOffset = 0u;
				IGPUBottomLevelAccelerationStructure::BuildRangeInfo* pRangeInfos[1u];
				pRangeInfos[0] = &buildRangeInfos[0];

				cmdbufBlas->buildAccelerationStructures({ &blasBuildInfo, 1 }, pRangeInfos);
			}

			{
				SMemoryBarrier memBarrier;
				memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
				memBarrier.srcAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
				memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
				memBarrier.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT;
				cmdbufBlas->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
			}

			cmdbufBlas->resetQueryPool(queryPool.get(), 0, 1);

			uint32_t queryCount = 0;
			const IGPUAccelerationStructure* ases[1u] = { gpuBlas.get() };
			cmdbufBlas->writeAccelerationStructureProperties({ ases, 1}, IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE, queryPool.get(), queryCount++);

			cmdbufSubmitAndWait(cmdbufBlas, getComputeQueue(), 39);

// TODO: SUBMIT THE BLAS BUILD FIRST AND AWAIT ITS COMPLETION WITH A SEMAPHORE SIGNAL

			auto cmdbufTlas = getSingleUseCommandBufferAndBegin(pool);

			//size_t asSizes[1];
			//m_device->getQueryPoolResults(queryPool.get(), 0, queryCount, asSizes, sizeof(size_t), IQueryPool::WAIT_BIT);
			//
			//auto cleanupBlas = gpuBlas;
			//{
			//	IGPUBuffer::SCreationParams params;
			//	params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
			//	params.size = asSizes[0];
			//	smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

			//	IGPUBottomLevelAccelerationStructure::SCreationParams blasParams;
			//	blasParams.bufferRange.buffer = asBuffer;
			//	blasParams.bufferRange.offset = 0u;
			//	blasParams.bufferRange.size = asSizes[0];
			//	blasParams.flags = IGPUBottomLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
			//	gpuBlas = m_device->createBottomLevelAccelerationStructure(std::move(blasParams));
			//}

			//IGPUBottomLevelAccelerationStructure::CopyInfo copyInfo;
			//copyInfo.src = cleanupBlas.get();
			//copyInfo.dst = gpuBlas.get();
			//copyInfo.mode = IGPUBottomLevelAccelerationStructure::COPY_MODE::COMPACT;
			//cmdbuf->copyAccelerationStructure(copyInfo);

			// build top level AS
			{
				const uint32_t instancesCount = 1;	// TODO: temporary for now
				IGPUTopLevelAccelerationStructure::DeviceInstance instances[instancesCount];
				core::matrix3x4SIMD identity;
				instances[0].blas.deviceAddress = gpuBlas->getReferenceForDeviceOperations().deviceAddress;

				{
					size_t bufSize = sizeof(IGPUTopLevelAccelerationStructure::DeviceStaticInstance);
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT |
						IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
					params.size = bufSize;
					instancesBuffer = createBuffer(params);	// does this need host visible memory?

					SBufferRange<IGPUBuffer> range = { .offset = 0u, .size = bufSize, .buffer = instancesBuffer };
					cmdbufTlas->updateBuffer(range, instances);
				}

				// make sure instances upload complete first
				{
					SMemoryBarrier memBarrier;
					memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
					memBarrier.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
					memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
					memBarrier.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
					cmdbufTlas->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
				}

				auto tlasFlags = bitflag(IGPUTopLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT);

				IGPUTopLevelAccelerationStructure::DeviceBuildInfo tlasBuildInfo;
				tlasBuildInfo.buildFlags = tlasFlags;
				tlasBuildInfo.srcAS = nullptr;
				tlasBuildInfo.dstAS = nullptr;
				tlasBuildInfo.instanceData.buffer = instancesBuffer;
				tlasBuildInfo.instanceData.offset = 0u;
				tlasBuildInfo.scratch = {};

				auto buildSizes = m_device->getAccelerationStructureBuildSizes(tlasFlags, 0, instancesCount);

				{
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
					params.size = buildSizes.accelerationStructureSize;
					smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

					IGPUTopLevelAccelerationStructure::SCreationParams tlasParams;
					tlasParams.bufferRange.buffer = asBuffer;
					tlasParams.bufferRange.offset = 0u;
					tlasParams.bufferRange.size = buildSizes.accelerationStructureSize;
					tlasParams.flags = IGPUTopLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
					gpuTlas = m_device->createTopLevelAccelerationStructure(std::move(tlasParams));
				}

				smart_refctd_ptr<IGPUBuffer> scratchBuffer;
				{
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
					params.size = buildSizes.buildScratchSize;
					scratchBuffer = createBuffer(params);
				}

				tlasBuildInfo.dstAS = gpuTlas.get();
				tlasBuildInfo.scratch.buffer = scratchBuffer;
				tlasBuildInfo.scratch.offset = 0u;

				IGPUTopLevelAccelerationStructure::BuildRangeInfo buildRangeInfo[1u];
				buildRangeInfo[0].instanceCount = instancesCount;
				buildRangeInfo[0].instanceByteOffset = 0u;
				IGPUTopLevelAccelerationStructure::BuildRangeInfo* pRangeInfos;
				pRangeInfos = &buildRangeInfo[0];

				cmdbufTlas->buildAccelerationStructures({ &tlasBuildInfo, 1 }, pRangeInfos);
			}

			cmdbufSubmitAndWait(cmdbufTlas, getComputeQueue(), 45);

//TODO : ERROR HANDLING FOR ALL THE CALLS ABOVE!
			return true;
		}


		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		video::CDumbPresentationOracle oracle;

		std::array<ReferenceObjectGpu, OT_COUNT> objectsGpu;
		ObjectDrawHookCpu object;

		smart_refctd_ptr<IGPUBottomLevelAccelerationStructure> gpuBlas;
		smart_refctd_ptr<IGPUTopLevelAccelerationStructure> gpuTlas;
		smart_refctd_ptr<IGPUBuffer> instancesBuffer;

		smart_refctd_ptr<IGPUBuffer> cameraBuffer;
		std::array<smart_refctd_ptr<IGPUImageView>, ISwapchain::MaxImages> outHDRImageViews;

		smart_refctd_ptr<IGPUComputePipeline> renderPipeline;
		std::array<smart_refctd_ptr<IGPUDescriptorSet>, ISwapchain::MaxImages> renderDs;
		smart_refctd_ptr<IDescriptorPool> renderPool;

		uint16_t gcIndex = {};

		void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
		{
			for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
			{
				auto ev = *eventIt;

				if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
					gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(ev.scrollEvent.verticalScroll)), int64_t(0), int64_t(OT_COUNT - (uint8_t)1u));
			}
		}
};

NBL_MAIN_FUNC(RayQueryGeometryApp)