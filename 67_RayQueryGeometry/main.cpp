// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"

class RayQueryGeometryApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
		using clock_t = std::chrono::steady_clock;

		constexpr static inline uint32_t WIN_W = 1280, WIN_H = 720;
		constexpr static inline uint32_t MaxFramesInFlight = 3u;

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
					params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
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
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
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

			auto gQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(gQueue, std::make_unique<ISimpleManagedSurface::ISwapchainResources>(), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			auto pool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

			for (auto i = 0u; i < MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
			m_surface->recreateSwapchain();

			// create output images
			outHDRImage = m_device->createImage({
				{
					.type = IGPUImage::ET_2D,
					.samples = asset::ICPUImage::ESCF_1_BIT,
					.format = asset::EF_R16G16B16A16_SFLOAT,
					.extent = {WIN_W, WIN_H, 1},
					.mipLevels = 1,
					.arrayLayers = 1,
					.flags = IImage::ECF_NONE,
					.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT
				}
			});
			if (!outHDRImage || !m_device->allocate(outHDRImage->getMemoryReqs(), outHDRImage.get()).isValid())
				return logFail("Could not create HDR Image");

			auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
			auto* geometryCreator = assetManager->getGeometryCreator();

			auto cQueue = getComputeQueue();

			// create geometry objects
			if (!createGeometries(gQueue, geometryCreator))
				return logFail("Could not create geometries from geometry creator");

			// create blas/tlas
			if (!createAccelerationStructures(cQueue))
				return logFail("Could not create acceleration structures");

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
				if (!shader)
					return logFail("Failed to create shader!");

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
						.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1,
					}
				};
				auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

				const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayoutPtrs = { descriptorSetLayout.get() };
				renderPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs.begin(), dsLayoutPtrs.end()));
				if (!renderPool)
					return logFail("Could not create descriptor pool");
				renderDs = renderPool->createDescriptorSet(descriptorSetLayout);
				if (!renderDs)
					return logFail("Could not create descriptor set");

				SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0u, .size = sizeof(SPushConstants)};
				auto pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

				IGPUComputePipeline::SCreationParams params = {};
				params.layout = pipelineLayout.get();
				params.shader.shader = shader.get();
				if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &renderPipeline))
					return logFail("Failed to create compute pipeline");
			}

			// write descriptors
			IGPUDescriptorSet::SDescriptorInfo infos[2];
			infos[0].desc = gpuTlas;
			infos[1].desc = m_device->createImageView({
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
				.image = outHDRImage,
				.viewType = IGPUImageView::E_TYPE::ET_2D,
				.format = asset::EF_R16G16B16A16_SFLOAT
			});
			if (!infos[1].desc)
				return logFail("Failed to create image view");
			infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
			IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				{.dstSet = renderDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = renderDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]}
			};
			m_device->updateDescriptorSets(std::span(writes, 2), {});

			// camera
			{
				core::vectorSIMDf cameraPosition(-5.81655884, 2.58630896, -4.23974705);
				core::vectorSIMDf cameraTarget(-0.349590302, -0.213266611, 0.317821503);
				matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 1000);
				camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);
			}

			m_winMgr->show(m_window.get());
			oracle.reportBeginFrameRecord();

			return true;
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());

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

			auto* queue = getGraphicsQueue();

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
				imageBarriers[0].image = outHDRImage.get();
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
			SPushConstants pc;
			pc.geometryInfoBuffer = geometryInfoBuffer->getDeviceAddress();

			const core::vector3df camPos = camera.getPosition().getAsVector3df();
			pc.camPos = { camPos.X, camPos.Y, camPos.Z };
			memcpy(&pc.invMVP, invModelViewProjectionMatrix.pointer(), sizeof(pc.invMVP));

			pc.scaleNDC = { 2.f / WIN_W, -2.f / WIN_H };
			pc.offsetNDC = { -1.f, 1.f };

			cmdbuf->bindComputePipeline(renderPipeline.get());
			cmdbuf->pushConstants(renderPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(SPushConstants), &pc);
			cmdbuf->bindDescriptorSets(EPBP_COMPUTE, renderPipeline->getLayout(), 0, 1, &renderDs.get());
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
				imageBarriers[0].image = outHDRImage.get();
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

				auto srcImg = outHDRImage.get();
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
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
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
					caption += ", displaying [all objects]";
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

			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
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

		bool createGeometries(video::CThreadSafeQueueAdapter* queue, const IGeometryCreator* gc)
		{
			auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!pool)
				return logFail("Couldn't create Command Pool for geometry creation!");

			std::array<ReferenceObjectCpu, OT_COUNT> objectsCpu;
			objectsCpu[OT_CUBE] = ReferenceObjectCpu{ .meta = {.type = OT_CUBE, .name = "Cube Mesh" }, .shadersType = GP_BASIC, .data = gc->createCubeMesh(nbl::core::vector3df(1.f, 1.f, 1.f)) };
			objectsCpu[OT_SPHERE] = ReferenceObjectCpu{ .meta = {.type = OT_SPHERE, .name = "Sphere Mesh" }, .shadersType = GP_BASIC, .data = gc->createSphereMesh(2, 16, 16) };
			objectsCpu[OT_CYLINDER] = ReferenceObjectCpu{ .meta = {.type = OT_CYLINDER, .name = "Cylinder Mesh" }, .shadersType = GP_BASIC, .data = gc->createCylinderMesh(2, 2, 20) };
			objectsCpu[OT_RECTANGLE] = ReferenceObjectCpu{ .meta = {.type = OT_RECTANGLE, .name = "Rectangle Mesh" }, .shadersType = GP_BASIC, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) };
			objectsCpu[OT_DISK] = ReferenceObjectCpu{ .meta = {.type = OT_DISK, .name = "Disk Mesh" }, .shadersType = GP_BASIC, .data = gc->createDiskMesh(2, 30) };
			objectsCpu[OT_ARROW] = ReferenceObjectCpu{ .meta = {.type = OT_ARROW, .name = "Arrow Mesh" }, .shadersType = GP_BASIC, .data = gc->createArrowMesh() };
			objectsCpu[OT_CONE] = ReferenceObjectCpu{ .meta = {.type = OT_CONE, .name = "Cone Mesh" }, .shadersType = GP_CONE, .data = gc->createConeMesh(2, 3, 10) };
			objectsCpu[OT_ICOSPHERE] = ReferenceObjectCpu{ .meta = {.type = OT_ICOSPHERE, .name = "Icosphere Mesh" }, .shadersType = GP_ICO, .data = gc->createIcoSphere(1, 3, true) };

			struct ScratchVIBindings
			{
				nbl::asset::SBufferBinding<ICPUBuffer> vertex, index;
			};
			std::array<ScratchVIBindings, OT_COUNT> scratchBuffers;
			//std::array<SGeomInfo, OT_COUNT> geomInfos;
			auto geomInfoBuffer = ICPUBuffer::create({ OT_COUNT * sizeof(SGeomInfo) });
			
			SGeomInfo* geomInfos = reinterpret_cast<SGeomInfo*>(geomInfoBuffer->getPointer());
			const uint32_t byteOffsets[OT_COUNT] = { 18, 24, 24, 20, 20, 24, 16, 12 };	// based on normals data position
			const uint32_t smoothNormals[OT_COUNT] = { 0, 1, 1, 0, 0, 1, 1, 1 };

			for (uint32_t i = 0; i < objectsCpu.size(); i++)
			{
				const auto& geom = objectsCpu[i];
				auto& obj = objectsGpu[i];
				auto& scratchObj = scratchBuffers[i];

				obj.meta.name = geom.meta.name;
				obj.meta.type = geom.meta.type;

				obj.indexCount = geom.data.indexCount;
				obj.indexType = geom.data.indexType;
				obj.vertexStride = geom.data.inputParams.bindings[0].stride;

				geomInfos[i].indexType = obj.indexType;
				geomInfos[i].vertexStride = obj.vertexStride;
				geomInfos[i].smoothNormals = smoothNormals[i];

				auto vBuffer = smart_refctd_ptr(geom.data.bindings[0].buffer); // no offset
				auto vUsage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | 
					IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				obj.bindings.vertex.offset = 0u;

				auto iBuffer = smart_refctd_ptr(geom.data.indexBuffer.buffer); // no offset
				auto iUsage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF |
					IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				obj.bindings.index.offset = 0u;

				vBuffer->addUsageFlags(vUsage);
				vBuffer->setContentHash(vBuffer->computeContentHash());
				scratchObj.vertex = { .offset = 0, .buffer = vBuffer };

				if (geom.data.indexType != EIT_UNKNOWN)
					if (iBuffer)
					{
						iBuffer->addUsageFlags(iUsage);
						iBuffer->setContentHash(iBuffer->computeContentHash());
					}
				scratchObj.index = { .offset = 0, .buffer = iBuffer };
			}

			auto cmdbuf = getSingleUseCommandBufferAndBegin(pool);
			cmdbuf->beginDebugMarker("Build geometry vertex and index buffers");

			smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });
			CAssetConverter::SInputs inputs = {};
			inputs.logger = m_logger.get();

			std::array<ICPUBuffer*, OT_COUNT * 2u> tmpBuffers;
			{
				for (uint32_t i = 0; i < objectsCpu.size(); i++)
				{
					tmpBuffers[2 * i + 0] = scratchBuffers[i].vertex.buffer.get();
					tmpBuffers[2 * i + 1] = scratchBuffers[i].index.buffer.get();
				}

				std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = tmpBuffers;
			}

			auto reservation = converter->reserve(inputs);
			{
				auto prepass = [&]<typename asset_type_t>(const auto & references) -> bool
				{
					auto objects = reservation.getGPUObjects<asset_type_t>();
					uint32_t counter = {};
					for (auto& object : objects)
					{
						auto gpu = object.value;
						auto* reference = references[counter];

						if (reference)
						{
							if (!gpu)
							{
								m_logger->log("Failed to convert a CPU object to GPU!", ILogger::ELL_ERROR);
								return false;
							}
						}
						counter++;
					}
					return true;
				};

				prepass.template operator() < ICPUBuffer > (tmpBuffers);
			}

			// not sure if need this (probably not, originally for transition img view)
			auto semaphore = m_device->createSemaphore(0u);

			std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1> cmdbufs = {};
			cmdbufs.front().cmdbuf = cmdbuf.get();

			SIntendedSubmitInfo transfer = {};
			transfer.queue = queue;
			transfer.scratchCommandBuffers = cmdbufs;
			transfer.scratchSemaphore = {
				.semaphore = semaphore.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			// convert
			{
				CAssetConverter::SConvertParams params = {};
				params.utilities = m_utils.get();
				params.transfer = &transfer;

				auto future = reservation.convert(params);
				if (future.copy() != IQueue::RESULT::SUCCESS)
				{
					m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
					return false;
				}

				// assign gpu objects to output
				auto&& buffers = reservation.getGPUObjects<ICPUBuffer>();
				for (uint32_t i = 0; i < objectsCpu.size(); i++)
				{
					auto& obj = objectsGpu[i];
					obj.bindings.vertex = { .offset = 0, .buffer = buffers[2 * i + 0].value };
					obj.bindings.index = { .offset = 0, .buffer = buffers[2 * i + 1].value };

					geomInfos[i].vertexBufferAddress = obj.bindings.vertex.buffer->getDeviceAddress() + byteOffsets[i];
					geomInfos[i].indexBufferAddress = obj.useIndex() ? obj.bindings.index.buffer->getDeviceAddress() : geomInfos[i].vertexBufferAddress;
				}
			}

			{
				IGPUBuffer::SCreationParams params;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				params.size = OT_COUNT * sizeof(SGeomInfo);
				m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{.queue = queue}, std::move(params), geomInfos).move_into(geometryInfoBuffer);
			}

			return true;
		}

		bool createAccelerationStructures(video::CThreadSafeQueueAdapter* queue)
		{
			IQueryPool::SCreationParams qParams{ .queryCount = OT_COUNT, .queryType = IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE };
			smart_refctd_ptr<IQueryPool> queryPool = m_device->createQueryPool(std::move(qParams));

			auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			if (!pool)
				return logFail("Couldn't create Command Pool for blas/tlas creation!");

			size_t totalScratchSize = 0;

			// build bottom level ASes
			{
				IGPUBottomLevelAccelerationStructure::DeviceBuildInfo blasBuildInfos[OT_COUNT];
				uint32_t primitiveCounts[OT_COUNT];
				IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer> triangles[OT_COUNT];
				uint32_t scratchSizes[OT_COUNT];

				for (uint32_t i = 0; i < objectsGpu.size(); i++)
				{
					const auto& obj = objectsGpu[i];

					const uint32_t vertexStride = obj.vertexStride;
					const uint32_t numVertices = obj.bindings.vertex.buffer->getSize() / vertexStride;
					if (obj.useIndex())
						primitiveCounts[i] = obj.indexCount / 3;
					else
						primitiveCounts[i] = numVertices / 3;

					triangles[i].vertexData[0] = obj.bindings.vertex;
					triangles[i].indexData = obj.useIndex() ? obj.bindings.index : obj.bindings.vertex;
					triangles[i].maxVertex = numVertices - 1;
					triangles[i].vertexStride = vertexStride;
					triangles[i].vertexFormat = EF_R32G32B32_SFLOAT;
					triangles[i].indexType = obj.indexType;
					triangles[i].geometryFlags = IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::OPAQUE_BIT;

					auto blasFlags = bitflag(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT) | IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_COMPACTION_BIT;
					if (m_physicalDevice->getProperties().limits.rayTracingPositionFetch)
						blasFlags |= IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_DATA_ACCESS_KHR;

					blasBuildInfos[i].buildFlags = blasFlags;
					blasBuildInfos[i].geometryCount = 1;	// only 1 geometry object per blas
					blasBuildInfos[i].srcAS = nullptr;
					blasBuildInfos[i].dstAS = nullptr;
					blasBuildInfos[i].triangles = &triangles[i];
					blasBuildInfos[i].scratch = {};

					ILogicalDevice::AccelerationStructureBuildSizes buildSizes;
					{
						const uint32_t maxPrimCount[1] = { primitiveCounts[i] };
						buildSizes = m_device->getAccelerationStructureBuildSizes(blasFlags, false, std::span{&triangles[i], 1}, maxPrimCount);
						if (!buildSizes)
							return logFail("Failed to get BLAS build sizes");
					}

					scratchSizes[i] = buildSizes.buildScratchSize;
					totalScratchSize += buildSizes.buildScratchSize;

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
						gpuBlas[i] = m_device->createBottomLevelAccelerationStructure(std::move(blasParams));
						if (!gpuBlas[i])
							return logFail("Could not create BLAS");
					}
				}

				auto cmdbufBlas = getSingleUseCommandBufferAndBegin(pool);
				cmdbufBlas->beginDebugMarker("Build BLAS");

				cmdbufBlas->resetQueryPool(queryPool.get(), 0, objectsGpu.size());

				smart_refctd_ptr<IGPUBuffer> scratchBuffer;
				{
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
					params.size = totalScratchSize;
					scratchBuffer = createBuffer(params);
				}

				uint32_t queryCount = 0;
				IGPUBottomLevelAccelerationStructure::BuildRangeInfo buildRangeInfos[OT_COUNT];
				IGPUBottomLevelAccelerationStructure::BuildRangeInfo* pRangeInfos[OT_COUNT];
				for (uint32_t i = 0; i < objectsGpu.size(); i++)
				{
					blasBuildInfos[i].dstAS = gpuBlas[i].get();
					blasBuildInfos[i].scratch.buffer = scratchBuffer;
					blasBuildInfos[i].scratch.offset = (i == 0) ? 0u : blasBuildInfos[i - 1].scratch.offset + scratchSizes[i - 1];

					buildRangeInfos[i].primitiveCount = primitiveCounts[i];
					buildRangeInfos[i].primitiveByteOffset = 0u;
					buildRangeInfos[i].firstVertex = 0u;
					buildRangeInfos[i].transformByteOffset = 0u;

					pRangeInfos[i] = &buildRangeInfos[i];
				}

				if (!cmdbufBlas->buildAccelerationStructures({ blasBuildInfos, OT_COUNT }, pRangeInfos))
					return logFail("Failed to build BLAS");

				{
					SMemoryBarrier memBarrier;
					memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
					memBarrier.srcAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
					memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
					memBarrier.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT;
					cmdbufBlas->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
				}

				const IGPUAccelerationStructure* ases[OT_COUNT];
				for (uint32_t i = 0; i < objectsGpu.size(); i++)
					ases[i] = gpuBlas[i].get();
				if (!cmdbufBlas->writeAccelerationStructureProperties({ ases, OT_COUNT }, IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE,
					queryPool.get(), queryCount++))
					return logFail("Failed to write acceleration structure properties!");

				cmdbufBlas->endDebugMarker();
				cmdbufSubmitAndWait(cmdbufBlas, getComputeQueue(), 39);
			}

			auto cmdbufCompact = getSingleUseCommandBufferAndBegin(pool);
			cmdbufCompact->beginDebugMarker("Compact BLAS");

			// compact blas
			{
				std::array<size_t, OT_COUNT> asSizes{ 0 };
				if (!m_device->getQueryPoolResults(queryPool.get(), 0, objectsGpu.size(), asSizes.data(), sizeof(size_t), IQueryPool::WAIT_BIT))
					return logFail("Could not get query pool results for AS sizes");

				std::array<smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>, OT_COUNT> cleanupBlas;
				for (uint32_t i = 0; i < objectsGpu.size(); i++)
				{
					cleanupBlas[i] = gpuBlas[i];
					{
						IGPUBuffer::SCreationParams params;
						params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
						params.size = asSizes[i];
						smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

						IGPUBottomLevelAccelerationStructure::SCreationParams blasParams;
						blasParams.bufferRange.buffer = asBuffer;
						blasParams.bufferRange.offset = 0u;
						blasParams.bufferRange.size = asSizes[i];
						blasParams.flags = IGPUBottomLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
						gpuBlas[i] = m_device->createBottomLevelAccelerationStructure(std::move(blasParams));
						if (!gpuBlas[i])
							return logFail("Could not create compacted BLAS");
					}

					IGPUBottomLevelAccelerationStructure::CopyInfo copyInfo;
					copyInfo.src = cleanupBlas[i].get();
					copyInfo.dst = gpuBlas[i].get();
					copyInfo.mode = IGPUBottomLevelAccelerationStructure::COPY_MODE::COMPACT;
					if (!cmdbufCompact->copyAccelerationStructure(copyInfo))
						return logFail("Failed to copy AS to compact");
				}
			}

			cmdbufCompact->endDebugMarker();
			cmdbufSubmitAndWait(cmdbufCompact, getComputeQueue(), 40);

			auto cmdbufTlas = getSingleUseCommandBufferAndBegin(pool);
			cmdbufTlas->beginDebugMarker("Build TLAS");

			// build top level AS
			{
				const uint32_t instancesCount = objectsGpu.size();
				IGPUTopLevelAccelerationStructure::DeviceStaticInstance instances[OT_COUNT];
				for (uint32_t i = 0; i < instancesCount; i++)
				{
					core::matrix3x4SIMD transform;
					transform.setTranslation(nbl::core::vectorSIMDf(5.f * i, 0, 0, 0));
					instances[i].base.blas.deviceAddress = gpuBlas[i]->getReferenceForDeviceOperations().deviceAddress;
					instances[i].base.mask = 0xFF;
					instances[i].base.instanceCustomIndex = i;
					instances[i].base.instanceShaderBindingTableRecordOffset = 0;
					instances[i].base.flags = static_cast<uint32_t>(IGPUTopLevelAccelerationStructure::INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT);
					instances[i].transform = transform;
				}

				{
					size_t bufSize = instancesCount * sizeof(IGPUTopLevelAccelerationStructure::DeviceStaticInstance);
					IGPUBuffer::SCreationParams params;
					params.usage = bitflag(IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT |
						IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
					params.size = bufSize;
					instancesBuffer = createBuffer(params);

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

				auto buildSizes = m_device->getAccelerationStructureBuildSizes(tlasFlags, false, instancesCount);
				if (!buildSizes)
					return logFail("Failed to get TLAS build sizes");

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
					if (!gpuTlas)
						return logFail("Could not create TLAS");
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

				if (!cmdbufTlas->buildAccelerationStructures({ &tlasBuildInfo, 1 }, pRangeInfos))
					return logFail("Failed to build TLAS");
			}

			cmdbufTlas->endDebugMarker();
			cmdbufSubmitAndWait(cmdbufTlas, getComputeQueue(), 45);

			return true;
		}


		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		video::CDumbPresentationOracle oracle;

		std::array<ReferenceObjectGpu, OT_COUNT> objectsGpu;

		std::array<smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>, OT_COUNT> gpuBlas;
		smart_refctd_ptr<IGPUTopLevelAccelerationStructure> gpuTlas;
		smart_refctd_ptr<IGPUBuffer> instancesBuffer;

		smart_refctd_ptr<IGPUBuffer> geometryInfoBuffer;
		smart_refctd_ptr<IGPUImage> outHDRImage;

		smart_refctd_ptr<IGPUComputePipeline> renderPipeline;
		smart_refctd_ptr<IGPUDescriptorSet> renderDs;
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