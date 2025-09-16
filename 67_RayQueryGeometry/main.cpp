// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "common.hpp"

class RayQueryGeometryApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
		using device_base_t = SimpleWindowedApplication;
		using asset_base_t = BuiltinResourcesApplication;
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

			auto cQueue = getComputeQueue();

			// create blas/tlas
			renderDs = 
//#define TRY_BUILD_FOR_NGFX // Validation errors on the fake Acquire-Presents, TODO fix
#ifdef TRY_BUILD_FOR_NGFX
			// Nsight is special and can't do debugger delay so you can debug your CPU stuff during a capture
			// Renderdoc-like Debugger Delay for NSight so that one may CPU debug applications launched from NSight
			if (m_api->runningInGraphicsDebugger()==IAPIConnection::EDebuggerType::NSight)
			{
				static volatile bool debugger_not_attached = true;
				while (debugger_not_attached)
					std::this_thread::yield();
			}
			// Nsight is special and can't capture anything not on the queue that performs the swapchain acquire/release
			createAccelerationStructureDS(gQueue);
#else
			createAccelerationStructureDS(cQueue);
#endif
			if (!renderDs)
				return logFail("Could not create acceleration structures and descriptor set");

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
				smart_refctd_ptr<IShader> shaderSrc = IAsset::castDown<IShader>(assets[0]);
				auto shader = m_device->compileShader({shaderSrc.get()});
				if (!shader)
					return logFail("Failed to create shader!");

				SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0u, .size = sizeof(SPushConstants)};
				auto pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, smart_refctd_ptr<const IGPUDescriptorSetLayout>(renderDs->getLayout()), nullptr, nullptr, nullptr);

				IGPUComputePipeline::SCreationParams params = {};
				params.layout = pipelineLayout.get();
				params.shader.shader = shader.get();
				params.shader.entryPoint = "main";
				if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &renderPipeline))
					return logFail("Failed to create compute pipeline");
			}

			// write descriptors
			{
				IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = m_device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
					.image = outHDRImage,
					.viewType = IGPUImageView::E_TYPE::ET_2D,
					.format = asset::EF_R16G16B16A16_SFLOAT
				});
				if (!info.desc)
					return logFail("Failed to create image view");
				info.info.image.imageLayout = IImage::LAYOUT::GENERAL;
				const IGPUDescriptorSet::SWriteDescriptorSet write = {.dstSet=renderDs.get(), .binding=1, .arrayElement=0, .count=1, .info=&info};
				m_device->updateDescriptorSets({&write,1}, {});
			}

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

			static bool first = true;
			if (first)
			{
				first = false;
			}

			auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->beginDebugMarker("RayQueryGeometryApp Frame");
			{
				camera.beginInputProcessing(nextPresentationTimestamp);
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, m_logger.get());
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
				camera.endInputProcessing(nextPresentationTimestamp);
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

		smart_refctd_ptr<IGPUDescriptorSet> createAccelerationStructureDS(video::CThreadSafeQueueAdapter* queue)
		{
			using namespace nbl::scene;

			// triangles geometries
			auto gc = make_smart_refctd_ptr<CGeometryCreator>();

			auto transform_i = 0;
			auto nextTransform = [&transform_i]()
			{
				core::matrix3x4SIMD transform;
				transform.setTranslation(nbl::core::vectorSIMDf(5.f * transform_i, 0, 0, 0));
				transform_i++;
				return transform;
			};

			std::vector<ReferenceObjectCpu> cpuObjects;

			const auto prism = gc->createPrism(2, 5, 9);
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = prism});
			const auto unweldedPrism = CPolygonGeometryManipulator::createUnweldedList(prism.get());
			const auto smoothedPrism = CPolygonGeometryManipulator::createSmoothVertexNormal(unweldedPrism.get(), true);
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = smoothedPrism});

			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createArrow() });
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = CPolygonGeometryManipulator::createTriangleListIndexing(gc->createDisk(1.0f, 12).get()) });
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createCube({1.f, 1.f, 1.f})});
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createSphere(2, 16, 16)});
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createCylinder(2, 2, 20)});
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createRectangle({1.5, 3})});
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createCone(2, 3, 10)});
			cpuObjects.push_back(ReferenceObjectCpu{ .transform = nextTransform(), .data = gc->createIcoSphere(1, 3, true)});

			const auto geometryCount = [&cpuObjects]
			{
				size_t count = 0;
				for (auto& cpuObject: cpuObjects)
				{
					const auto data = cpuObject.data;
					cpuObject.instanceID = count;
					if (std::holds_alternative<PolygonGeometryData>(data))
					{
						count += 1;
					} else if (std::holds_alternative<GeometryCollectionData>(data))
					{
						const auto colData = std::get<GeometryCollectionData>(data);
						count += colData->getGeometries()->size();
					}
				}
				return count;
			}();

			auto geomInfoBuffer = ICPUBuffer::create({ geometryCount * sizeof(SGeomInfo) });

			SGeomInfo* geomInfos = reinterpret_cast<SGeomInfo*>(geomInfoBuffer->getPointer());

			// get ICPUBuffers into ICPUBottomLevelAccelerationStructures
			std::vector<smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>> cpuBlas(cpuObjects.size());
			for (uint32_t blas_i = 0; blas_i < cpuBlas.size(); blas_i++)
			{
				auto& blas = cpuBlas[blas_i];
				blas = make_smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>();
				if (std::holds_alternative<PolygonGeometryData>(cpuObjects[blas_i].data))
				{
					const auto data = std::get<PolygonGeometryData>(cpuObjects[blas_i].data);

					auto triangles = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUBottomLevelAccelerationStructure::Triangles<ICPUBuffer>>>(1u);
					auto primitiveCounts = make_refctd_dynamic_array<smart_refctd_dynamic_array<uint32_t>>(1u);

					auto& tri = triangles->front();

					auto& primCount = primitiveCounts->front();
					primCount = data->getPrimitiveCount();

					tri = data->exportForBLAS();

					blas->setGeometries(std::move(triangles), std::move(primitiveCounts));

					
				} else if (std::holds_alternative<GeometryCollectionData>(cpuObjects[blas_i].data))
				{
					
					const auto data = std::get<GeometryCollectionData>(cpuObjects[blas_i].data);

					const auto& geometries = *data->getGeometries();
					const auto geometryCount = geometries.size();

					auto triangles = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUBottomLevelAccelerationStructure::Triangles<ICPUBuffer>>>(geometryCount);
					auto primitiveCounts = make_refctd_dynamic_array<smart_refctd_dynamic_array<uint32_t>>(geometryCount);

					for (auto geometry_i = 0u; geometry_i < geometryCount; geometry_i++)
					{
						const auto& geometry = geometries[geometry_i];
						const auto* polyGeo = static_cast<const ICPUPolygonGeometry*>(geometry.geometry.get());
						primitiveCounts->operator[](geometry_i) = polyGeo->getPrimitiveCount();
						auto& triangle = triangles->operator[](geometry_i);
						triangle = polyGeo->exportForBLAS();
						if (geometry.hasTransform())
							triangle.transform = geometry.transform;
					}

					blas->setGeometries(std::move(triangles), std::move(primitiveCounts));

				}
				auto blasFlags = bitflag(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT) | IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_COMPACTION_BIT;
				if (m_physicalDevice->getProperties().limits.rayTracingPositionFetch)
					blasFlags |= IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_DATA_ACCESS;

				blas->setBuildFlags(blasFlags);
				blas->setContentHash(blas->computeContentHash());
			}

			// get ICPUBottomLevelAccelerationStructure into ICPUTopLevelAccelerationStructure
			auto geomInstances = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUTopLevelAccelerationStructure::PolymorphicInstance>>(cpuObjects.size());
			{
				uint32_t i = 0;
				for (auto instance = geomInstances->begin(); instance != geomInstances->end(); instance++, i++)
				{
					ICPUTopLevelAccelerationStructure::StaticInstance inst;
					inst.base.blas = cpuBlas[i];
					inst.base.flags = static_cast<uint32_t>(IGPUTopLevelAccelerationStructure::INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT);
					inst.base.instanceCustomIndex = cpuObjects[i].instanceID;
					inst.base.instanceShaderBindingTableRecordOffset = 0;
					inst.base.mask = 0xFF;
					inst.transform = cpuObjects[i].transform;
					instance->instance = inst;
				}
			}

			auto cpuTlas = make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();
			cpuTlas->setInstances(std::move(geomInstances));
			cpuTlas->setBuildFlags(IGPUTopLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT);
			
			// descriptor set and layout
			ICPUDescriptorSetLayout::SBinding bindings[] = {
				{
					.binding = 0,
					.type = asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
					.createFlags = IDescriptorSetLayoutBase::SBindingBase::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1,
				},
				{
					.binding = 1,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = IDescriptorSetLayoutBase::SBindingBase::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1,
				}
			};
			auto descriptorSet = core::make_smart_refctd_ptr<ICPUDescriptorSet>(core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings));
			descriptorSet->getDescriptorInfos(IDescriptorSetLayoutBase::CBindingRedirect::binding_number_t{0},IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE).front().desc = cpuTlas;

//#define TEST_REBAR_FALLBACK
			// convert with asset converter
			smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });
			struct MyInputs : CAssetConverter::SInputs
			{
#ifndef TEST_REBAR_FALLBACK
				inline uint32_t constrainMemoryTypeBits(const size_t groupCopyID, const IAsset* canonicalAsset, const blake3_hash_t& contentHash, const IDeviceMemoryBacked* memoryBacked) const override
				{
					assert(memoryBacked);
					return memoryBacked->getObjectType()!=IDeviceMemoryBacked::EOT_BUFFER ? (~0u):rebarMemoryTypes;
				}
#endif
				uint32_t rebarMemoryTypes;
			} inputs = {};
			inputs.logger = m_logger.get();
			inputs.rebarMemoryTypes = m_physicalDevice->getDirectVRAMAccessMemoryTypeBits();
#ifndef TEST_REBAR_FALLBACK
			struct MyAllocator final : public IDeviceMemoryAllocator
			{
				ILogicalDevice* getDeviceForAllocations() const override {return device;}

				SAllocation allocate(const SAllocateInfo& info) override
				{
					auto retval = device->allocate(info);
					// map what is mappable by default so ReBAR checks succeed
					if (retval.isValid() && retval.memory->isMappable())
						retval.memory->map({.offset=0,.length=info.size});
					return retval;
				}

				ILogicalDevice* device;
			} myalloc;
			myalloc.device = m_device.get();
			inputs.allocator = &myalloc;
#endif
			
			CAssetConverter::patch_t<ICPUTopLevelAccelerationStructure> tlasPatch = {};
			tlasPatch.compactAfterBuild = true;
			std::vector<CAssetConverter::patch_t<ICPUBottomLevelAccelerationStructure>> tmpBLASPatches(cpuObjects.size());
			std::vector<ICPUPolygonGeometry*> tmpGeometries;
			tmpGeometries.reserve(geometryCount);
			std::vector<CAssetConverter::patch_t<asset::ICPUPolygonGeometry>> tmpGeometryPatches;
			tmpGeometryPatches.reserve(geometryCount);
			{
				tmpBLASPatches.front().compactAfterBuild = true;
				std::fill(tmpBLASPatches.begin(),tmpBLASPatches.end(),tmpBLASPatches.front());
				//
				for (uint32_t i = 0; i < cpuObjects.size(); i++)
				{
					const auto data = cpuObjects[i].data;
					if (std::holds_alternative<PolygonGeometryData>(data))
					{
						const auto polygonData = std::get<PolygonGeometryData>(data);
						tmpGeometries.push_back(polygonData.get());
						tmpGeometryPatches.push_back({});
						tmpGeometryPatches.back().indexBufferUsages = IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
					} else if (std::holds_alternative<GeometryCollectionData>(data))
					{
						const auto collectionData = std::get<GeometryCollectionData>(data);
						for (const auto& geometryRef : *collectionData->getGeometries())
						{
							auto* polyGeo = static_cast<ICPUPolygonGeometry*>(geometryRef.geometry.get());
							tmpGeometries.push_back(polyGeo);
							tmpGeometryPatches.push_back({});
							tmpGeometryPatches.back().indexBufferUsages = IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
						}
					}
				}
				assert(tmpGeometries.size() == geometryCount);
				assert(tmpGeometryPatches.size() == geometryCount);

				std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = {&descriptorSet.get(),1};
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUTopLevelAccelerationStructure>>(inputs.assets) = {&cpuTlas.get(),1};
				std::get<CAssetConverter::SInputs::patch_span_t<ICPUTopLevelAccelerationStructure>>(inputs.patches) = {&tlasPatch,1};
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUBottomLevelAccelerationStructure>>(inputs.assets) = {&cpuBlas.data()->get(),cpuBlas.size()};
				std::get<CAssetConverter::SInputs::patch_span_t<ICPUBottomLevelAccelerationStructure>>(inputs.patches) = tmpBLASPatches;
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = tmpGeometries;
				std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = tmpGeometryPatches;
			}

			auto reservation = converter->reserve(inputs);

			constexpr auto XferBufferCount = 2;
			std::array<smart_refctd_ptr<IGPUCommandBuffer>,XferBufferCount> xferBufs = {};
			std::array<IQueue::SSubmitInfo::SCommandBufferInfo,XferBufferCount> xferBufInfos = {};
			{
				auto pool = m_device->createCommandPool(getTransferUpQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,xferBufs);
				xferBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				for (auto i=0; i<XferBufferCount; i++)
					xferBufInfos[i].cmdbuf = xferBufs[i].get();
			}
			auto xferSema = m_device->createSemaphore(0u);
			xferSema->setObjectDebugName("Transfer Semaphore");
			SIntendedSubmitInfo transfer = {};
			transfer.queue = getTransferUpQueue();
			transfer.scratchCommandBuffers = xferBufInfos;
			transfer.scratchSemaphore = {
				.semaphore = xferSema.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			
			constexpr auto CompBufferCount = 2;
			std::array<smart_refctd_ptr<IGPUCommandBuffer>,CompBufferCount> compBufs = {};
			std::array<IQueue::SSubmitInfo::SCommandBufferInfo,CompBufferCount> compBufInfos = {};
			{
				auto pool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,compBufs);
				compBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				for (auto i=0; i<CompBufferCount; i++)
					compBufInfos[i].cmdbuf = compBufs[i].get();
			}
			auto compSema = m_device->createSemaphore(0u);
			compSema->setObjectDebugName("Compute Semaphore");
			SIntendedSubmitInfo compute = {};
			compute.queue = getComputeQueue();
			compute.scratchCommandBuffers = compBufInfos;
			compute.scratchSemaphore = {
				.semaphore = compSema.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT|PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT
			};
			// convert
#ifdef TRY_BUILD_FOR_NGFX // NSight is "debugger-challenged" it can't capture anything not happenning "during a frame", so we need to trick it
			m_currentImageAcquire = m_surface->acquireNextImage();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = { {
					.semaphore = m_currentImageAcquire.semaphore,
					.value = m_currentImageAcquire.acquireCount,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				} };
				m_surface->present(m_currentImageAcquire.imageIndex,acquired);
			}
			m_currentImageAcquire = m_surface->acquireNextImage();
#endif
			m_api->startCapture();
			auto gQueue = getGraphicsQueue();
			{
				smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t> scratchAlloc;
				{
					constexpr auto MaxAlignment = 256;
					constexpr auto MinAllocationSize = 1024;
					const auto scratchSize = core::alignUp(reservation.getMinASBuildScratchSize(false),MaxAlignment);
					

					IGPUBuffer::SCreationParams creationParams = {};
					creationParams.size = scratchSize;
					creationParams.usage = IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT|IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT|IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
#ifdef TEST_REBAR_FALLBACK
					creationParams.usage |= IGPUBuffer::EUF_TRANSFER_DST_BIT;
					core::unordered_set<uint32_t> sharingSet = {compute.queue->getFamilyIndex(),transfer.queue->getFamilyIndex()};
					core::vector<uint32_t> sharingIndices(sharingSet.begin(),sharingSet.end());
					if (sharingIndices.size()>1)
						creationParams.queueFamilyIndexCount = sharingIndices.size();
					creationParams.queueFamilyIndices = sharingIndices.data();
#endif
					auto scratchBuffer = m_device->createBuffer(std::move(creationParams));

					auto reqs = scratchBuffer->getMemoryReqs();
#ifndef TEST_REBAR_FALLBACK
					reqs.memoryTypeBits &= m_physicalDevice->getDirectVRAMAccessMemoryTypeBits();
#endif
					auto allocation = m_device->allocate(reqs,scratchBuffer.get(),IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
#ifndef TEST_REBAR_FALLBACK
					allocation.memory->map({.offset=0,.length=reqs.size});
#endif

					scratchAlloc = make_smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t>(
						SBufferRange<video::IGPUBuffer>{0ull,scratchSize,std::move(scratchBuffer)},
						core::allocator<uint8_t>(),MaxAlignment,MinAllocationSize
					);
				}

				struct MyParams final : CAssetConverter::SConvertParams
				{
					inline uint32_t getFinalOwnerQueueFamily(const IGPUBuffer* buffer, const core::blake3_hash_t& createdFrom) override
					{
						return finalUser;
					}
					inline uint32_t getFinalOwnerQueueFamily(const IGPUAccelerationStructure* image, const core::blake3_hash_t& createdFrom) override
					{
						return finalUser;
					}

					uint8_t finalUser;
				} params = {};
				params.utilities = m_utils.get();
				params.transfer = &transfer;
				params.compute = &compute;
				params.scratchForDeviceASBuild = scratchAlloc.get();
				params.finalUser = gQueue->getFamilyIndex();

				auto future = reservation.convert(params);
				if (future.copy() != IQueue::RESULT::SUCCESS)
				{
					m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
					return {};
				}

				auto&& tlases = reservation.getGPUObjects<ICPUTopLevelAccelerationStructure>();
				m_gpuTlas = tlases[0].value;

				auto&& gpuPolygonGeometries = reservation.getGPUObjects<ICPUPolygonGeometry>();
				m_gpuPolygons.resize(gpuPolygonGeometries.size());

				// assign gpu objects to output
				for (uint32_t i = 0; i < gpuPolygonGeometries.size(); i++)
				{
					const auto& gpuPolygon = gpuPolygonGeometries[i].value;
					const auto gpuTriangles = gpuPolygon->exportForBLAS();

					const auto& vertexBufferBinding = gpuTriangles.vertexData[0];
					const uint64_t vertexBufferAddress = vertexBufferBinding.buffer->getDeviceAddress() + vertexBufferBinding.offset;

					const auto& normalView = gpuPolygon->getNormalView();
					const uint64_t normalBufferAddress = normalView ? normalView.src.buffer->getDeviceAddress() + normalView.src.offset : 0;

					auto normalType = NT_R32G32B32_SFLOAT;
					if (normalView && normalView.composed.format == EF_R8G8B8A8_SNORM)
						normalType = NT_R8G8B8A8_SNORM;

					const auto& indexBufferBinding = gpuTriangles.indexData;
					auto& geomInfo = geomInfos[i];
					geomInfo = {
						.vertexBufferAddress = vertexBufferAddress,
						.indexBufferAddress = indexBufferBinding.buffer ? indexBufferBinding.buffer->getDeviceAddress() + indexBufferBinding.offset : 0,
						.normalBufferAddress = normalBufferAddress,
						.normalType = normalType,
						.indexType = gpuTriangles.indexType,
					};

					m_gpuPolygons[i] = gpuPolygon;
				}
			}

			//
			{
				IGPUBuffer::SCreationParams params;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				params.size = geometryCount * sizeof(SGeomInfo);
				m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = gQueue }, std::move(params), geomInfos).move_into(geometryInfoBuffer);
			}

			// acquire ownership
			{
				smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
				{
					const auto gQFI = gQueue->getFamilyIndex();
					m_device->createCommandPool(gQFI,IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT)->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1});
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					{
						core::vector<IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>> bufBarriers;
						auto acquireBufferRange = [&bufBarriers](const uint8_t otherQueueFamilyIndex, const SBufferRange<IGPUBuffer>& bufferRange)
						{
							bufBarriers.push_back({
								.barrier = {
									.dep = {
										.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
										.srcAccessMask = ACCESS_FLAGS::NONE,
										.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
										// we don't care what exactly, uncomplex our code
										.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
									},
									.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
									.otherQueueFamilyIndex = otherQueueFamilyIndex
								},
								.range = bufferRange
							});
						};
#ifdef TEST_REBAR_FALLBACK
						if (const auto otherQueueFamilyIndex=transfer.queue->getFamilyIndex(); gQFI!=otherQueueFamilyIndex)
						for (const auto& buffer : reservation.getGPUObjects<ICPUBuffer>())
						{
							const auto& buff = buffer.value;
							if (buff)
								acquireBufferRange(otherQueueFamilyIndex,{.offset=0,.size=buff->getSize(),.buffer=buff});
						}
#endif
						if (const auto otherQueueFamilyIndex=compute.queue->getFamilyIndex(); gQFI!=otherQueueFamilyIndex)
						{
							auto acquireAS = [&acquireBufferRange,otherQueueFamilyIndex](const IGPUAccelerationStructure* as)
							{
								acquireBufferRange(otherQueueFamilyIndex,as->getCreationParams().bufferRange);
							};
							for (const auto& blas : reservation.getGPUObjects<ICPUBottomLevelAccelerationStructure>())
								acquireAS(blas.value.get());
							acquireAS(reservation.getGPUObjects<ICPUTopLevelAccelerationStructure>().front().value.get());
						}
						if (!bufBarriers.empty())
							cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers=bufBarriers});
					}
					cmdbuf->end();
				}
				if (!cmdbuf->empty())
				{
					const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = {
						.cmdbuf = cmdbuf.get()
					};
					const IQueue::SSubmitInfo::SSemaphoreInfo signal = {
						.semaphore = compute.scratchSemaphore.semaphore,
						.value = compute.getFutureScratchSemaphore().value,
						.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
					};
					auto wait = signal;
					wait.value--;
					const IQueue::SSubmitInfo info = {
						.waitSemaphores = {&wait,1}, // we already waited with the host on the AS build
						.commandBuffers = {&cmdbufInfo,1},
						.signalSemaphores = {&signal,1}
					};
					if (const auto retval=gQueue->submit({&info,1}); retval!=IQueue::RESULT::SUCCESS)
						m_logger->log("Failed to transfer ownership with code %d!",system::ILogger::ELL_ERROR,retval);
				}
			}
#undef TEST_REBAR_FALLBACK
			
#ifdef TRY_BUILD_FOR_NGFX
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = { {
					.semaphore = m_currentImageAcquire.semaphore,
					.value = m_currentImageAcquire.acquireCount,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				} };
				m_surface->present(m_currentImageAcquire.imageIndex,acquired);
			}
#endif
			m_api->endCapture();

			return reservation.getGPUObjects<ICPUDescriptorSet>().front().value;
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

		smart_refctd_ptr<IGPUBuffer> geometryInfoBuffer;
		smart_refctd_ptr<IGPUImage> outHDRImage;
		core::vector<smart_refctd_ptr<IGPUPolygonGeometry>> m_gpuPolygons;
		smart_refctd_ptr<IGPUTopLevelAccelerationStructure> m_gpuTlas;

		smart_refctd_ptr<IGPUComputePipeline> renderPipeline;
		smart_refctd_ptr<IGPUDescriptorSet> renderDs;

};

NBL_MAIN_FUNC(RayQueryGeometryApp)