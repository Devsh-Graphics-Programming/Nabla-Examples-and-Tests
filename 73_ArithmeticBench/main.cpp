#include "SimpleWindowedApplication.hpp"
#include "CEventCallback.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "app_resources/common.hlsl"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// method emulations on the CPU, to verify the results of the GPU methods
template<class Binop>
struct emulatedReduction
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		const type_t red = std::reduce(in,in+itemCount,Binop::identity,Binop());
		std::fill(out,out+itemCount,red);
	}

	static inline constexpr const char* name = "reduction";
};
template<class Binop>
struct emulatedScanInclusive
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		std::inclusive_scan(in,in+itemCount,out,Binop());
	}
	static inline constexpr const char* name = "inclusive_scan";
};
template<class Binop>
struct emulatedScanExclusive
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		std::exclusive_scan(in,in+itemCount,out,Binop::identity,Binop());
	}
	static inline constexpr const char* name = "exclusive_scan";
};

class ArithmeticBenchApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	constexpr static inline uint32_t WIN_W = 1280;
	constexpr static inline uint32_t WIN_H = 720;
	constexpr static inline uint32_t MaxFramesInFlight = 5;

public:
	ArithmeticBenchApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

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
				params.windowCaption = "ArithmeticBenchApp";
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

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
		if (!swapchainParams.deduceFormat(m_physicalDevice))
			return logFail("Could not choose a Surface Format for the Swapchain!");

		auto graphicsQueue = getGraphicsQueue();
		if (!m_surface || !m_surface->init(graphicsQueue, std::make_unique<ISimpleManagedSurface::ISwapchainResources>(), swapchainParams.sharedParams))
			return logFail("Could not create Window & Surface or initialize the Surface!");

		auto pool = m_device->createCommandPool(graphicsQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!pool)
				return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
		m_surface->recreateSwapchain();

		transferDownQueue = getTransferDownQueue();
		computeQueue = getComputeQueue();

		// TODO: get the element count from argv
		const uint32_t elementCount = Output<>::ScanElementCount;
		// populate our random data buffer on the CPU and create a GPU copy
		inputData = new uint32_t[elementCount];
		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			std::mt19937 randGenerator(0xdeadbeefu);
			for (uint32_t i = 0u; i < elementCount; i++)
				inputData[i] = randGenerator(); // TODO: change to using xoroshiro, then we can skip having the input buffer at all

			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(Output<>::data[0]) * elementCount;
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
			m_utils->createFilledDeviceLocalBufferOnDedMem(
				SIntendedSubmitInfo{.queue=getTransferUpQueue()},
				std::move(inputDataBufferCreationParams),
				inputData
			).move_into(gpuinputDataBuffer);
		}

		// create 8 buffers for 8 operations
		for (auto i=0u; i<OutputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t) + gpuinputDataBuffer->getSize();
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT;

			outputBuffers[i] = m_device->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, outputBuffers[i].get());
			assert(bufferMem.isValid());
		}

		// create dummy image
		dummyImg = m_device->createImage({
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
		if (!dummyImg || !m_device->allocate(dummyImg->getMemoryReqs(), dummyImg.get()).isValid())
			return logFail("Could not create HDR Image");

		// create Descriptor Sets and Pipeline Layouts
		smart_refctd_ptr<IGPUPipelineLayout> benchPplnLayout;
		{
			// create Descriptor Set Layout
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				IGPUDescriptorSetLayout::SBinding binding[2];
				for (uint32_t i = 0u; i < 2; i++)
					binding[i] = {{},i,IDescriptor::E_TYPE::ET_STORAGE_BUFFER,IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,IShader::E_SHADER_STAGE::ESS_COMPUTE,1u,nullptr };
				binding[1].count = OutputBufferCount;
				dsLayout = m_device->createDescriptorSetLayout(binding);
			}

			// set and transient pool
			auto descPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,{&dsLayout.get(),1});
			testDs = descPool->createDescriptorSet(smart_refctd_ptr(dsLayout));
			{
				IGPUDescriptorSet::SDescriptorInfo infos[1+OutputBufferCount];
				infos[0].desc = gpuinputDataBuffer;
				infos[0].info.buffer = { 0u,gpuinputDataBuffer->getSize() };
				for (uint32_t i = 1u; i <= OutputBufferCount; i++)
				{
					auto buff = outputBuffers[i - 1];
					infos[i].info.buffer = { 0u,buff->getSize() };
					infos[i].desc = std::move(buff); // save an atomic in the refcount
				}

				IGPUDescriptorSet::SWriteDescriptorSet writes[2];
				for (uint32_t i=0u; i<2; i++)
					writes[i] = {testDs.get(),i,0u,1u,infos+i};
				writes[1].count = OutputBufferCount;

				m_device->updateDescriptorSets(2, writes, 0u, nullptr);
			}
			testPplnLayout = m_device->createPipelineLayout({}, std::move(dsLayout));


			smart_refctd_ptr<IGPUDescriptorSetLayout> benchLayout;
			{
				IGPUDescriptorSetLayout::SBinding binding[3];
				for (uint32_t i = 0u; i < 2; i++)
					binding[i] = { {},i,IDescriptor::E_TYPE::ET_STORAGE_BUFFER,IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,IShader::E_SHADER_STAGE::ESS_COMPUTE,1u,nullptr };
				binding[1].count = OutputBufferCount;
				binding[2] = { {},2,IDescriptor::E_TYPE::ET_STORAGE_IMAGE,IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,IShader::E_SHADER_STAGE::ESS_COMPUTE,1u,nullptr };
				benchLayout = m_device->createDescriptorSetLayout(binding);
			}

			benchPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, { &benchLayout.get(),1 });
			benchDs = benchPool->createDescriptorSet(smart_refctd_ptr(benchLayout));
			{
				IGPUDescriptorSet::SDescriptorInfo infos[1 + OutputBufferCount];
				infos[0].desc = gpuinputDataBuffer;
				infos[0].info.buffer = { 0u,gpuinputDataBuffer->getSize() };
				for (uint32_t i = 1u; i <= OutputBufferCount; i++)
				{
					auto buff = outputBuffers[i - 1];
					infos[i].info.buffer = { 0u,buff->getSize() };
					infos[i].desc = std::move(buff); // save an atomic in the refcount
				}
				// write swapchain image descriptor in loop

				IGPUDescriptorSet::SWriteDescriptorSet writes[2];
				for (uint32_t i = 0u; i < 2; i++)
					writes[i] = { benchDs.get(),i,0u,1u,infos + i };
				writes[1].count = OutputBufferCount;

				m_device->updateDescriptorSets(2, writes, 0u, nullptr);
			}
			benchPplnLayout = m_device->createPipelineLayout({}, std::move(benchLayout));
		}

		const auto spirv_isa_cache_path = localOutputCWD/"spirv_isa_cache.bin";
		// enclose to make sure file goes out of scope and we can reopen it
		{
			smart_refctd_ptr<const IFile> spirv_isa_cache_input;
			// try to load SPIR-V to ISA cache
			{
				ISystem::future_t<smart_refctd_ptr<IFile>> fileCreate;
				m_system->createFile(fileCreate,spirv_isa_cache_path,IFile::ECF_READ|IFile::ECF_MAPPABLE|IFile::ECF_COHERENT);
				if (auto lock=fileCreate.acquire())
					spirv_isa_cache_input = *lock;
			}
			// create the cache
			{
				std::span<const uint8_t> spirv_isa_cache_data = {};
				if (spirv_isa_cache_input)
					spirv_isa_cache_data = {reinterpret_cast<const uint8_t*>(spirv_isa_cache_input->getMappedPointer()),spirv_isa_cache_input->getSize()};
				else
					m_logger->log("Failed to load SPIR-V 2 ISA cache!",ILogger::ELL_PERFORMANCE);
				// Normally we'd deserialize a `ICPUPipelineCache` properly and pass that instead
				m_spirv_isa_cache = m_device->createPipelineCache(spirv_isa_cache_data);
			}
		}
		{
			// TODO: rename `deleteDirectory` to just `delete`? and a `IFile::setSize()` ?
			m_system->deleteDirectory(spirv_isa_cache_path);
			ISystem::future_t<smart_refctd_ptr<IFile>> fileCreate;
			m_system->createFile(fileCreate,spirv_isa_cache_path,IFile::ECF_WRITE);
			// I can be relatively sure I'll succeed to acquire the future, the pointer to created file might be null though.
			m_spirv_isa_cache_output=*fileCreate.acquire();
			if (!m_spirv_isa_cache_output)
				logFail("Failed to Create SPIR-V to ISA cache file.");
		}

		// load shader source from file
		auto getShaderSource = [&](const char* filePath) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams = {};
			lparams.logger = m_logger.get();
			lparams.workingDirectory = "";
			auto bundle = m_assetMgr->getAsset(filePath, lparams);
			if (bundle.getContents().empty() || bundle.getAssetType()!=IAsset::ET_SHADER)
			{
				m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
				exit(-1);
			}
			auto firstAssetInBundle = bundle.getContents()[0];
			return smart_refctd_ptr_static_cast<ICPUShader>(firstAssetInBundle);
		};

		auto subgroupTestSource = getShaderSource("app_resources/testSubgroup.comp.hlsl");
		auto subgroupBenchSource = getShaderSource("app_resources/benchmarkSubgroup.comp.hlsl");
		//auto workgroupTestSource = getShaderSource("app_resources/testWorkgroup.comp.hlsl");
		// now create or retrieve final resources to run our tests
		sema = m_device->createSemaphore(timelineValue);
		resultsBuffer = ICPUBuffer::create({ outputBuffers[0]->getSize() });
		smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1}))
			{
				logFail("Failed to create Command Buffers!\n");
				return false;
			}
		}
		
		// TODO variable items per invocation?
		const uint32_t NumLoops = 1000u;
		const std::array<uint32_t, 3> workgroupSizes = { 256, 512, 1024 };
		// const auto MaxWorkgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
		const auto MinSubgroupSize = m_physicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
		
		if (b_runTests)
		{
			runTests(cmdbuf.get(), subgroupTestSource, elementCount, ItemsPerInvocation, MinSubgroupSize, MaxSubgroupSize, workgroupSizes);

			m_logger->log("==========Result==========", ILogger::ELL_INFO);
			m_logger->log("Fail Count: %u", ILogger::ELL_INFO, totalFailCount);
		}

		// for each workgroup size (manually adjust items per invoc, operation else uses up a lot of ram)
		for (uint32_t i = 0; i < workgroupSizes.size(); i++)
			benchSets[i] = createBenchmarkPipelines<emulatedScanInclusive>(subgroupBenchSource, benchPplnLayout.get(), elementCount, hlsl::findMSB(MinSubgroupSize), workgroupSizes[i], ItemsPerInvocation, NumLoops);

		m_winMgr->show(m_window.get());

		return true;
	}

	virtual bool onAppTerminated() override
	{
		delete[] inputData;
		return true;
	}

	// the unit test is carried out on init
	void workLoopBody() override
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

		m_currentImageAcquire = m_surface->acquireNextImage();
		if (!m_currentImageAcquire)
			return;

		auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		// barrier transition to GENERAL
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
			imageBarriers[0].image = dummyImg.get();
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

		// bind dummy image
		IGPUImageView::SCreationParams viewParams = {
			.flags = IGPUImageView::ECF_NONE,
			.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
			.image = dummyImg,
			.viewType = IGPUImageView::ET_2D,
			.format = dummyImg->getCreationParameters().format
		};
		auto dummyImgView = m_device->createImageView(std::move(viewParams));

		video::IGPUDescriptorSet::SDescriptorInfo dsInfo;
		dsInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;
		dsInfo.desc = dummyImgView;

		IGPUDescriptorSet::SWriteDescriptorSet dsWrites[1u] =
		{
			{
				.dstSet = benchDs.get(),
				.binding = 2u,
				.arrayElement = 0u,
				.count = 1u,
				.info = &dsInfo,
			}
		};
		m_device->updateDescriptorSets(1u, dsWrites, 0u, nullptr);

		const uint32_t elementCount = Output<>::ScanElementCount;
		const auto MinSubgroupSize = m_physicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;

		const auto SubgroupSizeLog2 = hlsl::findMSB(MinSubgroupSize);

		bool passed = true;
		passed = runBenchmark<emulatedScanInclusive>(cmdbuf, benchSets[0], elementCount, SubgroupSizeLog2);
		passed = runBenchmark<emulatedScanInclusive>(cmdbuf, benchSets[1], elementCount, SubgroupSizeLog2);
		passed = runBenchmark<emulatedScanInclusive>(cmdbuf, benchSets[2], elementCount, SubgroupSizeLog2);


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
			imageBarriers[0].image = dummyImg.get();
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
			IGPUCommandBuffer::SImageBlit regions[] = { {
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
			} };

			auto srcImg = dummyImg.get();
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			auto dstImg = scRes->getImage(m_currentImageAcquire.imageIndex);

			cmdbuf->blitImage(srcImg, IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, dstImg, IImage::LAYOUT::TRANSFER_DST_OPTIMAL, regions, ISampler::ETF_NEAREST);
		}

		// barrier transition to PRESENT
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
			imageBarriers[0].barrier = {
				   .dep = {
					   .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
					   .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
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

		cmdbuf->end();

		// submit
		{
			auto* queue = getGraphicsQueue();
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

		numSubmits++;
	}

	//
	bool keepRunning() override { return numSubmits < MaxNumSubmits; }

private:
	void logTestOutcome(bool passed, uint32_t workgroupSize)
	{
		if (passed)
			m_logger->log("Passed test #%u", ILogger::ELL_INFO, workgroupSize);
		else
		{
			totalFailCount++;
			m_logger->log("Failed test #%u", ILogger::ELL_ERROR, workgroupSize);
		}
	}

	void runTests(IGPUCommandBuffer* cmdbuf, smart_refctd_ptr<ICPUShader> subgroupTestSource, uint32_t elementCount, uint32_t itemsPerInvocation, uint32_t MinSubgroupSize, uint32_t MaxSubgroupSize, const std::array<uint32_t, 3>& workgroupSizes)
	{
		for (auto subgroupSize = MinSubgroupSize; subgroupSize <= MaxSubgroupSize; subgroupSize *= 2u)
		{
			const uint8_t subgroupSizeLog2 = hlsl::findMSB(subgroupSize);
			for (const auto& workgroupSize : workgroupSizes)
			{
				// make sure renderdoc captures everything for debugging
				m_api->startCapture();
				m_logger->log("Testing Workgroup Size %u with Subgroup Size %u", ILogger::ELL_INFO, workgroupSize, subgroupSize);

				bool passed = true;
				// TODO async the testing
				passed = runTest<emulatedReduction, false>(cmdbuf, subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, ~0u, itemsPerInvocation) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedScanInclusive, false>(cmdbuf, subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, ~0u, itemsPerInvocation) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedScanExclusive, false>(cmdbuf, subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, ~0u, itemsPerInvocation) && passed;
				logTestOutcome(passed, workgroupSize);
				//for (uint32_t itemsPerWG = workgroupSize; itemsPerWG > workgroupSize - subgroupSize; itemsPerWG--)
				//{
				//	m_logger->log("Testing Item Count %u", ILogger::ELL_INFO, itemsPerWG);
				//	passed = runTest<emulatedReduction, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG) && passed;
				//	logTestOutcome(passed, itemsPerWG);
				//	passed = runTest<emulatedScanInclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG) && passed;
				//	logTestOutcome(passed, itemsPerWG);
				//	passed = runTest<emulatedScanExclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG) && passed;
				//	logTestOutcome(passed, itemsPerWG);
				//}
				m_api->endCapture();

				// save cache every now and then	
				{
					auto cpu = m_spirv_isa_cache->convertToCPUCache();
					// Normally we'd beautifully JSON serialize the thing, allow multiple devices & drivers + metadata
					auto bin = cpu->getEntries().begin()->second.bin;
					IFile::success_t success;
					m_spirv_isa_cache_output->write(success, bin->data(), 0ull, bin->size());
					if (!success)
						logFail("Could not write Create SPIR-V to ISA cache to disk!");
				}
			}
		}
	}

	// create pipeline (specialized every test) [TODO: turn into a future/async]
	smart_refctd_ptr<IGPUComputePipeline> createPipeline(const ICPUShader* overridenUnspecialized, const IGPUPipelineLayout* layout, const uint8_t subgroupSizeLog2)
	{
		auto shader = m_device->createShader(overridenUnspecialized);
		IGPUComputePipeline::SCreationParams params = {};
		params.layout = layout;
		params.shader = {
			.entryPoint = "main",
			.shader = shader.get(),
			.entries = nullptr,
			.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(subgroupSizeLog2),
			.requireFullSubgroups = true
		};
		core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		if (!m_device->createComputePipelines(m_spirv_isa_cache.get(),{&params,1},&pipeline))
			return nullptr;
		return pipeline;
	}

	struct BenchmarkSet
	{
		smart_refctd_ptr<IGPUComputePipeline> pipeline;
		uint32_t workgroupSize;
		uint32_t itemsPerInvocation;
	};

	template<template<class> class Arithmetic>
	BenchmarkSet createBenchmarkPipelines(const smart_refctd_ptr<const ICPUShader>&source, const IGPUPipelineLayout* layout, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, uint32_t itemsPerInvoc = 1u, uint32_t numLoops = 8u)
	{
		std::string arith_name = Arithmetic<bit_xor<uint32_t>>::name;	// TODO all operations

		//smart_refctd_ptr<ICPUShader> overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
		//	source.get(), "#define OPERATION %s\n#define WORKGROUP_SIZE %d\n#define ITEMS_PER_INVOCATION %d\n#define SUBGROUP_SIZE_LOG2 %d\n",
		//	(("subgroup2::") + arith_name).c_str(), workgroupSize, itemsPerInvoc, subgroupSizeLog2
		//);

		auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
		CHLSLCompiler::SOptions options = {};
		options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
		options.spirvOptimizer = nullptr;
//#ifndef _NBL_DEBUG
//		ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
//		auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
//		options.spirvOptimizer = opt.get();
//#endif
		options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
		options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
		options.preprocessorOptions.logger = m_logger.get();

		auto* includeFinder = compiler->getDefaultIncludeFinder();
		includeFinder->addSearchPath("nbl/builtin/hlsl/jit", core::make_smart_refctd_ptr<CJITIncludeLoader>(m_physicalDevice->getLimits(), m_device->getEnabledFeatures()));
		options.preprocessorOptions.includeFinder = includeFinder;

		const std::string definitions[5] = { 
			"subgroup2::" + arith_name,
			std::to_string(workgroupSize),
			std::to_string(itemsPerInvoc),
			std::to_string(subgroupSizeLog2),
			std::to_string(numLoops)
		};

		const IShaderCompiler::SMacroDefinition defines[5] = {
			{ "OPERATION", definitions[0] },
			{ "WORKGROUP_SIZE", definitions[1] },
			{ "ITEMS_PER_INVOCATION", definitions[2] },
			{ "SUBGROUP_SIZE_LOG2", definitions[3] },
			{ "NUM_LOOPS", definitions[4] },
		};
		options.preprocessorOptions.extraDefines = { defines, defines + 5 };

		smart_refctd_ptr<ICPUShader> overridenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);

		BenchmarkSet set;
		set.pipeline = createPipeline(overridenUnspecialized.get(), layout, subgroupSizeLog2);
		set.workgroupSize = workgroupSize;
		set.itemsPerInvocation = itemsPerInvoc;

		return set;
	};

	template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(IGPUCommandBuffer* cmdbuf, const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, uint32_t itemsPerWG = ~0u, uint32_t itemsPerInvoc = 1u)
	{
		std::string arith_name = Arithmetic<bit_xor<uint32_t>>::name;

		smart_refctd_ptr<ICPUShader> overridenUnspecialized;
		//if constexpr (WorkgroupTest)
		//{
		//	overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
		//		source.get(), "#define OPERATION %s\n#define WORKGROUP_SIZE %d\n#define ITEMS_PER_WG %d\n",
		//		(("workgroup::") + arith_name).c_str(), workgroupSize, itemsPerWG
		//	);
		//}
		//else
		//{
			itemsPerWG = workgroupSize;
			overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define OPERATION %s\n#define WORKGROUP_SIZE %d\n#define ITEMS_PER_INVOCATION %d\n#define SUBGROUP_SIZE_LOG2 %d\n",
				(("subgroup2::") + arith_name).c_str(), workgroupSize, itemsPerInvoc, subgroupSizeLog2
			);
		//}
		auto pipeline = createPipeline(overridenUnspecialized.get(),testPplnLayout.get(), subgroupSizeLog2);

		// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
		const uint32_t workgroupCount = elementCount / (itemsPerWG * itemsPerInvoc);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);
		cmdbuf->bindComputePipeline(pipeline.get());
		cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &testDs.get());
		cmdbuf->dispatch(workgroupCount, 1, 1);
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t memoryBarrier[OutputBufferCount];
			for (auto i=0u; i<OutputBufferCount; i++)
			{
				memoryBarrier[i] = {
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
							// in theory we don't need the HOST BITS cause we block on a semaphore but might as well add them
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT|PIPELINE_STAGE_FLAGS::HOST_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS|ACCESS_FLAGS::HOST_READ_BIT
						}
					},
					.range = {0ull,outputBuffers[i]->getSize(),outputBuffers[i]}
				};
			}
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {.memBarriers={},.bufBarriers=memoryBarrier};
			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE,info);
		}
		cmdbuf->end();

		const IQueue::SSubmitInfo::SSemaphoreInfo signal[1] = {{.semaphore=sema.get(),.value=++timelineValue}};
		const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{.cmdbuf=cmdbuf}};
		const IQueue::SSubmitInfo submits[1] = {{.commandBuffers=cmdbufs,.signalSemaphores=signal}};
		computeQueue->submit(submits);
		const ISemaphore::SWaitInfo wait[1] = {{.semaphore=sema.get(),.value=timelineValue}};
		m_device->blockForSemaphores(wait);

		// check results
		bool passed = validateResults<Arithmetic, bit_and<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc);
		passed = validateResults<Arithmetic, bit_xor<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, bit_or<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, plus<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, multiplies<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, minimum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, maximum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		//if constexpr (WorkgroupTest)
		//	passed = validateResults<Arithmetic, ballot<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;

		return passed;
	}

	//returns true if result matches
	template<template<class> class Arithmetic, class Binop, bool WorkgroupTest>
	bool validateResults(const uint32_t itemsPerWG, const uint32_t workgroupCount, uint32_t itemsPerInvoc = 1u)
	{
		bool success = true;

		// download data
		const SBufferRange<IGPUBuffer> bufferRange = {0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex]};
		m_utils->downloadBufferRangeViaStagingBufferAutoSubmit(SIntendedSubmitInfo{.queue=transferDownQueue},bufferRange,resultsBuffer->getPointer());

		using type_t = typename Binop::type_t;
		const auto dataFromBuffer = reinterpret_cast<const uint32_t*>(resultsBuffer->getPointer());
		const auto subgroupSize = dataFromBuffer[0];
		if (subgroupSize<nbl::hlsl::subgroup::MinSubgroupSize || subgroupSize>nbl::hlsl::subgroup::MaxSubgroupSize)
		{
			m_logger->log("Unexpected Subgroup Size %u", ILogger::ELL_ERROR, subgroupSize);
			return false;
		}

		const auto testData = reinterpret_cast<const type_t*>(dataFromBuffer + 1);
		// TODO: parallel for (the temporary values need to be threadlocal or what?)
		// now check if the data obtained has valid values
		type_t* tmp = new type_t[itemsPerWG * itemsPerInvoc];
		//type_t* ballotInput = new type_t[itemsPerWG];
		for (uint32_t workgroupID = 0u; success && workgroupID < workgroupCount; workgroupID++)
		{
			const auto workgroupOffset = workgroupID * itemsPerWG * itemsPerInvoc;

			//if constexpr (WorkgroupTest)
			//{
			//	if constexpr (std::is_same_v<ballot<type_t>, Binop>)
			//	{
			//		for (auto i = 0u; i < itemsPerWG; i++)
			//			ballotInput[i] = inputData[i + workgroupOffset] & 0x1u;
			//		Arithmetic<Binop>::impl(tmp, ballotInput, itemsPerWG);
			//	}
			//	else
			//		Arithmetic<Binop>::impl(tmp, inputData + workgroupOffset, itemsPerWG);
			//}
			//else
			//{
				for (uint32_t pseudoSubgroupID = 0u; pseudoSubgroupID < itemsPerWG; pseudoSubgroupID += subgroupSize)
					Arithmetic<Binop>::impl(tmp + pseudoSubgroupID * itemsPerInvoc, inputData + workgroupOffset + pseudoSubgroupID * itemsPerInvoc, subgroupSize * itemsPerInvoc);
			//}

			for (uint32_t localInvocationIndex = 0u; localInvocationIndex < itemsPerWG; localInvocationIndex++)
			{
				const auto localOffset = localInvocationIndex * itemsPerInvoc;
				const auto globalInvocationIndex = workgroupOffset + localOffset;

				for (uint32_t itemInvocationIndex = 0u; itemInvocationIndex < itemsPerInvoc; itemInvocationIndex++)
				{
					const auto cpuVal = tmp[localOffset + itemInvocationIndex];
					const auto gpuVal = testData[globalInvocationIndex + itemInvocationIndex];
					if (cpuVal != gpuVal)
					{
						m_logger->log(
							"Failed test #%d  (%s)  (%s) Expected %u got %u for workgroup %d and localinvoc %d and iteminvoc %d",
							ILogger::ELL_ERROR, itemsPerWG, WorkgroupTest ? "workgroup" : "subgroup", Binop::name,
							cpuVal, gpuVal, workgroupID, localInvocationIndex, itemInvocationIndex
						);
						success = false;
						break;
					}
				}
			}
		}
		//delete[] ballotInput;
		delete[] tmp;

		return success;
	}


	template<template<class> class Arithmetic>
	bool runBenchmark(IGPUCommandBuffer* cmdbuf, const BenchmarkSet& set, const uint32_t elementCount, const uint8_t subgroupSizeLog2)
	{
		const uint32_t workgroupCount = elementCount / (set.workgroupSize * set.itemsPerInvocation);

		cmdbuf->bindComputePipeline(set.pipeline.get());
		cmdbuf->bindDescriptorSets(EPBP_COMPUTE, set.pipeline->getLayout(), 0u, 1u, &benchDs.get());
		cmdbuf->dispatch(workgroupCount, 1, 1);
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t memoryBarrier[OutputBufferCount];
			for (auto i = 0u; i < OutputBufferCount; i++)
			{
				memoryBarrier[i] = {
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
							// in theory we don't need the HOST BITS cause we block on a semaphore but might as well add them
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT | PIPELINE_STAGE_FLAGS::HOST_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS | ACCESS_FLAGS::HOST_READ_BIT
						}
					},
					.range = {0ull,outputBuffers[i]->getSize(),outputBuffers[i]}
				};
			}
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = { .memBarriers = {},.bufBarriers = memoryBarrier };
			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, info);
		}

		return true;
	}

	IQueue* transferDownQueue;
	IQueue* computeQueue;
	smart_refctd_ptr<IGPUPipelineCache> m_spirv_isa_cache;
	smart_refctd_ptr<IFile> m_spirv_isa_cache_output;

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	smart_refctd_ptr<InputSystem> m_inputSystem;

	smart_refctd_ptr<IGPUImage> dummyImg;

	std::array<BenchmarkSet, 3> benchSets;
	smart_refctd_ptr<IGPUComputePipeline> benchPipeline;	// TODO array
	smart_refctd_ptr<IDescriptorPool> benchPool;
	smart_refctd_ptr<IGPUDescriptorSet> benchDs;

	smart_refctd_ptr<IGPUDescriptorSet> testDs;
	smart_refctd_ptr<IGPUPipelineLayout> testPplnLayout;

	constexpr static inline uint32_t MaxNumSubmits = 30;
	uint32_t numSubmits = 0;

	bool b_runTests = false;
	uint32_t* inputData = nullptr;
	uint32_t ItemsPerInvocation = 4u;
	constexpr static inline uint32_t OutputBufferCount = 8u;
	smart_refctd_ptr<IGPUBuffer> outputBuffers[OutputBufferCount];

	smart_refctd_ptr<ISemaphore> sema;
	uint64_t timelineValue = 0;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	uint32_t totalFailCount = 0;
};

NBL_MAIN_FUNC(ArithmeticBenchApp)