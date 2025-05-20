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

struct PushConstantData
{
	uint64_t inputBufAddress;
	uint64_t outputAddressBufAddress;
};

// NOTE added swapchain + drawing frames to be able to profile with Nsight, which still doesn't support profiling headless compute shaders
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
		{
			std::mt19937 randGenerator(0xdeadbeefu);
			for (uint32_t i = 0u; i < elementCount; i++)
				inputData[i] = randGenerator(); // TODO: change to using xoroshiro, then we can skip having the input buffer at all

			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(Output<>::data[0]) * elementCount;
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
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
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			outputBuffers[i] = m_device->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, outputBuffers[i].get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			assert(bufferMem.isValid());
		}

		// create buffer to store BDA of output buffers
		{
			std::array<uint64_t, OutputBufferCount> outputAddresses;
			for (uint32_t i = 0; i < OutputBufferCount; i++)
				outputAddresses[i] = outputBuffers[i]->getDeviceAddress();

			IGPUBuffer::SCreationParams params;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			params.size = OutputBufferCount * sizeof(uint64_t);
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = getTransferUpQueue() }, std::move(params), outputAddresses.data()).move_into(gpuOutputAddressesBuffer);
		}
		pc.inputBufAddress = gpuinputDataBuffer->getDeviceAddress();
		pc.outputAddressBufAddress = gpuOutputAddressesBuffer->getDeviceAddress();

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
			smart_refctd_ptr<IGPUDescriptorSetLayout> benchLayout;
			{
				IGPUDescriptorSetLayout::SBinding binding[1];
				binding[0] = { {},2,IDescriptor::E_TYPE::ET_STORAGE_IMAGE,IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,IShader::E_SHADER_STAGE::ESS_COMPUTE,1u,nullptr };
				benchLayout = m_device->createDescriptorSetLayout(binding);
			}

			benchPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, { &benchLayout.get(),1 });
			benchDs = benchPool->createDescriptorSet(smart_refctd_ptr(benchLayout));

			SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0,.size = sizeof(PushConstantData) };
			benchPplnLayout = m_device->createPipelineLayout({ &pcRange, 1 }, std::move(benchLayout));
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

		auto subgroupBenchSource = getShaderSource("app_resources/benchmarkSubgroup.comp.hlsl");
		auto workgroupBenchSource = getShaderSource("app_resources/benchmarkWorkgroup.comp.hlsl");
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

		// const auto MaxWorkgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
		const auto MinSubgroupSize = m_physicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;

		// for each workgroup size (manually adjust items per invoc, operation else uses up a lot of ram)
		if constexpr (DoWorkgroupBenchmarks)
		{
			for (uint32_t i = 0; i < workgroupSizes.size(); i++)
				benchSets[i] = createBenchmarkPipelines<ArithmeticOp, DoWorkgroupBenchmarks>(workgroupBenchSource, benchPplnLayout.get(), elementCount, hlsl::findMSB(MinSubgroupSize), workgroupSizes[i], ItemsPerInvocation, NumLoops);
		}
		else
		{
			for (uint32_t i = 0; i < workgroupSizes.size(); i++)
				benchSets[i] = createBenchmarkPipelines<ArithmeticOp, DoWorkgroupBenchmarks>(subgroupBenchSource, benchPplnLayout.get(), elementCount, hlsl::findMSB(MinSubgroupSize), workgroupSizes[i], ItemsPerInvocation, NumLoops);
		}

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

		cmdbuf->bindDescriptorSets(EPBP_COMPUTE, benchSets[0].pipeline->getLayout(), 0u, 1u, &benchDs.get());
		cmdbuf->pushConstants(benchSets[0].pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(PushConstantData), &pc);

		for (uint32_t i = 0; i < benchSets.size(); i++)
			runBenchmark<DoWorkgroupBenchmarks>(cmdbuf, benchSets[i], elementCount, SubgroupSizeLog2);


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
		if (!m_device->createComputePipelines(nullptr,{&params,1},&pipeline))
			return nullptr;
		return pipeline;
	}

	struct BenchmarkSet
	{
		smart_refctd_ptr<IGPUComputePipeline> pipeline;
		uint32_t workgroupSize;
		uint32_t itemsPerInvocation;
	};

	template<template<class> class Arithmetic, bool WorkgroupBench>
	BenchmarkSet createBenchmarkPipelines(const smart_refctd_ptr<const ICPUShader>&source, const IGPUPipelineLayout* layout, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, uint32_t itemsPerInvoc = 1u, uint32_t numLoops = 8u)
	{
		std::string arith_name = Arithmetic<plus<uint32_t>>::name;

		auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
		CHLSLCompiler::SOptions options = {};
		options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
		options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
		ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
		auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
		options.spirvOptimizer = opt.get();
#else
		options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
#endif
		options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
		options.preprocessorOptions.logger = m_logger.get();

		auto* includeFinder = compiler->getDefaultIncludeFinder();
		includeFinder->addSearchPath("nbl/builtin/hlsl/jit", core::make_smart_refctd_ptr<CJITIncludeLoader>(m_physicalDevice->getLimits(), m_device->getEnabledFeatures()));
		options.preprocessorOptions.includeFinder = includeFinder;

		const uint32_t subgroupSize = 0x1u << subgroupSizeLog2;
		const uint32_t itemsPerWG = workgroupSize <= subgroupSize ? workgroupSize * itemsPerInvoc : itemsPerInvoc * max(workgroupSize >> subgroupSizeLog2, subgroupSize) << subgroupSizeLog2;	// TODO use Config somehow
		smart_refctd_ptr<ICPUShader> overriddenUnspecialized;
		if constexpr (WorkgroupBench)
		{
			const uint32_t workgroupSizeLog2 = hlsl::findMSB(workgroupSize);
			const std::string definitions[6] = {
				"workgroup2::" + arith_name,
				std::to_string(workgroupSizeLog2),
				std::to_string(itemsPerWG),
				std::to_string(itemsPerInvoc),
				std::to_string(subgroupSizeLog2),
				std::to_string(numLoops)
			};

			const IShaderCompiler::SMacroDefinition defines[6] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_SIZE_LOG2", definitions[1] },
				{ "ITEMS_PER_WG", definitions[2] },
				{ "ITEMS_PER_INVOCATION", definitions[3] },
				{ "SUBGROUP_SIZE_LOG2", definitions[4] },
				{ "NUM_LOOPS", definitions[5] }
			};
			options.preprocessorOptions.extraDefines = { defines, defines + 6 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}
		else
		{
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
				{ "NUM_LOOPS", definitions[4] }
			};
			options.preprocessorOptions.extraDefines = { defines, defines + 5 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}
		
		BenchmarkSet set;
		set.pipeline = createPipeline(overriddenUnspecialized.get(), layout, subgroupSizeLog2);
		if constexpr (WorkgroupBench)
		{
			set.workgroupSize = itemsPerWG;
		}
		else
		{
			set.workgroupSize = workgroupSize;
		}
		set.itemsPerInvocation = itemsPerInvoc;

		return set;
	};

	template<bool WorkgroupBench>
	void runBenchmark(IGPUCommandBuffer* cmdbuf, const BenchmarkSet& set, const uint32_t elementCount, const uint8_t subgroupSizeLog2)
	{
		uint32_t workgroupCount;
		if constexpr (WorkgroupBench)
			workgroupCount = elementCount / set.workgroupSize;
		else
			workgroupCount = elementCount / (set.workgroupSize * set.itemsPerInvocation);

		cmdbuf->bindComputePipeline(set.pipeline.get());
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
	}

	IQueue* transferDownQueue;
	IQueue* computeQueue;

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	smart_refctd_ptr<InputSystem> m_inputSystem;

	smart_refctd_ptr<IGPUImage> dummyImg;

	constexpr static inline uint32_t MaxNumSubmits = 30;
	uint32_t numSubmits = 0;

	/* PARAMETERS TO CHANGE FOR DIFFERENT BENCHMARKS */

	constexpr static inline bool DoWorkgroupBenchmarks = true;
	uint32_t ItemsPerInvocation = 4u;
	constexpr static inline uint32_t NumLoops = 1000u;
	constexpr static inline uint32_t NumBenchmarks = 6u;
	constexpr static inline std::array<uint32_t, NumBenchmarks> workgroupSizes = { 32, 64, 128, 256, 512, 1024 };
	template<class BinOp>
	using ArithmeticOp = emulatedReduction<BinOp>;	// change this to test other arithmetic ops

	std::array<BenchmarkSet, NumBenchmarks> benchSets;
	smart_refctd_ptr<IDescriptorPool> benchPool;
	smart_refctd_ptr<IGPUDescriptorSet> benchDs;

	uint32_t* inputData = nullptr;
	smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
	constexpr static inline uint32_t OutputBufferCount = 8u;
	smart_refctd_ptr<IGPUBuffer> outputBuffers[OutputBufferCount];
	smart_refctd_ptr<IGPUBuffer> gpuOutputAddressesBuffer;
	PushConstantData pc;

	smart_refctd_ptr<ISemaphore> sema;
	uint64_t timelineValue = 0;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	uint32_t totalFailCount = 0;
};

NBL_MAIN_FUNC(ArithmeticBenchApp)