#include "nbl/examples/examples.hpp"
#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic_config.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_params.hlsl"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;


template<typename SwapchainResources> requires std::is_base_of_v<ISimpleManagedSurface::ISwapchainResources, SwapchainResources>
class CExplicitSurfaceFormatResizeSurface final : public ISimpleManagedSurface
{
public:
	using this_t = CExplicitSurfaceFormatResizeSurface<SwapchainResources>;

	// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `ICallback` declared just above
	template<typename Surface> requires std::is_base_of_v<CSurface<typename Surface::window_t, typename Surface::immediate_base_t>, Surface>
	static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<Surface>&& _surface)
	{
		if (!_surface)
			return nullptr;

		auto _window = _surface->getWindow();
		ICallback* cb = nullptr;
		if (_window)
			cb = dynamic_cast<ICallback*>(_window->getEventCallback());

		return core::smart_refctd_ptr<this_t>(new this_t(std::move(_surface), cb), core::dont_grab);
	}

	// Factory method so we can fail, requires a `_surface` created from a native surface
	template<typename Surface> requires std::is_base_of_v<CSurfaceNative<typename Surface::window_t, typename Surface::immediate_base_t>, Surface>
	static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<Surface>&& _surface, ICallback* cb)
	{
		if (!_surface)
			return nullptr;

		return core::smart_refctd_ptr<this_t>(new this_t(std::move(_surface), cb), core::dont_grab);
	}

	//
	inline bool init(CThreadSafeQueueAdapter* queue, std::unique_ptr<SwapchainResources>&& scResources, const ISwapchain::SSharedCreationParams& sharedParams = {})
	{
		if (!scResources || !base_init(queue))
			return init_fail();

		m_sharedParams = sharedParams;
		if (!m_sharedParams.deduce(queue->getOriginDevice()->getPhysicalDevice(), getSurface()))
			return init_fail();

		m_swapchainResources = std::move(scResources);
		return true;
	}

	// Can be public because we don't need to worry about mutexes unlike the Smooth Resize class
	inline ISwapchainResources* getSwapchainResources() override { return m_swapchainResources.get(); }

	// need to see if the swapchain is invalidated (e.g. because we're starting from 0-area old Swapchain) and try to recreate the swapchain
	inline SAcquireResult acquireNextImage()
	{
		if (!isWindowOpen())
		{
			becomeIrrecoverable();
			return {};
		}

		if (!m_swapchainResources || (m_swapchainResources->getStatus() != ISwapchainResources::STATUS::USABLE && !recreateSwapchain(m_surfaceFormat)))
			return {};

		return ISimpleManagedSurface::acquireNextImage();
	}

	// its enough to just foward though
	inline bool present(const uint8_t imageIndex, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores)
	{
		return ISimpleManagedSurface::present(imageIndex, waitSemaphores);
	}

	//
	inline bool recreateSwapchain(const ISurface::SFormat& explicitSurfaceFormat)
	{
		assert(m_swapchainResources);
		// dont assign straight to `m_swapchainResources` because of complex refcounting and cycles
		core::smart_refctd_ptr<ISwapchain> newSwapchain;
		// TODO: This block of code could be rolled up into `ISimpleManagedSurface::ISwapchainResources` eventually
		{
			auto* surface = getSurface();
			auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());
			// 0s are invalid values, so they indicate we want them deduced
			m_sharedParams.width = 0;
			m_sharedParams.height = 0;
			// Question: should we re-query the supported queues, formats, present modes, etc. just-in-time??
			auto* swapchain = m_swapchainResources->getSwapchain();
			if (swapchain ? swapchain->deduceRecreationParams(m_sharedParams) : m_sharedParams.deduce(device->getPhysicalDevice(), surface))
			{
				// super special case, we can't re-create the swapchain but its possible to recover later on
				if (m_sharedParams.width == 0 || m_sharedParams.height == 0)
				{
					// we need to keep the old-swapchain around, but can drop the rest
					m_swapchainResources->invalidate();
					return false;
				}
				// now lets try to create a new swapchain
				if (swapchain)
					newSwapchain = swapchain->recreate(m_sharedParams);
				else
				{
					ISwapchain::SCreationParams params = {
						.surface = core::smart_refctd_ptr<ISurface>(surface),
						.surfaceFormat = explicitSurfaceFormat,
						.sharedParams = m_sharedParams
						// we're not going to support concurrent sharing in this simple class
					};
					m_surfaceFormat = explicitSurfaceFormat;
					newSwapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device), std::move(params));
				}
			}
			else // parameter deduction failed
				return false;
		}

		if (newSwapchain)
		{
			m_swapchainResources->invalidate();
			return m_swapchainResources->onCreateSwapchain(getAssignedQueue()->getFamilyIndex(), std::move(newSwapchain));
		}
		else
			becomeIrrecoverable();

		return false;
	}

protected:
	using ISimpleManagedSurface::ISimpleManagedSurface;

	//
	inline void deinit_impl() override final
	{
		becomeIrrecoverable();
	}

	//
	inline void becomeIrrecoverable() override { m_swapchainResources = nullptr; }

	// gets called when OUT_OF_DATE upon an acquire
	inline SAcquireResult handleOutOfDate() override final
	{
		// recreate swapchain and try to acquire again
		if (recreateSwapchain(m_surfaceFormat))
			return ISimpleManagedSurface::acquireNextImage();
		return {};
	}

private:
	// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
	ISwapchain::SSharedCreationParams m_sharedParams = {};
	// The swapchain might not be possible to create or recreate right away, so this might be
	// either nullptr before the first successful acquire or the old to-be-retired swapchain.
	std::unique_ptr<SwapchainResources> m_swapchainResources = {};

	ISurface::SFormat m_surfaceFormat = {};
};

// NOTE added swapchain + drawing frames to be able to profile with Nsight, which still doesn't support profiling headless compute shaders
class ArithmeticBenchApp final : public examples::SimpleWindowedApplication, public examples::BuiltinResourcesApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = examples::BuiltinResourcesApplication;

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
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CExplicitSurfaceFormatResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
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
		asset::E_FORMAT preferredFormats[] = { asset::EF_R8G8B8A8_UNORM };
		if (!swapchainParams.deduceFormat(m_physicalDevice, preferredFormats))
			return logFail("Could not choose a Surface Format for the Swapchain!");

		swapchainParams.sharedParams.imageUsage = IGPUImage::E_USAGE_FLAGS::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT;

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
		m_surface->recreateSwapchain(swapchainParams.surfaceFormat);

		transferDownQueue = getTransferDownQueue();
		computeQueue = getComputeQueue();

		// create 2 buffers for 2 operations
		for (auto i=0u; i<OutputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t) * (ElementCount+1);
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			outputBuffers[i] = m_device->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, outputBuffers[i].get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			assert(bufferMem.isValid());
		}
		for (auto i = 0u; i < OutputBufferCount; i++)
			pc.pOutputBuf[i] = outputBuffers[i]->getDeviceAddress();

		// create image views for swapchain images
		for (uint32_t i = 0; i < ISwapchain::MaxImages; i++)
		{
			IGPUImage* scImg = m_surface->getSwapchainResources()->getImage(i);
			if (scImg == nullptr)
				continue;
			IGPUImageView::SCreationParams viewParams = {
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
				.image = smart_refctd_ptr<IGPUImage>(scImg),
				.viewType = IGPUImageView::ET_2D,
				.format = scImg->getCreationParameters().format
			};
			swapchainImageViews[i] = m_device->createImageView(std::move(viewParams));
		}

		// create Descriptor Sets and Pipeline Layouts
		smart_refctd_ptr<IGPUPipelineLayout> benchPplnLayout;
		{
			// set and transient pool
			smart_refctd_ptr<IGPUDescriptorSetLayout> benchLayout;
			{
				IGPUDescriptorSetLayout::SBinding binding[1];
				binding[0] = { {},2,IDescriptor::E_TYPE::ET_STORAGE_IMAGE,IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,IShader::E_SHADER_STAGE::ESS_COMPUTE,1u,nullptr };
				benchLayout = m_device->createDescriptorSetLayout(binding);
			}

			const uint32_t setCount = ISwapchain::MaxImages;
			benchPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, { &benchLayout.get(),1 }, &setCount);
			for (auto i = 0u; i < ISwapchain::MaxImages; i++)
			{
			    benchDs[i] = benchPool->createDescriptorSet(smart_refctd_ptr(benchLayout));
				if (!benchDs[i])
					return logFail("Could not create Descriptor Set!");
			}

			SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0,.size = sizeof(PushConstantData) };
			benchPplnLayout = m_device->createPipelineLayout({ &pcRange, 1 }, std::move(benchLayout));
		}
		if (UseNativeArithmetic && !m_physicalDevice->getProperties().limits.shaderSubgroupArithmetic)
		{
			logFail("UseNativeArithmetic is true but device does not support shaderSubgroupArithmetic!");
			return false;
		}

		IGPUDescriptorSet::SWriteDescriptorSet dsWrites[ISwapchain::MaxImages];
		for (auto i = 0u; i < ISwapchain::MaxImages; i++)
		{
			if (swapchainImageViews[i].get() == nullptr)
				continue;

			video::IGPUDescriptorSet::SDescriptorInfo dsInfo;
			dsInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;
			dsInfo.desc = swapchainImageViews[i];

			dsWrites[i] =
			{
				.dstSet = benchDs[i].get(),
				.binding = 2u,
				.arrayElement = 0u,
				.count = 1u,
				.info = &dsInfo,
			};
			m_device->updateDescriptorSets(1u, &dsWrites[i], 0u, nullptr);
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
			return smart_refctd_ptr_static_cast<IShader>(firstAssetInBundle);
		};

		// for each workgroup size (manually adjust items per invoc, operation else uses up a lot of ram)
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
		smart_refctd_ptr<IShader> shaderSource;
		if constexpr (DoWorkgroupBenchmarks)
			shaderSource = getShaderSource("app_resources/benchmarkWorkgroup.comp.hlsl");
		else
			shaderSource = getShaderSource("app_resources/benchmarkSubgroup.comp.hlsl");

		for (uint32_t op = 0; op < arithmeticOperations.size(); op++)
			for (uint32_t i = 0; i < workgroupSizes.size(); i++)
				benchSets[op*workgroupSizes.size()+i] = createBenchmarkPipelines<DoWorkgroupBenchmarks>(shaderSource, benchPplnLayout.get(), ElementCount, arithmeticOperations[op], hlsl::findMSB(MaxSubgroupSize), workgroupSizes[i], ItemsPerInvocation, NumLoops);

		m_winMgr->show(m_window.get());

		return true;
	}

	virtual bool onAppTerminated() override
	{
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

		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
		const auto SubgroupSizeLog2 = hlsl::findMSB(MaxSubgroupSize);

		cmdbuf->bindDescriptorSets(EPBP_COMPUTE, benchSets[0].pipeline->getLayout(), 0u, 1u, &benchDs[m_currentImageAcquire.imageIndex].get());
		cmdbuf->pushConstants(benchSets[0].pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(PushConstantData), &pc);

		for (uint32_t i = 0; i < benchSets.size(); i++)
			runBenchmark<DoWorkgroupBenchmarks>(cmdbuf, benchSets[i], ElementCount, SubgroupSizeLog2);

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
			imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
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

			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}

		numSubmits++;
	}

	//
	bool keepRunning() override { return numSubmits < MaxNumSubmits; }

private:
	// create pipeline (specialized every test) [TODO: turn into a future/async]
	smart_refctd_ptr<IGPUComputePipeline> createPipeline(const IShader* overridenUnspecialized, const IGPUPipelineLayout* layout, const uint8_t subgroupSizeLog2)
	{
		auto shader = m_device->compileShader({ overridenUnspecialized });
		IGPUComputePipeline::SCreationParams params = {};
		params.layout = layout;
		params.shader = {
			.shader = shader.get(),
			.entryPoint = "main",
			.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(subgroupSizeLog2),
			.entries = nullptr,
		};
		params.cached.requireFullSubgroups = true;
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

	template<bool WorkgroupBench>
	BenchmarkSet createBenchmarkPipelines(const smart_refctd_ptr<const IShader>&source, const IGPUPipelineLayout* layout, const uint32_t elementCount, const std::string& arith_name, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, uint32_t itemsPerInvoc = 1u, uint32_t numLoops = 8u)
	{
		auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
		CHLSLCompiler::SOptions options = {};
		options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
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
		options.preprocessorOptions.includeFinder = includeFinder;

		const uint32_t subgroupSize = 0x1u << subgroupSizeLog2;
		const uint32_t workgroupSizeLog2 = hlsl::findMSB(workgroupSize);
		hlsl::workgroup2::SArithmeticConfiguration wgConfig;
	    wgConfig.init(workgroupSizeLog2, subgroupSizeLog2, itemsPerInvoc);
		const uint32_t itemsPerWG = wgConfig.VirtualWorkgroupSize * wgConfig.ItemsPerInvocation_0;
		smart_refctd_ptr<IShader> overriddenUnspecialized;
		if constexpr (WorkgroupBench)
		{
			const std::string definitions[4] = {
				"workgroup2::" + arith_name,
				wgConfig.getConfigTemplateStructString(),
				std::to_string(numLoops),
				std::to_string(arith_name=="reduction")
			};

			const IShaderCompiler::SMacroDefinition defines[5] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_CONFIG_T", definitions[1] },
				{ "NUM_LOOPS", definitions[2] },
				{ "IS_REDUCTION", definitions[3] },
				{ "TEST_NATIVE", "1" }
			};
			if (UseNativeArithmetic)
				options.preprocessorOptions.extraDefines = { defines, defines + 5 };
			else
				options.preprocessorOptions.extraDefines = { defines, defines + 4 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}
		else
		{
			hlsl::subgroup2::SArithmeticParams sgParams;
			sgParams.init(subgroupSizeLog2, itemsPerInvoc);

			const std::string definitions[4] = { 
				"subgroup2::" + arith_name,
				std::to_string(workgroupSize),
				sgParams.getParamTemplateStructString(),
				std::to_string(numLoops)
			};

			const IShaderCompiler::SMacroDefinition defines[5] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_SIZE", definitions[1] },
				{ "SUBGROUP_CONFIG_T", definitions[2] },
				{ "NUM_LOOPS", definitions[3] },
				{ "TEST_NATIVE", "1" }
			};
			if (UseNativeArithmetic)
				options.preprocessorOptions.extraDefines = { defines, defines + 5 };
			else
				options.preprocessorOptions.extraDefines = { defines, defines + 4 };

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
	smart_refctd_ptr<CExplicitSurfaceFormatResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	smart_refctd_ptr<InputSystem> m_inputSystem;

	std::array<smart_refctd_ptr<IGPUImageView>, ISwapchain::MaxImages> swapchainImageViews;

	constexpr static inline uint32_t MaxNumSubmits = 30;
	uint32_t numSubmits = 0;
	constexpr static inline uint32_t ElementCount = 1024 * 1024;

	/* PARAMETERS TO CHANGE FOR DIFFERENT BENCHMARKS */
	constexpr static inline bool DoWorkgroupBenchmarks = true;
	constexpr static inline bool UseNativeArithmetic = true;
	uint32_t ItemsPerInvocation = 4u;
	constexpr static inline uint32_t NumLoops = 1000u;
	constexpr static inline uint32_t NumBenchmarks = 6u;
	std::array<uint32_t, NumBenchmarks> workgroupSizes = { 32, 64, 128, 256, 512, 1024 };
	std::array<std::string, 3u> arithmeticOperations = { "reduction", "inclusive_scan", "exclusive_scan" };


	std::array<BenchmarkSet, NumBenchmarks*3u> benchSets;
	smart_refctd_ptr<IDescriptorPool> benchPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ISwapchain::MaxImages> benchDs;

	constexpr static inline uint32_t OutputBufferCount = 2u;
	smart_refctd_ptr<IGPUBuffer> outputBuffers[OutputBufferCount];
	smart_refctd_ptr<IGPUBuffer> gpuOutputAddressesBuffer;
	PushConstantData pc;

	uint64_t timelineValue = 0;
};

NBL_MAIN_FUNC(ArithmeticBenchApp)