#include <nabla.h>

#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"
#include "../common/Camera.hpp"

#include "app_resources/common.hlsl"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// struct Particle defined in shader

class CSwapchainFramebuffersAndDepth final : public nbl::video::CDefaultSwapchainFramebuffers
{
	using scbase_t = CDefaultSwapchainFramebuffers;
public:
	template<typename... Args>
	inline CSwapchainFramebuffersAndDepth(ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args)
		: CDefaultSwapchainFramebuffers(device, std::forward<Args>(args)...)
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

		const auto retval = scbase_t::onCreateSwapchain_impl(qFam);
		m_depthBuffer = nullptr;
		return retval;
	}

	inline smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params) override
	{
		params.depthStencilAttachments = &m_depthBuffer.get();
		return m_device->createFramebuffer(std::move(params));
	}

	E_FORMAT m_depthFormat;
	smart_refctd_ptr<IGPUImageView> m_depthBuffer;
};

class FLIPFluidsApp final : public examples::SimpleWindowedApplication
{
	using base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_WIDTH = 1280, WIN_HEIGHT = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

public:
	inline FLIPFluidsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				//auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params{
					.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>(),
					.x = 32,
					.y = 32,
					.width = WIN_WIDTH,
					.height = WIN_HEIGHT,
					.flags = IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE,
					.windowCaption = "FLIPFluidsApp"
				};
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
		}

		if (m_surface)
			return { { m_surface->getSurface() } };

		return {};
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		m_semaphore = m_device->createSemaphore(m_submitIx);
		if (!m_semaphore)
			return logFail("Failed to create semaphore!");

		ISwapchain::SCreationParams swapchainParams{
			.surface = smart_refctd_ptr<video::ISurface>(m_surface->getSurface())
		};
		if (!swapchainParams.deduceFormat(m_physicalDevice))
			return logFail("Could not choose a surface format for the swapchain!");

		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
				.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT
			}
		},
			// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			}
		},
		IGPURenderpass::SCreationParams::DependenciesEnd
		};

		auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), EF_D16_UNORM, swapchainParams.surfaceFormat.format, dependencies);
		auto* renderpass = scResources->getRenderpass();
		if (!renderpass)
			return logFail("Failed to create renderpass!");

		// init shaders
		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
		CSPIRVIntrospector introspector;
		auto compiledShader = compileShaderAndIntrospect("app_resources/test.comp.hlsl", introspector, assetManager);
		auto source = compiledShader.first;
		auto shaderIntrospection = compiledShader.second;

		ICPUShader::SSpecInfo specInfo;
		specInfo.entryPoint = "main";
		specInfo.shader = source.get();

		smart_refctd_ptr<ICPUComputePipeline> cpuPipeline = introspector.createApproximateComputePipelineFromIntrospection(specInfo); ///< what does this do?

		smart_refctd_ptr<nbl::video::IGPUShader> exampleShader = m_device->createShader(source.get());
		if (!exampleShader)
			return logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");

		std::array<std::vector<IGPUDescriptorSetLayout::SBinding>, IGPUPipelineLayout::DESCRIPTOR_SET_COUNT> bindings;
		for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
		{
			const auto& introspectionBindings = shaderIntrospection->getDescriptorSetInfo(i);
			bindings[i].resize(introspectionBindings.size());

			for (const auto& introspectionBinding : introspectionBindings)
			{
				auto& binding = bindings[i].emplace_back();

				binding.binding = introspectionBinding.binding;
				binding.type = introspectionBinding.type;
				binding.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE;
				binding.stageFlags = IGPUShader::ESS_COMPUTE;
				assert(introspectionBinding.count.countMode == CSPIRVIntrospector::CIntrospectionData::SDescriptorArrayInfo::DESCRIPTOR_COUNT::STATIC);
				binding.count = introspectionBinding.count.count;
			}
		}

		const std::array<core::smart_refctd_ptr<IGPUDescriptorSetLayout>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> testDsLayouts = {
			bindings[0].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[0]),
			bindings[1].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[1]),
			bindings[2].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[2]),
			bindings[3].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[3]),
		};

		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout = m_device->createPipelineLayout(
			{},
			core::smart_refctd_ptr(testDsLayouts[0]),
			core::smart_refctd_ptr(testDsLayouts[1]),
			core::smart_refctd_ptr(testDsLayouts[2]),
			core::smart_refctd_ptr(testDsLayouts[3])
		);
		if (!pipelineLayout)
			return logFail("Failed to create compute pipeline layout!\n");

		// init pipeline(s)
		smart_refctd_ptr<video::IGPUComputePipeline> pipeline;
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pipelineLayout.get();
			params.shader.entryPoint = "main";
			params.shader.shader = exampleShader.get();
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
				return logFail("Failed to create pipelines (compile & link shaders)!\n");
		}

		// init and write descriptor
		constexpr uint32_t maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
		const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
			!testDsLayouts[0] ? nullptr : testDsLayouts[0].get(),
			!testDsLayouts[1] ? nullptr : testDsLayouts[1].get(),
			!testDsLayouts[2] ? nullptr : testDsLayouts[2].get(),
			!testDsLayouts[3] ? nullptr : testDsLayouts[3].get()
		};
		std::array<smart_refctd_ptr<IGPUDescriptorSet>, maxDescriptorSets> descriptorSets;
		auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
		pool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), descriptorSets.data());

		// init buffers
		constexpr size_t workgroupCount = 4096;
		constexpr size_t bufferSize = sizeof(uint32_t) * WorkgroupSize * workgroupCount;

		video::IDeviceMemoryAllocator::SAllocation allocation = {};

		{
			video::IGPUBuffer::SCreationParams params = {};
			params.size = bufferSize;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			smart_refctd_ptr<IGPUBuffer> testBuffer = m_device->createBuffer(std::move(params));
			if (!testBuffer)
				return logFail("Failed to create GPU buffer of size %d!\n", params.size);

			testBuffer->setObjectDebugName("test output buffer");

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = testBuffer->getMemoryReqs();
			reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

			allocation = m_device->allocate(reqs, testBuffer.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!allocation.isValid())
				return logFail("Failed to allocate device memory compatible with gpu buffer!\n");

			{
				IGPUDescriptorSet::SDescriptorInfo info[1];
				info[0].desc = smart_refctd_ptr(testBuffer);
				info[0].info.buffer = {.offset = 0, .size = bufferSize};
				IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
					{.dstSet = descriptorSets[0].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = info}
				};
				m_device->updateDescriptorSets(std::span(writes, 1), {});
			}
		}

		if (!allocation.memory->map({0ull, allocation.memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ))
			return logFail("Failed to map the device memory!\n");

		// create command buffer and pool
		smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf;
		IQueue* const queue = getComputeQueue();
		m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
		{
			logFail("Failed to create command buffers!\n");
			return false;
		}

		constexpr auto StartedValue = 0;
		constexpr auto FinishedValue = 45;
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(StartedValue);
		{
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->bindComputePipeline(pipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pipeline->getLayout(), 0, descriptorSets.size(), &descriptorSets.begin()->get());
			cmdbuf->dispatch(workgroupCount, 1, 1);
			cmdbuf->end();

			IQueue::SSubmitInfo submitInfo = {};
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = {{.cmdbuf = cmdbuf.get()}};
			submitInfo.commandBuffers = cmdbufs;
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {{.semaphore = progress.get(), .value = FinishedValue, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}};
			submitInfo.signalSemaphores = signals;

			queue->startCapture();
			queue->submit({{submitInfo}});
			queue->endCapture();
		}

		const ISemaphore::SWaitInfo waitInfos[] = {{
				.semaphore = progress.get(),
				.value = FinishedValue
			}};
		m_device->blockForSemaphores(waitInfos);

		auto buffData = reinterpret_cast<const uint32_t*>(allocation.memory->getMappedPointer());
		assert(allocation.offset==0); // simpler than writing out all the pointer arithmetic
		for (auto i=0; i<WorkgroupSize*workgroupCount; i++)
		if (buffData[i]!=i)
			return logFail("DWORD at position %d doesn't match!\n",i);
		// This allocation would unmap itself in the dtor anyway, but lets showcase the API usage
		allocation.memory->unmap();

		/*
		{
			core::vectorSIMDf cameraPosition(-5.81655884, 2.58630896, -4.23974705);
			core::vectorSIMDf cameraTarget(-0.349590302, -0.213266611, 0.317821503);
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_WIDTH) / WIN_HEIGHT, 0.1, 10000);
			camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);
		}
		*/

		m_winMgr->show(m_window.get());

		m_device->waitIdle();

		return true;
	}

	inline void workLoopBody()
	{
		/*
		for (uint32_t i = 0; i < m_substepsPerFrame; i++)
		{
			dispatchUpdateFluidCells();			// particle to grid
			dispatchApplyBodyForces(i == 0);	// external forces, e.g. gravity
			dispatchApplyDiffusion();
			dispatchApplyPressure();
			dispatchExtrapolateVelocities();	// grid -> particle vel
			dispatchAdvection();				// update/advect fluid
		}
		*/

		// renderFluid();		// TODO: mesh or particles?
	}

	bool keepRunning() override { return false; }

	void dispatchUpdateFluidCells()
	{
	}
	
	void dispatchApplyBodyForces(bool isFirstSubstep)
	{
	}
	
	void dispatchApplyDiffusion()
	{
	}
	
	void dispatchApplyPressure()
	{
	}
	
	void dispatchExtrapolateVelocities()
	{
	}
			
	void dispatchAdvection()
	{
	}

private:
	std::pair<smart_refctd_ptr<ICPUShader>, smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData>> compileShaderAndIntrospect(
		const std::string& filePath, CSPIRVIntrospector& introspector, smart_refctd_ptr<IAssetManager> assetManager)
	{
		IAssetLoader::SAssetLoadParams lparams = {};
		lparams.logger = m_logger.get();
		lparams.workingDirectory = "";
		auto bundle = assetManager->getAsset(filePath, lparams);
		if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
		{
			m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
			exit(-1);
		}
		
		const auto assets = bundle.getContents();
		assert(assets.size() == 1);
		smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);

		smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData> introspection;
		{
			auto* compilerSet = assetManager->getCompilerSet();

			nbl::asset::IShaderCompiler::SCompilerOptions options = {};
			options.stage = shaderSrc->getStage();
			if (!(options.stage == IShader::ESS_COMPUTE || options.stage == IShader::ESS_FRAGMENT))
				options.stage = IShader::ESS_VERTEX;
			options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
			options.spirvOptimizer = nullptr;
			options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
			options.preprocessorOptions.sourceIdentifier = shaderSrc->getFilepathHint();
			options.preprocessorOptions.logger = m_logger.get();
			options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(shaderSrc->getContentType())->getDefaultIncludeFinder();

			auto spirvUnspecialized = compilerSet->compileToSPIRV(shaderSrc.get(), options);
			const CSPIRVIntrospector::CStageIntrospectionData::SParams inspectParams = {
				.entryPoint = "main",
				.shader = spirvUnspecialized
			};

			introspection = introspector.introspect(inspectParams);
			introspection->debugPrint(m_logger.get());

			if (!introspection)
			{
				logFail("SPIR-V Introspection failed, probably the required SPIR-V compilation failed first!");
				return std::pair(nullptr, nullptr);
			}

			{
				auto* shaderContent = spirvUnspecialized->getContent();

				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_physicalDevice->getSystem()->createFile(future, system::path("../shaders/compiled.spv"), system::IFileBase::ECF_WRITE);
				if (auto file = future.acquire(); file && bool(*file))
				{
					system::IFile::success_t success;
					(*file)->write(success, shaderContent->getPointer(), 0, shaderContent->getSize());
					success.getBytesProcessed(true);
				}
			}

			shaderSrc = std::move(spirvUnspecialized);
		}
		return std::pair(shaderSrc, introspection);
	}

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_graphicsPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_computePipeline;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
	uint64_t m_realFrameIx : 59 = 0;
	uint64_t m_submitIx : 59 = 0;
	uint64_t m_maxFramesInFlight : 5;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	// input system?

	Camera camera = Camera(core::vectorSIMDf(0,0,0), core::vectorSIMDf(0,0,0), core::matrix4SIMD());

	smart_refctd_ptr<video::IDescriptorPool> m_descriptorPool;
	smart_refctd_ptr<video::IGPUDescriptorSet> m_gpuDescriptorSet;	// porbably need more

	// simulation constants
	uint32_t m_substepsPerFrame = 1;

	// buffers
	smart_refctd_ptr<IGPUBuffer> particleBuffer;		// Particle
	
	smart_refctd_ptr<IGPUBuffer> gridParticleIDBuffer;	// uint2
	smart_refctd_ptr<IGPUBuffer> gridCellTypeBuffer;	// uint, fluid or solid
	smart_refctd_ptr<IGPUBuffer> velocityFieldBuffer;	// float3
	smart_refctd_ptr<IGPUBuffer> prevVelocityFieldBuffer;// float3
	smart_refctd_ptr<IGPUBuffer> gridDiffusionBuffer;	// float3
	smart_refctd_ptr<IGPUBuffer> gridAxisTypeBuffer;	// uint3
	smart_refctd_ptr<IGPUBuffer> divergenceBuffer;		// float
	smart_refctd_ptr<IGPUBuffer> pressureBuffer;		// float
	smart_refctd_ptr<IGPUBuffer> gridWeightBuffer;		// float
	smart_refctd_ptr<IGPUBuffer> gridUintWeightBuffer;	// uint
	smart_refctd_ptr<IGPUBuffer> gridDensityPressureBuffer;// float
	smart_refctd_ptr<IGPUBuffer> positionModifyBuffer;	// float3
	smart_refctd_ptr<IGPUBuffer> zeroBuffer;			// float

};

NBL_MAIN_FUNC(FLIPFluidsApp)