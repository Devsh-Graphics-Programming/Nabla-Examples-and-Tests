// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/present/CWindowPresenter.h"

namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::hlsl;
using namespace nbl::ui;
using namespace nbl::video;

constexpr auto SessionImageWritingStages = PIPELINE_STAGE_FLAGS::CLEAR_BIT|PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT|PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;

constexpr IGPURenderpass::SCreationParams::SSubpassDependency CWindowPresenter::dependencies[3] =
{
	{
		.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
		.dstSubpass = 0,
		.memoryBarrier =
		{
			.srcStageMask = SessionImageWritingStages,
			.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT|ACCESS_FLAGS::STORAGE_WRITE_BIT,
			// fragment shader that draws them
			.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
			.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
		}
	},
	{
		.srcSubpass = 0,
		.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
		.memoryBarrier =
		{
			// the output to swapchain image
			.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
			// we only worry about next compute dispatch not overwriting our presented image
			.dstStageMask = SessionImageWritingStages,
			// but there are no writes from present to make available to it
			.dstAccessMask = ACCESS_FLAGS::NONE
			// swapchain present of image index I synchronises with the next acquire of image I so no need to worry about the reuse of that
			// note that there's no extra destination stages or accesses because they're not needed for a swapchain present
		}
	},
	IGPURenderpass::SCreationParams::DependenciesEnd
};

//
smart_refctd_ptr<CWindowPresenter> CWindowPresenter::create(SCreationParams&& _params)
{
	if (!_params)
	{
		_params.logger.log("`CWindowPresenter::SCreationParams` are invalidl!",ILogger::ELL_ERROR);
		return nullptr;
	}
	CWindowPresenter::SConstructorParams params = {std::move(_params)};

	{
		const auto& primDpyInfo = params.winMgr->getPrimaryDisplayInfo();
		// subtract window border/decoration elements
		params.maxResolution = hlsl::max<int32_t2>(int32_t2(primDpyInfo.resX,primDpyInfo.resY)-int32_t2(32,16),int32_t2(0,0));
		// we add an additional constraint that any dimension of maxResolution cannot be less than any dimension of minResolution
		// e.g. max resolution Height cannot be less than min resolution width 
		if (hlsl::any(hlsl::less<uint16_t4>()(params.maxResolution.xxyy,params.minResolution.xyxy)))
		{
			params.logger.log(
				"`CWindowPresenter::create` desktop resolution must allow for at least a %d x %d window!",
				ILogger::ELL_ERROR,params.minResolution.x,params.minResolution.y
			);
			return nullptr;
		}
		params.aspectRatioRange[0] = float64_t(params.minResolution.x)/float64_t(params.maxResolution.y);
		params.aspectRatioRange[1] = float64_t(params.maxResolution.x)/float64_t(params.minResolution.y);
	}

	// create the window
	smart_refctd_ptr<IWindow> window;
	{
		IWindow::SCreationParams winParams = {};
		winParams.width = 64;
		winParams.height = 64;
		winParams.x = 32;
		winParams.y = 32;
		winParams.flags = IWindow::ECF_HIDDEN|IWindow::ECF_BORDERLESS|IWindow::ECF_RESIZABLE;
		winParams.windowCaption = _params.initialWindowCaption;
		winParams.callback = std::move(_params.callback);
		window = params.winMgr->createWindow(std::move(winParams));
	}
	if (!window)
	{
		params.logger.log("`CWindowPresenter::create` failed to create a window!",ILogger::ELL_ERROR);
		return nullptr;
	}
	params.window = window.get();
	params.cursorControl = window->getCursorControl();

	// create surface
	{
		auto surface = CSurfaceVulkanWin32::create(std::move(_params.api),move_and_static_cast<IWindowWin32>(window));
		params.surface = surface_t::create(std::move(surface));
	}
	if (!params.surface)
	{
		params.logger.log("`CWindowPresenter::create` failed to create a surface!",ILogger::ELL_ERROR);
		return nullptr;
	}

	return smart_refctd_ptr<CWindowPresenter>(new CWindowPresenter(std::move(params)),dont_grab);
}

bool CWindowPresenter::init(CRenderer* renderer)
{
	auto& logger = m_creation.logger;
	auto* device = renderer->getDevice();

	// create swapchain and its resources (renderpass, etc.)
	{
		ISurface* const tmp = getSurface();
		ISwapchain::SCreationParams swapchainParams = {.surface=smart_refctd_ptr<ISurface>(tmp)};
		if (!swapchainParams.deduceFormat(device->getPhysicalDevice()))
		{
			logger.log("Could not choose a Surface Format for the Swapchain!",ILogger::ELL_ERROR);
			return false;
		}

		auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(device,swapchainParams.surfaceFormat.format,dependencies,IGPURenderpass::LOAD_OP::DONT_CARE);
		if (!scResources || !scResources->getRenderpass())
		{
			logger.log("Failed to create Renderpass!",ILogger::ELL_ERROR);
			return false;
		}

		if (!m_construction.surface->init(renderer->getCreationParams().graphicsQueue,std::move(scResources),swapchainParams.sharedParams))
		{
			logger.log("Could not create Window & Surface or initialize the Surface!",ILogger::ELL_ERROR);
			return false;
		}
	}

	//
	ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_creation.assMan.get(),device,logger.get().get());
	if (!fsTriProtoPPln)
	{
		logger.log("`CWindowPresenter::create` failed to create Full Screen Triangle protopipeline or load its vertex shader!",ILogger::ELL_ERROR);
		return false;
	}

	// present pipeline layout
	smart_refctd_ptr<IGPUPipelineLayout> layout;
	{
		const SPushConstantRange pcRange[] = {
			{.stageFlags=ShaderStage::ESS_FRAGMENT,.offset=0,.size=sizeof(m_pushConstants)}
		};
		if (!(layout=device->createPipelineLayout(pcRange,renderer->getConstructionParams().sensorDSLayout)))
		{
			logger.log("`CWindowPresenter::create` failed to create Pipeline Layout!",ILogger::ELL_ERROR);
			return false;
		}
	}

	// present pipeline
	if (auto shader=renderer->loadPrecompiledShader<"present_default">(m_creation.assMan.get(),device,logger.get().get()); shader)
	{
		const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
			.shader = shader.get(),
			.entryPoint = "present_default",
		};
		m_present = fsTriProtoPPln.createPipeline(fragSpec,layout.get(),getRenderpass());
		if (!m_present)
			logger.log("`CWindowPresenter::create` failed to create Graphics Pipeline!",ILogger::ELL_ERROR);
	}
	else
	{
		logger.log("`CWindowPresenter::create` failed to load shader!",ILogger::ELL_ERROR);
		return false;
	}

	return bool(m_present);
}

auto CWindowPresenter::acquire(const ISwapchain::SAcquireInfo& info, const CSession* session) -> clock_t::time_point
{
	auto expectedPresent = clock_t::time_point::min(); // invalid value
	if (!session)
		return expectedPresent;
	const auto& sessionParams = session->getConstructionParams();
	m_pushConstants.isCubemap = sessionParams.type==CSession::sensor_type_e::Env;

	const auto maxResolution = m_construction.maxResolution;
	uint16_t2 targetResolution = m_pushConstants.isCubemap ? maxResolution:sessionParams.cropResolution;
	const auto aspectRatio = double(targetResolution.x)/double(targetResolution.y);
	if (m_pushConstants.isCubemap)
	{
		// TODO: build default perspective projection matrix given aspect ratio and smaller axis (or diagonal) FOV of the viewer
//		m_pushConstants.cubemap.invProjView = ;
	}
	else
	{
		m_pushConstants.regular.crop = float32_t2(sessionParams.cropOffsets)*sessionParams.uniforms.rcpPixelSize;
		// this we shall adjust to take care of aspect ratio mismatch
		m_pushConstants.regular.scale = float32_t2(sessionParams.cropResolution)*sessionParams.uniforms.rcpPixelSize;
		m_pushConstants.regular.limit = m_pushConstants.regular.scale+m_pushConstants.regular.crop;
		// prevent extreme window size
		const auto clampedAspectRatio = hlsl::clamp(aspectRatio,m_construction.aspectRatioRange[0],m_construction.aspectRatioRange[1]);
		const float64_t asConv = core::min(1.0/clampedAspectRatio,clampedAspectRatio);
		const uint8_t largeDim = targetResolution.x<targetResolution.y ? 1:0;
		const uint8_t smallDim = largeDim^0x1u;
		const auto minResolution = m_creation.minResolution;
		targetResolution[largeDim] = hlsl::clamp(targetResolution[largeDim],minResolution[largeDim],maxResolution[largeDim]);
		// and aspect ratios
		targetResolution[smallDim] = hlsl::ceil(targetResolution[largeDim]*asConv);
		{
			const auto aspectChange = clampedAspectRatio/aspectRatio;
			// >1.0 makes us wider (adds width), <1.0 makes us narrower (adds height)
			if (aspectChange>1.0)
				m_pushConstants.regular.scale[0] *= aspectChange;
			else
				m_pushConstants.regular.scale[1] /= aspectChange;
		}
		// `CWindowPresenter::create` aspect ratio ranges and min/max relationships help us stay valid
		assert(all(minResolution<=targetResolution)&&all(targetResolution<=maxResolution));
	}

	// handle session resolution change
	auto& winMgr = m_creation.winMgr;
	auto* const window = m_construction.window;
	if (const uint16_t2 currentResolution={window->getWidth(),window->getHeight()}; currentResolution!=targetResolution)
	{
		if (!winMgr->setWindowSize(window,targetResolution.x,targetResolution.y))
			return expectedPresent;
		m_construction.surface->recreateSwapchain();
	}
	if (window->isHidden())
		winMgr->show(window);


	m_pushConstants.layer = 0; // TODO: cubemaps and RWMC debug
	m_pushConstants.imageIndex = 0;

	if (!(m_currentImageAcquire=m_construction.surface->acquireNextImage()))
	{
		return expectedPresent;
	}

	// TODO: Do this properly with present timing extension and a better oracle
	expectedPresent = clock_t::now() + std::chrono::microseconds(16666);

	return expectedPresent;
}

bool CWindowPresenter::beginRenderpass(IGPUCommandBuffer* cb)
{
	auto* const scRes = getSwapchainResources();
	auto* const framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex);
	const uint16_t2 resolution = { framebuffer->getCreationParameters().width,framebuffer->getCreationParameters().height};

	bool success = cb->beginDebugMarker("Present");
	const SViewport viewport[] = {{
		.x = 0u, .y = 0u,
		.width = static_cast<float>(resolution.x),
		.height = static_cast<float>(resolution.y),
		.minDepth = 1.f, .maxDepth = 0.f
	}};
	success = success && cb->setViewport(viewport,0);
	{
		const VkRect2D defaultScisors[] = {{
			.offset = {static_cast<int32_t>(viewport->x), static_cast<int32_t>(viewport->y)},
			.extent = {resolution.x,resolution.y}
		}};
		cb->setScissor(defaultScisors);
		const VkRect2D currentRenderArea = {.offset = {0,0}, .extent = defaultScisors->extent};
		const IGPUCommandBuffer::SRenderpassBeginInfo info =
		{
			.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
			.colorClearValues = nullptr,
			.depthStencilClearValues = nullptr,
			.renderArea = currentRenderArea
		};
		success = success && cb->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
	}

	success = success && cb->bindGraphicsPipeline(m_present.get());

	const auto* layout = m_present->getLayout();
//	success = success && cb->bindDescriptorSets(EPBP_GRAPHICS,layout,0,1u,&m_presentDs.get());
	success = success && cb->pushConstants(layout,ShaderStage::ESS_FRAGMENT,0,sizeof(m_pushConstants),&m_pushConstants);
	ext::FullScreenTriangle::recordDrawCall(cb);

	success = success && cb->endDebugMarker();
	return success;
}
bool CWindowPresenter::endRenderpassAndPresent(IGPUCommandBuffer* cb, ISemaphore* presentBeginSignal)
{
	bool success = cb->endRenderPass();
	success = success && cb->end();
/*
	const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
	{
		{.cmdbuf = cb}
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
	
	if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
		m_realFrameIx--;
*/

//	m_construction.surface->present(m_currentImageAcquire.imageIndex,rendered);
	return false;
}

}

#if 0
		{
			const IGPUDescriptorSetLayout::SBinding bindings[] = {
			  {
				.binding = 0u,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.count = 1u,
				.immutableSamplers = &defaultSampler
			  }
			};
			auto gpuPresentDescriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
			const video::IGPUDescriptorSetLayout* const layouts[] = { gpuPresentDescriptorSetLayout.get() };
			const uint32_t setCounts[] = { 1u };
			m_presentDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
			m_presentDs = m_presentDsPool->createDescriptorSet(gpuPresentDescriptorSetLayout);

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
			  .shader = fragmentShader.get(),
			  .entryPoint = "main",
			};

			auto presentLayout = m_device->createPipelineLayout(
				{},
				core::smart_refctd_ptr(gpuPresentDescriptorSetLayout),
				nullptr,
				nullptr,
				nullptr
			);
			m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass());
			if (!m_presentPipeline)
				return logFail("Could not create Graphics Pipeline!");
		}

		// write descriptors
		IGPUDescriptorSet::SDescriptorInfo infos[3];
		infos[0].desc = m_gpuTlas;

		infos[1].desc = m_hdrImageView;
		if (!infos[1].desc)
			return logFail("Failed to create image view");
		infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;

		infos[2].desc = m_hdrImageView;
		infos[2].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

		IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
			{.dstSet = m_rayTracingDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
			{.dstSet = m_rayTracingDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
			{.dstSet = m_presentDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[2] },
		};
		m_device->updateDescriptorSets(std::span(writes), {});
#endif