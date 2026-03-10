// TODO: Copyright notice
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "nbl/examples/examples.hpp"

using namespace nbl::hlsl;
using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "DrawResourcesFiller.h"

#include "nbl/builtin/hlsl/math/linalg/transform.hlsl"
#include "nbl/builtin/hlsl/math/thin_lens_projection.hlsl"

class CEventCallback : public ISimpleManagedSurface::ICallback
{
public:
	CEventCallback(nbl::core::smart_refctd_ptr<nbl::examples::InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
	CEventCallback() {}

	void setLogger(nbl::system::logger_opt_smart_ptr& logger)
	{
		m_logger = logger;
	}
	void setInputSystem(nbl::core::smart_refctd_ptr<nbl::examples::InputSystem>&& m_inputSystem)
	{
		m_inputSystem = std::move(m_inputSystem);
	}
private:

	void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
	}
	void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
	}

private:
	nbl::core::smart_refctd_ptr<nbl::examples::InputSystem> m_inputSystem = nullptr;
	nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};

class CSwapchainResources : public ISimpleManagedSurface::ISwapchainResources
{
	public:
		CSwapchainResources() = default;

		inline E_FORMAT deduceRenderpassFormat(ISurface* surface, IPhysicalDevice* physDev)
		{
			ISwapchain::SCreationParams swapchainParams = {.surface=smart_refctd_ptr<ISurface>(surface), };
			// Need to choose a surface format
			if (!swapchainParams.deduceFormat(physDev, getPreferredFormats(), getPreferredEOTFs(), getPreferredColorPrimaries()))
				return EF_UNKNOWN;
			return swapchainParams.surfaceFormat.format;
		}

		// When needing to recreate the framebuffer, We need to have access to a renderpass compatible to renderpass used to render to the framebuffer
		inline void setCompatibleRenderpass(core::smart_refctd_ptr<IGPURenderpass> renderpass)
		{
			m_renderpass = renderpass;
		}

		inline IGPUFramebuffer* getFramebuffer(const uint8_t imageIx)
		{
			if (imageIx<m_framebuffers.size())
				return m_framebuffers[imageIx].get();
			return nullptr;
		}

	protected:
		virtual inline void invalidate_impl()
		{
			std::fill(m_framebuffers.begin(),m_framebuffers.end(),nullptr);
		}

		// For creating extra per-image or swapchain resourcesCollection you might need
		virtual inline bool onCreateSwapchain_impl(const uint8_t qFam)
		{
			auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

			const auto swapchain = getSwapchain();
			const auto count = swapchain->getImageCount();
			const auto& sharedParams = swapchain->getCreationParameters().sharedParams;
			for (uint8_t i=0u; i<count; i++)
			{
				auto imageView = device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = core::smart_refctd_ptr<IGPUImage>(getImage(i)),
					.viewType = IGPUImageView::ET_2D,
					.format = getImage(i)->getCreationParameters().format
				});
				m_framebuffers[i] = device->createFramebuffer({ {
					.renderpass = core::smart_refctd_ptr(m_renderpass),
					.colorAttachments = &imageView.get(),
					// TODO:
					//.depthStencilAttachments = &depthImageView.get(),
					.width = sharedParams.width,
					.height = sharedParams.height
				}});
				if (!m_framebuffers[i])
					return false;
			}
			return true;
		}

		// Per-swapchain
		core::smart_refctd_ptr<IGPURenderpass> m_renderpass;
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>,ISwapchain::MaxImages> m_framebuffers;
};

class ComputerAidedDesign final : public nbl::examples::SimpleWindowedApplication, public nbl::examples::BuiltinResourcesApplication
{
	using device_base_t = nbl::examples::SimpleWindowedApplication;
	using asset_base_t = nbl::examples::BuiltinResourcesApplication;
	using clock_t = std::chrono::steady_clock;
	
	constexpr static uint32_t WindowWidthRequest = 1600u;
	constexpr static uint32_t WindowHeightRequest = 900u;
	constexpr static uint32_t MaxFramesInFlight = 3u;
	constexpr static uint32_t MaxSubmitsInFlight = 16u;
public:

	void allocateResources()
	{
		// TODO: currently using the same utils for buffers and images, make them separate staging buffers
		drawResourcesFiller = DrawResourcesFiller(core::smart_refctd_ptr(m_device), core::smart_refctd_ptr(m_utils), getGraphicsQueue(), core::smart_refctd_ptr(m_logger));

		// Just wanting to try memory type indices with device local flag, TODO: later improve to prioritize pure device local
		std::vector<uint32_t> deviceLocalMemoryTypeIndices;
		for (uint32_t i = 0u; i < m_physicalDevice->getMemoryProperties().memoryTypeCount; ++i)
		{
			const auto& memType = m_physicalDevice->getMemoryProperties().memoryTypes[i];
			if (memType.propertyFlags.hasFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT))
				deviceLocalMemoryTypeIndices.push_back(i);
		}

		size_t maxImagesMemSize = 1024ull * 1024ull * 1024ull; // 1024 MB
		size_t maxBufferMemSize = 1024ull * 1024ull * 1024ull; // 1024 MB

		drawResourcesFiller.allocateDrawResourcesWithinAvailableVRAM(m_device.get(), maxImagesMemSize, maxBufferMemSize, deviceLocalMemoryTypeIndices);

		{
			IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = sizeof(Globals);
			globalsCreationParams.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			m_globalsBuffer = m_device->createBuffer(std::move(globalsCreationParams));

			IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = m_globalsBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = m_device->allocate(memReq, m_globalsBuffer.get());
		}
		
		// pseudoStencil
		{
			asset::E_FORMAT pseudoStencilFormat = asset::EF_R32_UINT;
			{
				IGPUImage::SCreationParams imgInfo;
				imgInfo.format = pseudoStencilFormat;
				imgInfo.type = IGPUImage::ET_2D;
				imgInfo.extent.width = m_window->getWidth();
				imgInfo.extent.height = m_window->getHeight();
				imgInfo.extent.depth = 1u;
				imgInfo.mipLevels = 1u;
				imgInfo.arrayLayers = 1u;
				imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
				imgInfo.flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE;
				imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
				// [VKTODO] imgInfo.initialLayout = IGPUImage::EL_UNDEFINED;
				imgInfo.tiling = IGPUImage::TILING::OPTIMAL;

				auto image = m_device->createImage(std::move(imgInfo));
				auto imageMemReqs = image->getMemoryReqs();
				imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				m_device->allocate(imageMemReqs, image.get());

				image->setObjectDebugName("pseudoStencil Image");

				IGPUImageView::SCreationParams imgViewInfo;
				imgViewInfo.image = std::move(image);
				imgViewInfo.format = pseudoStencilFormat;
				imgViewInfo.viewType = IGPUImageView::ET_2D;
				imgViewInfo.flags = IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
				imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				imgViewInfo.subresourceRange.baseArrayLayer = 0u;
				imgViewInfo.subresourceRange.baseMipLevel = 0u;
				imgViewInfo.subresourceRange.layerCount = 1u;
				imgViewInfo.subresourceRange.levelCount = 1u;

				pseudoStencilImageView = m_device->createImageView(std::move(imgViewInfo));
			}
		}
		
		// colorStorage
		{
			asset::E_FORMAT colorStorageFormat = asset::EF_R32_UINT;
			{
				IGPUImage::SCreationParams imgInfo;
				imgInfo.format = colorStorageFormat;
				imgInfo.type = IGPUImage::ET_2D;
				imgInfo.extent.width = m_window->getWidth();
				imgInfo.extent.height = m_window->getHeight();
				imgInfo.extent.depth = 1u;
				imgInfo.mipLevels = 1u;
				imgInfo.arrayLayers = 1u;
				imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
				imgInfo.flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE;
				imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
				// [VKTODO] imgInfo.initialLayout = IGPUImage::EL_UNDEFINED;
				imgInfo.tiling = IGPUImage::TILING::OPTIMAL;

				auto image = m_device->createImage(std::move(imgInfo));
				auto imageMemReqs = image->getMemoryReqs();
				imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				m_device->allocate(imageMemReqs, image.get());

				image->setObjectDebugName("colorStorage Image");

				IGPUImageView::SCreationParams imgViewInfo;
				imgViewInfo.image = std::move(image);
				imgViewInfo.format = colorStorageFormat;
				imgViewInfo.viewType = IGPUImageView::ET_2D;
				imgViewInfo.flags = IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
				imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				imgViewInfo.subresourceRange.baseArrayLayer = 0u;
				imgViewInfo.subresourceRange.baseMipLevel = 0u;
				imgViewInfo.subresourceRange.layerCount = 1u;
				imgViewInfo.subresourceRange.levelCount = 1u;

				colorStorageImageView = m_device->createImageView(std::move(imgViewInfo));
			}
		}

		// Initial Pipeline Transitions and Clearing of PseudoStencil and ColorStorage
		// Recorded to Temporary CommandBuffer, Submitted to Graphics Queue, and Blocked on here
		{
			auto cmdPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			smart_refctd_ptr<IGPUCommandBuffer> tmpCmdBuffer;
			cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &tmpCmdBuffer, 1 });
			auto tmpJobFinishedSema = m_device->createSemaphore(0ull);

			tmpCmdBuffer->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			{
				// Clear pseudoStencil
				auto pseudoStencilImage = pseudoStencilImageView->getCreationParameters().image;
				auto colorStorageImage = colorStorageImageView->getCreationParameters().image;

				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t beforeClearImageBarrier[] =
				{
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, // previous top of pipe -> top_of_pipe in first scope = none
								.srcAccessMask = ACCESS_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
								.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT, // could be ALL_TRANSFER but let's be specific we only want to CLEAR right now
							}
							// .ownershipOp. No queueFam ownership transfer
						},
						.image = pseudoStencilImage.get(),
						.subresourceRange = {
							.aspectMask = IImage::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = 1u,
							.baseArrayLayer = 0u,
							.layerCount = 1u,
						},
						.oldLayout = IImage::LAYOUT::UNDEFINED,
						.newLayout = IImage::LAYOUT::GENERAL,
					}
				};

				tmpCmdBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeClearImageBarrier });

				uint32_t pseudoStencilInvalidValue = core::bitfieldInsert<uint32_t>(0u, 16777215, 8, 24);
				IGPUCommandBuffer::SClearColorValue clear = {};
				clear.uint32[0] = pseudoStencilInvalidValue;

				asset::IImage::SSubresourceRange subresourceRange = {};
				subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				subresourceRange.baseArrayLayer = 0u;
				subresourceRange.baseMipLevel = 0u;
				subresourceRange.layerCount = 1u;
				subresourceRange.levelCount = 1u;

				tmpCmdBuffer->clearColorImage(pseudoStencilImage.get(), asset::IImage::LAYOUT::GENERAL, &clear, 1u, &subresourceRange);

				// prepare pseudoStencilImage for usage in drawcall

				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t beforeUsageImageBarriers[] =
				{
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS, // could be ALL_TRANSFER but let's be specific we only want to CLEAR right now
							}
							// .ownershipOp. No queueFam ownership transfer
						},
						.image = pseudoStencilImage.get(),
						.subresourceRange = {
							.aspectMask = IImage::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = 1u,
							.baseArrayLayer = 0u,
							.layerCount = 1u,
						},
						.oldLayout = IImage::LAYOUT::GENERAL,
						.newLayout = IImage::LAYOUT::GENERAL,
					}, 
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
								.srcAccessMask = ACCESS_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS, // could be ALL_TRANSFER but let's be specific we only want to CLEAR right now
							}
							// .ownershipOp. No queueFam ownership transfer
						},
						.image = colorStorageImage.get(),
						.subresourceRange = {
							.aspectMask = IImage::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = 1u,
							.baseArrayLayer = 0u,
							.layerCount = 1u,
						},
						.oldLayout = IImage::LAYOUT::UNDEFINED,
						.newLayout = IImage::LAYOUT::GENERAL,
					}
				};

				tmpCmdBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeUsageImageBarriers });
			}
			tmpCmdBuffer->end();

			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1u] = { {.cmdbuf = tmpCmdBuffer.get() } };
			IQueue::SSubmitInfo::SSemaphoreInfo singalSemaphores[1] = {};
			singalSemaphores[0].semaphore = tmpJobFinishedSema.get();
			singalSemaphores[0].stageMask = asset::PIPELINE_STAGE_FLAGS::NONE;
			singalSemaphores[0].value = 1u;

			IQueue::SSubmitInfo submitInfo = {};
			submitInfo.commandBuffers = cmdbufs;
			submitInfo.waitSemaphores = {};
			submitInfo.signalSemaphores = singalSemaphores;

			getGraphicsQueue()->submit({ &submitInfo, 1u });

			ISemaphore::SWaitInfo waitTmpJobFinish = { .semaphore = tmpJobFinishedSema.get(), .value = 1u};
			m_device->blockForSemaphores({ &waitTmpJobFinish, 1u });
		}
	}
	
	smart_refctd_ptr<IGPURenderpass> createRenderpass(
		E_FORMAT colorAttachmentFormat,
		IGPURenderpass::LOAD_OP loadOp,
		IImage::LAYOUT initialLayout,
		IImage::LAYOUT finalLayout)
	{		
		const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
			{{
				{
					.format = colorAttachmentFormat,
					.samples = IGPUImage::ESCF_1_BIT,
					.mayAlias = false
				},
				/*.loadOp = */loadOp,
				/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
				/*.initialLayout = */initialLayout,
				/*.finalLayout = */finalLayout
			}},
			IGPURenderpass::SCreationParams::ColorAttachmentsEnd
		};

		// TODO:
		//IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
		//	{{
		//		{
		//			.format = asset::EF_D32_SFLOAT,
		//			.samples = IGPUImage::ESCF_1_BIT,
		//			.mayAlias = false
		//		},
		//		/*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
		//		/*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
		//		/*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED},
		//		/*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
		//	}},
		//	IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
		//};

		IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
			{},
			IGPURenderpass::SCreationParams::SubpassesEnd
		};

		subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
		// TODO:
		//subpasses[0].depthStencilAttachment = {{.render = {.attachmentIndex=0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}};
		
		// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
		const IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
			// wipe-transition to ATTACHMENT_OPTIMAL
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = {
					// we can have NONE as Sources because ????
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
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
		
		smart_refctd_ptr<IGPURenderpass> renderpass;
		IGPURenderpass::SCreationParams params = {};
		params.colorAttachments = colorAttachments;
		// TODO:
		//params.depthStencilAttachments = depthAttachments;
		params.subpasses = subpasses;
		params.dependencies = dependencies;
		renderpass = m_device->createRenderpass(params);
		if (!renderpass)
			logFail("Failed to Create a Renderpass!");
		return renderpass;
	}


	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	inline ComputerAidedDesign(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}
	
	// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		// So let's create our Window and Surface then!
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params = {};
				params.callback = windowCallback;
				params.width = WindowWidthRequest;
				params.height = WindowHeightRequest;
				params.x = 32;
				params.y = 32;
				// Don't want to have a window lingering about before we're ready so create it hidden.
				// Only programmatic resize, not regular.
				params.flags = IWindow::ECF_BORDERLESS|IWindow::ECF_RESIZABLE;
				params.windowCaption = "CAD 3D Playground";
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}
			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<CSwapchainResources>::create(std::move(surface));
		}
		if (m_surface)
			return {{m_surface->getSurface()/*,EQF_NONE*/}};
		return {};
	}
	
	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<nbl::examples::InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;
		
		// Let's just use the same queue since there's no need for async present
		if (!m_surface)
			return logFail("Could not create Window & Surface!");
		
		auto scResources = std::make_unique<CSwapchainResources>();
		const auto format = scResources->deduceRenderpassFormat(m_surface->getSurface(), m_physicalDevice); // TODO: DO I need to recreate render passes if swapchain gets recreated with different format?
		renderpassInitial = createRenderpass(format, IGPURenderpass::LOAD_OP::CLEAR, IImage::LAYOUT::UNDEFINED, IImage::LAYOUT::ATTACHMENT_OPTIMAL);
		renderpassInBetween = createRenderpass(format, IGPURenderpass::LOAD_OP::LOAD, IImage::LAYOUT::ATTACHMENT_OPTIMAL, IImage::LAYOUT::ATTACHMENT_OPTIMAL);
		renderpassFinal = createRenderpass(format, IGPURenderpass::LOAD_OP::LOAD, IImage::LAYOUT::ATTACHMENT_OPTIMAL, IImage::LAYOUT::PRESENT_SRC);
		const auto compatibleRenderPass = renderpassInitial; // all 3 above are compatible

		scResources->setCompatibleRenderpass(compatibleRenderPass);

		if (!m_surface->init(getGraphicsQueue(),std::move(scResources),{}))
			return logFail("Could not initialize the Surface!");

		allocateResources();

		const asset::SPushConstantRange range = {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.offset = 0,
			.size = sizeof(PushConstants)
		};

		m_pipelineLayout = m_device->createPipelineLayout({ &range,1 }, nullptr, nullptr, nullptr, nullptr);

		smart_refctd_ptr<IShader> mainPipelineFragmentShaders = {};
		smart_refctd_ptr<IShader> mainPipelineVertexShader = {};
		{
			// Load Custom Shader
			auto loadPrecompiledShader = [&]<core::StringLiteral ShaderKey>() -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "app_resources";

				auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
				auto assetBundle = m_assetMgr->getAsset(key.data(), lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
				{
					m_logger->log("Failed to load a precompiled shader of key \"%s\".", ILogger::ELL_ERROR, ShaderKey);
					return nullptr;
				}
					

				auto shader = IAsset::castDown<IShader>(assets[0]);
				return shader;
			};

			mainPipelineFragmentShaders = loadPrecompiledShader.operator()<"main_pipeline_fragment_shader">(); // "../shaders/main_pipeline/fragment_shader.hlsl"
			mainPipelineVertexShader = loadPrecompiledShader.operator()<"main_pipeline_vertex_shader">(); // "../shaders/main_pipeline/vertex_shader.hlsl"
		}

		IGPUGraphicsPipeline::SCreationParams mainGraphicsPipelineParams = {};
		mainGraphicsPipelineParams.layout = m_pipelineLayout.get();
		mainGraphicsPipelineParams.cached = {
			.vertexInput = {},
			.primitiveAssembly = {
				.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST,
			},
			.rasterization = {
				.polygonMode = EPM_FILL,
				.faceCullingMode = EFCM_NONE,
				.depthWriteEnable = false,
			},
			.blend = {},
		};
		mainGraphicsPipelineParams.renderpass = compatibleRenderPass.get();

		// Create Main Graphics Pipelines 
		{
			video::IGPUPipelineBase::SShaderSpecInfo specInfo[2] = {
				{ .shader = mainPipelineVertexShader.get(), .entryPoint = "vtxMain" },
				{ .shader = mainPipelineFragmentShaders.get(), .entryPoint = "fragMain" },
			};
			
			IGPUGraphicsPipeline::SCreationParams params[1] = { mainGraphicsPipelineParams };
			params[0].vertexShader = specInfo[0];
			params[0].fragmentShader = specInfo[1];

			if (!m_device->createGraphicsPipelines(nullptr,params,&m_graphicsPipeline))
				return logFail("Graphics Pipeline Creation Failed.");
		}
		
		// Create the commandbuffers and pools, this time properly 1 pool per FIF
		m_graphicsCommandPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		if (!m_graphicsCommandPool)
			return logFail("Couldn't create Command Pool!");
		if (!m_graphicsCommandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_commandBuffersInFlight.data(),MaxSubmitsInFlight}))
			return logFail("Couldn't create Command Buffers!");
		
		// Create the Semaphores
		m_renderSemaphore = m_device->createSemaphore(0ull);
		m_renderSemaphore->setObjectDebugName("m_renderSemaphore");
		m_overflowSubmitScratchSemaphore = m_device->createSemaphore(0ull);
		m_overflowSubmitScratchSemaphore->setObjectDebugName("m_overflowSubmitScratchSemaphore");
		if (!m_renderSemaphore || !m_overflowSubmitScratchSemaphore)
			return logFail("Failed to Create Semaphores!");

		// Set Queue and ScratchSemaInfo -> wait semaphores and command buffers will be modified by workLoop each frame
		m_intendedNextSubmit.queue = getGraphicsQueue();
		m_intendedNextSubmit.scratchSemaphore = {
				.semaphore = m_overflowSubmitScratchSemaphore.get(),
				.value = 0ull,
		};
		for (uint32_t i = 0; i < MaxSubmitsInFlight; ++i)
			m_commandBufferInfos[i] = { .cmdbuf = m_commandBuffersInFlight[i].get() };
		m_intendedNextSubmit.scratchCommandBuffers = m_commandBufferInfos;
		m_currentRecordingCommandBufferInfo = &m_commandBufferInfos[0];
		
		return true;
	}

	// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
	inline void workLoopBody() override
	{
		auto now = std::chrono::high_resolution_clock::now();
		double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
		lastTime = now;
		m_timeElapsed += dt;

		m_inputSystem->getDefaultMouse(&mouse);
		m_inputSystem->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
			}
		, m_logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
				{
				}
			}
		, m_logger.get());

		if (!beginFrameRender())
			return;

		const IQueue::SSubmitInfo::SSemaphoreInfo acquired = {
			.semaphore = m_currentImageAcquire.semaphore,
			.value = m_currentImageAcquire.acquireCount,
			.stageMask = asset::PIPELINE_STAGE_FLAGS::NONE // NONE for Acquire, right? Yes, the Spec Says so!
		};

		// prev frame done using the scene data (is in post process stage)
		const IQueue::SSubmitInfo::SSemaphoreInfo prevFrameRendered = {
			.semaphore = m_renderSemaphore.get(),
			.value = m_realFrameIx,
			.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
		};

		IQueue::SSubmitInfo::SSemaphoreInfo waitSems[2u] = { acquired, prevFrameRendered };
		m_intendedNextSubmit.waitSemaphores = waitSems;

		addObjects(m_intendedNextSubmit);

		endFrameRender(m_intendedNextSubmit);
	}
	
	bool beginFrameRender()
	{
		// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
		const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
		// We block for semaphores for 2 reasons here:
			// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
			// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
		if (m_realFrameIx>=framesInFlight)
		{
			const ISemaphore::SWaitInfo cmdbufDonePending[] = {
				{ 
					.semaphore = m_renderSemaphore.get(),
					.value = m_realFrameIx+1-framesInFlight
				}
			};
			if (m_device->blockForSemaphores(cmdbufDonePending)!=ISemaphore::WAIT_RESULT::SUCCESS)
				return false;
		}

		// Acquire
		m_currentImageAcquire = m_surface->acquireNextImage();
		if (!m_currentImageAcquire)
			return false;
		
		const bool beganSuccess = m_intendedNextSubmit.beginNextCommandBuffer(m_currentRecordingCommandBufferInfo);
		assert(beganSuccess);
		auto* cb = m_currentRecordingCommandBufferInfo->cmdbuf;

		// safe to proceed
		// no need to reset and begin new command buffers as SIntendedSubmitInfo already handled that.
		// cb->reset(video::IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		// cb->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cb->beginDebugMarker("Frame");
		
		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		auto scRes = static_cast<CSwapchainResources*>(m_surface->getSwapchainResources());
		const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.68f, 0.85f, 0.90f, 1.0f} };
		{
			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			beginInfo = {
				.renderpass = renderpassInitial.get(),
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};
		}

		cb->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		cb->endRenderPass();

		return true;
	}
	
	void submitDraws(SIntendedSubmitInfo& intendedSubmitInfo, bool inBetweenSubmit)
	{
		drawResourcesFiller.pushAllUploads(intendedSubmitInfo);

		m_currentRecordingCommandBufferInfo = intendedSubmitInfo.getCommandBufferForRecording(); // drawResourcesFiller.pushAllUploads might've overflow submitted and changed the current recording command buffer

		// Use the current recording command buffer of the intendedSubmitInfos scratchCommandBuffers, it should be in recording state
		auto* cb = m_currentRecordingCommandBufferInfo->cmdbuf;
		
		const auto& resourcesCollection = drawResourcesFiller.getResourcesCollection();
		const auto& resourcesGPUBuffer = drawResourcesFiller.getResourcesGPUBuffer();

		float64_t4x4 viewProjection;
		{
			// TODO: create a proper camera

			// animated camera which rotates around and always looks at the center
			const double animationFactor = m_timeElapsed * 0.0003;
			const float32_t3 cameraPosition = { 300.0f * std::cos(animationFactor), 300.0f, 300.0f * std::sin(animationFactor) };

			auto view = hlsl::math::linalg::rhLookAt<float64_t>(cameraPosition, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
			const float64_t aspectRatio = static_cast<float64_t>(m_window->getWidth()) / static_cast<float64_t>(m_window->getHeight());
			auto proj = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float64_t>(hlsl::radians(60.0f), aspectRatio, 0.1f, 2000.0f);

			viewProjection = hlsl::mul(proj, nbl::hlsl::math::linalg::promote_affine<4, 4>(view));
		}

		Globals globalData = {};
		uint64_t baseAddress = resourcesGPUBuffer->getDeviceAddress();
		globalData.pointers = {
			.drawObjects			= baseAddress + resourcesCollection.drawObjects.bufferOffset,
			.geometryBuffer			= baseAddress + resourcesCollection.geometryInfo.bufferOffset,
		};
		SBufferRange<IGPUBuffer> globalBufferUpdateRange = { .offset = 0ull, .size = sizeof(Globals), .buffer = m_globalsBuffer};
		bool updateSuccess = cb->updateBuffer(globalBufferUpdateRange, &globalData);
		assert(updateSuccess);

		asset::SViewport vp =
		{
			.x = 0u,
			.y = 0u,
			.width = static_cast<float>(m_window->getWidth()),
			.height = static_cast<float>(m_window->getHeight()),
			.minDepth = 1.f,
			.maxDepth = 0.f,
		};
		cb->setViewport(0u, 1u, &vp);

		VkRect2D scissor =
		{
			.offset = { 0, 0 },
			.extent = { m_window->getWidth(), m_window->getHeight() },
		};
		cb->setScissor(0u, 1u, &scissor);

		// pipelineBarriersBeforeDraw
		{	
			constexpr uint32_t MaxBufferBarriersCount = 2u;
			uint32_t bufferBarriersCount = 0u;
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t bufferBarriers[MaxBufferBarriersCount];
			
			const auto& resourcesCollection = drawResourcesFiller.getResourcesCollection();

			if (m_globalsBuffer->getSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT | PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::UNIFORM_READ_BIT;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = m_globalsBuffer->getSize(),
					.buffer = m_globalsBuffer,
				};
			}
			if (drawResourcesFiller.getCopiedResourcesSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_INPUT_BITS | PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT | PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = drawResourcesFiller.getCopiedResourcesSize(),
					.buffer = drawResourcesFiller.getResourcesGPUBuffer(),
				};
			}
			cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .bufBarriers = {bufferBarriers, bufferBarriersCount}, .imgBarriers = {} });
		}

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		VkRect2D currentRenderArea;
		const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,0.f} };
		{
			auto scRes = static_cast<CSwapchainResources*>(m_surface->getSwapchainResources());
			currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};
			beginInfo = {
				.renderpass = (inBetweenSubmit) ? renderpassInBetween.get():renderpassFinal.get(),
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};
		}
		cb->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		
		cb->bindGraphicsPipeline(m_graphicsPipeline.get());

		for (auto& drawCall : drawResourcesFiller.getDrawCalls())
		{
			cb->bindIndexBuffer({ .offset = resourcesCollection.geometryInfo.bufferOffset + drawCall.indexBufferOffset, .buffer = drawResourcesFiller.getResourcesGPUBuffer()}, asset::EIT_32BIT);

			PushConstants pc = {
				.triangleMeshVerticesBaseAddress = drawCall.triangleMeshVerticesBaseAddress + resourcesGPUBuffer->getDeviceAddress() + resourcesCollection.geometryInfo.bufferOffset,
				.triangleMeshMainObjectIndex = drawCall.triangleMeshMainObjectIndex,
				.viewProjectionMatrix = viewProjection
			};
			cb->pushConstants(m_graphicsPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0, sizeof(PushConstants), &pc);

			cb->drawIndexed(drawCall.indexCount, 1u, 0u, 0u, 0u);
		}

		cb->endRenderPass();

		if (!inBetweenSubmit)
			cb->endDebugMarker();
		
		drawResourcesFiller.markFrameUsageComplete(intendedSubmitInfo.getFutureScratchSemaphore().value);

		if (inBetweenSubmit)
		{
			if (intendedSubmitInfo.overflowSubmit(m_currentRecordingCommandBufferInfo) != IQueue::RESULT::SUCCESS)
			{
				m_logger->log("overflow submit failed.", ILogger::ELL_ERROR);
			}
		}
		else
		{
			// cb->end();
			
			const auto nextFrameIx = m_realFrameIx+1u;
			const IQueue::SSubmitInfo::SSemaphoreInfo thisFrameRendered = {
				.semaphore = m_renderSemaphore.get(),
				.value = nextFrameIx,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
			};
			if (intendedSubmitInfo.submit(m_currentRecordingCommandBufferInfo, { &thisFrameRendered,1 }) == IQueue::RESULT::SUCCESS)
			{
				m_realFrameIx = nextFrameIx;
				
				IQueue::SSubmitInfo::SSemaphoreInfo presentWait = thisFrameRendered;
				// the stages for a wait semaphore operation are about what stage you WAIT in, not what stage you wait for
				presentWait.stageMask = PIPELINE_STAGE_FLAGS::NONE; // top of pipe, there's no explicit presentation engine stage
				m_surface->present(m_currentImageAcquire.imageIndex,{&presentWait,1});
			}
			else
			{
				m_logger->log("regular submit failed.", ILogger::ELL_ERROR);
			}
		}
	}

	void endFrameRender(SIntendedSubmitInfo& intendedSubmitInfo)
	{
		submitDraws(intendedSubmitInfo, false);
	}

	inline bool keepRunning() override
	{
		if (duration_cast<decltype(timeout)>(clock_t::now()-start)>timeout)
			return false;

		return m_surface && !m_surface->irrecoverable();
	}

	virtual bool onAppTerminated() override
	{
		m_currentRecordingCommandBufferInfo->cmdbuf->end();

		// We actually want to wait for all the frames to finish rendering, otherwise our destructors will run out of order late
		m_device->waitIdle();

		// This is optional, but the window would close AFTER we return from this function
		m_surface = nullptr;
		
		return device_base_t::onAppTerminated();
	}
		
	virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable() override
	{
		auto retval = base_t::getAPIFeaturesToEnable();
		// We only support one swapchain mode, surface, the other one is Display which we have not implemented yet.
		retval.swapchainMode = video::E_SWAPCHAIN_MODE::ESM_SURFACE;
		retval.validations = true;
		retval.synchronizationValidation = false;
		return retval;
	}

protected:

	void addObjects(SIntendedSubmitInfo& intendedNextSubmit)
	{
		drawResourcesFiller.setSubmitDrawsFunction(
			[&](SIntendedSubmitInfo& intendedNextSubmit)
			{
				return submitDraws(intendedNextSubmit, true);
			}
		);
		drawResourcesFiller.reset();

		core::vector<TriangleMeshVertex> vertices = {
			{ float64_t3(0.0, 100.0, 0.0) },
			{ float64_t3(-200.0, 10.0, -200.0) },
			{ float64_t3(200.0, 10.0, -100.0) },
			{ float64_t3(0.0, 100.0, 0.0) },
			{ float64_t3(200.0, 10.0, -100.0) },
			{ float64_t3(200.0, -20.0, 200.0) },
			{ float64_t3(0.0, 100.0, 0.0) },
			{ float64_t3(200.0, -20.0, 200.0) },
			{ float64_t3(-200.0, 10.0, 200.0) },
			{ float64_t3(0.0, 100.0, 0.0) },
			{ float64_t3(-200.0, 10.0, 200.0) },
			{ float64_t3(-200.0, 10.0, -200.0) },
		};

		core::vector<uint32_t> indices = {
			0, 1, 2,
			3, 4, 5,
			6, 7, 8,
			9, 10, 11
		};

		CTriangleMesh mesh;
		mesh.setVertices(core::vector<TriangleMeshVertex>(vertices));
		mesh.setIndices(std::move(indices));

		// pyramid A
		drawResourcesFiller.drawTriangleMesh(mesh, intendedNextSubmit);

		// pyramid B
		float64_t3 offset = { 500.0f, 0.0f, 0.0f };
		for (auto& vertex : vertices)
			vertex.pos += offset;
		mesh.setVertices(std::move(vertices));
		drawResourcesFiller.drawTriangleMesh(mesh, intendedNextSubmit);
	}

protected:
	clock_t::time_point start; // TODO: am i missing somehting? why is it never initialized
	std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);

	double m_timeElapsed = 0.0;
	std::chrono::steady_clock::time_point lastTime;

	core::smart_refctd_ptr<nbl::examples::InputSystem> m_inputSystem;
	nbl::examples::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	nbl::examples::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	smart_refctd_ptr<IGPURenderpass> renderpassInitial; // this renderpass will clear the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	smart_refctd_ptr<IGPURenderpass> renderpassInBetween; // this renderpass will load the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	smart_refctd_ptr<IGPURenderpass> renderpassFinal; // this renderpass will load the attachment and transition it to PRESENT
	
	smart_refctd_ptr<IGPUCommandPool> m_graphicsCommandPool;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>,	MaxSubmitsInFlight>	m_commandBuffersInFlight; 
	// ref to above cmd buffers, these go into SIntendedSubmitInfo as command buffers available for recording.
	std::array<IQueue::SSubmitInfo::SCommandBufferInfo,	MaxSubmitsInFlight>	m_commandBufferInfos;
	// pointer to one of the command buffer infos from above, this is the only command buffer used to record current submit in current frame, it will be updated by SIntendedSubmitInfo
	IQueue::SSubmitInfo::SCommandBufferInfo const * m_currentRecordingCommandBufferInfo; // pointer can change, value cannot

	smart_refctd_ptr<IGPUBuffer> m_globalsBuffer;
	DrawResourcesFiller drawResourcesFiller; // you can think of this as the scene data needed to draw everything, we only have one instance so let's use a timeline semaphore to sync all renders

	smart_refctd_ptr<ISemaphore> m_renderSemaphore; // timeline semaphore to sync frames together
	
	// timeline semaphore used for overflows (they need to be on their own timeline to count overflows)
	smart_refctd_ptr<ISemaphore> m_overflowSubmitScratchSemaphore; 
	SIntendedSubmitInfo m_intendedNextSubmit;
	
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	uint64_t m_realFrameIx = 0u;

	smart_refctd_ptr<IGPUPipelineLayout> m_pipelineLayout;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_graphicsPipeline;

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CSwapchainResources>> m_surface;
	smart_refctd_ptr<IGPUImageView> pseudoStencilImageView;
	smart_refctd_ptr<IGPUImageView> colorStorageImageView;
};

NBL_MAIN_FUNC(ComputerAidedDesign)

