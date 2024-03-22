

#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"
#include "nbl/video/utilities/CSimpleResizeSurface.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include "curves.h"
#include "Hatch.h"
#include "Polyline.h"
#include "DrawBuffers.h"

#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

static constexpr bool DebugMode = false;
static constexpr bool DebugRotatingViewProj = false;
static constexpr bool FragmentShaderPixelInterlock = true;

enum class ExampleMode
{
	CASE_0, // Simple Line, Camera Zoom In/Out
	CASE_1,	// Overdraw Fragment Shader Stress Test
	CASE_2, // hatches
	CASE_3, // CURVES AND LINES
	CASE_4, // STIPPLE PATTERN
	CASE_5, // Advanced Styling
};

constexpr ExampleMode mode = ExampleMode::CASE_5;

using namespace nbl::hlsl;
using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class Camera2D
{
public:
	Camera2D()
	{}

	void setOrigin(const float64_t2& origin)
	{
		m_origin = origin;
	}

	void setAspectRatio(const double& aspectRatio)
	{
		m_aspectRatio = aspectRatio;
	}

	void setSize(const double size)
	{
		m_bounds = float64_t2{ size * m_aspectRatio, size };
	}

	float64_t2 getBounds() const
	{
		return m_bounds;
	}

	float64_t3x3 constructViewProjection()
	{
		auto ret = float64_t3x3();

		ret[0][0] = 2.0 / m_bounds.x;
		ret[1][1] = -2.0 / m_bounds.y;
		ret[2][2] = 1.0;
		
		ret[0][2] = (-2.0 * m_origin.x) / m_bounds.x;
		ret[1][2] = (2.0 * m_origin.y) / m_bounds.y;

		return ret;
	}

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
			{
				m_bounds = m_bounds + float64_t2{ (double)ev.scrollEvent.verticalScroll * -0.1 * m_aspectRatio, (double)ev.scrollEvent.verticalScroll * -0.1};
				m_bounds = float64_t2{ core::max(m_aspectRatio, m_bounds.x), core::max(1.0, m_bounds.y) };
			}
		}
	}

	void keyboardProcess(const IKeyboardEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_W)
			{
				m_origin.y += m_bounds.y / 100.0;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_A)
			{
				m_origin.x -= m_bounds.x / 100.0;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_S)
			{
				m_origin.y -= m_bounds.y / 100.0;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_D)
			{
				m_origin.x += m_bounds.x / 100.0;
			}
		}
	}
private:

	double m_aspectRatio = 0.0;
	float64_t2 m_bounds = {};
	float64_t2 m_origin = {};
};

class CEventCallback : public ISimpleManagedSurface::ICallback
{
public:
	CEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)){}
	CEventCallback() {}
	
	void setLogger(nbl::system::logger_opt_smart_ptr& logger)
	{
		m_logger = logger;
	}
	void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
	{
		m_inputSystem = std::move(m_inputSystem);
	}
private:
		
	void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse,std::move(mch));
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse,mch);
	}
	void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard,std::move(kbch));
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard,kbch);
	}

private:
	nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
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

		inline IGPUFramebuffer* getFrambuffer(const uint8_t imageIx)
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

		// For creating extra per-image or swapchain resources you might need
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
				m_framebuffers[i] = device->createFramebuffer({{
					.renderpass = core::smart_refctd_ptr(m_renderpass),
					.colorAttachments = &imageView.get(),
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

class ComputerAidedDesign final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;
	
	constexpr static uint32_t WindowWidthRequest = 1600u;
	constexpr static uint32_t WindowHeightRequest = 900u;
	constexpr static uint32_t MaxFramesInFlight = 8u;
public:
	
	void initCADResources(uint32_t maxObjects)
	{
		drawBuffer = DrawBuffersFiller(core::smart_refctd_ptr(m_utils));

		uint32_t maxIndices = maxObjects * 6u * 2u;
		drawBuffer.allocateIndexBuffer(m_device.get(), maxIndices);
		drawBuffer.allocateMainObjectsBuffer(m_device.get(), maxObjects);
		drawBuffer.allocateDrawObjectsBuffer(m_device.get(), maxObjects * 5u);
		drawBuffer.allocateStylesBuffer(m_device.get(), 32u);
		drawBuffer.allocateCustomClipProjectionBuffer(m_device.get(), 128u);

		// * 3 because I just assume there is on average 3x beziers per actual object (cause we approximate other curves/arcs with beziers now)
		size_t geometryBufferSize = maxObjects * sizeof(QuadraticBezierInfo) * 3;
		drawBuffer.allocateGeometryBuffer(m_device.get(), geometryBufferSize);

		for (uint32_t i = 0; i < m_framesInFlight; ++i)
		{
			IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = sizeof(Globals);
			globalsCreationParams.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			globalsBuffer[i] = m_device->createBuffer(std::move(globalsCreationParams));

			IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer[i]->getMemoryReqs();
			memReq.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = m_device->allocate(memReq, globalsBuffer[i].get());
		}

		// pseudoStencil
		asset::E_FORMAT pseudoStencilFormat = asset::EF_R32_UINT;

		IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
		promotionRequest.originalFormat = asset::EF_R32_UINT;
		promotionRequest.usages = {};
		promotionRequest.usages.storageImageAtomic = true;
		pseudoStencilFormat = m_physicalDevice->promoteImageFormat(promotionRequest, IGPUImage::TILING::OPTIMAL);

		for (uint32_t i = 0u; i < m_framesInFlight; ++i)
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

			pseudoStencilImageViews[i] = m_device->createImageView(std::move(imgViewInfo));
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
		
		IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
			{},
			IGPURenderpass::SCreationParams::SubpassesEnd
		};
		subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
		
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
				params.windowCaption = "CAD Playground";
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}
			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<CSwapchainResources>::create(std::move(surface));
		}
		if (m_surface)
			return {{m_surface->getSurface()/*,EQF_NONE*/}};
		return {};
	}
	
	double dt = 0;
	double m_timeElapsed = 0.0;
	std::chrono::steady_clock::time_point lastTime;
	uint32_t m_hatchDebugStep = 0u;

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		fragmentShaderInterlockEnabled = m_device->getEnabledFeatures().fragmentShaderPixelInterlock;
		
		// Create the Semaphores
		m_renderSemaphore = m_device->createSemaphore(0ull);
		m_overflowSubmitScratchSemaphore = m_device->createSemaphore(0ull);
		if (!m_renderSemaphore || !m_overflowSubmitScratchSemaphore)
			return logFail("Failed to Create Semaphores!");

		m_overflowSubmitsScratchSemaphoreInfo.semaphore = m_overflowSubmitScratchSemaphore.get();
		m_overflowSubmitsScratchSemaphoreInfo.value = 1ull;

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

		m_framesInFlight = min(m_surface->getMaxFramesInFlight(), MaxFramesInFlight);
		
		// Shaders
		std::array<smart_refctd_ptr<IGPUShader>, 4u> shaders = {};
		{
			constexpr auto vertexShaderPath = "../vertex_shader.hlsl";
			constexpr auto fragmentShaderPath = "../fragment_shader.hlsl";
			constexpr auto debugfragmentShaderPath = "../fragment_shader_debug.hlsl";
			constexpr auto resolveAlphasShaderPath = "../resolve_alphas.hlsl";
			
			// Load Custom Shader
			auto loadCompileAndCreateShader = [&](const std::string& relPath, IShader::E_SHADER_STAGE stage) -> smart_refctd_ptr<IGPUShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset(relPath,lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return nullptr;

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto cpuShader = IAsset::castDown<ICPUShader>(assets[0]);
				cpuShader->setShaderStage(stage);
				if (!cpuShader)
					return nullptr;

				return m_device->createShader(cpuShader.get());
			};
			shaders[0] = loadCompileAndCreateShader(vertexShaderPath, IShader::ESS_VERTEX);
			shaders[1] = loadCompileAndCreateShader(fragmentShaderPath, IShader::ESS_FRAGMENT);
			shaders[2] = loadCompileAndCreateShader(debugfragmentShaderPath, IShader::ESS_FRAGMENT);
			shaders[3] = loadCompileAndCreateShader(resolveAlphasShaderPath, IShader::ESS_FRAGMENT);

			initCADResources(40960u);
		}

		// Create DescriptorSetLayout, PipelineLayout and update DescriptorSets
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[] = {
				{
					.binding = 0u,
					.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 1u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 2u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 3u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 4u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 5u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::ESS_VERTEX,
					.count = 1u,
				},
			};
			descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
			if (!descriptorSetLayout)
				return logFail("Failed to Create Descriptor Layout");

			smart_refctd_ptr<IDescriptorPool> descriptorPool = nullptr;
			{
				const uint32_t setCount = m_framesInFlight;
				descriptorPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE,{&descriptorSetLayout.get(),1},&setCount);
				if (!descriptorPool)
					return logFail("Failed to Create Descriptor Pool");
			}

			for (size_t i = 0; i < m_framesInFlight; i++)
			{
				descriptorSets[i] = descriptorPool->createDescriptorSet(smart_refctd_ptr(descriptorSetLayout));
				constexpr uint32_t DescriptorCount = 6u;
				video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[DescriptorCount] = {};
				descriptorInfos[0u].info.buffer.offset = 0u;
				descriptorInfos[0u].info.buffer.size = globalsBuffer[i]->getCreationParams().size;
				descriptorInfos[0u].desc = globalsBuffer[i];

				descriptorInfos[1u].info.buffer.offset = 0u;
				descriptorInfos[1u].info.buffer.size = drawBuffer.gpuDrawBuffers.drawObjectsBuffer->getCreationParams().size;
				descriptorInfos[1u].desc = drawBuffer.gpuDrawBuffers.drawObjectsBuffer;

				descriptorInfos[2u].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				descriptorInfos[2u].info.image.sampler = nullptr;
				descriptorInfos[2u].desc = pseudoStencilImageViews[i];

				descriptorInfos[3u].info.buffer.offset = 0u;
				descriptorInfos[3u].info.buffer.size = drawBuffer.gpuDrawBuffers.lineStylesBuffer->getCreationParams().size;
				descriptorInfos[3u].desc = drawBuffer.gpuDrawBuffers.lineStylesBuffer;

				descriptorInfos[4u].info.buffer.offset = 0u;
				descriptorInfos[4u].info.buffer.size = drawBuffer.gpuDrawBuffers.mainObjectsBuffer->getCreationParams().size;
				descriptorInfos[4u].desc = drawBuffer.gpuDrawBuffers.mainObjectsBuffer;

				descriptorInfos[5u].info.buffer.offset = 0u;
				descriptorInfos[5u].info.buffer.size = drawBuffer.gpuDrawBuffers.customClipProjectionBuffer->getCreationParams().size;
				descriptorInfos[5u].desc = drawBuffer.gpuDrawBuffers.customClipProjectionBuffer;

				video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[6u] = {};
				descriptorUpdates[0u].dstSet = descriptorSets[i].get();
				descriptorUpdates[0u].binding = 0u;
				descriptorUpdates[0u].arrayElement = 0u;
				descriptorUpdates[0u].count = 1u;
				descriptorUpdates[0u].info = &descriptorInfos[0u];

				descriptorUpdates[1u].dstSet = descriptorSets[i].get();
				descriptorUpdates[1u].binding = 1u;
				descriptorUpdates[1u].arrayElement = 0u;
				descriptorUpdates[1u].count = 1u;
				descriptorUpdates[1u].info = &descriptorInfos[1u];

				descriptorUpdates[2u].dstSet = descriptorSets[i].get();
				descriptorUpdates[2u].binding = 2u;
				descriptorUpdates[2u].arrayElement = 0u;
				descriptorUpdates[2u].count = 1u;
				descriptorUpdates[2u].info = &descriptorInfos[2u];

				descriptorUpdates[3u].dstSet = descriptorSets[i].get();
				descriptorUpdates[3u].binding = 3u;
				descriptorUpdates[3u].arrayElement = 0u;
				descriptorUpdates[3u].count = 1u;
				descriptorUpdates[3u].info = &descriptorInfos[3u];

				descriptorUpdates[4u].dstSet = descriptorSets[i].get();
				descriptorUpdates[4u].binding = 4u;
				descriptorUpdates[4u].arrayElement = 0u;
				descriptorUpdates[4u].count = 1u;
				descriptorUpdates[4u].info = &descriptorInfos[4u];

				descriptorUpdates[5u].dstSet = descriptorSets[i].get();
				descriptorUpdates[5u].binding = 5u;
				descriptorUpdates[5u].arrayElement = 0u;
				descriptorUpdates[5u].count = 1u;
				descriptorUpdates[5u].info = &descriptorInfos[5u];

				m_device->updateDescriptorSets(DescriptorCount, descriptorUpdates, 0u, nullptr);
			}

			graphicsPipelineLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);
		}
		
		// Shared Blend Params between pipelines
		SBlendParams blendParams = {};
		blendParams.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
		blendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		blendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
		blendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;
		
		// Create Alpha Resovle Pipeline
		{
			// Load FSTri Shader
			ext::FullScreenTriangle::ProtoPipeline fsTriangleProtoPipe(m_assetMgr.get(),m_device.get(),m_logger.get());
			
			const IGPUShader::SSpecInfo fragSpec = {
				.entryPoint = "main",
				.shader = shaders[3u].get()
			};

			resolveAlphaPipeLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);
			resolveAlphaGraphicsPipeline = fsTriangleProtoPipe.createPipeline(fragSpec, resolveAlphaPipeLayout.get(), compatibleRenderPass.get(), 0u, blendParams);
			if (!resolveAlphaGraphicsPipeline)
				return logFail("Graphics Pipeline Creation Failed.");

		}
		
		// Create Main Graphics Pipelines 
		{
			
			IGPUShader::SSpecInfo specInfo[2] = {
				{.shader=shaders[0u].get() },
				{.shader=shaders[1u].get() },
			};

			IGPUGraphicsPipeline::SCreationParams params[1] = {};
			params[0].layout = graphicsPipelineLayout.get();
			params[0].shaders = specInfo;
			params[0].cached = {
				.vertexInput = {},
				.primitiveAssembly = {
					.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST,
				},
				.rasterization = {
					.polygonMode = EPM_FILL,
					.faceCullingMode = EFCM_NONE,
					.depthWriteEnable = false,
				},
				.blend = blendParams,
			};
			params[0].renderpass = compatibleRenderPass.get();
			
			if (!m_device->createGraphicsPipelines(nullptr,params,&graphicsPipeline))
				return logFail("Graphics Pipeline Creation Failed.");

			if constexpr (DebugMode)
			{
				specInfo[1u].shader = shaders[2u].get(); // change only fragment shader to fragment_shader_debug.hlsl
				params[0].cached.rasterization.polygonMode = asset::EPM_LINE;
				
				if (!m_device->createGraphicsPipelines(nullptr,params,&debugGraphicsPipeline))
					return logFail("Debug Graphics Pipeline Creation Failed.");
			}
		}

		// Create the commandbuffers and pools, this time properly 1 pool per FIF
		for (auto i=0u; i<m_framesInFlight; i++)
		{
			// non-individually-resettable commandbuffers have an advantage over invidually-resettable
			// mainly that the pool can use a "cheaper", faster allocator internally
			m_graphicsCommandPools[i] = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!m_graphicsCommandPools[i])
				return logFail("Couldn't create Command Pool!");
			if (!m_graphicsCommandPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_commandBuffers.data()+i,1}))
				return logFail("Couldn't create Command Buffer!");
		}
		
		m_Camera.setOrigin({ 0.0, 0.0 });
		m_Camera.setAspectRatio((double)m_window->getWidth() / m_window->getHeight());
		m_Camera.setSize(10.0);
		if constexpr (mode == ExampleMode::CASE_2)
		{
			m_Camera.setSize(200.0);
		}

		m_timeElapsed = 0.0;
		
		return true;
	}

	// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
	inline void workLoopBody() override
	{
		const auto resourceIx = m_realFrameIx%m_framesInFlight;

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
		lastTime = now;
		m_timeElapsed += dt;
		if constexpr (mode == ExampleMode::CASE_0)
		{
			m_Camera.setSize(20.0 + abs(cos(m_timeElapsed * 0.001)) * 600);
		}

		m_inputSystem->getDefaultMouse(&mouse);
		m_inputSystem->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				m_Camera.mouseProcess(events);
			}
		, m_logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				m_Camera.keyboardProcess(events);

				for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
				{
					auto ev = *eventIt;

					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_E)
					{
						m_hatchDebugStep++;
					}
					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_Q)
					{
						m_hatchDebugStep--;
					}
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

		const IQueue::SSubmitInfo::SSemaphoreInfo thisFrameRendered = {
			.semaphore = m_renderSemaphore.get(),
			.value = m_realFrameIx + 1u,
			.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
		};

		IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1u] = { {.cmdbuf = m_commandBuffers[resourceIx].get() } };
		IQueue::SSubmitInfo::SSemaphoreInfo waitSems[2u] = { acquired, prevFrameRendered };
		IQueue::SSubmitInfo::SSemaphoreInfo singalSems[2u] = { m_overflowSubmitsScratchSemaphoreInfo, thisFrameRendered };

		SIntendedSubmitInfo intendedNextSubmit;
		intendedNextSubmit.frontHalf.queue = getGraphicsQueue();
		intendedNextSubmit.frontHalf.commandBuffers = cmdbufs;
		intendedNextSubmit.frontHalf.waitSemaphores = waitSems;
		intendedNextSubmit.signalSemaphores = singalSems;

		addObjects(intendedNextSubmit);
		
		endFrameRender(intendedNextSubmit);
	}
	
	bool beginFrameRender()
	{
		// Can't reset a cmdbuffer before the previous use of commandbuffer is finished!
		if (m_realFrameIx>=m_framesInFlight)
		{
			const ISemaphore::SWaitInfo cmdbufDonePending[] = {
				{ 
					.semaphore = m_renderSemaphore.get(),
					.value = m_realFrameIx+1-m_framesInFlight
				}
			};
			if (m_device->blockForSemaphores(cmdbufDonePending)!=ISemaphore::WAIT_RESULT::SUCCESS)
				return false;
		}
		
		// Acquire
		m_currentImageAcquire = m_surface->acquireNextImage();
		if (!m_currentImageAcquire)
			return false;

		const auto resourceIx = m_realFrameIx%m_framesInFlight;
		auto& cb = m_commandBuffers[resourceIx];
		auto& commandPool = m_graphicsCommandPools[resourceIx];
		
		// safe to proceed
		cb->reset(video::IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cb->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cb->beginDebugMarker("Frame");

		float64_t3x3 projectionToNDC;
		projectionToNDC = m_Camera.constructViewProjection();
		
		Globals globalData = {};
		globalData.antiAliasingFactor = 1.0;// +abs(cos(m_timeElapsed * 0.0008)) * 20.0f;
		globalData.resolution = uint32_t2{ m_window->getWidth(), m_window->getHeight() };
		globalData.defaultClipProjection.projectionToNDC = projectionToNDC;
		globalData.defaultClipProjection.minClipNDC = float32_t2(-1.0, -1.0);
		globalData.defaultClipProjection.maxClipNDC = float32_t2(+1.0, +1.0);
		auto screenToWorld = getScreenToWorldRatio(globalData.defaultClipProjection.projectionToNDC, globalData.resolution);
		globalData.screenToWorldRatio = screenToWorld;
		globalData.worldToScreenRatio = (1.0/screenToWorld);
		globalData.miterLimit = 10.0f;
		SBufferRange<IGPUBuffer> globalBufferUpdateRange = { .offset = 0ull, .size = sizeof(Globals), .buffer = globalsBuffer[resourceIx].get() };
		bool updateSuccess = cb->updateBuffer(globalBufferUpdateRange, &globalData);
		assert(updateSuccess);
		
		// Clear pseudoStencil
		{
			auto pseudoStencilImage = pseudoStencilImageViews[resourceIx]->getCreationParameters().image;

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[] =
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

			cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,  { .imgBarriers = imageBarriers  });

			uint32_t pseudoStencilInvalidValue = core::bitfieldInsert<uint32_t>(0u, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
			IGPUCommandBuffer::SClearColorValue clear = {};
			clear.uint32[0] = pseudoStencilInvalidValue;

			asset::IImage::SSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			subresourceRange.baseArrayLayer = 0u;
			subresourceRange.baseMipLevel = 0u;
			subresourceRange.layerCount = 1u;
			subresourceRange.levelCount = 1u;

			cb->clearColorImage(pseudoStencilImage.get(), asset::IImage::LAYOUT::GENERAL, &clear, 1u, &subresourceRange);
		}

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			auto scRes = static_cast<CSwapchainResources*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,0.f} };
			beginInfo = {
				.renderpass = renderpassInitial.get(),
				.framebuffer = scRes->getFrambuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};
		}

		// you could do this later but only use renderpassInitial on first draw
		cb->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		// Wait what's going on here? empty Renderpass!?
		cb->endRenderPass();

		return true;
	}
	
	void submitDraws(SIntendedSubmitInfo& intendedSubmitInfo, bool inBetweenSubmit)
	{
		const auto resourceIx = m_realFrameIx%m_framesInFlight;
		auto* cb = intendedSubmitInfo.frontHalf.getScratchCommandBuffer();
		auto&r = drawBuffer;
		
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
			// prepare pseudoStencilImage for usage in drawcall
			auto pseudoStencilImage = pseudoStencilImageViews[resourceIx]->getCreationParameters().image;
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[] =
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
				}
			};

			
			constexpr uint32_t MaxBufferBarriersCount = 6u;
			uint32_t bufferBarriersCount = 0u;
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t bufferBarriers[MaxBufferBarriersCount];

			// Index Buffer Copy Barrier -> Remove after Filling up the index buffer at init time
			if (drawBuffer.getCurrentIndexBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_INPUT_BITS;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::INDEX_READ_BIT;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = drawBuffer.getCurrentIndexBufferSize(),
					.buffer = drawBuffer.gpuDrawBuffers.indexBuffer,
				};
			}
			if (globalsBuffer[resourceIx]->getSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::UNIFORM_READ_BIT;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = globalsBuffer[resourceIx]->getSize(),
					.buffer = globalsBuffer[resourceIx],
				};
			}
			if (drawBuffer.getCurrentDrawObjectsBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = drawBuffer.getCurrentDrawObjectsBufferSize(),
					.buffer = drawBuffer.gpuDrawBuffers.drawObjectsBuffer,
				};
			}
			if (drawBuffer.getCurrentGeometryBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = drawBuffer.getCurrentGeometryBufferSize(),
					.buffer = drawBuffer.gpuDrawBuffers.geometryBuffer,
				};
			}
			if (drawBuffer.getCurrentLineStylesBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = drawBuffer.getCurrentLineStylesBufferSize(),
					.buffer = drawBuffer.gpuDrawBuffers.lineStylesBuffer,
				};
			}
			if (drawBuffer.getCurrentCustomClipProjectionBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = drawBuffer.getCurrentCustomClipProjectionBufferSize(),
					.buffer = drawBuffer.gpuDrawBuffers.customClipProjectionBuffer,
				};
			}
			cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .bufBarriers = {bufferBarriers, bufferBarriersCount}, .imgBarriers = imageBarriers });
		}

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			auto scRes = static_cast<CSwapchainResources*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,0.f} };
			beginInfo = {
				.renderpass = (inBetweenSubmit) ? renderpassInBetween.get():renderpassFinal.get(),
				.framebuffer = scRes->getFrambuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};
		}
		cb->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

		const uint32_t currentIndexCount = drawBuffer.getIndexCount();
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[resourceIx].get());
		cb->bindIndexBuffer({ .offset = 0u, .buffer = drawBuffer.gpuDrawBuffers.indexBuffer.get() }, asset::EIT_32BIT);
		cb->bindGraphicsPipeline(graphicsPipeline.get());
		cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if (fragmentShaderInterlockEnabled)
		{
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS, resolveAlphaPipeLayout.get(), 0u, 1u, &descriptorSets[resourceIx].get());
			cb->bindGraphicsPipeline(resolveAlphaGraphicsPipeline.get());
			nbl::ext::FullScreenTriangle::recordDrawCall(cb);
		}

		if constexpr (DebugMode)
		{
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}
		
		cb->endRenderPass();

		if (!inBetweenSubmit)
			cb->endDebugMarker();

		cb->end();

		if (inBetweenSubmit)
		{
			intendedSubmitInfo.overflowSubmit();
			drawBuffer.reset();
		}
		else
		{
			IQueue::SSubmitInfo submitInfo = static_cast<IQueue::SSubmitInfo>(intendedSubmitInfo);
			if (getGraphicsQueue()->submit({ &submitInfo, 1u }) == IQueue::RESULT::SUCCESS)
			{
				m_realFrameIx++;
				intendedSubmitInfo.advanceScratchSemaphoreValue(); // last submits needs to also advance scratch sema value like overflowSubmit() does
				
				IQueue::SSubmitInfo::SSemaphoreInfo renderFinished =
				{
					.semaphore = m_renderSemaphore.get(),
					.value = m_realFrameIx,
					.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
				};
				m_surface->present(m_currentImageAcquire.imageIndex, { &renderFinished, 1u });
			}
		}

		m_overflowSubmitsScratchSemaphoreInfo.value = intendedSubmitInfo.getScratchSemaphoreNextWait().value; // because we need this info consistent within frames
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
		// We actually want to wait for all the frames to finish rendering, otherwise our destructors will run out of order late
		m_device->waitIdle();

		// This is optional, but the window would close AFTER we return from this function
		m_surface = nullptr;
		
		return device_base_t::onAppTerminated();
	}
	
	// virtual function so you can override as needed for some example father down the line
	virtual SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		retval.fragmentShaderPixelInterlock = FragmentShaderPixelInterlock;
		return retval;
	}
		
protected:
	
	void addObjects(SIntendedSubmitInfo& intendedNextSubmit)
	{
		// we record upload of our objects and if we failed to allocate we submit everything
		if (!intendedNextSubmit.valid())
		{
			// log("intendedNextSubmit is invalid.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
			return;
		}

		// Use the last command buffer in intendedNextSubmit, it should be in recording state
		auto* cmdbuf = intendedNextSubmit.frontHalf.getScratchCommandBuffer();

		assert(cmdbuf->getState() == video::IGPUCommandBuffer::STATE::RECORDING && cmdbuf->isResettable());
		assert(cmdbuf->getRecordingFlags().hasFlags(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT));

		auto* cmdpool = cmdbuf->getPool();

		drawBuffer.setSubmitDrawsFunction(
			[&](SIntendedSubmitInfo& intendedNextSubmit)
			{
				return submitDraws(intendedNextSubmit, true);
			}
		);
		drawBuffer.reset();

		if constexpr (mode == ExampleMode::CASE_0)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPolyline polyline;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, -50.0 });
				linePoints.push_back({ 50.0, 50.0 });
				polyline.addLinePoints(linePoints);
			}

			drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_1)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 0.8f;
			style.color = float32_t4(0.619f, 0.325f, 0.709f, 0.2f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 0.0f;
			style2.worldSpaceLineWidth = 0.8f;
			style2.color = float32_t4(0.119f, 0.825f, 0.709f, 0.5f);

			// drawBuffer.drawPolyline(bigPolyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			// drawBuffer.drawPolyline(bigPolyline2, style2, UseDefaultClipProjectionIdx, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_2)
		{
			auto debug = [&](CPolyline polyline, CPULineStyle lineStyle)
			{
				drawBuffer.drawPolyline(polyline, lineStyle, UseDefaultClipProjectionIdx, intendedNextSubmit);
			};
			
			int32_t hatchDebugStep = m_hatchDebugStep;

			if (hatchDebugStep > 0)
			{
#include "bike_hatch.h"
				for (uint32_t i = 0; i < polylines.size(); i++)
				{
					CPULineStyle lineStyle = {};
					lineStyle.screenSpaceLineWidth = 5.0;
					lineStyle.color = float32_t4(float(i) / float(polylines.size()), 1.0 - (float(i) / float(polylines.size())), 0.0, 0.2);
					// assert(polylines[i].checkSectionsContunuity());
					//drawBuffer.drawPolyline(polylines[i], lineStyle, UseDefaultClipProjectionIdx, intendedNextSubmit);
				}
				//printf("hatchDebugStep = %d\n", hatchDebugStep);
				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				Hatch hatch(polylines, SelectedMajorAxis, hatchDebugStep, debug);
				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				// std::cout << "Hatch::Hatch time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
				std::sort(hatch.intersectionAmounts.begin(), hatch.intersectionAmounts.end());

				auto percentile = [&](float percentile)
					{
						return hatch.intersectionAmounts[uint32_t(round(percentile * float(hatch.intersectionAmounts.size() - 1)))];
					};
				//printf(std::format(
				//	"Intersection amounts: 10%%: {}, 25%%: {}, 50%%: {}, 75%%: {}, 90%%: {}, 100%% (max): {}\n",
				//	percentile(0.1), percentile(0.25), percentile(0.5), percentile(0.75), percentile(0.9), hatch.intersectionAmounts[hatch.intersectionAmounts.size() - 1]
				//).c_str());
				drawBuffer.drawHatch(hatch, float32_t4(0.6, 0.6, 0.1, 1.0f), UseDefaultClipProjectionIdx, intendedNextSubmit);
			}

			if (hatchDebugStep > 0)
			{
				std::vector <CPolyline> polylines;
				auto line = [&](float64_t2 begin, float64_t2 end) {
					std::vector<float64_t2> points = {
						begin, end
					};
					CPolyline polyline;
					polyline.addLinePoints(points);
					polylines.push_back(polyline);
				};
				{
					CPolyline polyline;
					std::vector<nbl::hlsl::shapes::QuadraticBezier<float64_t>> beziers;

					// new test case with messed up intersection
					beziers.push_back({ float64_t2(-26, 160), float64_t2(-10, 160), float64_t2(-20, 175.0), });
					beziers.push_back({ float64_t2(-26, 160), float64_t2(-5, 160), float64_t2(-29, 175.0), });

					beziers.push_back({ float64_t2(-26, 120), float64_t2(23, 120), float64_t2(20.07, 145.34), });
					beziers.push_back({ float64_t2(-26, 120), float64_t2(19.73, 120), float64_t2(27.76, 138.04), });
					line(float64_t2(20.07, 145.34), float64_t2(27.76, 138.04));

					beziers.push_back({ float64_t2(25, 70), float64_t2(25, 86), float64_t2(30, 90), });
					beziers.push_back({ float64_t2(25, 70), float64_t2(25, 86), float64_t2(20, 90), });
					line(float64_t2(30, 90), float64_t2(20, 90));

					beziers.push_back({ float64_t2(26, 20), float64_t2(37.25, 29.15), float64_t2(34.9, 42.75), });
					beziers.push_back({ float64_t2(26, 20), float64_t2(33.8, 26.35), float64_t2(15.72, 40.84), });
					line(float64_t2(34.9, 42.75), float64_t2(15.72, 40.84));

					beziers.push_back({ float64_t2(22.5, -20), float64_t2(35, -20), float64_t2(35, 0), });
					beziers.push_back({ float64_t2(22.5, -20), float64_t2(10, -20), float64_t2(10, 0), });
					line(float64_t2(35, 0), float64_t2(10, 0));

					polyline.addQuadBeziers(beziers);

					polylines.push_back(polyline);
				}

				Hatch hatch(polylines, SelectedMajorAxis, hatchDebugStep, debug);
				drawBuffer.drawHatch(hatch, float32_t4(0.0, 1.0, 0.1, 1.0f), UseDefaultClipProjectionIdx, intendedNextSubmit);
			}

			if (hatchDebugStep > 0)
			{
				std::vector <CPolyline> polylines;
				auto circleThing = [&](float64_t2 offset)
				{
					CPolyline polyline;
					std::vector<shapes::QuadraticBezier<double>> beziers;

					beziers.push_back({ float64_t2(0, -1), float64_t2(-1, -1),float64_t2(-1, 0) });
					beziers.push_back({ float64_t2(0, -1), float64_t2(1, -1),float64_t2(1, 0) });
					beziers.push_back({ float64_t2(-1, 0), float64_t2(-1, 1),float64_t2(0, 1) });
					beziers.push_back({ float64_t2(1, 0), float64_t2(1, 1),float64_t2(0, 1) });

					for (uint32_t i = 0; i < beziers.size(); i++)
					{
						beziers[i].P0 = (beziers[i].P0 * 200.0) + offset;
						beziers[i].P1 = (beziers[i].P1 * 200.0) + offset;
						beziers[i].P2 = (beziers[i].P2 * 200.0) + offset;
					}

					polyline.addQuadBeziers(beziers);

					polylines.push_back(polyline);
				};
				circleThing(float64_t2(-500, 0));
				circleThing(float64_t2(500, 0));
				circleThing(float64_t2(0, -500));
				circleThing(float64_t2(0, 500));

				Hatch hatch(polylines, SelectedMajorAxis, hatchDebugStep, debug);
				drawBuffer.drawHatch(hatch, float32_t4(1.0, 0.1, 0.1, 1.0f), UseDefaultClipProjectionIdx, intendedNextSubmit);
			}

			if (hatchDebugStep > 0)
			{
				std::vector <CPolyline> polylines;
				auto line = [&](float64_t2 begin, float64_t2 end) {
					std::vector<float64_t2> points = {
						begin, end
					};
					CPolyline polyline;
					polyline.addLinePoints(points);
					polylines.push_back(polyline);
				};
				{
					CPolyline polyline;
					std::vector<shapes::QuadraticBezier<double>> beziers;

					// new test case with messed up intersection
					beziers.push_back({ float64_t2(-26, 160), float64_t2(-10, 160), float64_t2(-20, 175.0), });
					beziers.push_back({ float64_t2(-26, 160), float64_t2(-5, 160), float64_t2(-29, 175.0), });

					beziers.push_back({ float64_t2(-26, 120), float64_t2(23, 120), float64_t2(20.07, 145.34), });
					beziers.push_back({ float64_t2(-26, 120), float64_t2(19.73, 120), float64_t2(27.76, 138.04), });
					line(float64_t2(20.07, 145.34), float64_t2(27.76, 138.04));

					beziers.push_back({ float64_t2(25, 70), float64_t2(25, 86), float64_t2(30, 90), });
					beziers.push_back({ float64_t2(25, 70), float64_t2(25, 86), float64_t2(20, 90), });
					line(float64_t2(30, 90), float64_t2(20, 90));

					beziers.push_back({ float64_t2(26, 20), float64_t2(37.25, 29.15), float64_t2(34.9, 42.75), });
					beziers.push_back({ float64_t2(26, 20), float64_t2(33.8, 26.35), float64_t2(15.72, 40.84), });
					line(float64_t2(34.9, 42.75), float64_t2(15.72, 40.84));

					beziers.push_back({ float64_t2(22.5, -20), float64_t2(35, -20), float64_t2(35, 0), });
					beziers.push_back({ float64_t2(22.5, -20), float64_t2(10, -20), float64_t2(10, 0), });
					line(float64_t2(35, 0), float64_t2(10, 0));

					polyline.addQuadBeziers(beziers);
				}
			}

			if (hatchDebugStep > 0)
			{
				std::vector <CPolyline> polylines;
				{
					std::vector<float64_t2> points = {
						float64_t2(119.196, -152.568),
						float64_t2(121.566, -87.564),
						float64_t2(237.850, -85.817),
						float64_t2(236.852, -152.194),
						float64_t2(206.159, -150.447),
						float64_t2(205.785, -125.618),
						float64_t2(205.785, -125.618),
						float64_t2(196.180, -122.051),
						float64_t2(186.820, -124.870),
						float64_t2(185.733, -136.350),
						float64_t2(185.822, -149.075),
						float64_t2(172.488, -155.349),
						float64_t2(159.621, -150.447),
						float64_t2(159.638, -137.831),
						float64_t2(159.246, -125.618),
						float64_t2(149.309, -121.398),
						float64_t2(139.907, -123.872),
						float64_t2(140.281, -149.075),
						float64_t2(140.281, -149.075),
						float64_t2(119.196, -152.568)
					};
					CPolyline polyline;
					polyline.addLinePoints(points);
					polylines.push_back(polyline);
				}
				{
					std::vector<float64_t2> points = {
						float64_t2(110.846, -97.918),
						float64_t2(113.217, -32.914),
						float64_t2(229.501, -31.167),
						float64_t2(228.503, -97.544),
						float64_t2(197.810, -95.797),
						float64_t2(197.435, -70.968),
						float64_t2(197.435, -70.968),
						float64_t2(187.831, -67.401),
						float64_t2(178.471, -70.220),
						float64_t2(177.384, -81.700),
						float64_t2(177.473, -94.425),
						float64_t2(164.138, -100.699),
						float64_t2(151.271, -95.797),
						float64_t2(151.289, -83.181),
						float64_t2(150.897, -70.968),
						float64_t2(140.960, -66.748),
						float64_t2(131.558, -69.222),
						float64_t2(131.932, -94.425),
						float64_t2(131.932, -94.425),
						float64_t2(110.846, -97.918)
					};
					CPolyline polyline;
					polyline.addLinePoints(points);
					polylines.push_back(polyline);
				}
				{
					std::vector<float64_t2> points = {
						float64_t2(50.504, -128.469),
						float64_t2(52.874, -63.465),
						float64_t2(169.158, -61.718),
						float64_t2(168.160, -128.095),
						float64_t2(137.467, -126.348),
						float64_t2(137.093, -101.519),
						float64_t2(137.093, -101.519),
						float64_t2(127.488, -97.952),
						float64_t2(118.128, -100.771),
						float64_t2(117.041, -112.251),
						float64_t2(117.130, -124.976),
						float64_t2(103.796, -131.250),
						float64_t2(90.929, -126.348),
						float64_t2(90.946, -113.732),
						float64_t2(90.554, -101.519),
						float64_t2(80.617, -97.298),
						float64_t2(71.215, -99.772),
						float64_t2(71.589, -124.976),
						float64_t2(71.589, -124.976),
						float64_t2(50.504, -128.469)
					};
					CPolyline polyline;
					polyline.addLinePoints(points);
					polylines.push_back(polyline);
				}
				{
					std::vector<float64_t2> points = {
						float64_t2(98.133, -111.581),
						float64_t2(100.503, -46.577),
						float64_t2(216.787, -44.830),
						float64_t2(215.789, -111.206),
						float64_t2(185.096, -109.460),
						float64_t2(184.722, -84.631),
						float64_t2(184.722, -84.631),
						float64_t2(175.117, -81.064),
						float64_t2(165.757, -83.882),
						float64_t2(164.670, -95.363),
						float64_t2(164.759, -108.087),
						float64_t2(151.425, -114.361),
						float64_t2(138.558, -109.460),
						float64_t2(138.575, -96.843),
						float64_t2(138.183, -84.631),
						float64_t2(128.246, -80.410),
						float64_t2(118.844, -82.884),
						float64_t2(119.218, -108.087),
						float64_t2(119.218, -108.087),
						float64_t2(98.133, -111.581)
					};
					CPolyline polyline;
					polyline.addLinePoints(points);
					polylines.push_back(polyline);
				}
				Hatch hatch(polylines, SelectedMajorAxis, hatchDebugStep, debug);
				drawBuffer.drawHatch(hatch, float32_t4(0.0, 0.0, 1.0, 1.0f), UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			if (hatchDebugStep > 0)
			{
				std::vector<float64_t2> points;
				double sqrt3 = sqrt(3.0);
				points.push_back(float64_t2(0, 1));
				points.push_back(float64_t2(sqrt3 / 2, 0.5));
				points.push_back(float64_t2(sqrt3 / 2, -0.5));
				points.push_back(float64_t2(0, -1));
				points.push_back(float64_t2(-sqrt3 / 2, -0.5));
				points.push_back(float64_t2(-sqrt3 / 2, 0.5));
				points.push_back(float64_t2(0, 1));

				std::vector<shapes::QuadraticBezier<double>> beziers;
				beziers.push_back({
					float64_t2(-0.5, -0.25),
					float64_t2(-sqrt3 / 2, 0.0),
					float64_t2(-0.5, 0.25) });
				beziers.push_back({
					float64_t2(0.5, -0.25),
					float64_t2(sqrt3 / 2, 0.0),
					float64_t2(0.5, 0.25) });
			
				for (uint32_t i = 0; i < points.size(); i++)
					points[i] = float64_t2(-200.0, 0.0) + float64_t2(10.0 + abs(cos(m_timeElapsed * 0.00008)) * 150.0f, 100.0) * points[i];
				for (uint32_t i = 0; i < beziers.size(); i++)
				{
					beziers[i].P0 = float64_t2(-200.0, 0.0) + float64_t2(10.0 + abs(cos(m_timeElapsed * 0.00008)) * 150.0f, 100.0) * beziers[i].P0;
					beziers[i].P1 = float64_t2(-200.0, 0.0) + float64_t2(10.0 + abs(cos(m_timeElapsed * 0.00008)) * 150.0f, 100.0) * beziers[i].P1;
					beziers[i].P2 = float64_t2(-200.0, 0.0) + float64_t2(10.0 + abs(cos(m_timeElapsed * 0.00008)) * 150.0f, 100.0) * beziers[i].P2;
				}

				CPolyline polyline;
				polyline.addLinePoints(points);
				polyline.addQuadBeziers(beziers);

				Hatch hatch({&polyline, 1u}, SelectedMajorAxis, hatchDebugStep, debug);
				drawBuffer.drawHatch(hatch, float32_t4(1.0f, 0.325f, 0.103f, 1.0f), UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			if (hatchDebugStep > 0)
			{
				CPolyline polyline;
				std::vector<shapes::QuadraticBezier<double>> beziers;
				beziers.push_back({
					100.0 * float64_t2(-0.4, 0.13),
					100.0 * float64_t2(7.7, 3.57),
					100.0 * float64_t2(8.8, 7.27) });
				beziers.push_back({
					100.0 * float64_t2(6.6, 0.13),
					100.0 * float64_t2(-1.97, 3.2),
					100.0 * float64_t2(3.7, 7.27) });
				polyline.addQuadBeziers(beziers);
			
				Hatch hatch({&polyline, 1u}, SelectedMajorAxis, hatchDebugStep, debug);
				drawBuffer.drawHatch(hatch, float32_t4(0.619f, 0.325f, 0.709f, 0.9f), UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
		}
		else if (mode == ExampleMode::CASE_3)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 2.0f;
			style2.worldSpaceLineWidth = 0.0f;
			style2.color = float32_t4(0.2f, 0.6f, 0.2f, 1.0f);


			CPolyline originalPolyline;
			{
				// float64_t2 endPoint = { cos(m_timeElapsed * 0.0005), sin(m_timeElapsed * 0.0005) };
				float64_t2 endPoint = { 0.0, 0.0 };
				originalPolyline.setClosed(true);
				std::vector<float64_t2> linePoints;

				{
					linePoints.push_back(endPoint);
					linePoints.push_back({ 1.25, -0.625 });
					linePoints.push_back({ 2.5, -1.25 });
					linePoints.push_back({ 5.0, -2.5 });
					linePoints.push_back({ 10.0, -5.0 });
					linePoints.push_back({ 20.0, 0.0 });
					linePoints.push_back({ 20.0, 5.0 });
					originalPolyline.addLinePoints(linePoints);
				}

				{
					std::vector<shapes::QuadraticBezier<double>> quadBeziers;
					shapes::QuadraticBezier<double>  quadratic1;
					quadratic1.P0 = float64_t2(20.0, 5.0);
					quadratic1.P1 = float64_t2(30.0, 20.0);
					quadratic1.P2 = float64_t2(40.0, 5.0);
					quadBeziers.push_back(quadratic1);
					originalPolyline.addQuadBeziers(quadBeziers);
				}

				{
					linePoints.clear();
					linePoints.push_back({ 40.0, 5.0 });
					linePoints.push_back({ 50.0, -10.0 });
					originalPolyline.addLinePoints(linePoints);
				}

				{
					std::vector<shapes::QuadraticBezier<double>> quadBeziers;
					curves::EllipticalArcInfo myCurve;
					{
						myCurve.majorAxis = { -20.0, 0.0 };
						myCurve.center = { 30, -10.0 };
						myCurve.angleBounds = {
							nbl::core::PI<double>() * 1.0,
							nbl::core::PI<double>() * 0.0
						};
						myCurve.eccentricity = 1.0;
					}

					curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
						{
							quadBeziers.push_back(info);
						};

					curves::Subdivision::adaptive(myCurve, 1e-5, addToBezier, 10u);
					originalPolyline.addQuadBeziers(quadBeziers);
					// ellipse arc ends on 10, -10
				}

				{
					std::vector<shapes::QuadraticBezier<double>> quadBeziers;
					curves::EllipticalArcInfo myCurve;
					{
						myCurve.majorAxis = { -10.0, 5.0 };
						myCurve.center = { 0, -5.0 };
						myCurve.angleBounds = {
							nbl::core::PI<double>() * 1.0,
							nbl::core::PI<double>() * 0.0
							};
						myCurve.eccentricity = 1.0;
					}

					curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
						{
							quadBeziers.push_back(info);
						};

					curves::Subdivision::adaptive(myCurve, 1e-5, addToBezier, 10u);
					originalPolyline.addQuadBeziers(quadBeziers);
					// ellipse arc ends on -10, 0.0
				}

				{
					linePoints.clear();
					linePoints.push_back({ -10.0, 0.0 });
					linePoints.push_back({ -5.0, -5.0 });
					linePoints.push_back({ -3.0, -3.0 });
					linePoints.push_back({ -1.0, -1.0 });
					linePoints.push_back(endPoint);
					originalPolyline.addLinePoints(linePoints);
				}
			}

			drawBuffer.drawPolyline(originalPolyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			CPolyline offsettedPolyline = originalPolyline.generateParallelPolyline(+0.0 - 3.0 * abs(cos(m_timeElapsed * 0.0009)));
			CPolyline offsettedPolyline2 = originalPolyline.generateParallelPolyline(+0.0 + 3.0 * abs(cos(m_timeElapsed * 0.0009)));
			drawBuffer.drawPolyline(offsettedPolyline, style2, UseDefaultClipProjectionIdx, intendedNextSubmit);
			drawBuffer.drawPolyline(offsettedPolyline2, style2, UseDefaultClipProjectionIdx, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_4)
		{
			constexpr uint32_t CURVE_CNT = 16u;
			constexpr uint32_t SPECIAL_CASE_CNT = 6u;

			CPULineStyle cpuLineStyle = {};
			cpuLineStyle.screenSpaceLineWidth = 7.0f;
			cpuLineStyle.worldSpaceLineWidth = 0.0f;
			cpuLineStyle.color = float32_t4(0.7f, 0.3f, 0.7f, 0.8f);
			cpuLineStyle.isRoadStyleFlag = false;

			std::vector<CPULineStyle> cpuLineStyles(CURVE_CNT, cpuLineStyle);
			std::vector<CPolyline> polylines(CURVE_CNT);

			{
				std::vector<shapes::QuadraticBezier<double>> quadratics(CURVE_CNT);

				// setting controll points
				{
					float64_t2 P0(-90, 68);
					float64_t2 P1(-41, 118);
					float64_t2 P2(88, 19);

					const float64_t2 translationVector(0, -5);

					uint32_t curveIdx = 0;
					while(curveIdx < CURVE_CNT - SPECIAL_CASE_CNT)
					{
						quadratics[curveIdx].P0 = P0;
						quadratics[curveIdx].P1 = P1;
						quadratics[curveIdx].P2 = P2;

						P0 += translationVector;
						P1 += translationVector;
						P2 += translationVector;

						curveIdx++;
					}

					// special case 0 (line, evenly spaced points)
					const double prevLineLowestY = quadratics[curveIdx - 1].P2.y;
					double lineY = prevLineLowestY - 10.0;

					quadratics[curveIdx].P0 = float64_t2(-100, lineY);
					quadratics[curveIdx].P1 = float64_t2(0, lineY);
					quadratics[curveIdx].P2 = float64_t2(100, lineY);
					cpuLineStyles[curveIdx].color = float64_t4(0.7f, 0.3f, 0.1f, 0.5f);

					// special case 1 (line, not evenly spaced points)
					lineY -= 10.0;
					curveIdx++;

					quadratics[curveIdx].P0 = float64_t2(-100, lineY);
					quadratics[curveIdx].P1 = float64_t2(20, lineY);
					quadratics[curveIdx].P2 = float64_t2(100, lineY);

					// special case 2 (folded line)
					lineY -= 10.0;
					curveIdx++;

					quadratics[curveIdx].P0 = float64_t2(-100, lineY);
					quadratics[curveIdx].P1 = float64_t2(100, lineY);
					quadratics[curveIdx].P2 = float64_t2(50, lineY);

					// oblique line
					curveIdx++;
					quadratics[curveIdx].P0 = float64_t2(-100, 100);
					quadratics[curveIdx].P1 = float64_t2(50.0, -50.0);
					quadratics[curveIdx].P2 = float64_t2(100, -100);

					// special case 3 (A.x == 0)
					curveIdx++;
					quadratics[curveIdx].P0 = float64_t2(0.0, 0.0);
					quadratics[curveIdx].P1 = float64_t2(3.0, 4.14);
					quadratics[curveIdx].P2 = float64_t2(6.0, 4.0);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

						// make sure A.x == 0
					float64_t2 A = quadratics[curveIdx].P0 - 2.0 * quadratics[curveIdx].P1 + quadratics[curveIdx].P2;
					assert(A.x == 0);

					// special case 4 (symetric parabola)
					curveIdx++;
					quadratics[curveIdx].P0 = float64_t2(-150.0, 1.0);
					quadratics[curveIdx].P1 = float64_t2(2000.0, 0.0);
					quadratics[curveIdx].P2 = float64_t2(-150.0, -1.0);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
				}

				std::array<core::vector<double>, CURVE_CNT> stipplePatterns;

				// TODO: fix uninvited circles at beggining and end of curves, solve with clipping (precalc tMin, tMax)

					// test case 0: test curve
				stipplePatterns[0] = { 0.0f, -5.0f, 2.0f, -5.0f };
					// test case 1: lots of redundant values, should look exactly like stipplePattern[0]
				stipplePatterns[1] = { 1.0f, 2.0f, 2.0f, -4.0f, -1.0f, 1.0f, -3.0f, -1.5f, -0.3f, -0.2f }; 
					// test case 2:stipplePattern[0] but shifted curve but shifted to left by 2.5f
				stipplePatterns[2] = { 2.5f, -5.0f, 1.0f, -5.0f, 2.5f };
					// test case 3: starts and ends with negative value, stipplePattern[2] reversed (I'm suspisious about that, need more testing)
				stipplePatterns[3] = { -2.5f, 5.0f, -1.0f, 5.0f, -2.5f };
					// test case 4: starts with "don't draw section"
				stipplePatterns[4] = { -5.0f, 5.0f };
					// test case 5: invisible curve (shouldn't be send to GPU)
				stipplePatterns[5] = { -1.0f };
					// test case 6: invisible curve (shouldn't be send to GPU)
				stipplePatterns[6] = { -1.0f, -5.0f, -10.0f };
					// test case 7: continous curuve
				stipplePatterns[7] = { 25.0f, 25.0f };
					// test case 8: start with `0` pattern + 2 `0` patterns close together
				stipplePatterns[8] = { 0.0, -10.0f, 0.0, -1.0, 0.0, -7.0 };
					// test case 9: max pattern size
				stipplePatterns[9] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -2.0f };
					// test case 10: A = 0 (line), evenly distributed controll points
				stipplePatterns[10] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 11: A = 0 (line), not evenly distributed controll points
				stipplePatterns[11] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 12: A = 0 (line), folds itself
				stipplePatterns[12] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 13: oblique line 
				stipplePatterns[13] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 14: curve with A.x = 0
				stipplePatterns[14] = { 0.0f, -0.5f, 0.2f, -0.5f };
					// test case 15: long parabola
				stipplePatterns[15] = { 5.0f, -5.0f, 1.0f, -5.0f };

				std::vector<uint32_t> activIdx = { 10 };
				for (uint32_t i = 0u; i < CURVE_CNT; i++)
				{
					cpuLineStyles[i].setStipplePatternData(stipplePatterns[i]);
					cpuLineStyles[i].phaseShift += abs(cos(m_timeElapsed * 0.0003));
					polylines[i].addQuadBeziers({ &quadratics[i], &quadratics[i] + 1u });

					float64_t2 linePoints[2u] = {};
					linePoints[0] = { -200.0, 50.0 - 5.0 * i };
					linePoints[1] = { -100.0, 50.0 - 6.0 * i };
					polylines[i].addLinePoints(linePoints);

					activIdx.push_back(i);
					if (std::find(activIdx.begin(), activIdx.end(), i) == activIdx.end())
						cpuLineStyles[i].stipplePatternSize = -1;

					polylines[i].preprocessPolylineWithStyle(cpuLineStyles[i]);
				}
			}

			for (uint32_t i = 0u; i < CURVE_CNT; i++)
				drawBuffer.drawPolyline(polylines[i], cpuLineStyles[i], UseDefaultClipProjectionIdx, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_5)
		{
//#define CASE_5_POLYLINE_1 // animated stipple pattern
//#define CASE_5_POLYLINE_2 // miter test static
//#define CASE_5_POLYLINE_3 // miter test animated
//#define CASE_5_POLYLINE_4 // miter test animated (every angle)
//#define CASE_5_POLYLINE_5 // closed polygon
#define CASE_5_POLYLINE_6 // stretching
//#define CASE_5_POLYLINE_7 // wide non solid lines

#if defined(CASE_5_POLYLINE_1)
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
			style.isRoadStyleFlag = true;

			const double firstDrawSectionSize = (std::cos(m_timeElapsed * 0.00002) + 1.0f) * 10.0f;
			std::array<double, 4u> stipplePattern = { firstDrawSectionSize, -20.0f, 1.0f, -5.0f };
			style.setStipplePatternData(stipplePattern);

			CPolyline polyline;
			{
				// section 1: lines
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, -50.0 });
				linePoints.push_back({ 50.0, 50.0 });
				linePoints.push_back({ 50.0, -50.0 });
				linePoints.push_back({ 80.0, -50.0 });
				linePoints.push_back({ 80.0, 70.0 });
				linePoints.push_back({ 100.0, 70.0 });
				linePoints.push_back({ 120.0, 50.0 });
				polyline.addLinePoints(linePoints);

				// section 2: beziers
				std::vector<shapes::QuadraticBezier<double>> quadratics(2u);
				quadratics[0].P0 = { 120.0, 50.0 };
				quadratics[0].P1 = { 200.0, 80.0 };
				quadratics[0].P2 = { 140.0, 30.0 };
				quadratics[1].P0 = { 140.0, 30.0 };
				quadratics[1].P1 = { 100.0, 15.0 };
				quadratics[1].P2 = { 140.0, 0.0 };
				polyline.addQuadBeziers(quadratics);

				// section 3: lines
				std::vector<float64_t2> linePoints2;
				linePoints2.push_back({ 140.0, 0.0 });
				linePoints2.push_back({ 140.0, -80.0 });
				linePoints2.push_back({ -140.0, -80.0 });
				linePoints2.push_back({ -150.0, 20.0, });
				linePoints2.push_back({ -100.0, 50.0 });
				polyline.addLinePoints(linePoints2);

				// section 4: beziers
				std::vector<shapes::QuadraticBezier<double>> quadratics2(4u);
				quadratics2[0].P0 = { -100.0, 50.0 };
				quadratics2[0].P1 = { -80.0, 30.0 };
				quadratics2[0].P2 = { -60.0, 50.0 };
				quadratics2[1].P0 = { -60.0, 50.0 };
				quadratics2[1].P1 = { -40.0, 80.0 };
				quadratics2[1].P2 = { -20.0, 50.0 };
				quadratics2[2].P0 = { -20.0, 50.0 };
				quadratics2[2].P1 = { 0.0, 30.0 };
				quadratics2[2].P2 = { 20.0, 50.0 };
				quadratics2[3].P0 = { 20.0, 50.0 };
				quadratics2[3].P1 = { -80.0, 100.0 };
				quadratics2[3].P2 = { -100.0, 90.0 };
				polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<double>>(quadratics2.data(), quadratics2.data() + quadratics2.size()));

				polyline.preprocessPolylineWithStyle(style);
			}

			drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);

#elif defined(CASE_5_POLYLINE_2)

			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 2.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
			style.isRoadStyleFlag = true;

			//const double firstDrawSectionSize = (std::cos(m_timeElapsed * 0.0002) + 1.0f) * 10.0f;
			//std::array<float, 4u> stipplePattern = { firstDrawSectionSize, -20.0f, 1.0f, -5.0f };
			std::array<double, 1u> stipplePattern = { 1.0f };
			style.setStipplePatternData(stipplePattern);

			CPolyline polyline;
			{
				// section 0: beziers
				std::vector<shapes::QuadraticBezier<double>> quadratics(2u);
				quadratics[0].P0 = { -50.0, -100.0 };
				quadratics[0].P1 = { -25.0, -75.0 };
				quadratics[0].P2 = { 0.0, -100.0 };
				quadratics[1].P0 = { 0.0, -100.0 };
				quadratics[1].P1 = { -20.0, -75.0 };
				quadratics[1].P2 = { -50.0, -50.0 };
				polyline.addQuadBeziers(quadratics);

				// section 1: lines
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, -50.0 });
				linePoints.push_back({ 0.0, 0.0 });
				linePoints.push_back({ 50.0, -50.0 });
				linePoints.push_back({ 0.0, -50.0 });
				linePoints.push_back({ 50.0, 0.0 });
				linePoints.push_back({ 0.0, 50.0 });
				polyline.addLinePoints(linePoints);

				// section 2: beziers
				std::vector<shapes::QuadraticBezier<double>> quadratics2(3u);
				quadratics2[0].P0 = { 0.0, 50.0 };
				quadratics2[0].P1 = { -20.0, 30.0 };
				quadratics2[0].P2 = { -40.0, 50.0 };
				quadratics2[1].P0 = { -40.0, 50.0 };
				quadratics2[1].P1 = { -60.0, 35.0 };
				quadratics2[1].P2 = { -40.0, 20.0 };
				quadratics2[2].P0 = { -40.0, 20.0 };
				quadratics2[2].P1 = { -20.0, 30.0 };
				quadratics2[2].P2 = { 0.0, 20.0 };
				/*quadratics2[3].P0 = {20.0, 50.0};
				quadratics2[3].P1 = { -80.0, 100.0 };
				quadratics2[3].P2 = { -100.0, 90.0 };*/
				polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<double>>(quadratics2.data(), quadratics2.data() + quadratics2.size()));

				// section 3: lines
				std::vector<float64_t2> linePoints2;
				linePoints2.push_back({ 0.0, 20.0 });
				linePoints2.push_back({ 0.0, 10.0 });
				linePoints2.push_back({ -30.0, 10.0 });
				/*linePoints2.push_back({0.0, -50.0});
				linePoints2.push_back({ 50.0, 0.0 });
				linePoints2.push_back({ 0.0, 50.0 });*/
				polyline.addLinePoints(linePoints2);

				// section 4: beziers
				std::vector<shapes::QuadraticBezier<double>> quadratics3(1u);
				quadratics3[0].P0 = { -30.0, 10.0 };
				quadratics3[0].P1 = { -30.0, 0.0 };
				quadratics3[0].P2 = { -20.0, 5.0 };
				polyline.addQuadBeziers(quadratics3);

				polyline.preprocessPolylineWithStyle(style);
			}

			drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);

#elif defined(CASE_5_POLYLINE_3)
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
			style.isRoadStyleFlag = true;

			//const double firstDrawSectionSize = (std::cos(m_timeElapsed * 0.0002) + 1.0f) * 10.0f;
			//std::array<float, 4u> stipplePattern = { firstDrawSectionSize, -20.0f, 1.0f, -5.0f };
			std::array<double, 1u> stipplePattern = { 1.0f };
			style.setStipplePatternData(stipplePattern);

			CPolyline polyline;
			{
				std::vector<float64_t2> linePoints;
				const double animationFactor = std::cos(m_timeElapsed * 0.0003);
				linePoints.push_back({-200.0,  50.0 * animationFactor});
				linePoints.push_back({-150.0, -50.0 * animationFactor});
				linePoints.push_back({-100.0,  50.0 * animationFactor});
				linePoints.push_back({-50.0,  -50.0 * animationFactor});
				linePoints.push_back({ 0.0,    50.0 * animationFactor});
				linePoints.push_back({ 50.0,  -50.0 * animationFactor});
				linePoints.push_back({ 100.0,  50.0 * animationFactor});
				linePoints.push_back({ 150.0, -50.0 * animationFactor});
				linePoints.push_back({ 200.0,  50.0 * animationFactor});
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style);
			}

			drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);

#elif defined(CASE_5_POLYLINE_4)
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
			style.isRoadStyleFlag = true;

			//const double firstDrawSectionSize = (std::cos(m_timeElapsed * 0.0002) + 1.0f) * 10.0f;
			//std::array<float, 4u> stipplePattern = { firstDrawSectionSize, -20.0f, 1.0f, -5.0f };
			std::array<double, 1u> stipplePattern = { 1.0f };
			style.setStipplePatternData(stipplePattern);

			CPolyline polyline;
			CPolyline polyline2;
			{
				const float rotationAngle = m_timeElapsed * 0.0005;
				const float64_t rotationAngleCos = std::cos(rotationAngle);
				const float64_t rotationAngleSin = std::sin(rotationAngle);
				const float64_t2x2 rotationMatrix = float64_t2x2(rotationAngleCos, -rotationAngleSin, rotationAngleSin, rotationAngleCos);
				
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ 0.0, -50.0 });
				linePoints.push_back({ 0.0,  0.0 });
				linePoints.push_back(mul(rotationMatrix, float64_t2(0.0, 50.0)));
				polyline.addLinePoints(linePoints);

				std::vector<shapes::QuadraticBezier<double>> quadratics(2u);
				quadratics[0].P0 = { 0.0, -50.0 };
				quadratics[0].P1 = { 0.0, -25.0 };
				quadratics[0].P2 = { 0.0, 0.0 };

				quadratics[1].P0 = { 0.0, 0.0 };
				quadratics[1].P1 = { 0.0, 25.0 };
				quadratics[1].P2 = { mul(rotationMatrix, float64_t2(0.0, 50.0)) };
				polyline2.addQuadBeziers(quadratics);

				polyline.preprocessPolylineWithStyle(style);
				polyline2.preprocessPolylineWithStyle(style);
			}

			drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			//drawBuffer.drawPolyline(polyline2, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
#elif defined(CASE_5_POLYLINE_5)
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
			style.isRoadStyleFlag = true;

			const double firstDrawSectionSize = (std::cos(m_timeElapsed * 0.0002) + 1.0f) * 10.0f;
			std::array<double, 4u> stipplePattern = { firstDrawSectionSize, -20.0f, 1.0f, -5.0f };
			//std::array<float, 1u> stipplePattern = { 1.0f };
			style.setStipplePatternData(stipplePattern);

			CPolyline polyline;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({0.0, -50.0});
				linePoints.push_back({50.0, 0.0});
				linePoints.push_back({0.0, 50.0});
				linePoints.push_back({-50.0, 0.0});
				linePoints.push_back({ 0.0, -50.0});
				polyline.addLinePoints(linePoints);
				polyline.setClosed(true);
				polyline.preprocessPolylineWithStyle(style);
			}

			{
				/*std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 0.0 });
				linePoints.push_back({ 0.0, 0.0 });
				polyline.addLinePoints(linePoints);

				std::vector<shapes::QuadraticBezier<double>> quadratics(1u);
				quadratics[0].P0 = { 0.0, 0.0 };
				quadratics[0].P1 = { -25.0, -50.0 };
				quadratics[0].P2 = { -50.0, 0.0 };
				polyline.addQuadBeziers(quadratics);

				polyline.setClosed(true);
				polyline.preprocessPolylineWithStyle(style);*/
			}

			{
				/*std::vector<float64_t2> linePoints;
				linePoints.push_back({ 0.0, -50.0});
				linePoints.push_back({ 50.0, 0.0});
				linePoints.push_back({ 0.0, 50.0});
				linePoints.push_back({ -50.0, 0.0 });
				polyline.addLinePoints(linePoints);

				std::vector<shapes::QuadraticBezier<double>> quadratics(1u);
				quadratics[0].P0 = { -50.0, 0.0 };
				quadratics[0].P1 = { -25.0, 0.0 };
				quadratics[0].P2 = { 0.0, -50.0 };
				polyline.addQuadBeziers(quadratics);

				polyline.setClosed(true);
				polyline.preprocessPolylineWithStyle(style);*/
			}

			drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
#elif defined(CASE_5_POLYLINE_6)
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 3.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.85f, 0.1f, 0.1f, 0.5f);
			style.isRoadStyleFlag = false;

			CPULineStyle shapeStyle = style;
			CPolyline shapesPolyline;

			// double linesLength = 20.0;
			double linesLength = 10.0 + 20.0 * abs(cos(m_timeElapsed * 0.0001));
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 58.0 });
				linePoints.push_back({ -50.0 + linesLength, 58.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			// std::array<double, 4u> stipplePattern = { 0.0f, -5.0f, 5.0f, -2.5f };
			std::array<double, 2u> stipplePattern = { 5.0f, -5.0f };
			style.setStipplePatternData(stipplePattern);
			auto addShapesFunction = [&](const float64_t2& position, const float64_t2& direction, float32_t stretch) {
				std::vector<shapes::QuadraticBezier<double>> quadBeziers;
				curves::EllipticalArcInfo myCurve;
				{
					myCurve.majorAxis = glm::normalize(direction) * 0.4;
					myCurve.center = position;
					myCurve.angleBounds = {
						nbl::core::PI<double>() * 0.0,
						nbl::core::PI<double>() * 2.0
					};
					myCurve.eccentricity = 0.4;
				}

				curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
					{
						quadBeziers.push_back(info);
					};

				curves::Subdivision::adaptive(myCurve, 1e-2, addToBezier, 10u);
				shapesPolyline.addQuadBeziers(quadBeziers);
			};
			
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 54.0 });
				linePoints.push_back({ -50.0 + linesLength, 54.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}

			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 52.0 });
				linePoints.push_back({ -50.0 + linesLength, 52.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			style.setStipplePatternData(stipplePattern, 7.5, true, false);
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 50.0 });
				linePoints.push_back({ -50.0 + linesLength, 50.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style, addShapesFunction);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			style.setStipplePatternData(stipplePattern, 7.5, true, true);
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 48.0 });
				linePoints.push_back({ -50.0 + linesLength, 48.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style, addShapesFunction);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			std::array<double, 3u> stipplePattern2 = { 2.5f, -5.0f, 2.5f };
			style.setStipplePatternData(stipplePattern2, 5.0, true, true);
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 46.0 });
				linePoints.push_back({ -50.0 + linesLength, 46.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style, addShapesFunction);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			style.setStipplePatternData(stipplePattern, 7.5, true, false);
			{
				CPolyline polyline;

				std::vector<shapes::QuadraticBezier<double>> quadBeziers;
				curves::EllipticalArcInfo myCurve;
				{
					myCurve.majorAxis = { -50.0, 0.0 };
					myCurve.center = { 0.0, 25.0 };
					myCurve.angleBounds = {
						nbl::core::PI<double>() * 0.0,
						nbl::core::PI<double>() * abs(cos(m_timeElapsed * 0.00005))
					};
					myCurve.eccentricity = 1.0;
				}

				curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
					{
						quadBeziers.push_back(info);
					};

				curves::Subdivision::adaptive(myCurve, 1e-3, addToBezier, 10u);
				polyline.addQuadBeziers(quadBeziers);
				polyline.preprocessPolylineWithStyle(style, addShapesFunction);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}
			
			style.setStipplePatternData(stipplePattern, 7.5, true, true);
			style.color = float32_t4(0.3f, 0.7f, 0.7f, 0.5f);
			{
				CPolyline polyline;

				std::vector<shapes::QuadraticBezier<double>> quadBeziers;
				curves::EllipticalArcInfo myCurve;
				{
					myCurve.majorAxis = { -52.0, 0.0 };
					myCurve.center = { 0.0, 25.0 };
					myCurve.angleBounds = {
						nbl::core::PI<double>() * 0.0,
						nbl::core::PI<double>() * abs(cos(m_timeElapsed * 0.00005))
					};
					myCurve.eccentricity = 1.0;
				}

				curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
					{
						quadBeziers.push_back(info);
					};

				curves::Subdivision::adaptive(myCurve, 1e-3, addToBezier, 10u);
				polyline.addQuadBeziers(quadBeziers);
				polyline.preprocessPolylineWithStyle(style, addShapesFunction);
				drawBuffer.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
			}

			drawBuffer.drawPolyline(shapesPolyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);
#elif defined(CASE_5_POLYLINE_7)
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 2.0f;
			style2.worldSpaceLineWidth = 0.0f;
			style2.color = float32_t4(0.2f, 0.6f, 0.2f, 1.0f);


			CPolyline originalPolyline;
			{
				// float64_t2 endPoint = { cos(m_timeElapsed * 0.0005), sin(m_timeElapsed * 0.0005) };
				float64_t2 endPoint = { 0.0, 0.0 };
				originalPolyline.setClosed(true);
				std::vector<float64_t2> linePoints;

				{
					linePoints.push_back(endPoint);
					linePoints.push_back({ 1.25, -0.625 });
					linePoints.push_back({ 2.5, -1.25 });
					linePoints.push_back({ 5.0, -2.5 });
					linePoints.push_back({ 10.0, -5.0 });
					linePoints.push_back({ 20.0, 0.0 });
					linePoints.push_back({ 20.0, 5.0 });
					originalPolyline.addLinePoints(linePoints);
				}

				{
					std::vector<shapes::QuadraticBezier<double>> quadBeziers;
					shapes::QuadraticBezier<double>  quadratic1;
					quadratic1.P0 = float64_t2(20.0, 5.0);
					quadratic1.P1 = float64_t2(30.0, 20.0);
					quadratic1.P2 = float64_t2(40.0, 5.0);
					quadBeziers.push_back(quadratic1);
					originalPolyline.addQuadBeziers(quadBeziers);
				}

				{
					linePoints.clear();
					linePoints.push_back({ 40.0, 5.0 });
					linePoints.push_back({ 50.0, -10.0 });
					originalPolyline.addLinePoints(linePoints);
				}

				{
					std::vector<shapes::QuadraticBezier<double>> quadBeziers;
					curves::EllipticalArcInfo myCurve;
					{
						myCurve.majorAxis = { -20.0, 0.0 };
						myCurve.center = { 30, -10.0 };
						myCurve.angleBounds = {
							nbl::core::PI<double>() * 1.0,
							nbl::core::PI<double>() * 0.0
						};
						myCurve.eccentricity = 1.0;
					}

					curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
						{
							quadBeziers.push_back(info);
						};

					curves::Subdivision::adaptive(myCurve, 1e-5, addToBezier, 10u);
					originalPolyline.addQuadBeziers(quadBeziers);
					// ellipse arc ends on 10, -10
				}

				{
					std::vector<shapes::QuadraticBezier<double>> quadBeziers;
					curves::EllipticalArcInfo myCurve;
					{
						myCurve.majorAxis = { -10.0, 5.0 };
						myCurve.center = { 0, -5.0 };
						myCurve.angleBounds = {
							nbl::core::PI<double>() * 1.0,
							nbl::core::PI<double>() * 0.0
							};
						myCurve.eccentricity = 1.0;
					}

					curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
						{
							quadBeziers.push_back(info);
						};

					curves::Subdivision::adaptive(myCurve, 1e-5, addToBezier, 10u);
					originalPolyline.addQuadBeziers(quadBeziers);
					// ellipse arc ends on -10, 0.0
				}

				{
					linePoints.clear();
					linePoints.push_back({ -10.0, 0.0 });
					linePoints.push_back({ -5.0, -5.0 });
					linePoints.push_back({ -3.0, -3.0 });
					linePoints.push_back({ -1.0, -1.0 });
					linePoints.push_back(endPoint);
					originalPolyline.addLinePoints(linePoints);
				}
			}

			// drawBuffer.drawPolyline(originalPolyline, style, UseDefaultClipProjectionIdx, intendedNextSubmit);

			std::array<double, 2u> stipplePattern = { 2, -1 };
			style.setStipplePatternData(stipplePattern);
			
			style.phaseShift += abs(cos(m_timeElapsed * 0.0003));
			CPolyline offsetPolyline1, offsetPolyline2;
			originalPolyline.stippleBreakDown(style, [&](const CPolyline& smallPoly)
				{
					smallPoly.makeWideWhole(offsetPolyline1, offsetPolyline2, 0.1f, 1e-3);
					drawBuffer.drawPolyline(smallPoly, style2, UseDefaultClipProjectionIdx, intendedNextSubmit);
					drawBuffer.drawPolyline(offsetPolyline1, style2, UseDefaultClipProjectionIdx, intendedNextSubmit);
					drawBuffer.drawPolyline(offsetPolyline2, style2, UseDefaultClipProjectionIdx, intendedNextSubmit);
				});

#endif

		}

		drawBuffer.finalizeAllCopiesToGPU(intendedNextSubmit);
	}

	double getScreenToWorldRatio(const float64_t3x3& viewProjectionMatrix, uint32_t2 windowSize)
	{
		double idx_0_0 = viewProjectionMatrix[0u][0u] * (windowSize.x / 2.0);
		double idx_1_1 = viewProjectionMatrix[1u][1u] * (windowSize.y / 2.0);
		double det_2x2_mat = idx_0_0 * idx_1_1;
		return static_cast<float>(core::sqrt(core::abs(det_2x2_mat)));
	}

protected:

	std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
	clock_t::time_point start;

	bool fragmentShaderInterlockEnabled = false;

	core::smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	
	smart_refctd_ptr<IGPURenderpass> renderpassInitial; // this renderpass will clear the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	smart_refctd_ptr<IGPURenderpass> renderpassInBetween; // this renderpass will load the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	smart_refctd_ptr<IGPURenderpass> renderpassFinal; // this renderpass will load the attachment and transition it to PRESENT
	
	std::array<smart_refctd_ptr<IGPUCommandPool>,	MaxFramesInFlight>	m_graphicsCommandPools;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>,	MaxFramesInFlight>	m_commandBuffers;
	
	std::array<smart_refctd_ptr<IGPUImageView>,			MaxFramesInFlight>	pseudoStencilImageViews;
	std::array<smart_refctd_ptr<IGPUBuffer>,			MaxFramesInFlight>	globalsBuffer;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>,		MaxFramesInFlight>	descriptorSets;
	DrawBuffersFiller drawBuffer; // you can think of this as the scene data needed to draw everything, we only have one instance so let's use a timeline semaphore to sync all renders

	smart_refctd_ptr<ISemaphore> m_renderSemaphore; // timeline semaphore to sync frames together
	
	// timeline semaphore used for overflows (they need to be on their own timeline to count overflows)
	smart_refctd_ptr<ISemaphore> m_overflowSubmitScratchSemaphore; 
	// this is the semaphore info the overflows update the value for (the semaphore is set to the overflow semaphore above, and the value get's updated by SIntendedSubmitInfo)
	IQueue::SSubmitInfo::SSemaphoreInfo m_overflowSubmitsScratchSemaphoreInfo;
	
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	uint64_t m_realFrameIx : 59 = 0;
	// Maximum frames which can be simultaneously rendered
	uint64_t m_framesInFlight : 5;

	smart_refctd_ptr<IGPUGraphicsPipeline>		graphicsPipeline;
	smart_refctd_ptr<IGPUGraphicsPipeline>		debugGraphicsPipeline;
	smart_refctd_ptr<IGPUDescriptorSetLayout>	descriptorSetLayout;
	smart_refctd_ptr<IGPUPipelineLayout>		graphicsPipelineLayout;

	smart_refctd_ptr<IGPUGraphicsPipeline> resolveAlphaGraphicsPipeline;
	smart_refctd_ptr<IGPUPipelineLayout> resolveAlphaPipeLayout;

	Camera2D m_Camera;


	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CSwapchainResources>> m_surface;
};

NBL_MAIN_FUNC(ComputerAidedDesign)