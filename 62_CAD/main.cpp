// TODO: Copyright notice


#include "nbl/examples/examples.hpp"

using namespace nbl::hlsl;
using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

#include "nbl/examples/common/BuiltinResourcesApplication.hpp"
#include "nbl/examples/common/SimpleWindowedApplication.hpp"
#include "nbl/examples/common/InputSystem.hpp"
#include "nbl/video/utilities/CSimpleResizeSurface.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/TextRendering/TextRendering.h"
#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include "Curves.h"
#include "Hatch.h"
#include "Polyline.h"
#include "DrawResourcesFiller.h"
#include "SingleLineText.h"

#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "HatchGlyphBuilder.h"

#include <nbl/builtin/hlsl/tgmath.hlsl>

//TODO: remove
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/concepts/matrix.hlsl>

#include <chrono>
#define BENCHMARK_TILL_FIRST_FRAME

static constexpr bool DebugModeWireframe = false;
static constexpr bool DebugRotatingViewProj = false;
static constexpr bool FragmentShaderPixelInterlock = true;
static constexpr bool LargeGeoTextureStreaming = true;
static constexpr bool CacheAndReplay = false; // caches first frame resources (buffers and images) from DrawResourcesFiller  and replays in future frames, skiping CPU Logic
static constexpr bool testCameraRotation = false;

enum class ExampleMode
{
	CASE_0, // Simple Line, Camera Zoom In/Out
	CASE_1,	// Overdraw Fragment Shader Stress Test
	CASE_2, // hatches
	CASE_3, // CURVES AND LINES
	CASE_4, // STIPPLE PATTERN
	CASE_5, // Advanced Styling
	CASE_6, // Custom Clip Projections
	CASE_7, // Images
	CASE_8, // MSDF and Text
	CASE_9, // DTM
	CASE_10, // testing fixed geometry and emulated fp64 corner cases
	CASE_11, // grid DTM
	CASE_12, // Georeferenced streamed images
	CASE_COUNT
};

constexpr std::array<float, (uint32_t)ExampleMode::CASE_COUNT> cameraExtents =
{
	10.0,	// CASE_0
	10.0,	// CASE_1
	200.0,	// CASE_2
	10.0,	// CASE_3
	10.0,	// CASE_4
	10.0,	// CASE_5
	10.0,	// CASE_6
	10.0,	// CASE_7
	600.0,	// CASE_8
	600.0,	// CASE_9
	10.0,	// CASE_10
	1000.0,	// CASE_11
	10.0	// CASE_12
};

constexpr ExampleMode mode = ExampleMode::CASE_5;

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
				m_bounds = m_bounds + float64_t2{ (double)ev.scrollEvent.verticalScroll * -0.025 * m_aspectRatio, (double)ev.scrollEvent.verticalScroll * -0.025};
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
	CEventCallback(nbl::core::smart_refctd_ptr<nbl::examples::InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)){}
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


// TODO: Move this funcitons that help with creating a new promoted CPUImage
template<unsigned int SRC_CHANNELS>
struct PromotionComponentSwizzle
{
    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

        reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];

        if constexpr (SRC_CHANNELS > 1)
            reinterpret_cast<out_t*>(out)[1u] = reinterpret_cast<const in_t*>(in)[1u];
        else
            reinterpret_cast<out_t*>(out)[1u] = static_cast<in_t>(0);

        if constexpr (SRC_CHANNELS > 2)
            reinterpret_cast<out_t*>(out)[2u] = reinterpret_cast<const in_t*>(in)[2u];
        else
            reinterpret_cast<out_t*>(out)[2u] = static_cast<in_t>(0);

        if constexpr (SRC_CHANNELS > 3)
            reinterpret_cast<out_t*>(out)[3u] = reinterpret_cast<const in_t*>(in)[3u];
        else
            reinterpret_cast<out_t*>(out)[3u] = static_cast<in_t>(1);
    }
};
template<typename Filter>
bool performCopyUsingImageFilter(
    const core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage,
    const core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage)
{
    Filter filter;

	const uint32_t mipLevels = inCPUImage->getCreationParameters().mipLevels;
	
	for (uint32_t level = 0u; level < mipLevels; ++level)
	{
		const auto regions = inCPUImage->getRegions(level);

		for (auto& region : regions)
		{
			typename Filter::state_type state = {};
			state.extent = region.imageExtent;
			state.layerCount = region.imageSubresource.layerCount;
			state.inImage = inCPUImage.get();
			state.outImage = outCPUImage.get();
			state.inOffsetBaseLayer = core::vectorSIMDu32(region.imageOffset.x,  region.imageOffset.y, region.imageOffset.z, region.imageSubresource.baseArrayLayer);
			state.outOffsetBaseLayer = core::vectorSIMDu32(0u);
			state.inMipLevel = region.imageSubresource.mipLevel;
			state.outMipLevel = region.imageSubresource.mipLevel;

			if (!filter.execute(core::execution::par_unseq, &state))
				return false;
		}
	}
	return true;
}

bool performImageFormatPromotionCopy(const core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage, const core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage)
{
	asset::E_FORMAT srcImageFormat = inCPUImage->getCreationParameters().format;
	asset::E_FORMAT dstImageFormat = outCPUImage->getCreationParameters().format;

	// In = srcData, Out = stagingBuffer
	if (srcImageFormat == dstImageFormat)
		return false;

    auto srcChannelCount = asset::getFormatChannelCount(srcImageFormat);
    if (srcChannelCount == 1u)
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<1u>>>(inCPUImage, outCPUImage);
    else if (srcChannelCount == 2u)
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<2u>>>(inCPUImage, outCPUImage);
    else if (srcChannelCount == 3u)
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<3u>>>(inCPUImage, outCPUImage);
    else
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<4u>>>(inCPUImage, outCPUImage);
}

// Used by case 12
struct ImageLoader : public IImageRegionLoader
{
	ImageLoader(asset::IAssetManager* assetMgr, system::ILogger* logger, video::IPhysicalDevice* physicalDevice)
		: m_assetMgr(assetMgr), m_logger(logger), m_physicalDevice(physicalDevice) 
	{ 
		auto loadImage = [&](const std::string& imagePath) -> smart_refctd_ptr<ICPUImage>
			{
				system::path m_loadCWD = "..";
				constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
				const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger, m_loadCWD);
				auto bundle = m_assetMgr->getAsset(imagePath, loadParams);
				auto contents = bundle.getContents();
				if (contents.empty())
				{
					m_logger->log("Failed to load image with path %s, skipping!", ILogger::ELL_ERROR, (m_loadCWD / imagePath).c_str());
					return nullptr;
				}

				smart_refctd_ptr<ICPUImageView> cpuImgView;
				const auto& asset = contents[0];
				switch (asset->getAssetType())
				{
				case IAsset::ET_IMAGE:
				{
					auto image = smart_refctd_ptr_static_cast<ICPUImage>(asset);
					auto& flags = image->getCreationParameters().flags;
					// assert if asset is mutable
					const_cast<core::bitflag<asset::IImage::E_CREATE_FLAGS>&>(flags) |= asset::IImage::E_CREATE_FLAGS::ECF_MUTABLE_FORMAT_BIT;
					const auto format = image->getCreationParameters().format;

					ICPUImageView::SCreationParams viewParams = {
						.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
						.image = std::move(image),
						.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
						.format = format,
						.subresourceRange = {
							.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = ICPUImageView::remaining_mip_levels,
							.baseArrayLayer = 0u,
							.layerCount = ICPUImageView::remaining_array_layers
						}
					};

					cpuImgView = ICPUImageView::create(std::move(viewParams));
				} break;

				case IAsset::ET_IMAGE_VIEW:
					cpuImgView = smart_refctd_ptr_static_cast<ICPUImageView>(asset);
					break;
				default:
					m_logger->log("Failed to load ICPUImage or ICPUImageView got some other Asset Type, skipping!", ILogger::ELL_ERROR);
					return nullptr;
				}

				const auto loadedCPUImage = cpuImgView->getCreationParameters().image;
				const auto loadedCPUImageCreationParams = loadedCPUImage->getCreationParameters();

				// Promoting the image to a format GPU supports. (so that updateImageViaStagingBuffer doesn't have to handle that each frame if overflow-submit needs to happen)
				auto promotedCPUImageCreationParams = loadedCPUImage->getCreationParameters();

				promotedCPUImageCreationParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT | IGPUImage::EUF_SAMPLED_BIT;
				// promote format because RGB8 and friends don't actually exist in HW
				{
					const IPhysicalDevice::SImageFormatPromotionRequest request = {
						.originalFormat = promotedCPUImageCreationParams.format,
						.usages = IPhysicalDevice::SFormatImageUsages::SUsage(promotedCPUImageCreationParams.usage)
					};
					promotedCPUImageCreationParams.format = m_physicalDevice->promoteImageFormat(request, video::IGPUImage::TILING::OPTIMAL);
				}

				if (loadedCPUImageCreationParams.format != promotedCPUImageCreationParams.format)
				{
					smart_refctd_ptr<ICPUImage> promotedCPUImage = ICPUImage::create(promotedCPUImageCreationParams);
					core::rational<uint32_t> bytesPerPixel = asset::getBytesPerPixel(promotedCPUImageCreationParams.format);

					const auto extent = loadedCPUImageCreationParams.extent;
					const uint32_t mipLevels = loadedCPUImageCreationParams.mipLevels;
					const uint32_t arrayLayers = loadedCPUImageCreationParams.arrayLayers;

					// Only supporting 1 mip, it's just for test..
					const size_t byteSize = (bytesPerPixel * extent.width * extent.height * extent.depth * arrayLayers).getIntegerApprox(); // TODO: consider mips
					ICPUBuffer::SCreationParams bufferCreationParams = {};
					bufferCreationParams.size = byteSize;
					smart_refctd_ptr<ICPUBuffer> promotedCPUImageBuffer = ICPUBuffer::create(std::move(bufferCreationParams));

					auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
					ICPUImage::SBufferCopy& region = newRegions->front();
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					region.imageSubresource.mipLevel = 0u; // TODO
					region.imageSubresource.baseArrayLayer = 0u;
					region.imageSubresource.layerCount = arrayLayers;
					region.bufferOffset = 0u;
					region.bufferRowLength = 0u;
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = extent;
					promotedCPUImage->setBufferAndRegions(std::move(promotedCPUImageBuffer), newRegions);

					performImageFormatPromotionCopy(loadedCPUImage, promotedCPUImage);
					return promotedCPUImage;
				}
				else
				{
					return loadedCPUImage;
				}
			};

		// This is all hardcoded for the example
		const std::string basePath = "../../media/npot_geotex_mip_";
		smart_refctd_ptr<ICPUImage> img = loadImage(basePath + "0_base.png");

		// This is hardcoded
		const uint32_t maxMipLevel = 7;
		baseMipLevels.reserve(maxMipLevel + 1);
		baseMipLevels.emplace_back(std::move(img));
		for (auto i = 1u; i <= maxMipLevel; i++)
		{
			baseMipLevels.emplace_back(loadImage(basePath + std::to_string(i) + "_base.png"));
		}
		downsampledMipLevels.reserve(maxMipLevel + 1);
		for (auto i = 0u; i <= maxMipLevel; i++)
		{
			downsampledMipLevels.emplace_back(loadImage(basePath + std::to_string(i) + "_downsampled.png"));
		}
	}

	uint32_t2 getExtents(std::filesystem::path imagePath, uint32_t mipLevel)
	{
		return { baseMipLevels[mipLevel]->getCreationParameters().extent.width, baseMipLevels[mipLevel]->getCreationParameters().extent.height };
	}

	uint32_t2 getExtents(std::filesystem::path imagePath) override
	{
		return getExtents(imagePath, 0);
	}

	asset::E_FORMAT getFormat(std::filesystem::path imagePath) override
	{
		return baseMipLevels[0]->getCreationParameters().format;
	}

	bool hasPrecomputedMips(std::filesystem::path imagePath) const override
	{
		return true;
	}

private:

	// Assume offset always fits in the image, but maybe offset + extent doesn't
	// Example of a precomputed mip loader with 2x mip levels
	core::smart_refctd_ptr<ICPUBuffer> load_impl(std::filesystem::path imagePath, uint32_t2 offset, uint32_t2 extent, uint32_t mipLevel, bool downsample) override
	{
		// Hardcoded tile size that's not accessible
		auto mippedImageExtents = getExtents(imagePath, mipLevel);
		// If `offset + extent` exceeds the extent of the image at the current mip level, we clamp it
		extent = nbl::hlsl::min(mippedImageExtents - offset, extent);
		// Image path ignored for this hardcoded example
		const auto& image = downsample ? downsampledMipLevels[mipLevel] : baseMipLevels[mipLevel];
		const auto& imageBuffer = image->getBuffer();
		const core::rational<uint32_t> bytesPerPixel = asset::getBytesPerPixel(image->getCreationParameters().format);
		const size_t bytesPerRow = (bytesPerPixel * extent.x).getIntegerApprox();
		const size_t loadedImageBytes = bytesPerRow * extent.y;
		asset::IBuffer::SCreationParams bufCreationParams = { .size = loadedImageBytes, .usage = imageBuffer->getCreationParams().usage };
		ICPUBuffer::SCreationParams cpuBufCreationParams(std::move(bufCreationParams));
		core::smart_refctd_ptr<ICPUBuffer> retVal = ICPUBuffer::create(std::move(cpuBufCreationParams));

		// Copy row by row into the new buffer
		uint8_t* dataPtr = reinterpret_cast<uint8_t*>(retVal->getPointer());
		const uint8_t* imageBufferDataPtr = reinterpret_cast<uint8_t*>(imageBuffer->getPointer());
		const size_t bytesPerImageRow = (bytesPerPixel * image->getCreationParameters().extent.width).getIntegerApprox();
		for (auto row = 0u; row < extent.y; row++)
		{
			const size_t imageBufferOffset = bytesPerImageRow * (offset.y + row) + (bytesPerPixel * offset.x).getIntegerApprox();
			std::memcpy(dataPtr + row * bytesPerRow, imageBufferDataPtr + imageBufferOffset, bytesPerRow);
		}
		return retVal;
	}

	// These are here for the example, might not be class members when porting to n4ce
	asset::IAssetManager* m_assetMgr = {};
	system::ILogger* m_logger = {};
	video::IPhysicalDevice* m_physicalDevice = {};
	// We're going to fake it in the example so it's easier to work with, but the interface remains
	core::vector<smart_refctd_ptr<ICPUImage>> baseMipLevels = {};
	core::vector<smart_refctd_ptr<ICPUImage>> downsampledMipLevels = {};
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
		drawResourcesFiller = DrawResourcesFiller(core::smart_refctd_ptr(m_device), core::smart_refctd_ptr(m_utils), core::smart_refctd_ptr(m_utils), getGraphicsQueue(), core::smart_refctd_ptr(m_logger));

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
		drawResourcesFiller.allocateMSDFTextures(m_device.get(), 256u, uint32_t2(MSDFSize, MSDFSize));

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

		// MSDF Image Sampler
		{
			IGPUSampler::SParams samplerParams = {};
			samplerParams.TextureWrapU = IGPUSampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
			samplerParams.TextureWrapV = IGPUSampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
			samplerParams.TextureWrapW = IGPUSampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
			samplerParams.BorderColor = IGPUSampler::ETBC_FLOAT_OPAQUE_WHITE; // positive means outside shape
			samplerParams.MinFilter = IGPUSampler::ETF_LINEAR;
			samplerParams.MaxFilter = IGPUSampler::ETF_LINEAR;
			samplerParams.MipmapMode = IGPUSampler::ESMM_LINEAR;
			samplerParams.AnisotropicFilter = 3;
			samplerParams.CompareEnable = false;
			samplerParams.CompareFunc = ECO_GREATER;
			samplerParams.LodBias = 0.f;
			samplerParams.MinLod = -1000.f;
			samplerParams.MaxLod = 1000.f;
			msdfImageSampler = m_device->createSampler(samplerParams);
		}
		
		// Static Image Sampler
		{
			IGPUSampler::SParams samplerParams = {};
			samplerParams.TextureWrapU = IGPUSampler::E_TEXTURE_CLAMP::ETC_REPEAT;
			samplerParams.TextureWrapV = IGPUSampler::E_TEXTURE_CLAMP::ETC_REPEAT;
			samplerParams.TextureWrapW = IGPUSampler::E_TEXTURE_CLAMP::ETC_REPEAT;
			samplerParams.BorderColor = IGPUSampler::ETBC_FLOAT_TRANSPARENT_BLACK;
			samplerParams.MinFilter = IGPUSampler::ETF_LINEAR;
			samplerParams.MaxFilter = IGPUSampler::ETF_LINEAR;
			samplerParams.MipmapMode = IGPUSampler::ESMM_LINEAR;
			samplerParams.AnisotropicFilter = 3;
			samplerParams.CompareEnable = false;
			samplerParams.CompareFunc = ECO_GREATER;
			samplerParams.LodBias = 0.f;
			samplerParams.MinLod = -1000.f;
			samplerParams.MaxLod = 1000.f;
			staticImageSampler = m_device->createSampler(samplerParams);
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

				uint32_t pseudoStencilInvalidValue = core::bitfieldInsert<uint32_t>(0u, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
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
	uint32_t m_hatchDebugStep = 10u;
	E_HEIGHT_SHADING_MODE m_shadingModeExample = E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS;

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<nbl::examples::InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;
		
		fragmentShaderInterlockEnabled = m_device->getEnabledFeatures().fragmentShaderPixelInterlock;
		
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

		const bitflag<IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS> bindlessTextureFlags =
			IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT |
			IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT |
			IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_PARTIALLY_BOUND_BIT;

		// Create DescriptorSetLayout, PipelineLayout and update DescriptorSets
		const uint32_t imagesBinding = 3u;
		{
			video::IGPUDescriptorSetLayout::SBinding bindingsSet0[] = {
				{
					.binding = 0u,
					.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_VERTEX | asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 1u,
					.type = asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 2u,
					.type = asset::IDescriptor::E_TYPE::ET_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = imagesBinding,
					.type = asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
					.createFlags = bindlessTextureFlags,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = ImagesBindingArraySize,
				},
			};
			descriptorSetLayout0 = m_device->createDescriptorSetLayout(bindingsSet0);
			if (!descriptorSetLayout0)
				return logFail("Failed to Create Descriptor Layout 0");
			
			video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
				{
					.binding = 0u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 1u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				},
			};
			descriptorSetLayout1 = m_device->createDescriptorSetLayout(bindingsSet1);
			if (!descriptorSetLayout1)
				return logFail("Failed to Create Descriptor Layout 1");

			const video::IGPUDescriptorSetLayout* const layouts[2u] = { descriptorSetLayout0.get(), descriptorSetLayout1.get() };

			smart_refctd_ptr<IDescriptorPool> descriptorPool = nullptr;
			{
				const uint32_t setCounts[2u] = { 1u, 1u};
				descriptorPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, layouts, setCounts);
				if (!descriptorPool)
					return logFail("Failed to Create Descriptor Pool");
			}

			{
				descriptorSet0 = descriptorPool->createDescriptorSet(smart_refctd_ptr(descriptorSetLayout0));
				descriptorSet1 = descriptorPool->createDescriptorSet(smart_refctd_ptr(descriptorSetLayout1));
				constexpr uint32_t DescriptorCountSet0 = 3u;
				video::IGPUDescriptorSet::SDescriptorInfo descriptorInfosSet0[DescriptorCountSet0] = {};

				// Descriptors For Set 0:
				descriptorInfosSet0[0u].info.buffer.offset = 0u;
				descriptorInfosSet0[0u].info.buffer.size = m_globalsBuffer->getCreationParams().size;
				descriptorInfosSet0[0u].desc = m_globalsBuffer;

				descriptorInfosSet0[1u].info.combinedImageSampler.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
				descriptorInfosSet0[1u].info.combinedImageSampler.sampler = msdfImageSampler;
				descriptorInfosSet0[1u].desc = drawResourcesFiller.getMSDFsTextureArray();
				
				descriptorInfosSet0[2u].desc = staticImageSampler; // TODO[Erfan]: different sampler and make immutable?
				
				// This is bindless to we write to it later.
				// descriptorInfosSet0[3u].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
				// descriptorInfosSet0[3u].desc = drawResourcesFiller.getMSDFsTextureArray();

				// Descriptors For Set 1:
				constexpr uint32_t DescriptorCountSet1 = 2u;
				video::IGPUDescriptorSet::SDescriptorInfo descriptorInfosSet1[DescriptorCountSet1] = {};
				descriptorInfosSet1[0u].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				descriptorInfosSet1[0u].info.combinedImageSampler.sampler = nullptr;
				descriptorInfosSet1[0u].desc = pseudoStencilImageView;

				descriptorInfosSet1[1u].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				descriptorInfosSet1[1u].info.combinedImageSampler.sampler = nullptr;
				descriptorInfosSet1[1u].desc = colorStorageImageView;

				constexpr uint32_t DescriptorUpdatesCount = DescriptorCountSet0 + DescriptorCountSet1;
				video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[DescriptorUpdatesCount] = {};
				
				// Set 0 Updates:
					// globals
				descriptorUpdates[0u].dstSet = descriptorSet0.get();
				descriptorUpdates[0u].binding = 0u;
				descriptorUpdates[0u].arrayElement = 0u;
				descriptorUpdates[0u].count = 1u;
				descriptorUpdates[0u].info = &descriptorInfosSet0[0u];

					// mdfs textures
				descriptorUpdates[1u].dstSet = descriptorSet0.get();
				descriptorUpdates[1u].binding = 1u;
				descriptorUpdates[1u].arrayElement = 0u;
				descriptorUpdates[1u].count = 1u;
				descriptorUpdates[1u].info = &descriptorInfosSet0[1u];
				
					// general texture sampler	
				descriptorUpdates[2u].dstSet = descriptorSet0.get();
				descriptorUpdates[2u].binding = 2u;
				descriptorUpdates[2u].arrayElement = 0u;
				descriptorUpdates[2u].count = 1u;
				descriptorUpdates[2u].info = &descriptorInfosSet0[2u];

				// Set 1 Updates:
				descriptorUpdates[3u].dstSet = descriptorSet1.get();
				descriptorUpdates[3u].binding = 0u;
				descriptorUpdates[3u].arrayElement = 0u;
				descriptorUpdates[3u].count = 1u;
				descriptorUpdates[3u].info = &descriptorInfosSet1[0u];

				descriptorUpdates[4u].dstSet = descriptorSet1.get();
				descriptorUpdates[4u].binding = 1u;
				descriptorUpdates[4u].arrayElement = 0u;
				descriptorUpdates[4u].count = 1u;
				descriptorUpdates[4u].info = &descriptorInfosSet1[1u];

				m_device->updateDescriptorSets(DescriptorUpdatesCount, descriptorUpdates, 0u, nullptr);
			}

			const asset::SPushConstantRange range = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.offset = 0,
						.size = sizeof(PushConstants)
			};

			pipelineLayout = m_device->createPipelineLayout({ &range,1 }, core::smart_refctd_ptr(descriptorSetLayout0), core::smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);
		}

		drawResourcesFiller.setTexturesDescriptorSetAndBinding(core::smart_refctd_ptr(descriptorSet0), imagesBinding);

		smart_refctd_ptr<IShader> mainPipelineFragmentShaders = {};
		smart_refctd_ptr<IShader> mainPipelineVertexShader = {};
		std::array<smart_refctd_ptr<IShader>, 2u> geoTexturePipelineShaders = {};
		{
			smart_refctd_ptr<IShaderCompiler::CCache> shaderReadCache = nullptr;
			smart_refctd_ptr<IShaderCompiler::CCache> shaderWriteCache = core::make_smart_refctd_ptr<IShaderCompiler::CCache>();
			auto shaderCachePath = localOutputCWD / "main_pipeline_shader_cache.bin";

			{
				core::smart_refctd_ptr<system::IFile> shaderReadCacheFile;
				{
					system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
					m_system->createFile(future, shaderCachePath.c_str(), system::IFile::ECF_READ);
					if (future.wait())
					{
						future.acquire().move_into(shaderReadCacheFile);
						if (shaderReadCacheFile)
						{
							const size_t size = shaderReadCacheFile->getSize();
							if (size > 0ull)
							{
								std::vector<uint8_t> contents(size);
								system::IFile::success_t succ;
								shaderReadCacheFile->read(succ, contents.data(), 0, size);
								if (succ)
									shaderReadCache = IShaderCompiler::CCache::deserialize(contents);
							}
						}
					}
					else
						m_logger->log("Failed Openning Shader Cache File.", ILogger::ELL_ERROR);
				}

			}

			// Load Custom Shader
			auto loadCompileShader = [&](const std::string& relPath) -> smart_refctd_ptr<IShader>
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.logger = m_logger.get();
					lp.workingDirectory = ""; // virtual root
					auto assetBundle = m_assetMgr->getAsset(relPath, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
						return nullptr;

					// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
					auto source = IAsset::castDown<IShader>(assets[0]);
					if (!source)
						return nullptr;
	
					return m_device->compileShader( ILogicalDevice::SShaderCreationParameters { .source = source.get(), .readCache = shaderReadCache.get(), .writeCache = shaderWriteCache.get(), .stage = IShader::E_SHADER_STAGE::ESS_ALL_OR_LIBRARY });
				};

			mainPipelineFragmentShaders = loadCompileShader("../shaders/main_pipeline/fragment.hlsl");
			mainPipelineVertexShader = loadCompileShader("../shaders/main_pipeline/vertex_shader.hlsl");
			
			core::smart_refctd_ptr<system::IFile> shaderWriteCacheFile;
			{
				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_system->deleteFile(shaderCachePath); // temp solution instead of trimming, to make sure we won't have corrupted json
				m_system->createFile(future, shaderCachePath.c_str(), system::IFile::ECF_WRITE);
				if (future.wait())
				{
					future.acquire().move_into(shaderWriteCacheFile);
					if (shaderWriteCacheFile)
					{
						auto serializedCache = shaderWriteCache->serialize();
						if (shaderWriteCacheFile)
						{
							system::IFile::success_t succ;
							shaderWriteCacheFile->write(succ, serializedCache->getPointer(), 0, serializedCache->getSize());
							if (!succ)
								m_logger->log("Failed Writing To Shader Cache File.", ILogger::ELL_ERROR);
						}
					}
					else
						m_logger->log("Failed Creating Shader Cache File.", ILogger::ELL_ERROR);
				}
				else
					m_logger->log("Failed Creating Shader Cache File.", ILogger::ELL_ERROR);
			}
		}


		// Shared Blend Params between pipelines
		// Premultiplied over-blend (back-to-front)
		SBlendParams premultipliedOverBlendParams = {};
		premultipliedOverBlendParams.blendParams[0u].srcColorFactor = asset::EBF_ONE;
		premultipliedOverBlendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		premultipliedOverBlendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		premultipliedOverBlendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		premultipliedOverBlendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		premultipliedOverBlendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		premultipliedOverBlendParams.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;
		// Premultiplied UNDER-blend (front-to-back)
		SBlendParams premultipliedUnderBlendParams = {};
		premultipliedUnderBlendParams.blendParams[0u].srcColorFactor = asset::EBF_ONE_MINUS_DST_ALPHA;
		premultipliedUnderBlendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE;
		premultipliedUnderBlendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		premultipliedUnderBlendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		premultipliedUnderBlendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		premultipliedUnderBlendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		premultipliedUnderBlendParams.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;

		// Create Alpha Resovle Pipeline
		{
			// Load FSTri Shader
			ext::FullScreenTriangle::ProtoPipeline fsTriangleProtoPipe(m_assetMgr.get(),m_device.get(),m_logger.get());
			
			const video::IGPUPipelineBase::SShaderSpecInfo fragSpec = { .shader = mainPipelineFragmentShaders.get(), .entryPoint = "resolveAlphaMain" };

			resolveAlphaGraphicsPipeline = fsTriangleProtoPipe.createPipeline(fragSpec, pipelineLayout.get(), compatibleRenderPass.get(), 0u, premultipliedOverBlendParams);
			if (!resolveAlphaGraphicsPipeline)
				return logFail("Graphics Pipeline Creation Failed.");

		}
		
		// Create Main Graphics Pipelines 
		{
			
			video::IGPUPipelineBase::SShaderSpecInfo specInfo[2] = {
				{
					.shader = mainPipelineVertexShader.get(),
					.entryPoint = "vtxMain"
				},
				{
					.shader = mainPipelineFragmentShaders.get(),
					.entryPoint = "fragMain"
				},
			};

			IGPUGraphicsPipeline::SCreationParams params[1] = {};
			params[0].layout = pipelineLayout.get();
			params[0].vertexShader = specInfo[0];
			params[0].fragmentShader = specInfo[1];
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
				.blend = premultipliedOverBlendParams,
			};
			params[0].renderpass = compatibleRenderPass.get();
			
			if (!m_device->createGraphicsPipelines(nullptr,params,&graphicsPipeline))
				return logFail("Graphics Pipeline Creation Failed.");

			if constexpr (DebugModeWireframe)
			{
				specInfo[1u].entryPoint = "fragDebugMain"; // change only fragment shader entrypoint
				params[0].cached.rasterization.polygonMode = asset::EPM_LINE;
				
				if (!m_device->createGraphicsPipelines(nullptr,params,&debugGraphicsPipeline))
					return logFail("Debug Graphics Pipeline Creation Failed.");
			}
		}

		// Create the commandbuffers and pools, this time properly 1 pool per FIF
		m_graphicsCommandPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		if (!m_graphicsCommandPool)
			return logFail("Couldn't create Command Pool!");
		if (!m_graphicsCommandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_commandBuffersInFlight.data(),MaxSubmitsInFlight}))
			return logFail("Couldn't create Command Buffers!");
		
		m_Camera.setOrigin({ 0.0, 0.0 });
		m_Camera.setAspectRatio((double)m_window->getWidth() / m_window->getHeight());
		m_Camera.setSize(cameraExtents[uint32_t(mode)]);

		m_timeElapsed = 0.0;
		
		// Loading font stuff
		m_textRenderer = nbl::core::make_smart_refctd_ptr<TextRenderer>();

		m_font = FontFace::create(core::smart_refctd_ptr(m_textRenderer), std::string("C:\\Windows\\Fonts\\arial.ttf"));
	
		if (m_font->getFreetypeFace()->num_charmaps > 0)
			FT_Set_Charmap(m_font->getFreetypeFace(), m_font->getFreetypeFace()->charmaps[0]);
		
		const std::wstring str = L"MSDF: ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnoprstuvwxyz '1234567890-=\"!@#$%&*()_+";
		singleLineText = std::unique_ptr<SingleLineText>(new SingleLineText(
			m_font.get(),
			str));

		drawResourcesFiller.setGlyphMSDFTextureFunction(
			[&](nbl::ext::TextRendering::FontFace* face, uint32_t glyphIdx) -> core::smart_refctd_ptr<asset::ICPUImage>
			{
				return face->generateGlyphMSDF(MSDFPixelRange, glyphIdx, drawResourcesFiller.getMSDFResolution(), MSDFMips);
			}
		);

		drawResourcesFiller.setHatchFillMSDFTextureFunction(
			[&](HatchFillPattern pattern) -> core::smart_refctd_ptr<asset::ICPUImage>
			{
				return Hatch::generateHatchFillPatternMSDF(m_textRenderer.get(), pattern, drawResourcesFiller.getMSDFResolution());
			}
		);
		
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

		// Load image
		system::path m_loadCWD = "..";
		std::string imagePaths[] =
		{
			"../../media/color_space_test/R8G8B8_1.jpg",
			"../../media/color_space_test/R8G8B8_1.png",
			"../../media/color_space_test/R8G8B8A8_2.png",
			"../../media/color_space_test/R8G8B8A8_1.png",
		};

		/**
		* @param formatOverride override format of an image view, use special argument asset::E_FORMAT::EF_COUNT to don't override image view format and use one retrieved from the loaded image
		*/
		auto loadImage = [&](const std::string& imagePath) -> smart_refctd_ptr<ICPUImage>
		{
			constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
			const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get(), m_loadCWD);
			auto bundle = m_assetMgr->getAsset(imagePath, loadParams);
			auto contents = bundle.getContents();
			if (contents.empty())
			{
				m_logger->log("Failed to load image with path %s, skipping!", ILogger::ELL_ERROR, (m_loadCWD / imagePath).c_str());
				return nullptr;
			}

			smart_refctd_ptr<ICPUImageView> cpuImgView;
			const auto& asset = contents[0];
			switch (asset->getAssetType())
			{
			case IAsset::ET_IMAGE:
			{
				auto image = smart_refctd_ptr_static_cast<ICPUImage>(asset);
				auto& flags = image->getCreationParameters().flags;
				// assert if asset is mutable
				const_cast<core::bitflag<asset::IImage::E_CREATE_FLAGS>&>(flags) |= asset::IImage::E_CREATE_FLAGS::ECF_MUTABLE_FORMAT_BIT;
				const auto format = image->getCreationParameters().format;

				ICPUImageView::SCreationParams viewParams = {
					.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
					.image = std::move(image),
					.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
					.format = format,
					.subresourceRange = {
						.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
						.baseMipLevel = 0u,
						.levelCount = ICPUImageView::remaining_mip_levels,
						.baseArrayLayer = 0u,
						.layerCount = ICPUImageView::remaining_array_layers
					}
				};

				cpuImgView = ICPUImageView::create(std::move(viewParams));
			} break;

			case IAsset::ET_IMAGE_VIEW:
				cpuImgView = smart_refctd_ptr_static_cast<ICPUImageView>(asset);
				break;
			default:
				m_logger->log("Failed to load ICPUImage or ICPUImageView got some other Asset Type, skipping!", ILogger::ELL_ERROR);
				return nullptr;
			}

			const auto loadedCPUImage = cpuImgView->getCreationParameters().image;
			const auto loadedCPUImageCreationParams = loadedCPUImage->getCreationParameters();

			// Promoting the image to a format GPU supports. (so that updateImageViaStagingBuffer doesn't have to handle that each frame if overflow-submit needs to happen)
			auto promotedCPUImageCreationParams = loadedCPUImage->getCreationParameters();
			
			promotedCPUImageCreationParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT|IGPUImage::EUF_SAMPLED_BIT;
			// promote format because RGB8 and friends don't actually exist in HW
			{
				const IPhysicalDevice::SImageFormatPromotionRequest request = {
					.originalFormat = promotedCPUImageCreationParams.format,
					.usages = IPhysicalDevice::SFormatImageUsages::SUsage(promotedCPUImageCreationParams.usage)
				};
				promotedCPUImageCreationParams.format = m_physicalDevice->promoteImageFormat(request,video::IGPUImage::TILING::OPTIMAL);
			}

			if (loadedCPUImageCreationParams.format != promotedCPUImageCreationParams.format)
			{
				smart_refctd_ptr<ICPUImage> promotedCPUImage = ICPUImage::create(promotedCPUImageCreationParams);
				core::rational<uint32_t> bytesPerPixel = asset::getBytesPerPixel(promotedCPUImageCreationParams.format);

				const auto extent = loadedCPUImageCreationParams.extent;
				const uint32_t mipLevels = loadedCPUImageCreationParams.mipLevels;
				const uint32_t arrayLayers = loadedCPUImageCreationParams.arrayLayers;
				
				// Only supporting 1 mip, it's just for test..
				const size_t byteSize = (bytesPerPixel * extent.width * extent.height * extent.depth * arrayLayers).getIntegerApprox(); // TODO: consider mips
				ICPUBuffer::SCreationParams bufferCreationParams = {};
				bufferCreationParams.size = byteSize;
				smart_refctd_ptr<ICPUBuffer> promotedCPUImageBuffer = ICPUBuffer::create(std::move(bufferCreationParams));
				
				auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
				ICPUImage::SBufferCopy& region = newRegions->front();
				region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				region.imageSubresource.mipLevel = 0u; // TODO
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = arrayLayers;
				region.bufferOffset = 0u;
				region.bufferRowLength = 0u;
				region.bufferImageHeight = 0u;
				region.imageOffset = { 0u, 0u, 0u };
				region.imageExtent = extent;
				promotedCPUImage->setBufferAndRegions(std::move(promotedCPUImageBuffer), newRegions);

				performImageFormatPromotionCopy(loadedCPUImage, promotedCPUImage);
				return promotedCPUImage;
			}
			else
			{
				return loadedCPUImage;
			}
		};

		for (const auto& imagePath : imagePaths)
		{
			auto image = loadImage(imagePath);
			if (image)
				sampleImages.push_back(image);
		}

		gridDTMHeightMap = loadImage("../../media/gridDTMHeightMap.exr");

		// Create case 12 image loader
		if constexpr (mode == ExampleMode::CASE_12)
		{
			drawResourcesFiller.setGeoreferencedImageLoader(make_smart_refctd_ptr<ImageLoader>(m_assetMgr.get(), m_logger.get(), m_physicalDevice));
		}

		// set diagonals of cells to TOP_LEFT_TO_BOTTOM_RIGHT or BOTTOM_LEFT_TO_TOP_RIGHT randomly
		{
			// assumption is that format of the grid DTM height map is *_SRGB, I don't think we need any code to ensure that

			auto* region = gridDTMHeightMap->getRegion(0, core::vectorSIMDu32(0.0f));
			auto imageExtent = region->getExtent();
			auto imagePixelSize = asset::getBytesPerPixel(gridDTMHeightMap->getCreationParameters().format).getIntegerApprox();
			float* imageData = static_cast<float*>(gridDTMHeightMap->getBuffer()->getPointer()) + region->bufferOffset;
			const size_t imageByteSize = gridDTMHeightMap->getImageDataSizeInBytes();
			assert(imageByteSize % sizeof(float) == 0);

			std::random_device rd;
			std::mt19937 mt(rd());
			std::uniform_int_distribution<int> dist(0, 1);

			for (int i = 0; i < imageByteSize; i += sizeof(float))
			{
				const bool isTexelEven = static_cast<bool>(dist(mt));
				E_CELL_DIAGONAL diagonal = isTexelEven ? TOP_LEFT_TO_BOTTOM_RIGHT : BOTTOM_LEFT_TO_TOP_RIGHT;

				setDiagonalModeBit(imageData, diagonal);
				imageData++;
			}

		}

		assert(gridDTMHeightMap);

		return true;
	}

	// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
	inline void workLoopBody() override
	{
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
				if (m_window->hasMouseFocus())
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
					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_1)
					{
						m_shadingModeExample = E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS;
					}
					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_2)
					{
						m_shadingModeExample = E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS;
					}
					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_3)
					{
						m_shadingModeExample = E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS;
					}
				}
			}
		, m_logger.get());
		
		const bool isCachingDraw = CacheAndReplay && m_realFrameIx == 0u;
		if (isCachingDraw)
		{
			SIntendedSubmitInfo invalidSubmit = {};
			addObjects(invalidSubmit); // if any overflows happen here, it will add to our replay cache and not submit anything
			replayCaches.push_back(drawResourcesFiller.createReplayCache());
			finishedCachingDraw = true;
		}

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
		
		if (CacheAndReplay)
		{
			// to size-1u because we only want to submit overflows here.
			for (uint32_t i = 0u; i < replayCaches.size() - 1u; ++i)
			{
				drawResourcesFiller.setReplayCache(replayCaches[i].get());
				submitDraws(m_intendedNextSubmit, true);
				drawResourcesFiller.unsetReplayCache();
			}
			if (!replayCaches.empty())
				drawResourcesFiller.setReplayCache(replayCaches.back().get());
		}
		else
		{
			addObjects(m_intendedNextSubmit);
		}

		endFrameRender(m_intendedNextSubmit);

		if (CacheAndReplay)
			drawResourcesFiller.unsetReplayCache();

#ifdef BENCHMARK_TILL_FIRST_FRAME
		if (!stopBenchamrkFlag)
		{
			const std::chrono::steady_clock::time_point stopBenchmark = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopBenchmark - startBenchmark);
			std::cout << "Time taken till first render pass: " << duration.count() << " milliseconds" << std::endl;
			stopBenchamrkFlag = true;
		}
#endif
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
		const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,0.f} };
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

		// you could do this later but only use renderpassInitial on first draw
		cb->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		// Wait what's going on here? empty Renderpass!?
		cb->endRenderPass();

		return true;
	}
	
	void submitDraws(SIntendedSubmitInfo& intendedSubmitInfo, bool inBetweenSubmit)
	{
		const bool isCachingDraw = CacheAndReplay && m_realFrameIx == 0u && !finishedCachingDraw;
		if (isCachingDraw)
		{
			drawResourcesFiller.markFrameUsageComplete(intendedSubmitInfo.getFutureScratchSemaphore().value);
			replayCaches.push_back(drawResourcesFiller.createReplayCache());
			return; // we don't record, submit or do anything, just caching the draw resources
		}

		drawResourcesFiller.pushAllUploads(intendedSubmitInfo);

		m_currentRecordingCommandBufferInfo = intendedSubmitInfo.getCommandBufferForRecording(); // drawResourcesFiller.pushAllUploads might've overflow submitted and changed the current recording command buffer

		// Use the current recording command buffer of the intendedSubmitInfos scratchCommandBuffers, it should be in recording state
		auto* cb = m_currentRecordingCommandBufferInfo->cmdbuf;
		
		const auto& resourcesCollection = drawResourcesFiller.getResourcesCollection();
		const auto& resourcesGPUBuffer = drawResourcesFiller.getResourcesGPUBuffer();

		float64_t3x3 projectionToNDC;
		projectionToNDC = m_Camera.constructViewProjection();

		// TEST CAMERA ROTATION
		if constexpr (testCameraRotation)
			projectionToNDC = rotateBasedOnTime(projectionToNDC);

		Globals globalData = {};
		uint64_t baseAddress = resourcesGPUBuffer->getDeviceAddress();
		globalData.pointers = {
			.lineStyles				= baseAddress + resourcesCollection.lineStyles.bufferOffset,
			.dtmSettings			= baseAddress + resourcesCollection.dtmSettings.bufferOffset,
			.customProjections		= baseAddress + resourcesCollection.customProjections.bufferOffset,
			.customClipRects		= baseAddress + resourcesCollection.customClipRects.bufferOffset,
			.mainObjects			= baseAddress + resourcesCollection.mainObjects.bufferOffset,
			.drawObjects			= baseAddress + resourcesCollection.drawObjects.bufferOffset,
			.geometryBuffer			= baseAddress + resourcesCollection.geometryInfo.bufferOffset,
		};
		globalData.antiAliasingFactor = 1.0;// +abs(cos(m_timeElapsed * 0.0008)) * 20.0f;
		globalData.resolution = uint32_t2{ m_window->getWidth(), m_window->getHeight() };
		globalData.defaultProjectionToNDC = projectionToNDC;
		float screenToWorld = getScreenToWorldRatio(globalData.defaultProjectionToNDC, globalData.resolution);
		globalData.screenToWorldScaleTransform = float64_t3x3(	1.0f / screenToWorld, 0.0f, 0.0f,
																0.0f, 1.0f / screenToWorld, 0.0f,
																0.0f, 0.0f, 1.0f);
		globalData.miterLimit = 10.0f;
		globalData.currentlyActiveMainObjectIndex = drawResourcesFiller.getActiveMainObjectIndex();
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
		
		IGPUDescriptorSet* descriptorSets[] = { descriptorSet0.get(), descriptorSet1.get() };
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, pipelineLayout.get(), 0u, 2u, descriptorSets);
		
		cb->bindGraphicsPipeline(graphicsPipeline.get());

		for (auto& drawCall : drawResourcesFiller.getDrawCalls())
		{
			if (drawCall.isDTMRendering)
			{
				cb->bindIndexBuffer({ .offset = resourcesCollection.geometryInfo.bufferOffset + drawCall.dtm.indexBufferOffset, .buffer = drawResourcesFiller.getResourcesGPUBuffer()}, asset::EIT_32BIT);

				PushConstants pc = {
					.triangleMeshVerticesBaseAddress = drawCall.dtm.triangleMeshVerticesBaseAddress + resourcesGPUBuffer->getDeviceAddress() + resourcesCollection.geometryInfo.bufferOffset,
					.triangleMeshMainObjectIndex = drawCall.dtm.triangleMeshMainObjectIndex,
					.isDTMRendering = true
				};
				cb->pushConstants(graphicsPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0, sizeof(PushConstants), &pc);

				cb->drawIndexed(drawCall.dtm.indexCount, 1u, 0u, 0u, 0u);
			}
			else
			{
				PushConstants pc = {
					.isDTMRendering = false
				};
				cb->pushConstants(graphicsPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0, sizeof(PushConstants), &pc);

				const uint64_t indexOffset = drawCall.drawObj.drawObjectStart * 6u;
				const uint64_t indexCount = drawCall.drawObj.drawObjectCount * 6u;

				// assert(currentIndexCount == resourcesCollection.indexBuffer.getCount());
				cb->bindIndexBuffer({ .offset = resourcesCollection.indexBuffer.bufferOffset + indexOffset * sizeof(uint32_t), .buffer = resourcesGPUBuffer}, asset::EIT_32BIT);
				cb->drawIndexed(indexCount, 1u, 0u, 0u, 0u);
			}
		}

		if (fragmentShaderInterlockEnabled)
		{
			cb->bindGraphicsPipeline(resolveAlphaGraphicsPipeline.get());
			nbl::ext::FullScreenTriangle::recordDrawCall(cb);
		}
		
		if constexpr (DebugModeWireframe)
		{
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());

			for (auto& drawCall : drawResourcesFiller.getDrawCalls())
			{
				PushConstants pc = {
					.isDTMRendering = false
				};
				cb->pushConstants(debugGraphicsPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0, sizeof(PushConstants), &pc);

				const uint64_t indexOffset = drawCall.drawObj.drawObjectStart * 6u;
				const uint64_t indexCount = drawCall.drawObj.drawObjectCount * 6u;

				// assert(currentIndexCount == resourcesCollection.indexBuffer.getCount());
				cb->bindIndexBuffer({ .offset = resourcesCollection.indexBuffer.bufferOffset + indexOffset * sizeof(uint32_t), .buffer = resourcesGPUBuffer}, asset::EIT_32BIT);

				cb->drawIndexed(indexCount, 1u, 0u, 0u, 0u);
			}
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
	
	// virtual function so you can override as needed for some example father down the line
	virtual SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		retval.fragmentShaderPixelInterlock = FragmentShaderPixelInterlock;
		retval.nullDescriptor = true;
		return retval;
	}

	virtual video::SPhysicalDeviceLimits getRequiredDeviceLimits() const override
	{
		video::SPhysicalDeviceLimits retval = base_t::getRequiredDeviceLimits();
		retval.fragmentShaderBarycentric = true;

		return retval;
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

		if constexpr (mode == ExampleMode::CASE_0)
		{
			LineStyleInfo style = {};
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

			drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_1)
		{
			// For Taking
		}
		else if (mode == ExampleMode::CASE_2)
		{
			auto debug = [&](CPolyline polyline, LineStyleInfo lineStyle)
			{
				drawResourcesFiller.drawPolyline(polyline, lineStyle, intendedNextSubmit);
			};

			int32_t hatchDebugStep = m_hatchDebugStep;
			
			if (hatchDebugStep > 0)
			{
				// Degenerate and Corner cases for hatches
				{
					{
						float64_t miniGap = 1.0e-15;
						// degenerate major const line points:
						float64_t2 pointA = { 0.0, -50.0 + miniGap };
						float64_t2 pointB = { 50.0, -50.0 };
						CPolyline polyline;
						std::vector<float64_t2> linePoints;
						{
							linePoints.push_back({ 0.0, 0.0 });
							linePoints.push_back({ 50.0, 0.0 });
							linePoints.push_back(pointB);
							linePoints.push_back(pointA);
							linePoints.push_back({ 0.0, 0.0 });
						}
						polyline.addLinePoints(linePoints);
						Hatch hatch({ &polyline, 1u }, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
						drawResourcesFiller.drawHatch(hatch, float32_t4(0.4f, 1.0f, 0.1f, 1.0f), intendedNextSubmit);
						linePoints.clear();
						polyline.clearEverything();
						{
							linePoints.push_back(pointA);
							linePoints.push_back(pointB);
							linePoints.push_back({ 50.0, -100.0 });
							linePoints.push_back({ 0.0, -100.0 });
							linePoints.push_back(pointA);
						}
						polyline.addLinePoints(linePoints);
						Hatch hatch2({ &polyline, 1u }, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
						drawResourcesFiller.drawHatch(hatch2, float32_t4(0.4f, 0.6f, 0.8f, 1.0f), intendedNextSubmit);
					}
				}
				


				{
					{
						float64_t2 offset = { 150.0, 0.0 };
						float64_t miniGap = 1.0e-15 + abs(cos(m_timeElapsed * 0.00018)) * 15.0f;
						CPolyline polyline;
						std::vector<float64_t2> linePoints;
						{
							linePoints.push_back(offset + float64_t2{ 0.0, 0.0 });
							linePoints.push_back(offset + float64_t2{ 100.0, 0.0 });
							linePoints.push_back(offset + float64_t2{ 100.0, -100.0 });
							linePoints.push_back(offset + float64_t2{ 0.0, -100.0 + miniGap });
							linePoints.push_back(offset + float64_t2{ 0.0, 0.0 });
						}
						polyline.addLinePoints(linePoints);
						linePoints.clear();
						{
							linePoints.push_back(offset + float64_t2{ 20.0, -20.0 });
							linePoints.push_back(offset + float64_t2{ 80.0, -20.0 });
							linePoints.push_back(offset + float64_t2{ 80.0, -80.0 });
							linePoints.push_back(offset + float64_t2{ 20.0, -80.0 + miniGap });
							linePoints.push_back(offset + float64_t2{ 20.0, -20.0 });
						}
						polyline.addLinePoints(linePoints);
						Hatch hatch({ &polyline, 1u }, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
						drawResourcesFiller.drawHatch(hatch, float32_t4(0.4f, 1.0f, 0.1f, 1.0f), intendedNextSubmit);
					}
				}
			}
			if (hatchDebugStep > 0)
			{
#include "bike_hatch.h"
				for (uint32_t i = 0; i < polylines.size(); i++)
				{
					LineStyleInfo lineStyle = {};
					lineStyle.screenSpaceLineWidth = 5.0;
					lineStyle.color = float32_t4(float(i) / float(polylines.size()), 1.0 - (float(i) / float(polylines.size())), 0.0, 0.2);
					// assert(polylines[i].checkSectionsContunuity());
					//drawResourcesFiller.drawPolyline(polylines[i], lineStyle, intendedNextSubmit);
				}
				//printf("hatchDebugStep = %d\n", hatchDebugStep);
				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				Hatch hatch(polylines, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				//// std::cout << "Hatch::Hatch time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
				//std::sort(hatch.intersectionAmounts.begin(), hatch.intersectionAmounts.end());

				//auto percentile = [&](float percentile)
				//	{
				//		return hatch.intersectionAmounts[uint32_t(round(percentile * float(hatch.intersectionAmounts.size() - 1)))];
				//	};
				//printf(std::format(
				//	"Intersection amounts: 10%%: {}, 25%%: {}, 50%%: {}, 75%%: {}, 90%%: {}, 100%% (max): {}\n",
				//	percentile(0.1), percentile(0.25), percentile(0.5), percentile(0.75), percentile(0.9), hatch.intersectionAmounts[hatch.intersectionAmounts.size() - 1]
				//).c_str());
				drawResourcesFiller.drawHatch(hatch, float32_t4(0.6, 0.6, 0.1, 1.0f), intendedNextSubmit);
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

				Hatch hatch(polylines, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
				drawResourcesFiller.drawHatch(hatch, float32_t4(0.0, 1.0, 0.1, 1.0f), intendedNextSubmit);
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

				Hatch hatch(polylines, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
				drawResourcesFiller.drawHatch(hatch, float32_t4(1.0, 0.1, 0.1, 1.0f), intendedNextSubmit);
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
				Hatch hatch(polylines, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
				drawResourcesFiller.drawHatch(hatch, float32_t4(0.0, 0.0, 1.0, 1.0f), intendedNextSubmit);
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

				Hatch hatch({&polyline, 1u}, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
				drawResourcesFiller.drawHatch(hatch, float32_t4(1.0f, 0.325f, 0.103f, 1.0f), intendedNextSubmit);
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
			
				Hatch hatch({&polyline, 1u}, SelectedMajorAxis, logger_opt_smart_ptr(smart_refctd_ptr(m_logger)), &hatchDebugStep, debug);
				drawResourcesFiller.drawHatch(hatch, float32_t4(0.619f, 0.325f, 0.709f, 0.9f), intendedNextSubmit);
			}
		}
		else if (mode == ExampleMode::CASE_3)
		{
			// Testing Degenerate Cases Causing Bugs/Nan/Crashes
			// 0 Sized Rect --> generateOffsetPolyline shouldn't return nan
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				{
					linePoints.push_back({ 1.0, -20.0 });
					linePoints.push_back({ 1.0, -20.0 });
					linePoints.push_back({ 1.0, 0.0 });
					linePoints.push_back({ 1.0, 0.0 });
					linePoints.push_back({ 1.0, -20.0 });
				}
				polyline.addLinePoints(linePoints);
				auto parallelPoly = polyline.generateParallelPolyline(1.0);
				assert(!std::isnan(parallelPoly.getLinePointAt(0).p[0]));
			}

			LineStyleInfo style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.worldSpaceLineWidth = 2.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.1f);

			LineStyleInfo style2 = {};
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

			drawResourcesFiller.drawPolyline(originalPolyline, style, intendedNextSubmit);
			CPolyline offsettedPolyline = originalPolyline.generateParallelPolyline(+0.0 - 3.0 * abs(cos(10.0 * 0.0009)));
			CPolyline offsettedPolyline2 = originalPolyline.generateParallelPolyline(+0.0 + 3.0 * abs(cos(10.0 * 0.0009)));
			drawResourcesFiller.drawPolyline(offsettedPolyline, style2, intendedNextSubmit);
			drawResourcesFiller.drawPolyline(offsettedPolyline2, style2, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_4)
		{
			constexpr uint32_t CURVE_CNT = 16u;
			constexpr uint32_t SPECIAL_CASE_CNT = 6u;

			LineStyleInfo lineStyleInfo = {};
			lineStyleInfo.screenSpaceLineWidth = 7.0f;
			lineStyleInfo.worldSpaceLineWidth = 0.0f;
			lineStyleInfo.color = float32_t4(0.7f, 0.3f, 0.7f, 0.8f);
			lineStyleInfo.isRoadStyleFlag = false;

			std::vector<LineStyleInfo> lineStyleInfos(CURVE_CNT, lineStyleInfo);
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
					lineStyleInfos[curveIdx].color = float64_t4(0.7f, 0.3f, 0.1f, 0.5f);

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
					lineStyleInfos[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

						// make sure A.x == 0
					float64_t2 A = quadratics[curveIdx].P0 - 2.0 * quadratics[curveIdx].P1 + quadratics[curveIdx].P2;
					assert(A.x == 0);

					// special case 4 (symetric parabola)
					curveIdx++;
					quadratics[curveIdx].P0 = float64_t2(-150.0, 1.0);
					quadratics[curveIdx].P1 = float64_t2(2000.0, 0.0);
					quadratics[curveIdx].P2 = float64_t2(-150.0, -1.0);
					lineStyleInfos[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
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
				stipplePatterns[9] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f };
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
					lineStyleInfos[i].setStipplePatternData(stipplePatterns[i]);
					lineStyleInfos[i].phaseShift += abs(cos(m_timeElapsed * 0.0003));
					polylines[i].addQuadBeziers({ &quadratics[i], &quadratics[i] + 1u });

					float64_t2 linePoints[2u] = {};
					linePoints[0] = { -200.0, 50.0 - 5.0 * i };
					linePoints[1] = { -100.0, 50.0 - 6.0 * i };
					polylines[i].addLinePoints(linePoints);

					activIdx.push_back(i);
					if (std::find(activIdx.begin(), activIdx.end(), i) == activIdx.end())
						lineStyleInfos[i].stipplePatternSize = -1;

					polylines[i].preprocessPolylineWithStyle(lineStyleInfos[i]);
				}
			}

			for (uint32_t i = 0u; i < CURVE_CNT; i++)
				drawResourcesFiller.drawPolyline(polylines[i], lineStyleInfos[i], intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_5)
		{
//#define CASE_5_POLYLINE_1 // animated stipple pattern
#define CASE_5_POLYLINE_2 // miter test static
//#define CASE_5_POLYLINE_3 // miter test animated
//#define CASE_5_POLYLINE_4 // miter test animated (every angle)
//#define CASE_5_POLYLINE_5 // closed polygon
// #define CASE_5_POLYLINE_6 // stretching
//#define CASE_5_POLYLINE_7 // wide non solid lines

#if defined(CASE_5_POLYLINE_1)
			LineStyleInfo style = {};
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

			drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);

#elif defined(CASE_5_POLYLINE_2)

			LineStyleInfo style = {};
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
				polyline.addQuadBeziers(quadratics2);

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

			drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);

#elif defined(CASE_5_POLYLINE_3)
			LineStyleInfo style = {};
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

			drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);

#elif defined(CASE_5_POLYLINE_4)
			LineStyleInfo style = {};
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

			drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			//drawResourcesFiller.drawPolyline(polyline2, style, intendedNextSubmit);
#elif defined(CASE_5_POLYLINE_5)
			LineStyleInfo style = {};
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

			drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
#elif defined(CASE_5_POLYLINE_6)
			LineStyleInfo style = {};
			style.screenSpaceLineWidth = 3.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.85f, 0.1f, 0.1f, 0.5f);
			style.isRoadStyleFlag = false;

			LineStyleInfo shapeStyle = style;
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
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
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
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			}

			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 52.0 });
				linePoints.push_back({ -50.0 + linesLength, 52.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style);
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			}
			
			style.setStipplePatternData(stipplePattern, 7.5, true, false);
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 50.0 });
				linePoints.push_back({ -50.0 + linesLength, 50.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style, 1e-5, addShapesFunction);
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			}
			
			style.setStipplePatternData(stipplePattern, 7.5, true, true);
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 48.0 });
				linePoints.push_back({ -50.0 + linesLength, 48.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style, 1e-5, addShapesFunction);
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			}
			
			std::array<double, 3u> stipplePattern2 = { 2.5f, -5.0f, 2.5f };
			style.setStipplePatternData(stipplePattern2, 5.0, true, true);
			{
				CPolyline polyline;
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, 46.0 });
				linePoints.push_back({ -50.0 + linesLength, 46.0 });
				polyline.addLinePoints(linePoints);
				polyline.preprocessPolylineWithStyle(style, 1e-5, addShapesFunction);
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
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
				polyline.preprocessPolylineWithStyle(style, 1e-5, addShapesFunction);
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
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
				polyline.preprocessPolylineWithStyle(style, 1e-5, addShapesFunction);
				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			}

			drawResourcesFiller.drawPolyline(shapesPolyline, style, intendedNextSubmit);
#elif defined(CASE_5_POLYLINE_7)
			LineStyleInfo style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			LineStyleInfo style2 = {};
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

			// drawResourcesFiller.drawPolyline(originalPolyline, style, intendedNextSubmit);

			std::array<double, 2u> stipplePattern = { 2, -1 };
			style.setStipplePatternData(stipplePattern);
			
			style.phaseShift += abs(cos(m_timeElapsed * 0.0003));
			CPolyline offsetPolyline1, offsetPolyline2;
			originalPolyline.stippleBreakDown(style, [&](const CPolyline& smallPoly)
				{
					smallPoly.makeWideWhole(offsetPolyline1, offsetPolyline2, 0.1f, 1e-3);
					drawResourcesFiller.drawPolyline(smallPoly, style2, intendedNextSubmit);
					drawResourcesFiller.drawPolyline(offsetPolyline1, style2, intendedNextSubmit);
					drawResourcesFiller.drawPolyline(offsetPolyline2, style2, intendedNextSubmit);
				});

#endif

		}
		else if (mode == ExampleMode::CASE_6)
		{
			float64_t3x3 customProjection = float64_t3x3{
				1.0, 0.0, cos(m_timeElapsed * 0.0005) * 100.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, 1.0
			};

			/// [NOTE]: We set minClip and maxClip (in default worldspace) in such a way that minClip.y > maxClip.y so that minClipNDC.y < maxClipNDC.y
			// left half should be red and right half should be green
			WorldClipRect showLeft = {};
			showLeft.minClip  = float64_t2(-100.0, +1000.0);
			showLeft.maxClip  = float64_t2(0.0, -1000.0);
			WorldClipRect showRight = {};
			showRight.minClip = float64_t2(0.0, +1000.0);
			showRight.maxClip = float64_t2(100.0, -1000.0);

			LineStyleInfo leftLineStyle = {};
			leftLineStyle.screenSpaceLineWidth = 3.0f;
			leftLineStyle.worldSpaceLineWidth = 0.0f;
			leftLineStyle.color = float32_t4(1.0f, 0.1f, 0.1f, 0.9f);
			leftLineStyle.isRoadStyleFlag = false;
			LineStyleInfo rightLineStyle = {};
			rightLineStyle.screenSpaceLineWidth = 6.0f;
			rightLineStyle.worldSpaceLineWidth = 0.0f;
			rightLineStyle.color = float32_t4(0.1f, 1.0f, 0.1f, 0.9f);
			rightLineStyle.isRoadStyleFlag = false;
			
			CPolyline polyline3;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -20.0, 20.0 });
				linePoints.push_back({ 20.0, 70.0 });
				linePoints.push_back({ 20.0, -40.0 });
				polyline3.addLinePoints(linePoints);
			}
			CPolyline polyline1;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -20.0, 0.0 });
				linePoints.push_back({ -20.0, 50.0 });
				linePoints.push_back({ 20.0, 0.0 });
				linePoints.push_back({ 20.0, 50.0 });
				polyline1.addLinePoints(linePoints);
			}
			CPolyline polyline2;
			{

				std::vector<shapes::QuadraticBezier<double>> quadBeziers;
				curves::EllipticalArcInfo myCurve;
				{
					myCurve.majorAxis = { 20.0, 0.0 };
					myCurve.center = { 0.0, -25.0 };
					myCurve.angleBounds = {
						nbl::core::PI<double>() * 0.0,
						nbl::core::PI<double>() * 2.0
					};
					myCurve.eccentricity = 1.0;
				}

				curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
					{
						quadBeziers.push_back(info);
					};

				curves::Subdivision::adaptive(myCurve, 1e-3, addToBezier, 10u);
				polyline2.addQuadBeziers(quadBeziers);
			}

			// we do redundant and nested push/pops to test
			drawResourcesFiller.pushCustomClipRect(showLeft);
			{
				drawResourcesFiller.drawPolyline(polyline1, leftLineStyle, intendedNextSubmit);

				drawResourcesFiller.pushCustomClipRect(showRight);
				drawResourcesFiller.pushCustomProjection(customProjection);
				{
					drawResourcesFiller.drawPolyline(polyline1, rightLineStyle, intendedNextSubmit);
					drawResourcesFiller.drawPolyline(polyline2, rightLineStyle, intendedNextSubmit);
				}
				drawResourcesFiller.popCustomProjection();
				drawResourcesFiller.popCustomClipRect();
				
				drawResourcesFiller.drawPolyline(polyline2, leftLineStyle, intendedNextSubmit);

				drawResourcesFiller.pushCustomClipRect(showRight);
				{
					drawResourcesFiller.drawPolyline(polyline3, rightLineStyle, intendedNextSubmit);
					drawResourcesFiller.drawPolyline(polyline2, rightLineStyle, intendedNextSubmit);
					
					drawResourcesFiller.pushCustomClipRect(showLeft);
					{
					drawResourcesFiller.drawPolyline(polyline1, leftLineStyle, intendedNextSubmit);
					}
					drawResourcesFiller.popCustomClipRect();
				}
				drawResourcesFiller.popCustomClipRect();

				drawResourcesFiller.drawPolyline(polyline2, leftLineStyle, intendedNextSubmit);
			}
			drawResourcesFiller.popCustomClipRect();
			
		}
		else if (mode == ExampleMode::CASE_7)
		{
			for (uint32_t i = 0; i < sampleImages.size(); ++i)
			{
				uint64_t imageID = i * 69ull; // it can be hash or something of the file path the image was loaded from
				//printf(std::format("\n Image {} \n", i).c_str());
				drawResourcesFiller.ensureStaticImageAvailability({ imageID, sampleImages[i] }, intendedNextSubmit);
				drawResourcesFiller.addImageObject(imageID, { .topLeft = { 0.0 + (i) * 3.0, 0.0 }, .dirU = { 3.0 , 0.0 }, .aspectRatio = 1.0 }, intendedNextSubmit);
				//printf("\n");
			}

			LineStyleInfo lineStyle = 
			{
				.color = float32_t4(1.0f, 0.1f, 0.1f, 0.9f),
				.screenSpaceLineWidth = 3.0f,
				.worldSpaceLineWidth = 0.0f,
				.isRoadStyleFlag = false,
			};
			CPolyline polyline;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ 0.0, 0.0 });
				linePoints.push_back({ 1.0, 0.0 });
				linePoints.push_back({ 1.0, -1.0 });
				polyline.addLinePoints(linePoints);
			}
			drawResourcesFiller.drawPolyline(polyline, lineStyle, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_8)
		{
			constexpr double hatchFillShapeSize = 100.0;
			constexpr double hatchFillShapePadding = 10.0;

			// Hatches with MSDF Fill patterns
			for (uint32_t hatchFillShapeIdx = 1u; hatchFillShapeIdx < uint32_t(HatchFillPattern::COUNT); hatchFillShapeIdx++)
			{
				double totalShapesWidth = hatchFillShapeSize * double(uint32_t(HatchFillPattern::COUNT)) + hatchFillShapePadding * double(uint32_t(HatchFillPattern::COUNT) - 1);
				double offset = hatchFillShapeSize * double(hatchFillShapeIdx) + hatchFillShapePadding * double(hatchFillShapeIdx);
				// Center it
				offset -= totalShapesWidth / 2.0;

				{
					CPolyline squareBelow;
					{
						std::vector<float64_t2> points;
						auto addPt = [&](float64_t2 p)
						{
							auto point = p / 8.0;
							points.push_back(point * hatchFillShapeSize + float64_t2(offset, -300.0 - hatchFillShapeSize));
						};
						addPt(float64_t2(0.0, 0.0));
						addPt(float64_t2(8.0, 0.0));
						addPt(float64_t2(8.0, 8.0));
						addPt(float64_t2(0.0, 8.0));
						addPt(float64_t2(0.0, 0.0));
						squareBelow.addLinePoints(points);
					}

					Hatch filledHatch({&squareBelow, 1}, SelectedMajorAxis);
					// This draws a square that is textured with the fill pattern at hatchFillShapeIdx
					drawResourcesFiller.drawHatch(
						filledHatch, 
						float32_t4(1.0, 1.0, 1.0, 1.0f), 
						float32_t4(0.1, 0.1, 0.1, 1.0f), 
						HatchFillPattern(hatchFillShapeIdx), 
						intendedNextSubmit
					);
				}
			}

			if (singleLineText)
			{
				float32_t rotation = 0.0; // nbl::core::PI<float>()* abs(cos(m_timeElapsed * 0.00005));
				float32_t italicTiltAngle = nbl::core::PI<float>() / 9.0f;
				singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, m_font.get(), float64_t2(0.0,-100.0), float32_t2(1.0, 1.0), rotation, float32_t4(1.0, 1.0, 1.0, 1.0), 0.0f, 0.0f);
				singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, m_font.get(), float64_t2(0.0,-150.0), float32_t2(1.0, 1.0), rotation, float32_t4(1.0, 1.0, 1.0, 1.0), 0.0f, 0.5f);
				singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, m_font.get(), float64_t2(0.0,-200.0), float32_t2(1.0, 1.0), rotation, float32_t4(1.0, 1.0, 1.0, 1.0), italicTiltAngle, 0.0f);
				singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, m_font.get(), float64_t2(0.0,-250.0), float32_t2(1.0, 1.0), rotation, float32_t4(1.0, 1.0, 1.0, 1.0), italicTiltAngle, 0.5f);
				// singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, float64_t2(0.0,-200.0), float32_t2(1.0, 1.0), nbl::core::PI<float>() * abs(cos(m_timeElapsed * 0.00005)));
				// Smaller text to test level maps
				//singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, float64_t2(0.0,-130.0), float32_t2(0.4, 0.4), rotation);
				//singleLineText->Draw(drawResourcesFiller, intendedNextSubmit, float64_t2(0.0,-150.0), float32_t2(0.2, 0.2), rotation);
			}

			const bool drawTextHatches = true;
			if (drawTextHatches)
			{
				constexpr auto TestString = "Hatch: The quick brown fox jumps over the lazy dog. !@#$%&*()_+-";

				auto penX = -100.0;
				auto penY = -500.0;
				auto previous = 0;

				float64_t2 currentBaselineStart = float64_t2(0.0, 0.0);
				float64_t scale = 1.0 / 64.0;

				for (uint32_t i = 0; i < strlen(TestString); i++)
				{
					char k = TestString[i];
					auto glyphIndex = m_font->getGlyphIndex(wchar_t(k));
					const auto glyphMetrics = m_font->getGlyphMetrics(glyphIndex);
					const float64_t2 baselineStart = currentBaselineStart;

					currentBaselineStart += glyphMetrics.advance;
					
					struct TextGlyphBoundingBox
					{
						float64_t2 topLeft;
						float64_t2 dirU;
						float64_t2 dirV;
					} glyphBbox;

					if (glyphIndex != 0 || k != ' ')
					{
						glyphBbox.topLeft = baselineStart + glyphMetrics.horizontalBearing;
						glyphBbox.dirU = float64_t2(glyphMetrics.size.x, 0.0);
						glyphBbox.dirV = float64_t2(0.0, -glyphMetrics.size.y);

						{
							// Draw bounding box of the glyph
							LineStyleInfo bboxStyle = {};
							bboxStyle.screenSpaceLineWidth = 1.0f;
							bboxStyle.worldSpaceLineWidth = 0.0f;
							bboxStyle.color = float32_t4(0.619f, 0.325f, 0.709f, 0.5f);

							CPolyline newPoly = {};
							std::vector<float64_t2> points;
							points.push_back(glyphBbox.topLeft);
							points.push_back(glyphBbox.topLeft + glyphBbox.dirU);
							points.push_back(glyphBbox.topLeft + glyphBbox.dirU + glyphBbox.dirV);
							points.push_back(glyphBbox.topLeft + glyphBbox.dirV);
							points.push_back(glyphBbox.topLeft);
							newPoly.addLinePoints(points);
							drawResourcesFiller.drawPolyline(newPoly, bboxStyle, intendedNextSubmit);
						}

						{
							const auto msdfTextureIdx = uint32_t(k) - uint32_t(FirstGeneratedCharacter);

							FreetypeHatchBuilder hatchBuilder;
							{
								FT_Outline_Funcs ftFunctions;
								ftFunctions.move_to = &ftMoveTo;
								ftFunctions.line_to = &ftLineTo;
								ftFunctions.conic_to = &ftConicTo;
								ftFunctions.cubic_to = &ftCubicTo;
								ftFunctions.shift = 0;
								ftFunctions.delta = 0;
								auto error = FT_Outline_Decompose(&m_font->getGlyphSlot(glyphIndex)->outline, &ftFunctions, &hatchBuilder);
								assert(!error);
								hatchBuilder.finish();
							}
							msdfgen::Shape glyphShape;
							bool loadedGlyph = drawFreetypeGlyph(glyphShape, m_textRenderer->getFreetypeLibrary(), m_font->getFreetypeFace());
							assert(loadedGlyph);

							auto& shapePolylines = hatchBuilder.polylines;
							std::vector<CPolyline> transformedPolylines;

							auto transformPoint = [&](float64_t2 point)
							{
								auto shapeBounds = glyphShape.getBounds();
								float64_t2 scale = float64_t2(
									shapeBounds.r - shapeBounds.l,
									shapeBounds.t - shapeBounds.b
								);
								float64_t2 translate = float64_t2(-shapeBounds.l, -shapeBounds.b);
								
								auto pointIn0To1 = (point + translate) / scale;
								pointIn0To1.y = 1.0 - pointIn0To1.y; // Honestly not sure why this makes it go upside down
								auto aabbMin = glyphBbox.topLeft;
								auto aabbMax = glyphBbox.topLeft + glyphBbox.dirU + glyphBbox.dirV;
								return aabbMin + pointIn0To1 * (aabbMax - aabbMin);
							};

							for (uint32_t polylineIdx = 0; polylineIdx < shapePolylines.size(); polylineIdx++)
							{
								auto& polyline = shapePolylines[polylineIdx];
								if (polyline.getSectionsCount() == 0) continue;
								CPolyline transformedPolyline;
								for (uint32_t sectorIdx = 0; sectorIdx < polyline.getSectionsCount(); sectorIdx++)
								{
									auto& section = polyline.getSectionInfoAt(sectorIdx);
									if (section.type == ObjectType::LINE)
									{
										if (section.count == 0u) continue;

										std::vector<float64_t2> points;
										for (uint32_t i = section.index; i < section.index + section.count + 1; i++)
										{
											auto point = polyline.getLinePointAt(i).p;
											points.push_back(transformPoint(point));
										}
										transformedPolyline.addLinePoints(points);
									}
									else if (section.type == ObjectType::QUAD_BEZIER)
									{
										if (section.count == 0u) continue;

										std::vector<nbl::hlsl::shapes::QuadraticBezier<double>> beziers;
										for (uint32_t i = section.index; i < section.index + section.count; i++)
										{
											QuadraticBezierInfo bezier = polyline.getQuadBezierInfoAt(i);
											beziers.push_back(nbl::hlsl::shapes::QuadraticBezier<double>::construct(
												transformPoint(bezier.shape.P0),
												transformPoint(bezier.shape.P1),
												transformPoint(bezier.shape.P2)
											));
										}
										transformedPolyline.addQuadBeziers(beziers);
									}
								}
								transformedPolylines.push_back(transformedPolyline);
							}

							if (transformedPolylines.size() == 0) continue;
							Hatch hatch(transformedPolylines, SelectedMajorAxis);
							drawResourcesFiller.drawHatch(hatch, float32_t4(1.0, 0.8, 1.0, 1.0f), intendedNextSubmit);
						}

					}
				}
			}

		}
		else if (mode == ExampleMode::CASE_9)
		{
			// PYRAMID
			core::vector<TriangleMeshVertex> vertices = {
				//{ float64_t2(0.0, 0.0), 100.0 }, //0
				//{ float64_t2(-200.0, -200.0), 10.0 }, //1
				//{ float64_t2(200.0, -200.0), 10.0 }, //2
				//{ float64_t2(200.0, 200.0), -20.0 }, //3
				//{ float64_t2(-200.0, 200.0), 10.0 }, //4

				{ float64_t2(0.0, 0.0), 100.0 },
				{ float64_t2(-200.0, -200.0), 10.0 },
				{ float64_t2(200.0, -100.0), 10.0 },
				{ float64_t2(0.0, 0.0), 100.0 },
				{ float64_t2(200.0, -100.0), 10.0 },
				{ float64_t2(200.0, 200.0), -20.0 },
				{ float64_t2(0.0, 0.0), 100.0 },
				{ float64_t2(200.0, 200.0), -20.0 },
				{ float64_t2(-200.0, 200.0), 10.0 },
				{ float64_t2(0.0, 0.0), 100.0 },
				{ float64_t2(-200.0, 200.0), 10.0 },
				{ float64_t2(-200.0, -200.0), 10.0 },
			};

			core::vector<uint32_t> indices = {
				0, 1, 2,
				3, 4, 5,
				6, 7, 8,
				9, 10, 11
			};

			// SINGLE TRIANGLE
			/*core::vector<TriangleMeshVertex> vertices = {
				{ float64_t2(0.0, 0.0), -20.0 },
				{ float64_t2(-200.0, -200.0), 100.0 },
				{ float64_t2(200.0, -100.0), 80.0 },
			};

			core::vector<uint32_t> indices = {
				0, 1, 2
			};*/

			CTriangleMesh mesh;
			mesh.setVertices(std::move(vertices));
			mesh.setIndices(std::move(indices));

			DTMSettingsInfo dtmInfo{};
			//dtmInfo.mode |= E_DTM_MODE::OUTLINE;
			dtmInfo.mode |= E_DTM_MODE::HEIGHT_SHADING;
			dtmInfo.mode |= E_DTM_MODE::CONTOUR;

			dtmInfo.outlineStyleInfo.screenSpaceLineWidth = 0.0f;
			dtmInfo.outlineStyleInfo.worldSpaceLineWidth = 1.0f;
			dtmInfo.outlineStyleInfo.color = float32_t4(0.0f, 0.39f, 0.0f, 1.0f);
			std::array<double, 4> outlineStipplePattern = { 0.0f, -5.0f, 20.0f, -5.0f };
			dtmInfo.outlineStyleInfo.setStipplePatternData(outlineStipplePattern);

			dtmInfo.contourSettingsCount = 2u;
			dtmInfo.contourSettings[0u].startHeight = 20;
			dtmInfo.contourSettings[0u].endHeight = 90;
			dtmInfo.contourSettings[0u].heightInterval = 9.98;
			dtmInfo.contourSettings[0u].lineStyleInfo.screenSpaceLineWidth = 0.0f;
			dtmInfo.contourSettings[0u].lineStyleInfo.worldSpaceLineWidth = 1.0f;
			dtmInfo.contourSettings[0u].lineStyleInfo.color = float32_t4(0.0f, 0.0f, 1.0f, 0.7f);
			std::array<double, 4> contourStipplePattern = { 0.0f, -5.0f, 10.0f, -5.0f };
			dtmInfo.contourSettings[0u].lineStyleInfo.setStipplePatternData(contourStipplePattern);

			dtmInfo.contourSettings[1u] = dtmInfo.contourSettings[0u];
			dtmInfo.contourSettings[1u].startHeight += 5.0f;
			dtmInfo.contourSettings[1u].heightInterval = 13.0f;
			dtmInfo.contourSettings[1u].lineStyleInfo.color = float32_t4(0.8f, 0.4f, 0.3f, 1.0f);

			// PRESS 1, 2, 3 TO SWITCH HEIGHT SHADING MODE
			// 1 - DISCRETE_VARIABLE_LENGTH_INTERVALS
			// 2 - DISCRETE_FIXED_LENGTH_INTERVALS
			// 3 - CONTINOUS_INTERVALS
			float animatedAlpha = (std::cos(m_timeElapsed * 0.0005) + 1.0) * 0.5;
			switch (m_shadingModeExample)
			{
				case E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS:
				{
					dtmInfo.heightShadingInfo.heightShadingMode = E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS;

					dtmInfo.heightShadingInfo.addHeightColorMapEntry(-10.0f, float32_t4(0.5f, 1.0f, 1.0f, 1.0f));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(20.0f, float32_t4(0.0f, 1.0f, 0.0f, 1.0f));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(25.0f, float32_t4(1.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(70.0f, float32_t4(1.0f, 0.0f, 0.0f, 1.0f));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(90.0f, float32_t4(1.0f, 0.0f, 0.0f, 1.0f));

					break;
				}
				case E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS:
				{
					dtmInfo.heightShadingInfo.intervalLength = 10.0f;
					dtmInfo.heightShadingInfo.intervalIndexToHeightMultiplier = dtmInfo.heightShadingInfo.intervalLength;
					dtmInfo.heightShadingInfo.isCenteredShading = false;
					dtmInfo.heightShadingInfo.heightShadingMode = E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS;
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(0.0f, float32_t4(0.0f, 0.0f, 1.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(25.0f, float32_t4(0.0f, 1.0f, 1.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(50.0f, float32_t4(0.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(75.0f, float32_t4(1.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(100.0f, float32_t4(1.0f, 0.0f, 0.0f, animatedAlpha));

					break;
				}
				case E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS:
				{
					dtmInfo.heightShadingInfo.heightShadingMode = E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS;
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(0.0f, float32_t4(0.0f, 0.0f, 1.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(25.0f, float32_t4(0.0f, 1.0f, 1.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(50.0f, float32_t4(0.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(75.0f, float32_t4(1.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(90.0f, float32_t4(1.0f, 0.0f, 0.0f, animatedAlpha));

					break;
				}
			}

			drawResourcesFiller.drawTriangleMesh(mesh, dtmInfo, intendedNextSubmit);

			dtmInfo.contourSettings[0u].lineStyleInfo.color = float32_t4(1.0f, 0.39f, 0.0f, 1.0f);
			dtmInfo.outlineStyleInfo.color = float32_t4(0.0f, 0.39f, 1.0f, 1.0f);
			for (auto& v : mesh.m_vertices)
			{
				v.pos += float64_t2(450.0, 200.0);
				v.height -= 10.0;
			}

			drawResourcesFiller.drawTriangleMesh(mesh, dtmInfo, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_10)
		{
			CPolyline polyline;
			
			LineStyleInfo style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.color = float32_t4(0.619f, 0.325f, 0.709f, 0.5f);

			for (uint32_t i = 0; i < 128u; ++i)
			{
				std::vector<shapes::QuadraticBezier<double>> quadBeziers;
				curves::EllipticalArcInfo myCircle;
				{
					myCircle.majorAxis = { 0.05 , 0.0};
					myCircle.center = { 0.0 + i * 0.1, i * 0.1 };
					myCircle.angleBounds = {
						nbl::core::PI<double>() * 0.0,
						nbl::core::PI<double>() * 2.0
					};
					myCircle.eccentricity = 1.0;
				}

				curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
					{
						quadBeziers.push_back(info);
					};

				curves::Subdivision::adaptive(myCircle, 1e-5, addToBezier, 10u);
				polyline.addQuadBeziers(quadBeziers);
				// drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
				polyline.clearEverything();
			}
			
			// Testing Fixed Geometry
			{
				float64_t2 line0[2u] =
				{
					float64_t2(-1.0, 0.0),
					float64_t2(+1.0, 0.0),
				};
				float64_t2 line1[3u] =
				{
					float64_t2(0.0, -1.0),
					float64_t2(0.0, +1.0),
					float64_t2(+1.0, +1.0),
				};

				float64_t3x3 translateMat =
				{
					1.0, 0.0, 0.0,
					0.0, 1.0, 0.0,
					0.0, 0.0, 1.0
				};
				
				float64_t angle = m_timeElapsed * 0.001;
				float64_t2 dir = float64_t2{ cos(angle), sin(angle) };
				float64_t3x3 rotateMat =
				{
					dir.x, -dir.y, 0.0,
					dir.y, dir.x,  0.0,
					0.0, 0.0, 1.0
				};

				float64_t2 scale = float64_t2{ 100.0, 100.0 };
				float64_t3x3 scaleMat =
				{
					scale.x, 0.0, 0.0,
					0.0, scale.y, 0.0,
					0.0, 0.0, 1.0
				};

				float64_t3x3 transformation = nbl::hlsl::mul(translateMat, nbl::hlsl::mul(rotateMat, scaleMat));
				polyline.addLinePoints(line0);
				polyline.addLinePoints(line1);
				polyline.preprocessPolylineWithStyle(style);
				// drawResourcesFiller.drawPolyline(polyline, intendedNextSubmit);
				drawResourcesFiller.drawFixedGeometryPolyline(polyline, style, transformation, TransformationType::TT_FIXED_SCREENSPACE_SIZE, intendedNextSubmit);
			}
		}
		else if (mode == ExampleMode::CASE_11)
		{
			DTMSettingsInfo dtmInfo{};
			dtmInfo.mode |= E_DTM_MODE::OUTLINE;
			dtmInfo.mode |= E_DTM_MODE::HEIGHT_SHADING;
			dtmInfo.mode |= E_DTM_MODE::CONTOUR;

			dtmInfo.outlineStyleInfo.screenSpaceLineWidth = 0.0f;
			dtmInfo.outlineStyleInfo.worldSpaceLineWidth = 1.0f;
			dtmInfo.outlineStyleInfo.color = float32_t4(0.0f, 0.39f, 0.0f, 1.0f);
			//std::array<double, 4> outlineStipplePattern = { 0.0f, -5.0f, 20.0f, -5.0f };
			std::array<double, 4> outlineStipplePattern = { -10.0f, 10.0f };
			dtmInfo.outlineStyleInfo.setStipplePatternData(outlineStipplePattern);

			dtmInfo.contourSettingsCount = 2u;
			dtmInfo.contourSettings[0u].startHeight = 20;
			dtmInfo.contourSettings[0u].endHeight = 90;
			dtmInfo.contourSettings[0u].heightInterval = 10;
			dtmInfo.contourSettings[0u].lineStyleInfo.screenSpaceLineWidth = 0.0f;
			dtmInfo.contourSettings[0u].lineStyleInfo.worldSpaceLineWidth = 3.0f;
			dtmInfo.contourSettings[0u].lineStyleInfo.color = float32_t4(0.0f, 0.0f, 1.0f, 0.7f);
			std::array<double, 4> contourStipplePattern = { 0.0f, -5.0f, 10.0f, -5.0f };
			dtmInfo.contourSettings[0u].lineStyleInfo.setStipplePatternData(contourStipplePattern);

			dtmInfo.contourSettings[1u] = dtmInfo.contourSettings[0u];
			dtmInfo.contourSettings[1u].startHeight += 5.0f;
			dtmInfo.contourSettings[1u].heightInterval = 13.0f;
			dtmInfo.contourSettings[1u].lineStyleInfo.color = float32_t4(0.8f, 0.4f, 0.3f, 1.0f);

			// PRESS 1, 2, 3 TO SWITCH HEIGHT SHADING MODE
			// 1 - DISCRETE_VARIABLE_LENGTH_INTERVALS
			// 2 - DISCRETE_FIXED_LENGTH_INTERVALS
			// 3 - CONTINOUS_INTERVALS
			float animatedAlpha = (std::cos(m_timeElapsed * 0.0005) + 1.0) * 0.5;
			animatedAlpha = 1.0f;
			switch (m_shadingModeExample)
			{
				case E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS:
				{
					dtmInfo.heightShadingInfo.heightShadingMode = E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS;

					dtmInfo.heightShadingInfo.addHeightColorMapEntry(-10.0f, float32_t4(0.5f, 1.0f, 1.0f, 1.0f));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(20.0f, float32_t4(0.0f, 1.0f, 0.0f, 1.0f));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(25.0f, float32_t4(1.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(70.0f, float32_t4(1.0f, 0.0f, 0.0f, 1.0f));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(90.0f, float32_t4(1.0f, 0.0f, 0.0f, 1.0f));

					break;
				}
				case E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS:
				{
					dtmInfo.heightShadingInfo.intervalLength = 10.0f;
					dtmInfo.heightShadingInfo.intervalIndexToHeightMultiplier = dtmInfo.heightShadingInfo.intervalLength;
					dtmInfo.heightShadingInfo.isCenteredShading = false;
					dtmInfo.heightShadingInfo.heightShadingMode = E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS;
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(-20.0f, float32_t4(0.0f, 0.5f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(25.0f, float32_t4(0.0f, 0.7f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(50.0f, float32_t4(0.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(75.0f, float32_t4(1.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(100.0f, float32_t4(1.0f, 0.0f, 0.0f, animatedAlpha));

					break;
				}
				case E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS:
				{
					dtmInfo.heightShadingInfo.heightShadingMode = E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS;
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(0.0f, float32_t4(0.0f, 0.0f, 1.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(25.0f, float32_t4(0.0f, 1.0f, 1.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(50.0f, float32_t4(0.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(75.0f, float32_t4(1.0f, 1.0f, 0.0f, animatedAlpha));
					dtmInfo.heightShadingInfo.addHeightColorMapEntry(90.0f, float32_t4(1.0f, 0.0f, 0.0f, animatedAlpha));

					break;
				}
			}

			constexpr float HeightMapCellWidth = 20.0f;
			const auto heightMapExtent = gridDTMHeightMap->getCreationParameters().extent;
			assert(heightMapExtent.width > 0 && heightMapExtent.height > 0);

			float64_t2 worldSpaceExtents;
			const float64_t2 topLeft = { -400.0f, 400.0f };
			worldSpaceExtents.x = (heightMapExtent.width - 1) * HeightMapCellWidth;
			worldSpaceExtents.y = (heightMapExtent.height - 1) * HeightMapCellWidth;
			const uint64_t heightMapTextureID = 0ull;

			constexpr bool DrawGridOnly = false;
			
			if(DrawGridOnly)
			{
				dtmInfo.mode = E_DTM_MODE::OUTLINE;
				drawResourcesFiller.drawGridDTM(topLeft, worldSpaceExtents, HeightMapCellWidth, InvalidTextureIndex, dtmInfo, intendedNextSubmit);
			}
			else
			{
				StaticImageInfo heightMapStaticImageInfo = {
				.imageID = heightMapTextureID,
				.cpuImage = gridDTMHeightMap,
				.forceUpdate = false,
				.imageViewFormatOverride = asset::E_FORMAT::EF_R32G32B32A32_UINT // for now we use only R32G32B32A32_* anyway
				};

				if (!drawResourcesFiller.ensureStaticImageAvailability(heightMapStaticImageInfo, intendedNextSubmit))
					m_logger->log("Grid DTM height map texture unavailable!", ILogger::ELL_ERROR);
				drawResourcesFiller.drawGridDTM(topLeft, worldSpaceExtents, HeightMapCellWidth, heightMapTextureID, dtmInfo, intendedNextSubmit);
			}

			// draw test polyline
#if 0
			{
				LineStyleInfo style = {};
				style.screenSpaceLineWidth = 0.0f;
				style.worldSpaceLineWidth = 15.0f;
				style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

				CPolyline polyline;
				{
					std::vector<float64_t2> linePoints;
					linePoints.push_back(topLeft);
					linePoints.push_back(topLeft + float64_t2(worldSpaceExtents.x, 0.0));
					linePoints.push_back(topLeft + float64_t2(worldSpaceExtents.x, -worldSpaceExtents.y));
					linePoints.push_back(topLeft + float64_t2(0.0, -worldSpaceExtents.y));
					linePoints.push_back(topLeft);
					polyline.addLinePoints(linePoints);
				}

				drawResourcesFiller.drawPolyline(polyline, style, intendedNextSubmit);
			}
#endif
		}
		else if (mode == ExampleMode::CASE_12)
		{
			// placeholder, actual path is right now hardcoded into the loader
			const static std::string georeferencedImagePath = "../../media/tiled_grid_mip_0.exr";

			constexpr float64_t3 topLeftViewportH = float64_t3(-1.0, -1.0, 1.0);
			constexpr float64_t3 topRightViewportH = float64_t3(1.0, -1.0, 1.0);
			constexpr float64_t3 bottomLeftViewportH = float64_t3(-1.0, 1.0, 1.0);
			constexpr float64_t3 bottomRightViewportH = float64_t3(1.0, 1.0, 1.0);

			//GeoreferencedImageParams georeferencedImageParams;
			//georeferencedImageParams.storagePath = georeferencedImagePath;
			//georeferencedImageParams.format = drawResourcesFiller.queryGeoreferencedImageFormat(georeferencedImagePath);
			//georeferencedImageParams.imageExtents = drawResourcesFiller.queryGeoreferencedImageExtents(georeferencedImagePath);

			image_id georefImageID = 6996;
			// Position at topLeft viewport
			auto projectionToNDC = m_Camera.constructViewProjection();
			// TEST CAMERA ROTATION
			if constexpr (testCameraRotation)
				projectionToNDC = rotateBasedOnTime(projectionToNDC);
			auto inverseViewProj = nbl::hlsl::inverse(projectionToNDC);

			// Get 1 viewport pixel to match `startingImagePixelsPerViewportPixel` pixels of the image by choosing appropriate dirU
			const static float64_t startingImagePixelsPerViewportPixels = 1.0;
			const static auto startingViewportWidthVector = nbl::hlsl::mul(inverseViewProj, topRightViewportH - topLeftViewportH);
			const static auto dirU = startingViewportWidthVector * float64_t(drawResourcesFiller.queryGeoreferencedImageExtents(georeferencedImagePath).x) / float64_t(startingImagePixelsPerViewportPixels * m_window->getWidth());

			const static auto startingTopLeft = nbl::hlsl::mul(inverseViewProj, topLeftViewportH);
			const uint32_t2 imageExtents = drawResourcesFiller.queryGeoreferencedImageExtents(georeferencedImagePath);
			OrientedBoundingBox2D georefImageBB = { .topLeft = startingTopLeft, .dirU = dirU, .aspectRatio = float32_t(imageExtents.y) / imageExtents.x };

			auto streamingState = drawResourcesFiller.ensureGeoreferencedImageEntry(georefImageID, georefImageBB, uint32_t2(m_window->getWidth(), m_window->getHeight()), inverseViewProj, georeferencedImagePath);
			constexpr static WorldClipRect invalidClipRect = { .minClip = float64_t2(std::numeric_limits<float64_t>::signaling_NaN()) };
			drawResourcesFiller.launchGeoreferencedImageTileLoads(georefImageID, streamingState.get(), invalidClipRect);

			drawResourcesFiller.drawGeoreferencedImage(georefImageID, std::move(streamingState), intendedNextSubmit);

			drawResourcesFiller.finalizeGeoreferencedImageTileLoads(intendedNextSubmit);

			//drawResourcesFiller.ensureGeoreferencedImageAvailability_AllocateIfNeeded(georefImageID, std::move(georeferencedImageParams), intendedNextSubmit);

			//drawResourcesFiller.addGeoreferencedImage(georefImageID, inverseViewProj, intendedNextSubmit);
		}
	}

	double getScreenToWorldRatio(const float64_t3x3& viewProjectionMatrix, uint32_t2 windowSize)
	{
		double idx_0_0 = viewProjectionMatrix[0u][0u] * (windowSize.x / 2.0);
		double idx_1_0 = viewProjectionMatrix[1u][0u] * (windowSize.y / 2.0);
		return hlsl::length(float64_t2(idx_0_0, idx_1_0));
	}

	float64_t3x3 rotateBasedOnTime(const float64_t3x3& projectionMatrix)
	{
		double rotation = abs(cos(m_timeElapsed * 0.0004)) * 0.25 * PI<double>();
		float64_t2 rotationVec = float64_t2(cos(rotation), sin(rotation));
		float64_t3x3 rotationParameter = float64_t3x3{
			rotationVec.x, rotationVec.y, 0.0,
			-rotationVec.y, rotationVec.x, 0.0,
			0.0, 0.0, 1.0
		};
		return nbl::hlsl::mul(projectionMatrix, rotationParameter);
	}

protected:
	std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
	clock_t::time_point start;

	std::vector<std::unique_ptr<DrawResourcesFiller::ReplayCache>> replayCaches = {}; // vector because there can be overflow submits
	bool finishedCachingDraw = false;

	bool fragmentShaderInterlockEnabled = false;

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

	smart_refctd_ptr<IGPUSampler>		msdfImageSampler;
	smart_refctd_ptr<IGPUSampler>		staticImageSampler;

	smart_refctd_ptr<IGPUBuffer>		m_globalsBuffer;
	smart_refctd_ptr<IGPUDescriptorSet>	descriptorSet0;
	smart_refctd_ptr<IGPUDescriptorSet>	descriptorSet1;
	DrawResourcesFiller drawResourcesFiller; // you can think of this as the scene data needed to draw everything, we only have one instance so let's use a timeline semaphore to sync all renders

	smart_refctd_ptr<ISemaphore> m_renderSemaphore; // timeline semaphore to sync frames together
	
	// timeline semaphore used for overflows (they need to be on their own timeline to count overflows)
	smart_refctd_ptr<ISemaphore> m_overflowSubmitScratchSemaphore; 
	SIntendedSubmitInfo m_intendedNextSubmit;
	
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	uint64_t m_realFrameIx = 0u;

	smart_refctd_ptr<IGPUGraphicsPipeline>		debugGraphicsPipeline;
	smart_refctd_ptr<IGPUDescriptorSetLayout>	descriptorSetLayout0;
	smart_refctd_ptr<IGPUDescriptorSetLayout>	descriptorSetLayout1;
	smart_refctd_ptr<IGPUPipelineLayout>		pipelineLayout;
	smart_refctd_ptr<IGPUGraphicsPipeline>		resolveAlphaGraphicsPipeline;
	smart_refctd_ptr<IGPUGraphicsPipeline>		graphicsPipeline;

	Camera2D m_Camera;

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CSwapchainResources>> m_surface;
	smart_refctd_ptr<IGPUImageView> pseudoStencilImageView;
	smart_refctd_ptr<IGPUImageView> colorStorageImageView;
	smart_refctd_ptr<TextRenderer> m_textRenderer;
	smart_refctd_ptr<FontFace> m_font;
	std::unique_ptr<SingleLineText> singleLineText = nullptr;
	
	std::vector<std::unique_ptr<msdfgen::Shape>> m_shapeMSDFImages = {};

	std::vector<smart_refctd_ptr<ICPUImage>> sampleImages;
	smart_refctd_ptr<ICPUImage> gridDTMHeightMap;

	static constexpr char FirstGeneratedCharacter = ' ';
	static constexpr char LastGeneratedCharacter = '~';

	std::vector<CPolyline> m_glyphPolylines[(LastGeneratedCharacter + 1) - FirstGeneratedCharacter];

	#ifdef BENCHMARK_TILL_FIRST_FRAME
	const std::chrono::steady_clock::time_point startBenchmark = std::chrono::high_resolution_clock::now();
	bool stopBenchamrkFlag = false;
	#endif
};

NBL_MAIN_FUNC(ComputerAidedDesign)

