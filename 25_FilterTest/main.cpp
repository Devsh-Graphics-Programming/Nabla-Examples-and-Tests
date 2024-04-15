// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/MonoDeviceApplication.hpp"

// TODO: these should come from <nabla.h> and nbl/asset/asset.h find out why they don't
#include "nbl/asset/filters/CRegionBlockFunctorFilter.h"
//#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::core;
using namespace nbl::video;


// TODO: move inside some class
static inline asset::IImageView<asset::ICPUImage>::E_TYPE getImageViewTypeFromImageType_CPU(const asset::IImage::E_TYPE type)
{
	switch (type)
	{
	case asset::IImage::ET_1D:
		return asset::ICPUImageView::ET_1D;
	case asset::IImage::ET_2D:
		return asset::ICPUImageView::ET_2D;
	case asset::IImage::ET_3D:
		return asset::ICPUImageView::ET_3D;
	default:
		assert(!"Invalid code path.");
		return static_cast<asset::IImageView<asset::ICPUImage>::E_TYPE>(0u);
	}
}

// TODO: move inside some class
static inline video::IGPUImageView::E_TYPE getImageViewTypeFromImageType_GPU(const video::IGPUImage::E_TYPE type)
{
	switch (type)
	{
	case video::IGPUImage::ET_1D:
		return video::IGPUImageView::ET_1D_ARRAY;
	case video::IGPUImage::ET_2D:
		return video::IGPUImageView::ET_2D_ARRAY;
	case video::IGPUImage::ET_3D:
		return video::IGPUImageView::ET_3D;
	default:
		assert(!"Invalid code path.");
		return static_cast<video::IGPUImageView::E_TYPE>(0u);
	}
}

// TODO: inherit from BasicMultiQueue app
class BlitFilterTestApp final : public virtual application_templates::MonoDeviceApplication
{
	using base_t = nbl::application_templates::MonoDeviceApplication;

	constexpr static uint32_t SC_IMG_COUNT = 3u;

	class ITest
	{
	public:
		virtual bool run() = 0;

	protected:
		ITest(core::smart_refctd_ptr<asset::ICPUImage>&& inImage, BlitFilterTestApp* parentApp) : m_inImage(std::move(inImage)), m_parentApp(parentApp) { assert(m_parentApp); }

		core::smart_refctd_ptr<asset::ICPUImage> m_inImage = nullptr;
		BlitFilterTestApp* m_parentApp = nullptr;

		void writeImage(core::smart_refctd_ptr<asset::ICPUImage>&& image, const char* path)
		{
			asset::ICPUImageView::SCreationParams viewParams = {};
			viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
			viewParams.image = std::move(image);
			viewParams.format = viewParams.image->getCreationParameters().format;
			viewParams.viewType = asset::ICPUImageView::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = viewParams.image->getCreationParameters().arrayLayers;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;

			auto imageViewToWrite = asset::ICPUImageView::create(std::move(viewParams));
			if (!imageViewToWrite)
			{
				m_parentApp->m_logger->log("Failed to create image view for the output image to write it to disk.", system::ILogger::ELL_ERROR);
				return;
			}

			asset::IAssetWriter::SAssetWriteParams writeParams(imageViewToWrite.get());
			if (!m_parentApp->assetManager->writeAsset(path, writeParams))
			{
				m_parentApp->m_logger->log("Failed to write the output image.", system::ILogger::ELL_ERROR);
				return;
			}
		}
	};

	template <typename BlitUtilities>
	class CBlitImageFilterTest : public ITest
	{
		using blit_utils_t = BlitUtilities;

	public:
		CBlitImageFilterTest(
			core::smart_refctd_ptr<asset::ICPUImage>&&				inImage,
			BlitFilterTestApp*										parentApp,
			const core::vectorSIMDu32&								outImageDim,
			const asset::E_FORMAT									outImageFormat,
			const char*												writeImagePath,
			const typename blit_utils_t::convolution_kernels_t&		convolutionKernels,
			const IBlitUtilities::E_ALPHA_SEMANTIC					alphaSemantic = asset::IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED,
			const float												referenceAlpha = 0.5f,
			const uint32_t											alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
			: ITest(std::move(inImage), parentApp), m_outImageDim(outImageDim), m_outImageFormat(outImageFormat),
			m_convolutionKernels(convolutionKernels), m_writeImagePath(writeImagePath),
			m_alphaSemantic(alphaSemantic), m_referenceAlpha(referenceAlpha), m_alphaBinCount(alphaBinCount)
		{}

		bool run() override
		{
			const auto& inImageExtent = m_inImage->getCreationParameters().extent;
			const auto& inImageFormat = m_inImage->getCreationParameters().format;

			assert(m_outImageDim.w == m_inImage->getCreationParameters().arrayLayers);

			auto outImage = m_parentApp->createCPUImage(m_outImageDim, m_inImage->getCreationParameters().type, m_outImageFormat);
			if (!outImage)
			{
				m_parentApp->m_logger->log("Failed to create CPU image for output.", system::ILogger::ELL_ERROR);
				return false;
			}

			// enabled clamping so the test outputs don't look weird on Kaiser filters which ring
			using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle, asset::IdentityDither, void, true, BlitUtilities>;
			typename BlitFilter::state_type blitFilterState(m_convolutionKernels);

			blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
			blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, m_inImage->getCreationParameters().arrayLayers) + m_inImage->getMipSize();
			blitFilterState.inImage = m_inImage.get();

			blitFilterState.outImage = outImage.get();

			blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
			blitFilterState.outExtentLayerCount = m_outImageDim;

			blitFilterState.alphaSemantic = m_alphaSemantic;
			blitFilterState.alphaBinCount = m_alphaBinCount;
			blitFilterState.alphaRefValue = m_referenceAlpha;

			blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
			blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

			if (!blit_utils_t::computeScaledKernelPhasedLUT(blitFilterState.scratchMemory + BlitFilter::getScratchOffset(&blitFilterState, BlitFilter::ESU_SCALED_KERNEL_PHASED_LUT), blitFilterState.inExtentLayerCount, blitFilterState.outExtentLayerCount, blitFilterState.inImage->getCreationParameters().type, m_convolutionKernels))
			{
				m_parentApp->m_logger->log("Failed to compute the LUT for blitting", system::ILogger::ELL_ERROR);
				return false;
			}

			if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
			{
				m_parentApp->m_logger->log("Failed to blit", system::ILogger::ELL_ERROR);
				return false;
			}

			_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);

			writeImage(std::move(outImage), m_writeImagePath);

			return true;
		}

	private:
		const core::vectorSIMDu32					m_outImageDim;
		const asset::E_FORMAT						m_outImageFormat;
		const typename blit_utils_t::convolution_kernels_t	m_convolutionKernels;
		const char*									m_writeImagePath;
		const IBlitUtilities::E_ALPHA_SEMANTIC		m_alphaSemantic;
		const float									m_referenceAlpha;
		const uint32_t								m_alphaBinCount;
	};

	class CFlattenRegionsImageFilterTest : public ITest
	{
	public:
		CFlattenRegionsImageFilterTest(
			core::smart_refctd_ptr<asset::ICPUImage>&& inImage,
			BlitFilterTestApp* parentApp,
			const bool enablePrefill,
			const asset::IImageFilter::IState::ColorValue& fillColor,
			const char* writeImagePath)
			: ITest(std::move(inImage), parentApp), m_enablePrefill(enablePrefill), m_writeImagePath(writeImagePath)
		{
			memcpy(&m_fillColorValue, &fillColor, sizeof(asset::IImageFilter::IState::ColorValue));
		}

		bool run() override
		{
			const auto& inImageExtent = m_inImage->getCreationParameters().extent;
			const auto& inImageFormat = m_inImage->getCreationParameters().format;
			const uint32_t inImageMipCount = m_inImage->getCreationParameters().mipLevels;

			core::smart_refctd_ptr<ICPUImage> flattenInImage;
			{
				const uint64_t bufferSizeNeeded = (inImageExtent.width * inImageExtent.height * inImageExtent.depth * asset::getTexelOrBlockBytesize(inImageFormat)) / 2ull;

				IImage::SCreationParams imageParams = {};
				imageParams.type = asset::ICPUImage::ET_2D;
				imageParams.format = inImageFormat;
				imageParams.extent = { inImageExtent.width, inImageExtent.height, inImageExtent.depth };
				imageParams.mipLevels = 1u;
				imageParams.arrayLayers = 1u;
				imageParams.samples = asset::ICPUImage::ESCF_1_BIT;

				auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(2ull);
				std::fill(imageRegions->begin(),imageRegions->end(),*m_inImage->getRegion(0,core::vectorSIMDu32(0,0,0)));
				{
					auto& region = (*imageRegions)[0];
					region.bufferOffset = 0ull;
					if (!region.bufferRowLength)
						region.bufferRowLength = region.imageExtent.width;
					region.imageExtent = { imageParams.extent.width / 2, imageParams.extent.height / 2, core::max(imageParams.extent.depth / 2, 1) };
					region.imageOffset = { 0u, 0u, 0u };
				}
				{
					auto& region = (*imageRegions)[1];
					if (!region.bufferRowLength)
						region.bufferRowLength = region.imageExtent.width;
					region.imageExtent = { imageParams.extent.width / 2, imageParams.extent.height / 2, core::max(imageParams.extent.depth / 2, 1) };
					region.imageOffset = { imageParams.extent.width / 2, imageParams.extent.height / 2, 0u };
					const auto blockSize = asset::getBlockDimensions(inImageFormat);
					region.bufferOffset += ((region.bufferRowLength/blockSize[0])*(region.imageOffset.y/blockSize[1])+region.imageOffset.x/blockSize[0])*asset::getTexelOrBlockBytesize(inImageFormat);
				}

				flattenInImage = ICPUImage::create(std::move(imageParams));
				if (!flattenInImage)
				{
					m_parentApp->m_logger->log("Failed to create the flatten input image.", system::ILogger::ELL_ERROR);
					return false;
				}

				flattenInImage->setBufferAndRegions(core::smart_refctd_ptr<ICPUBuffer>(m_inImage->getBuffer()), imageRegions);
			}

			asset::CFlattenRegionsImageFilter::CState filterState;
			filterState.inImage = flattenInImage.get();
			filterState.outImage = nullptr;
			filterState.preFill = m_enablePrefill;

			if (filterState.preFill)
			{
				const auto inImageFormat = filterState.inImage->getCreationParameters().format;
				if (asset::isBlockCompressionFormat(inImageFormat))
					memcpy(filterState.fillValue.asCompressedBlock, m_fillColorValue.asCompressedBlock, asset::getTexelOrBlockBytesize(inImageFormat));
				else if (asset::isFloatingPointFormat(inImageFormat))
					filterState.fillValue.asFloat = m_fillColorValue.asFloat;
				else
					_NBL_TODO();
			} 

			if (!asset::CFlattenRegionsImageFilter::execute(&filterState))
			{
				m_parentApp->m_logger->log("CFlattenRegionsImageFilter failed.", system::ILogger::ELL_ERROR);
				return false;
			}

			writeImage(core::smart_refctd_ptr(filterState.outImage), m_writeImagePath);

			return true;
		}

	private:
		const bool m_enablePrefill;
		asset::IImageFilter::IState::ColorValue m_fillColorValue;
		const char* m_writeImagePath;
	};

	template <typename Dither = IdentityDither, typename Normalization = void, bool Clamp = false>
	class CSwizzleAndConvertTest : public ITest
	{
	public:
		CSwizzleAndConvertTest(core::smart_refctd_ptr<asset::ICPUImage>&& inImage,
			BlitFilterTestApp* parentApp,
			const asset::E_FORMAT outFormat,
			core::vectorSIMDu32 inOffsetBaseLayer,
			core::vectorSIMDu32 outOffsetBaseLayer,
			asset::ICPUImageView::SComponentMapping swizzle,
			const char* writeImagePath)
			: ITest(std::move(inImage), parentApp), m_outFormat(outFormat), m_inOffsetBaseLayer(inOffsetBaseLayer), m_outOffsetBaseLayer(outOffsetBaseLayer), m_swizzle(swizzle), m_writeImagePath(writeImagePath)
		{}

		bool run() override
		{
			if (!m_inImage)
				return false;

			const auto& inImageExtent = m_inImage->getCreationParameters().extent;
			const auto& inImageFormat = m_inImage->getCreationParameters().format;

			auto outImage = m_parentApp->createCPUImage(core::vectorSIMDu32(inImageExtent.width, inImageExtent.height, inImageExtent.depth, m_inImage->getCreationParameters().arrayLayers), m_inImage->getCreationParameters().type, m_outFormat);
			if (!outImage)
				return false;

			using convert_filter_t = asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, asset::DefaultSwizzle, Dither, Normalization, Clamp>;

			typename convert_filter_t::state_type filterState = {};
			filterState.extentLayerCount = core::vectorSIMDu32(inImageExtent.width, inImageExtent.height, inImageExtent.depth, m_inImage->getCreationParameters().arrayLayers) - m_inOffsetBaseLayer;
			assert((static_cast<core::vectorSIMDi32>(filterState.extentLayerCount) > core::vectorSIMDi32(0)).all());

			filterState.inOffsetBaseLayer = m_inOffsetBaseLayer;
			filterState.outOffsetBaseLayer = m_outOffsetBaseLayer;
			filterState.inMipLevel = 0;
			filterState.outMipLevel = 0;
			filterState.inImage = m_inImage.get();
			filterState.outImage = outImage.get();
			filterState.swizzle = m_swizzle;

			if constexpr (std::is_same_v<Dither, asset::CWhiteNoiseDither>)
			{
				asset::CWhiteNoiseDither::CState ditherState;
				ditherState.texelRange.offset = filterState.inOffset;
				ditherState.texelRange.extent = filterState.extent;

				filterState.dither = asset::CWhiteNoiseDither();
				filterState.ditherState = &ditherState;
			}

			if (!convert_filter_t::execute(&filterState))
				return false;

			writeImage(std::move(outImage), m_writeImagePath);

			return true;
		}

	private:
		asset::E_FORMAT m_outFormat = asset::EF_UNKNOWN;
		core::vectorSIMDu32 m_inOffsetBaseLayer;
		core::vectorSIMDu32 m_outOffsetBaseLayer;
		asset::ICPUImageView::SComponentMapping m_swizzle;
		const char* m_writeImagePath;
	};

	template <typename BlitUtilities>
	class CComputeBlitTest : public ITest
	{
		using blit_utils_t = BlitUtilities;

	public:
		CComputeBlitTest(
			core::smart_refctd_ptr<asset::ICPUImage>&&			inImage,
			BlitFilterTestApp*									parentApp,
			const core::vectorSIMDu32&							outImageDim,
			const typename blit_utils_t::convolution_kernels_t& convolutionKernels,
			const IBlitUtilities::E_ALPHA_SEMANTIC				alphaSemantic = asset::IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED,
			const float											referenceAlpha = 0.f,
			const uint32_t										alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
			: ITest(std::move(inImage), parentApp), m_outImageDim(outImageDim), m_convolutionKernels(convolutionKernels),
			m_alphaSemantic(alphaSemantic), m_referenceAlpha(referenceAlpha), m_alphaBinCount(alphaBinCount)
		{}

		bool run() override
		{
			assert(m_inImage->getCreationParameters().mipLevels == 1);
			// GPU clamps when storing to a texture, so the CPU needs to as well
			using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle, asset::IdentityDither, void, true, blit_utils_t>;

			const asset::E_FORMAT inImageFormat = m_inImage->getCreationParameters().format;
			const asset::E_FORMAT outImageFormat = inImageFormat;
			const auto layerCount = m_inImage->getCreationParameters().arrayLayers;
			assert(m_outImageDim.w == layerCount);

			auto computeAlphaCoverage = [this](const double referenceAlpha, asset::ICPUImage* image) -> float
			{
				const uint32_t mipLevel = 0u;

				uint32_t alphaTestPassCount = 0u;

				const auto& extent = image->getCreationParameters().extent;
				const auto layerCount = image->getCreationParameters().arrayLayers;

				for (auto layer = 0; layer < layerCount; ++layer)
				{
					for (uint32_t z = 0u; z < extent.depth; ++z)
					{
						for (uint32_t y = 0u; y < extent.height; ++y)
						{
							for (uint32_t x = 0u; x < extent.width; ++x)
							{
								const core::vectorSIMDu32 texCoord(x, y, z, layer);
								core::vectorSIMDu32 dummy;
								const void* encodedPixel = image->getTexelBlockData(mipLevel, texCoord, dummy);

								double decodedPixel[4];
								asset::decodePixelsRuntime(image->getCreationParameters().format, &encodedPixel, decodedPixel, dummy.x, dummy.y);

								if (decodedPixel[3] > referenceAlpha)
									++alphaTestPassCount;
							}
						}
					}
				}

				const float alphaCoverage = float(alphaTestPassCount) / float(extent.width * extent.height * extent.depth * layerCount);
				return alphaCoverage;
			};

			// CPU
			core::vector<uint8_t> cpuOutput(static_cast<uint64_t>(m_outImageDim[0]) * m_outImageDim[1] * m_outImageDim[2] * asset::getTexelOrBlockBytesize(outImageFormat) * layerCount);
			{
				auto outImageCPU = m_parentApp->createCPUImage(m_outImageDim, m_inImage->getCreationParameters().type, outImageFormat, 1);
				if (!outImageCPU)
					return false;

				typename BlitFilter::state_type blitFilterState = typename BlitFilter::state_type(m_convolutionKernels);

				blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
				blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, layerCount) + m_inImage->getMipSize();
				blitFilterState.inImage = m_inImage.get();
				blitFilterState.outImage = outImageCPU.get();

				blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
				blitFilterState.outExtentLayerCount = m_outImageDim;

				blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
				blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
				blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
				blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

				blitFilterState.alphaSemantic = m_alphaSemantic;
				blitFilterState.alphaBinCount = m_alphaBinCount;
				blitFilterState.alphaRefValue = m_referenceAlpha;

				blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
				blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

				if (!BlitFilter::blit_utils_t::computeScaledKernelPhasedLUT(blitFilterState.scratchMemory + BlitFilter::getScratchOffset(&blitFilterState, BlitFilter::ESU_SCALED_KERNEL_PHASED_LUT), blitFilterState.inExtentLayerCount, blitFilterState.outExtentLayerCount, blitFilterState.inImage->getCreationParameters().type, m_convolutionKernels))
					m_parentApp->m_logger->log("Failed to compute the LUT for blitting\n", system::ILogger::ELL_ERROR);

				m_parentApp->m_logger->log("CPU begin..");
				if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
					m_parentApp->m_logger->log("Failed to blit\n", system::ILogger::ELL_ERROR);
				m_parentApp->m_logger->log("CPU end..");

				if (m_alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
					m_parentApp->m_logger->log("CPU alpha coverage: %f", system::ILogger::ELL_DEBUG, computeAlphaCoverage(m_referenceAlpha, outImageCPU.get()));

				// TODO: this is silly and just slows us down
				memcpy(cpuOutput.data(), outImageCPU->getBuffer()->getPointer(), cpuOutput.size());

				_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);
			}

			// GPU
			core::vector<uint8_t> gpuOutput(static_cast<uint64_t>(m_outImageDim[0]) * m_outImageDim[1] * m_outImageDim[2] * asset::getTexelOrBlockBytesize(outImageFormat) * layerCount);
			{
				constexpr auto BlitWorkgroupSize = video::CComputeBlit::DefaultBlitWorkgroupSize;

				assert(m_inImage->getCreationParameters().mipLevels == 1);

				auto transitionImageLayout = [this](core::smart_refctd_ptr<video::IGPUImage>&& image, const asset::IImage::E_LAYOUT finalLayout)
				{
					core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
					m_parentApp->m_device->createCommandBuffers(m_parentApp->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

					auto fence = m_parentApp->m_device->createFence(video::IGPUFence::ECF_UNSIGNALED);

					video::IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
					barrier.oldLayout = asset::IImage::EL_UNDEFINED;
					barrier.newLayout = finalLayout;
					barrier.srcQueueFamilyIndex = ~0u;
					barrier.dstQueueFamilyIndex = ~0u;
					barrier.image = image;
					barrier.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
					barrier.subresourceRange.levelCount = image->getCreationParameters().mipLevels;
					barrier.subresourceRange.layerCount = image->getCreationParameters().arrayLayers;

					cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
					cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
					cmdbuf->end();

					video::IGPUQueue::SSubmitInfo submitInfo = {};
					submitInfo.commandBufferCount = 1u;
					submitInfo.commandBuffers = &cmdbuf.get();
					m_parentApp->queue->submit(1u, &submitInfo, fence.get());
					m_parentApp->m_device->blockForFences(1u, &fence.get());
				};

				core::smart_refctd_ptr<video::IGPUImage> inImageGPU = nullptr;
				{
					m_parentApp->cpu2gpuParams.beginCommandBuffers();
					auto gpuArray = m_parentApp->cpu2gpu.getGPUObjectsFromAssets(&m_inImage, &m_inImage+ 1ull, m_parentApp->cpu2gpuParams);
					m_parentApp->cpu2gpuParams.waitForCreationToComplete();
					if (!gpuArray || gpuArray->size() < 1ull || (!(*gpuArray)[0]))
					{
						m_parentApp->m_logger->log("Cannot convert the inpute CPU image to GPU image", system::ILogger::ELL_ERROR);
						return false;
					}

					inImageGPU = gpuArray->begin()[0];

					// Do layout transition to SHADER_READ_ONLY_OPTIMAL 
					// I think it might be a good idea to allow the user to change asset::ICPUImage's initialLayout and have the asset converter
					// do the layout transition for them.
					transitionImageLayout(core::smart_refctd_ptr(inImageGPU), asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL);
				}

				core::smart_refctd_ptr<video::IGPUImage> outImageGPU = nullptr;
				{
					video::IGPUImage::SCreationParams creationParams = {};
					creationParams.flags = video::IGPUImage::ECF_MUTABLE_FORMAT_BIT;
					creationParams.type = inImageGPU->getCreationParameters().type;
					creationParams.format = outImageFormat;
					creationParams.extent = { m_outImageDim.x, m_outImageDim.y, m_outImageDim.z };
					creationParams.mipLevels = m_inImage->getCreationParameters().mipLevels; // Asset converter will make the mip levels 10 for inImage, so use the original value of m_inImage
					creationParams.arrayLayers = layerCount;
					creationParams.samples = video::IGPUImage::ESCF_1_BIT;
					creationParams.tiling = video::IGPUImage::ET_OPTIMAL;
					creationParams.usage = static_cast<video::IGPUImage::E_USAGE_FLAGS>(video::IGPUImage::EUF_STORAGE_BIT | video::IGPUImage::EUF_TRANSFER_SRC_BIT | video::IGPUImage::EUF_SAMPLED_BIT);

					outImageGPU = m_parentApp->m_device->createImage(std::move(creationParams));
					auto memReqs = outImageGPU->getMemoryReqs();
					memReqs.memoryTypeBits &= m_parentApp->m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					m_parentApp->m_device->allocate(memReqs, outImageGPU.get());

					transitionImageLayout(core::smart_refctd_ptr(outImageGPU), asset::IImage::EL_GENERAL);
				}

				// Create resources needed to do the blit
				auto blitFilter = video::CComputeBlit::create(core::smart_refctd_ptr(m_parentApp->m_device));

				const asset::E_FORMAT outImageViewFormat = blitFilter->getOutImageViewFormat(outImageFormat);

				const auto layersToBlit = layerCount;
				core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
				core::smart_refctd_ptr<video::IGPUImageView> outImageView = nullptr;
				{
					video::IGPUImageView::SCreationParams creationParams = {};
					creationParams.image = inImageGPU;
					creationParams.viewType = getImageViewTypeFromImageType_GPU(inImageGPU->getCreationParameters().type);
					creationParams.format = inImageGPU->getCreationParameters().format;
					creationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
					creationParams.subresourceRange.baseMipLevel = 0;
					creationParams.subresourceRange.levelCount = 1;
					creationParams.subresourceRange.baseArrayLayer = 0;
					creationParams.subresourceRange.layerCount = layersToBlit;

					video::IGPUImageView::SCreationParams outCreationParams = creationParams;
					outCreationParams.image = outImageGPU;
					outCreationParams.format = outImageViewFormat;

					inImageView = m_parentApp->m_device->createImageView(std::move(creationParams));
					outImageView = m_parentApp->m_device->createImageView(std::move(outCreationParams));
				}

				core::smart_refctd_ptr<video::IGPUImageView> normalizationInImageView = outImageView;
				core::smart_refctd_ptr<video::IGPUImage> normalizationInImage = outImageGPU;
				auto normalizationInFormat = outImageFormat;
				if (m_alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
				{
					normalizationInFormat = video::CComputeBlit::getCoverageAdjustmentIntermediateFormat(outImageFormat);

					if (normalizationInFormat != outImageFormat)
					{
						video::IGPUImage::SCreationParams creationParams;
						creationParams = outImageGPU->getCreationParameters();
						creationParams.format = normalizationInFormat;
						creationParams.usage = static_cast<video::IGPUImage::E_USAGE_FLAGS>(video::IGPUImage::EUF_STORAGE_BIT | video::IGPUImage::EUF_SAMPLED_BIT);
						normalizationInImage = m_parentApp->m_device->createImage(std::move(creationParams));
						auto memReqs = normalizationInImage->getMemoryReqs();
						memReqs.memoryTypeBits &= m_parentApp->m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
						m_parentApp->m_device->allocate(memReqs, normalizationInImage.get());
						transitionImageLayout(core::smart_refctd_ptr(normalizationInImage), asset::IImage::EL_GENERAL); // First we do the blit which requires storage image so starting layout is GENERAL

						video::IGPUImageView::SCreationParams viewCreationParams = {};
						viewCreationParams.image = normalizationInImage;
						viewCreationParams.viewType = getImageViewTypeFromImageType_GPU(inImageGPU->getCreationParameters().type);
						viewCreationParams.format = normalizationInImage->getCreationParameters().format;
						viewCreationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
						viewCreationParams.subresourceRange.baseMipLevel = 0;
						viewCreationParams.subresourceRange.levelCount = 1;
						viewCreationParams.subresourceRange.baseArrayLayer = 0;
						viewCreationParams.subresourceRange.layerCount = layersToBlit;

						normalizationInImageView = m_parentApp->m_device->createImageView(std::move(viewCreationParams));
					}
				}

				const core::vectorSIMDu32 inExtent(inImageGPU->getCreationParameters().extent.width, inImageGPU->getCreationParameters().extent.height, inImageGPU->getCreationParameters().extent.depth, 1);
				const auto inImageType = inImageGPU->getCreationParameters().type;

				// create scratch buffer
				core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer = nullptr;
				{
					const size_t scratchSize = blitFilter->getCoverageAdjustmentScratchSize(m_alphaSemantic, inImageType, m_alphaBinCount, layersToBlit);
					if (scratchSize > 0)
					{
						video::IGPUBuffer::SCreationParams creationParams = {};
						creationParams.size = scratchSize;
						creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);

						coverageAdjustmentScratchBuffer = m_parentApp->m_device->createBuffer(std::move(creationParams));
						auto memReqs = coverageAdjustmentScratchBuffer->getMemoryReqs();
						memReqs.memoryTypeBits &= m_parentApp->m_physicalDevice->getDeviceLocalMemoryTypeBits();
						m_parentApp->m_device->allocate(memReqs, coverageAdjustmentScratchBuffer.get());

						asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
						bufferRange.offset = 0ull;
						bufferRange.size = coverageAdjustmentScratchBuffer->getSize();
						bufferRange.buffer = coverageAdjustmentScratchBuffer;

						core::vector<uint32_t> fillValues(scratchSize / sizeof(uint32_t), 0u);
						m_parentApp->utilities->updateBufferRangeViaStagingBufferAutoSubmit(bufferRange, fillValues.data(), m_parentApp->queue);
					}
				}

				// create scaledKernelPhasedLUT and its view
				core::smart_refctd_ptr<video::IGPUBufferView> scaledKernelPhasedLUTView = nullptr;
				{
					const auto lutSize = blit_utils_t::getScaledKernelPhasedLUTSize(inExtent, m_outImageDim, inImageType, m_convolutionKernels);

					uint8_t* lutMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(lutSize, 32));
					if (!blit_utils_t::computeScaledKernelPhasedLUT(lutMemory, inExtent, m_outImageDim, inImageType, m_convolutionKernels))
					{
						m_parentApp->m_logger->log("Failed to compute scaled kernel phased LUT for the GPU case!", system::ILogger::ELL_ERROR);
						return false;
					}

					video::IGPUBuffer::SCreationParams creationParams = {};
					creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
					creationParams.size = lutSize;
					auto scaledKernelPhasedLUT = m_parentApp->m_device->createBuffer(std::move(creationParams));
					auto memReqs = scaledKernelPhasedLUT->getMemoryReqs();
					memReqs.memoryTypeBits &= m_parentApp->m_physicalDevice->getDeviceLocalMemoryTypeBits();
					m_parentApp->m_device->allocate(memReqs, scaledKernelPhasedLUT.get());

					// fill it up with data
					asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
					bufferRange.offset = 0ull;
					bufferRange.size = lutSize;
					bufferRange.buffer = scaledKernelPhasedLUT;
					m_parentApp->utilities->updateBufferRangeViaStagingBufferAutoSubmit(bufferRange, lutMemory, m_parentApp->queue);

					asset::E_FORMAT bufferViewFormat;
					if constexpr (std::is_same_v<blit_utils_t::lut_value_type, uint16_t>)
						bufferViewFormat = asset::EF_R16G16B16A16_SFLOAT;
					else if constexpr (std::is_same_v<blit_utils_t::lut_value_type, float>)
						bufferViewFormat = asset::EF_R32G32B32A32_SFLOAT;
					else
						assert(false);

					scaledKernelPhasedLUTView = m_parentApp->m_device->createBufferView(scaledKernelPhasedLUT.get(), bufferViewFormat, 0ull, scaledKernelPhasedLUT->getSize());

					_NBL_ALIGNED_FREE(lutMemory);
				}

				auto blitDSLayout = blitFilter->getDefaultBlitDescriptorSetLayout(m_alphaSemantic);
				auto kernelWeightsDSLayout = blitFilter->getDefaultKernelWeightsDescriptorSetLayout();
				auto blitPipelineLayout = blitFilter->getDefaultBlitPipelineLayout(m_alphaSemantic);

				video::IGPUDescriptorSetLayout* blitDSLayouts_raw[] = { blitDSLayout.get(), kernelWeightsDSLayout.get() };
				uint32_t dsCounts[] = { 2, 1 };
				auto descriptorPool = m_parentApp->m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, blitDSLayouts_raw, blitDSLayouts_raw + 2ull, dsCounts);

				core::smart_refctd_ptr<video::IGPUComputePipeline> blitPipeline = nullptr;
				core::smart_refctd_ptr<video::IGPUDescriptorSet> blitDS = nullptr;
				core::smart_refctd_ptr<video::IGPUDescriptorSet> blitWeightsDS = nullptr;

				core::smart_refctd_ptr<video::IGPUComputePipeline> alphaTestPipeline = nullptr;
				core::smart_refctd_ptr<video::IGPUComputePipeline> normalizationPipeline = nullptr;
				core::smart_refctd_ptr<video::IGPUDescriptorSet> normalizationDS = nullptr;

				if (m_alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
				{
					alphaTestPipeline = blitFilter->getAlphaTestPipeline(m_alphaBinCount, inImageType);
					normalizationPipeline = blitFilter->getNormalizationPipeline(normalizationInImage->getCreationParameters().type, outImageFormat, m_alphaBinCount);

					normalizationDS = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(blitDSLayout));
					blitFilter->updateDescriptorSet(normalizationDS.get(), nullptr, normalizationInImageView, outImageView, coverageAdjustmentScratchBuffer, nullptr);
				}

				blitPipeline = blitFilter->getBlitPipeline<BlitUtilities>(outImageFormat, inImageType, inExtent, m_outImageDim, m_alphaSemantic, m_convolutionKernels, BlitWorkgroupSize, m_alphaBinCount);
				blitDS = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(blitDSLayout));
				blitWeightsDS = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(kernelWeightsDSLayout));

				blitFilter->updateDescriptorSet(blitDS.get(), blitWeightsDS.get(), inImageView, normalizationInImageView, coverageAdjustmentScratchBuffer, scaledKernelPhasedLUTView);

				m_parentApp->m_logger->log("GPU begin..");
				m_parentApp->queue->startCapture();
				blitFilter->blit<BlitUtilities>(
					m_parentApp->queue, m_alphaSemantic,
					blitDS.get(), alphaTestPipeline.get(),
					blitDS.get(), blitWeightsDS.get(), blitPipeline.get(),
					normalizationDS.get(), normalizationPipeline.get(),
					inExtent, inImageType, inImageFormat, normalizationInImage, m_convolutionKernels,
					layersToBlit,
					coverageAdjustmentScratchBuffer, m_referenceAlpha,
					m_alphaBinCount, BlitWorkgroupSize);
				m_parentApp->queue->endCapture();
				m_parentApp->m_logger->log("GPU end..");

				if (m_alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
				if (outImageGPU->getCreationParameters().type == asset::IImage::ET_2D) // TODO: why alpha coverage only for 2D ?
				{
					if (layerCount > 1)
					{
						// This can be removed once ext::ScreenShot::createScreenShot works for multiple layers.
						m_parentApp->m_logger->log("Layer count (%d) is greater than 1 for a 2D image, not calculating GPU alpha coverage..", system::ILogger::ELL_WARNING, layerCount);
					}
					else
					{
						auto outCPUImageView = ext::ScreenShot::createScreenShot(
							m_parentApp->m_device.get(),
							m_parentApp->queue,
							nullptr,
							outImageView.get(),
							asset::EAF_NONE,
							asset::IImage::EL_GENERAL);

						m_parentApp->m_logger->log("GPU alpha coverage: %f", system::ILogger::ELL_DEBUG, computeAlphaCoverage(m_referenceAlpha, outCPUImageView->getCreationParameters().image.get()));
					}
				}

				// download results to check
				{
					const size_t downloadSize = gpuOutput.size();

					video::IGPUBuffer::SCreationParams creationParams = {};
					creationParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
					creationParams.size = downloadSize;
					core::smart_refctd_ptr<video::IGPUBuffer> downloadBuffer = m_parentApp->m_device->createBuffer(std::move(creationParams));

					auto memReqs = downloadBuffer->getMemoryReqs();
					memReqs.memoryTypeBits &= m_parentApp->m_physicalDevice->getDownStreamingMemoryTypeBits();
					m_parentApp->m_device->allocate(memReqs, downloadBuffer.get());

					core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
					m_parentApp->m_device->createCommandBuffers(m_parentApp->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
					auto fence = m_parentApp->m_device->createFence(video::IGPUFence::ECF_UNSIGNALED);

					cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

					asset::ICPUImage::SBufferCopy downloadRegion = {};
					downloadRegion.imageSubresource.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
					downloadRegion.imageSubresource.layerCount = layerCount;
					downloadRegion.imageExtent = outImageGPU->getCreationParameters().extent;

					// Todo(achal): Transition layout to TRANSFER_SRC_OPTIMAL
					cmdbuf->copyImageToBuffer(outImageGPU.get(), asset::IImage::EL_GENERAL, downloadBuffer.get(), 1u, &downloadRegion);

					cmdbuf->end();

					video::IGPUQueue::SSubmitInfo submitInfo = {};
					submitInfo.commandBufferCount = 1u;
					submitInfo.commandBuffers = &cmdbuf.get();
					m_parentApp->queue->submit(1u, &submitInfo, fence.get());

					m_parentApp->m_device->blockForFences(1u, &fence.get());

					video::IDeviceMemoryAllocation::MappedMemoryRange memoryRange = {};
					memoryRange.memory = downloadBuffer->getBoundMemory();
					memoryRange.length = downloadSize;
					uint8_t* mappedGPUData = reinterpret_cast<uint8_t*>(m_parentApp->m_device->mapMemory(memoryRange));

					memcpy(gpuOutput.data(), mappedGPUData, gpuOutput.size());
					m_parentApp->m_device->unmapMemory(downloadBuffer->getBoundMemory());

					// TODO: also save the gpu image to disk!
				}
			}

			assert(gpuOutput.size() == cpuOutput.size());

			const uint32_t outChannelCount = asset::getFormatChannelCount(outImageFormat);

			double sqErr = 0.0;
			uint8_t* cpuBytePtr = cpuOutput.data();
			uint8_t* gpuBytePtr = gpuOutput.data();
			const auto layerSize = m_outImageDim[2] * m_outImageDim[1] * m_outImageDim[0] * asset::getTexelOrBlockBytesize(outImageFormat);

			for (auto layer = 0; layer < layerCount; ++layer)
			{
				for (uint64_t k = 0u; k < m_outImageDim[2]; ++k)
				{
					for (uint64_t j = 0u; j < m_outImageDim[1]; ++j)
					{
						for (uint64_t i = 0; i < m_outImageDim[0]; ++i)
						{
							const uint64_t pixelIndex = (k * m_outImageDim[1] * m_outImageDim[0]) + (j * m_outImageDim[0]) + i;
							core::vectorSIMDu32 dummy;

							const void* cpuEncodedPixel = cpuBytePtr + (layer * layerSize) + pixelIndex * asset::getTexelOrBlockBytesize(outImageFormat);
							const void* gpuEncodedPixel = gpuBytePtr + (layer * layerSize) + pixelIndex * asset::getTexelOrBlockBytesize(outImageFormat);

							double cpuDecodedPixel[4];
							asset::decodePixelsRuntime(outImageFormat, &cpuEncodedPixel, cpuDecodedPixel, dummy.x, dummy.y);

							double gpuDecodedPixel[4];
							asset::decodePixelsRuntime(outImageFormat, &gpuEncodedPixel, gpuDecodedPixel, dummy.x, dummy.y);

							for (uint32_t ch = 0u; ch < outChannelCount; ++ch)
							{
								// TODO: change to logs
#if 1
								if (std::isnan(cpuDecodedPixel[ch]) || std::isinf(cpuDecodedPixel[ch]))
									__debugbreak();

								if (std::isnan(gpuDecodedPixel[ch]) || std::isinf(gpuDecodedPixel[ch]))
									__debugbreak();

								const auto diff = std::abs(cpuDecodedPixel[ch]-gpuDecodedPixel[ch]) / core::max(core::max(core::abs(cpuDecodedPixel[ch]),core::abs(gpuDecodedPixel[ch])),exp2(-16.f));
								if (diff>0.01f)
									__debugbreak();
#endif

								sqErr += (cpuDecodedPixel[ch] - gpuDecodedPixel[ch]) * (cpuDecodedPixel[ch] - gpuDecodedPixel[ch]);
							}
						}
					}
				}
			}

			// compute alpha coverage
			const uint64_t totalPixelCount = static_cast<uint64_t>(m_outImageDim[2]) * m_outImageDim[1] * m_outImageDim[0] * layerCount;
			const double RMSE = core::sqrt(sqErr / totalPixelCount);
			m_parentApp->m_logger->log("RMSE: %f", system::ILogger::ELL_INFO, RMSE);

			constexpr double MaxAllowedRMSE = 0.0046; // arbitrary

			return (RMSE <= MaxAllowedRMSE) && !std::isnan(RMSE);
		}

	private:
		const core::vectorSIMDu32							m_outImageDim;
		const IBlitUtilities::E_ALPHA_SEMANTIC				m_alphaSemantic;
		const typename blit_utils_t::convolution_kernels_t	m_convolutionKernels;
		const float											m_referenceAlpha = 0.f;
		const uint32_t										m_alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount;
	};

	class CRegionBlockFunctorFilterTest : public ITest
	{
	public:
		CRegionBlockFunctorFilterTest(core::smart_refctd_ptr<asset::ICPUImage>&& inImage, BlitFilterTestApp* parentApp, const char* writeImagePath)
			: ITest(std::move(inImage), parentApp), m_writeImagePath(writeImagePath)
		{}

		bool run() override
		{
			auto outImage = core::smart_refctd_ptr_static_cast<ICPUImage>(m_inImage->clone());
			if (!outImage)
				return false;

			const auto format = m_inImage->getCreationParameters().format;
			TexelBlockInfo blockInfo(format);

			const auto strides = m_inImage->getRegions().begin()->getByteStrides(blockInfo);

			auto copyFromLevel0 = [this, &outImage, format, &blockInfo, &strides](uint64_t dstByteOffset, core::vectorSIMDu32 coord)
			{
				const uint64_t srcByteOffset = m_inImage->getRegions().begin()->getByteOffset(coord, strides);

				uint8_t* src = reinterpret_cast<uint8_t*>(m_inImage->getBuffer()->getPointer()) + srcByteOffset;
				uint8_t* dst = reinterpret_cast<uint8_t*>(outImage->getBuffer()->getPointer()) + dstByteOffset;
				memcpy(dst, src, asset::getTexelOrBlockBytesize(format));
			};

			using region_block_filter_t = asset::CRegionBlockFunctorFilter<decltype(copyFromLevel0), false>;

			region_block_filter_t::CState filterState(copyFromLevel0, outImage.get(), outImage->getRegions().begin() + 1);

			for (uint32_t i = 1; i < outImage->getCreationParameters().mipLevels; ++i)
			{
				filterState.regionIterator = outImage->getRegions().begin() + i;

				if (!region_block_filter_t::execute(&filterState))
				{
					m_parentApp->m_logger->log("CRegionBlockFunctorFilter failed for mip level %u", system::ILogger::ELL_ERROR, i);
					return false;
				}
			}

			writeImage(std::move(outImage), m_writeImagePath);

			return true;
		}

	private:
		const char* m_writeImagePath;
	};

public:
	using base_t::base_t;
	BlitFilterTestApp() = default;

	virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
	{
		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		queue = getComputeQueue();
		commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		assetManager = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));
		utilities = make_smart_refctd_ptr<video::IUtilities>(smart_refctd_ptr(m_device));

		core::smart_refctd_ptr<IGPUCommandBuffer> transferCmdBuffer;
		core::smart_refctd_ptr<IGPUCommandBuffer> computeCmdBuffer;

		m_device->createCommandBuffers(commandPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmdBuffer);
		m_device->createCommandBuffers(commandPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &computeCmdBuffer);

		cpu2gpuParams.assetManager = assetManager.get();
		cpu2gpuParams.device = m_device.get();
		cpu2gpuParams.finalQueueFamIx = queue->getFamilyIndex();
		cpu2gpuParams.pipelineCache = nullptr;
		cpu2gpuParams.utilities = utilities.get();
		cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
		cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue;
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf = transferCmdBuffer;
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf = computeCmdBuffer;


		constexpr bool TestCPUBlitFilter = true;
		constexpr bool TestFlattenFilter = true;
		constexpr bool TestSwizzleAndConvertFilter = true;
		constexpr bool TestGPUBlitFilter = true;
		constexpr bool TestRegionBlockFunctorFilter = true;

		auto loadImage = [this](const char* path) -> core::smart_refctd_ptr<asset::ICPUImage>
		{
			constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
			asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
			auto imageBundle = assetManager->getAsset(path, loadParams);
			auto imageContents = imageBundle.getContents();

			if (imageContents.empty())
			{
				m_logger->log("Failed to load image at path %s", system::ILogger::ELL_ERROR, path);
				return nullptr;
			}

			auto asset = *imageContents.begin();

			core::smart_refctd_ptr<asset::ICPUImage> result;
			{
				if (asset->getAssetType() == asset::IAsset::ET_IMAGE_VIEW)
					result = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset)->getCreationParameters().image;
				else if (asset->getAssetType() == asset::IAsset::ET_IMAGE)
					result = std::move(core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset));
				else
					assert(!"Invalid code path.");
			}

			result->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);

			return result;
		};

		auto runTests = [this](const uint32_t count, std::unique_ptr<ITest>* tests)
		{
			assert(tests);

			for (uint32_t i = 0; i < count; ++i)
			{
				if (tests[i])
				{
					if (!tests[i]->run())
						m_logger->log("Test #%u failed.", system::ILogger::ELL_ERROR, i);
					else
						m_logger->log("Test #%u passed.", system::ILogger::ELL_INFO, i);
				}
			}
		};

		if (TestCPUBlitFilter)
		{
			using namespace asset;

			m_logger->log("CBlitImageFilter", system::ILogger::ELL_INFO);

			constexpr uint32_t TestCount = 2;
			std::unique_ptr<ITest> tests[TestCount] = { nullptr };

			// Test 0: Non-uniform downscale 2D BC format image with Mitchell
			{
				const char* path = "../../media/GLI/kueken7_rgba_dxt1_unorm.dds";
				auto inImage = loadImage(path);

				if (inImage)
				{
					const auto& inExtent = inImage->getCreationParameters().extent;
					const auto outImageDim = core::vectorSIMDu32(inExtent.width/2, inExtent.height/4, 1, 1);
					const auto outImageFormat = asset::EF_R8G8B8A8_SRGB;

					using BlitUtilities = CBlitUtilities<CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(core::vectorSIMDu32(inExtent.width, inExtent.height, inExtent.depth, 1), outImageDim);

					tests[0] = std::make_unique<CBlitImageFilterTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						outImageFormat,
						"CBlitImageFilter_0.png",
						convolutionKernels
					);
				}
			}

			// Test 1: Non-uniform upscale 2D BC format image with Kaiser
			{
				const char* path = "../../media/GLI/kueken7_rgba_dxt5_unorm.dds";
				auto inImage = loadImage(path);

				if (inImage)
				{
					const auto& inExtent = inImage->getCreationParameters().extent;
					const auto outImageDim = core::vectorSIMDu32(inExtent.width*2, inExtent.height*4, 1, 1);
					const auto outImageFormat = asset::EF_R32G32B32A32_SFLOAT;

					using BlitUtilities = CBlitUtilities<CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SKaiserFunction>>(core::vectorSIMDu32(inExtent.width, inExtent.height, inExtent.depth, 1), outImageDim);

					tests[1] = std::make_unique<CBlitImageFilterTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						outImageFormat,
						"CBlitImageFilter_1.exr",
						convolutionKernels
					);
				}
			}

			runTests(TestCount, tests);
		}

		if (TestFlattenFilter)
		{
			asset::IImageFilter::IState::ColorValue fillColorValue;
			auto getFillValueAsFirstBlockOrTexel = [&fillColorValue](asset::ICPUImage* image)
			{
				const auto format = image->getCreationParameters().format;
				if (asset::isBlockCompressionFormat(format))
					memcpy(fillColorValue.asCompressedBlock, image->getBuffer()->getPointer(), asset::getTexelOrBlockBytesize(format));
				else if (asset::isFloatingPointFormat(format))
					fillColorValue.asFloat.set(reinterpret_cast<float*>(image->getBuffer()->getPointer()));
				else
					_NBL_TODO();				
			};

			m_logger->log("CFlattenRegionsImageFilter", system::ILogger::ELL_INFO);

			constexpr uint32_t TestCount = 4;
			std::unique_ptr<ITest> tests[TestCount] = { nullptr };

			// Test 0: BC format with prefill
			{
				const char* path = "../../media/GLI/kueken7_rgba_dxt1_unorm.dds";
				auto inImage = loadImage(path);

				if (inImage)
				{
					const auto inImageFormat = inImage->getCreationParameters().format;
					getFillValueAsFirstBlockOrTexel(inImage.get());

					tests[0] = std::make_unique<CFlattenRegionsImageFilterTest>
					(
						std::move(inImage),
						this,
						true,
						fillColorValue,
						"CFlattenRegionsImageFilter_0.dds"
					);
				}
			}

			// Test 1: Non BC format with prefill
			{
				const char* path = "../../media/colorexr.exr";
				auto inImage = loadImage(path);

				if (inImage)
				{
					const auto inImageFormat = inImage->getCreationParameters().format;
					getFillValueAsFirstBlockOrTexel(inImage.get());

					tests[1] = std::make_unique<CFlattenRegionsImageFilterTest>
					(
						std::move(inImage),
						this,
						true,
						fillColorValue,
						"CFlattenRegionsImageFilter_1.exr"
					);
				}
			}

			// Test 2: BC format without prefill
			{
				const char* path = "../../media/GLI/kueken7_rgba_dxt5_unorm.dds";
				auto inImage = loadImage(path);
				
				if (inImage)
				{
					tests[2] = std::make_unique<CFlattenRegionsImageFilterTest>
					(
						std::move(inImage),
						this,
						false,
						asset::IImageFilter::IState::ColorValue(),
						"CFlattenRegionsImageFilter_2.dds"
					);
				}
			}

			// Test 3: Non BC format without prefill
			{
				const char* path = "../../media/color_space_test/R8G8B8_2.jpg";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[3] = std::make_unique<CFlattenRegionsImageFilterTest>
					(
						std::move(inImage),
						this,
						false,
						asset::IImageFilter::IState::ColorValue(),
						"CFlattenRegionsImageFilter_3.jpg"
					);
				}
			}

			runTests(TestCount, tests);
		}

		if (TestSwizzleAndConvertFilter)
		{
			m_logger->log("CSwizzleAndConvertImageFilter", system::ILogger::ELL_INFO);

			constexpr uint32_t TestCount = 6;
			std::unique_ptr<ITest> tests[TestCount] = { nullptr };

			// Test 0: Simple format conversion
			{
				const char* path = "../../media/GLI/kueken7_rgba_dxt1_unorm.dds";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[0] = std::make_unique<CSwizzleAndConvertTest<>>
					(
						std::move(inImage),
						this,
						asset::EF_R8G8B8A8_SRGB,
						core::vectorSIMDu32(0, 0, 0, 0),
						core::vectorSIMDu32(0, 0, 0, 0),
						asset::ICPUImageView::SComponentMapping(),
						"CSwizzleAndConvertImageFilter_0.png"
					);
				}
			}

			// Test 1: Non-trivial offsets
			{
				const char* path = "../../media/GLI/kueken7_rgba_dxt5_unorm.dds";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[1] = std::make_unique<CSwizzleAndConvertTest<>>
					(
						std::move(inImage),
						this,
						asset::EF_R32G32B32A32_SFLOAT,
						core::vectorSIMDu32(64, 64, 0, 0),
						core::vectorSIMDu32(64, 0, 0, 0),
						asset::ICPUImageView::SComponentMapping(),
						"CSwizzleAndConvertImageFilter_1.exr"
					);
				}
			}

			// Test 2: Non-trivial swizzle
			{
				const char* path = "../../media/GLI/dice_bc3.dds";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[2] = std::make_unique<CSwizzleAndConvertTest<>>
					(
						std::move(inImage),
						this,
						asset::EF_R32G32B32A32_SFLOAT,
						core::vectorSIMDu32(0, 0, 0, 0),
						core::vectorSIMDu32(0, 0, 0, 0),
						asset::ICPUImageView::SComponentMapping(asset::ICPUImageView::SComponentMapping::ES_G, asset::ICPUImageView::SComponentMapping::ES_B, asset::ICPUImageView::SComponentMapping::ES_R, asset::ICPUImageView::SComponentMapping::ES_A),
						"CSwizzleAndConvertImageFilter_2.exr"
					);
				}
			}

			// Test 3: Non-trivial dithering
			{
				const char* path = "../../media/GLI/kueken7_rgb_dxt1_unorm.ktx";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[3] = std::make_unique<CSwizzleAndConvertTest<asset::CWhiteNoiseDither>>
					(
						std::move(inImage),
						this,
						asset::EF_R8G8B8_SRGB,
						core::vectorSIMDu32(0, 0, 0, 0),
						core::vectorSIMDu32(0, 0, 0, 0),
						asset::ICPUImageView::SComponentMapping(),
						"CSwizzleAndConvertImageFilter_3.jpg"
					);
				}
			}

			// Test 4: Non-trivial normalization (warning, supposed to look like crap)
			{
				const char* path = "../../media/envmap/envmap_0.exr";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[4] = std::make_unique<CSwizzleAndConvertTest<asset::IdentityDither, asset::CGlobalNormalizationState>>
					(
						std::move(inImage),
						this,
						asset::EF_R32G32B32A32_SFLOAT,
						core::vectorSIMDu32(0, 0, 0, 0),
						core::vectorSIMDu32(0, 0, 0, 0),
						asset::ICPUImageView::SComponentMapping(),
						"CSwizzleAndConvertImageFilter_4.exr"
					);
				}
			}

			// Test 5: Non-trivial clamping
			{
				const char* path = "../../media/envmap/envmap_1.exr";
				auto inImage = loadImage(path);

				if (inImage)
				{
					tests[5] = std::make_unique<CSwizzleAndConvertTest<asset::IdentityDither, void, true>>
					(
						std::move(inImage),
						this,
						asset::EF_R8G8B8A8_SRGB,
						core::vectorSIMDu32(0, 0, 0, 0),
						core::vectorSIMDu32(0, 0, 0, 0),
						asset::ICPUImageView::SComponentMapping(),
						"CSwizzleAndConvertImageFilter_5.png"
					);
				}
			}

			runTests(TestCount, tests);
		}

		if (TestGPUBlitFilter)
		{
			using namespace asset;

			m_logger->log("CComputeBlit", system::ILogger::ELL_INFO);

			constexpr uint32_t TestCount = 6;
			std::unique_ptr<ITest> tests[TestCount] = { nullptr };

			// Test 0: Resize 1D image with Mitchell
			{
				const auto layerCount = 10;
				const core::vectorSIMDu32 inImageDim(59u, 1u, 1u, layerCount);
				const asset::IImage::E_TYPE inImageType = asset::IImage::ET_1D;
				const asset::E_FORMAT inImageFormat = asset::EF_R32_SFLOAT;
				auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

				if (inImage)
				{
					const core::vectorSIMDu32 outImageDim(800u, 1u, 1u, layerCount);

					auto reconstructionX = asset::CWeightFunction1D<asset::SMitchellFunction<>>();
					reconstructionX.stretchAndScale(0.35f);

					auto resamplingX = asset::CWeightFunction1D<asset::SMitchellFunction<>>();
					resamplingX.stretchAndScale(0.35f);

					using LutDataType = uint16_t;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(inImageDim, outImageDim, std::move(reconstructionX), std::move(resamplingX));

					tests[0] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						convolutionKernels
					);
				}
			}

			// Test 1: Resize 2D image with Kaiser
			{
				const char* path = "../../media/colorexr.exr";
				auto inImage = loadImage(path);
				if (inImage)
				{
					const auto& inExtent = inImage->getCreationParameters().extent;
					const auto layerCount = inImage->getCreationParameters().arrayLayers;
					const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth, layerCount);

					using LutDataType = float;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>,
						LutDataType
					>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SKaiserFunction>>(core::vectorSIMDu32(inExtent.width, inExtent.height, inExtent.depth, 1), outImageDim);

					tests[1] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						convolutionKernels
					);
				}
			}

			// Test 2: Resize 3D image with Box
			{
				const auto layerCount = 1u;
				const core::vectorSIMDu32 inImageDim(2u, 3u, 4u, layerCount);
				const asset::IImage::E_TYPE inImageType = asset::IImage::ET_3D;
				const asset::E_FORMAT inImageFormat = asset::EF_R32G32B32A32_SFLOAT;
				auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

				if (inImage)
				{
					const core::vectorSIMDu32 outImageDim(3u, 4u, 2u, layerCount);

					auto reconstructionX = asset::CWeightFunction1D<asset::SBoxFunction>();
					reconstructionX.stretchAndScale(0.35f);
					auto resamplingX = asset::CWeightFunction1D<asset::SBoxFunction>();
					resamplingX.stretchAndScale(0.35f);

					auto reconstructionY = asset::CWeightFunction1D<asset::SBoxFunction>();
					reconstructionY.stretchAndScale(9.f/16.f);
					auto resamplingY = asset::CWeightFunction1D<asset::SBoxFunction>();
					resamplingY.stretchAndScale(9.f/16.f);

					using LutDataType = uint16_t;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>>,
						LutDataType
					>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SBoxFunction>>(inImageDim, outImageDim, std::move(reconstructionX), std::move(resamplingX), std::move(reconstructionY), std::move(resamplingY));

					tests[2] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						convolutionKernels
					);
				}
			}

			// Test 3: Resize 2D image with alpha coverage adjustment
			{
				// We should find a better image for testing coverage adjustment
				const char* path = "../../media/colorexr.exr";
				auto inImage = loadImage(path);
				if (inImage)
				{
					const auto& inExtent = inImage->getCreationParameters().extent;
					const auto layerCount = inImage->getCreationParameters().arrayLayers;
					const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth, layerCount);
					const auto alphaSemantic = IBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
					const float referenceAlpha = 0.5f;
					const auto alphaBinCount = 1024;

					using LutDataType = float;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType
					>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(core::vectorSIMDu32(inExtent.width, inExtent.height, inExtent.depth, 1), outImageDim);

					tests[3] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						convolutionKernels,
						alphaSemantic,
						referenceAlpha,
						alphaBinCount
					);
				}
			}

			// Test 4: A larger 3D image with an atypical format
			{
				const auto layerCount = 1;
				const core::vectorSIMDu32 inImageDim(257u, 129u, 63u, layerCount);
				const asset::IImage::E_TYPE inImageType = asset::IImage::ET_3D;
				const asset::E_FORMAT inImageFormat = asset::EF_B10G11R11_UFLOAT_PACK32;
				auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

				if (inImage)
				{
					const core::vectorSIMDu32 outImageDim(256u, 128u, 64u, layerCount);

					using LutDataType = uint16_t;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType
					>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(inImageDim, outImageDim);

					tests[4] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						convolutionKernels
					);
				}
			}

			// Test 5: A 2D image with atypical dimensions and alpha coverage adjustment
			{
				const auto layerCount = 7;
				const core::vectorSIMDu32 inImageDim(511u, 1024u, 1u, layerCount);
				const asset::IImage::E_TYPE inImageType = asset::IImage::ET_2D;
				const asset::E_FORMAT inImageFormat = EF_R16G16B16A16_SNORM;
				auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

				if (inImage)
				{
					const core::vectorSIMDu32 outImageDim(512u, 257u, 1u, layerCount);
					const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
					const float referenceAlpha = 0.5f;
					const auto alphaBinCount = 4096;

					using LutDataType = float;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType
					>;

					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(inImageDim, outImageDim);

					tests[5] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						std::move(inImage),
						this,
						outImageDim,
						convolutionKernels,
						alphaSemantic,
						referenceAlpha,
						alphaBinCount
					);
				}
			}

			runTests(TestCount, tests);
		}

		if (TestRegionBlockFunctorFilter)
		{
			m_logger->log("CRegionBlockFunctorFilter", system::ILogger::ELL_INFO);

			constexpr uint32_t TestCount = 1;
			std::unique_ptr<ITest> tests[TestCount] = { nullptr };

			// Test 0: Copy the first NxM texels of the 0th mip level to the ith mip level, where N and M are dimensions of the ith mip level.
			{
				auto inImage = loadImage("../../media/GLI/kueken7_rgba_dxt5_unorm.dds");
				if (inImage)
					tests[0] = std::make_unique<CRegionBlockFunctorFilterTest>(std::move(inImage), this, "CRegionBlockFunctorFilter_0.dds");
			}

			runTests(TestCount, tests);
		}
	}

	bool onAppTerminated() override
	{
		m_device->waitIdle();
		return base_t::onAppTerminated();
	}

	void workLoopBody() override
	{
	}

	bool keepRunning() override
	{
		return false;
	}

	core::vector<queue_req_t> getQueueRequirements() const override
	{
		core::vector<queue_req_t> retval;

		using flags_t = video::IPhysicalDevice::E_QUEUE_FLAGS;
		// IGPUObjectFromAssetConverter requires the queue to support graphics as well.
		retval.push_back({ .requiredFlags = flags_t::EQF_COMPUTE_BIT | flags_t::EQF_TRANSFER_BIT | flags_t::EQF_GRAPHICS_BIT,.disallowedFlags = flags_t::EQF_NONE,.queueCount = 1,.maxImageTransferGranularity = {1,1,1} });

		return retval;
	}

private:
	// dims[3] is layer count
	core::smart_refctd_ptr<ICPUImage> createCPUImage(const core::vectorSIMDu32& dims, const asset::IImage::E_TYPE imageType, const asset::E_FORMAT format, const bool fillWithTestData = false)
	{
		IImage::SCreationParams imageParams = {};
		imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(asset::IImage::ECF_MUTABLE_FORMAT_BIT | asset::IImage::ECF_EXTENDED_USAGE_BIT);
		imageParams.type = imageType;
		imageParams.format = format;
		imageParams.extent = { dims[0], dims[1], dims[2] };
		imageParams.mipLevels = 1;
		imageParams.arrayLayers = dims[3];
		imageParams.samples = asset::ICPUImage::ESCF_1_BIT;
		imageParams.usage = asset::IImage::EUF_SAMPLED_BIT;

		auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull);
		auto& region = (*imageRegions)[0];
		region.bufferImageHeight = 0u;
		region.bufferOffset = 0ull;
		region.bufferRowLength = dims[0];
		region.imageExtent = { dims[0], dims[1], dims[2] };
		region.imageOffset = { 0u, 0u, 0u };
		region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = imageParams.arrayLayers;
		region.imageSubresource.mipLevel = 0;

		size_t bufferSize = imageParams.arrayLayers * asset::getTexelOrBlockBytesize(imageParams.format) * static_cast<size_t>(region.imageExtent.width) * region.imageExtent.height * region.imageExtent.depth;
		auto imageBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufferSize);

		core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imageParams));
		if (!image)
		{
			m_logger->log("Failed to create a CPU image", system::ILogger::ELL_ERROR);
			return nullptr;
		}

		image->setBufferAndRegions(core::smart_refctd_ptr(imageBuffer), imageRegions);

		if (fillWithTestData)
		{
			double pixelValueUpperBound = 20.0;
			if (asset::isNormalizedFormat(format) || format == asset::EF_B10G11R11_UFLOAT_PACK32)
				pixelValueUpperBound = 1.00000000001;

			std::uniform_real_distribution<double> dist(0.0, pixelValueUpperBound);
			std::mt19937 prng;

			uint8_t* bytePtr = reinterpret_cast<uint8_t*>(image->getBuffer()->getPointer());
			const auto layerSize = bufferSize / imageParams.arrayLayers;

			double dummyVal = 1.0;
			for (auto layer = 0; layer < image->getCreationParameters().arrayLayers; ++layer)
			{
				for (uint64_t k = 0u; k < dims[2]; ++k)
				{
					for (uint64_t j = 0u; j < dims[1]; ++j)
					{
						for (uint64_t i = 0; i < dims[0]; ++i)
						{
							double decodedPixel[4] = { 0 };
							for (uint32_t ch = 0u; ch < asset::getFormatChannelCount(format); ++ch)
								decodedPixel[ch] = dist(prng);

							const uint64_t pixelIndex = (k * dims[1] * dims[0]) + (j * dims[0]) + i;
							asset::encodePixelsRuntime(format, bytePtr + layer * layerSize + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
						}
					}
				}
			}
		}

		return image;
	}

	video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = base_t::getRequiredDeviceFeatures();

		retval.vulkanMemoryModelDeviceScope = true;

		return retval;
	}

	core::smart_refctd_ptr<asset::IAssetManager> assetManager;
	video::IGPUQueue* queue;
	core::smart_refctd_ptr<video::IGPUCommandPool> commandPool;
	core::smart_refctd_ptr<video::IUtilities> utilities;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	video::IGPUObjectFromAssetConverter cpu2gpu;
};

NBL_MAIN_FUNC(BlitFilterTestApp)