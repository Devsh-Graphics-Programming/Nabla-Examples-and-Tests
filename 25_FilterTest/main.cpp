// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/BasicMultiQueueApplication.hpp"

// TODO: these should come from <nabla.h> and nbl/asset/asset.h find out why they don't
#include "nbl/asset/filters/CRegionBlockFunctorFilter.h"
//#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;


// TODO: inherit from BasicMultiQueue app
class BlitFilterTestApp final : public virtual application_templates::BasicMultiQueueApplication
{
		static smart_refctd_ptr<ICPUImage> createCPUImage(const hlsl::uint32_t3 extent, const uint32_t layers, const IImage::E_TYPE imageType, const E_FORMAT format, const bool fillWithTestData = false)
		{
			IImage::SCreationParams imageParams = {};
			imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(asset::IImage::ECF_MUTABLE_FORMAT_BIT | asset::IImage::ECF_EXTENDED_USAGE_BIT);
			imageParams.type = imageType;
			imageParams.format = format;
			imageParams.extent = { extent[0], extent[1], extent[2] };
			imageParams.mipLevels = 1;
			imageParams.arrayLayers = layers;
			imageParams.samples = ICPUImage::ESCF_1_BIT;
			imageParams.usage = IImage::EUF_SAMPLED_BIT;

			smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imageParams));
			assert(image);
			
			const size_t bufferSize = (((static_cast<size_t>(layers) * extent[0]) * extent[1]) * extent[2]) * getTexelOrBlockBytesize(format);
			{
				auto imageRegions = make_refctd_dynamic_array<smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull);
				auto& region = imageRegions->front();
				region.bufferImageHeight = 0u;
				region.bufferOffset = 0ull;
				region.bufferRowLength = extent[0];
				region.imageExtent = { extent[0], extent[1], extent[2] };
				region.imageOffset = { 0u, 0u, 0u };
				region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = layers;
				region.imageSubresource.mipLevel = 0;

				image->setBufferAndRegions(make_smart_refctd_ptr<ICPUBuffer>(bufferSize),std::move(imageRegions));
			}

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
				for (auto layer = 0; layer < layers; ++layer)
				for (uint64_t k = 0u; k < extent[2]; ++k)
				for (uint64_t j = 0u; j < extent[1]; ++j)
				for (uint64_t i = 0; i < extent[0]; ++i)
				{
					double decodedPixel[4] = { 0 };
					for (uint32_t ch = 0u; ch < asset::getFormatChannelCount(format); ++ch)
						decodedPixel[ch] = dist(prng);

					const uint64_t pixelIndex = (k * extent[1] * extent[0]) + (j * extent[0]) + i;
					asset::encodePixelsRuntime(format, bytePtr + layer * layerSize + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
				}
			}

			return image;
		}

		using base_t = application_templates::BasicMultiQueueApplication;

		constexpr static uint32_t SC_IMG_COUNT = 3u;

		// Class to unify test inputs and writing out of the result
		class ITest
		{
			public:
				virtual bool run() = 0;

			protected:
				ITest(smart_refctd_ptr<ICPUImage>&& inImage, BlitFilterTestApp* parentApp) : m_inImage(std::move(inImage)), m_parentApp(parentApp) { assert(m_parentApp); }

				smart_refctd_ptr<ICPUImage> m_inImage = nullptr;
				BlitFilterTestApp* m_parentApp = nullptr;

				void writeImage(const ICPUImage* image, const std::string_view path)
				{
					const auto& params = image->getCreationParameters();

					ICPUImageView::SCreationParams viewParams = {};
					viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
					viewParams.image = core::smart_refctd_ptr<ICPUImage>(const_cast<ICPUImage*>(image));
					viewParams.format = params.format;
					switch (params.type)
					{
						case ICPUImage::ET_1D:
							viewParams.viewType = params.arrayLayers>1 ? ICPUImageView::ET_1D_ARRAY:ICPUImageView::ET_1D;
							break;
						case ICPUImage::ET_2D:
							viewParams.viewType = params.arrayLayers>1 ? ICPUImageView::ET_2D_ARRAY:ICPUImageView::ET_2D;
							break;
						default:
							viewParams.viewType = ICPUImageView::ET_3D;
							break;
					}
					viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
					viewParams.subresourceRange.baseArrayLayer = 0u;
					viewParams.subresourceRange.layerCount = params.arrayLayers;
					viewParams.subresourceRange.baseMipLevel = 0u;
					viewParams.subresourceRange.levelCount = params.mipLevels;

					auto imageViewToWrite = ICPUImageView::create(std::move(viewParams));
					if (!imageViewToWrite)
					{ 
						m_parentApp->m_logger->log("Failed to create image view for the output image to write it to disk.", system::ILogger::ELL_ERROR);
						return;
					}

					asset::IAssetWriter::SAssetWriteParams writeParams(imageViewToWrite.get());
					if (!m_parentApp->assetManager->writeAsset(core::string(path),writeParams))
					{
						m_parentApp->m_logger->log("Failed to write the output image.", system::ILogger::ELL_ERROR);
						return;
					}
				}
		};

		// CPU Blit test
		template <typename BlitUtilities> requires std::is_base_of_v<asset::IBlitUtilities,BlitUtilities>
		class CBlitImageFilterTest : public ITest
		{
				using blit_utils_t = BlitUtilities;
				using convolution_kernels_t = typename blit_utils_t::convolution_kernels_t;

			public:
				CBlitImageFilterTest(
					BlitFilterTestApp*						parentApp,
					smart_refctd_ptr<ICPUImage>&&			inImage,
					const hlsl::uint32_t3&					outImageDim,
					const uint32_t&							outImageLayers,
					const E_FORMAT							outImageFormat,
					const char*								writeImagePath,
					const convolution_kernels_t&			convolutionKernels,
					const IBlitUtilities::E_ALPHA_SEMANTIC	alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED,
					const float								referenceAlpha = 0.5f,
					const uint32_t							alphaBinCount = IBlitUtilities::DefaultAlphaBinCount)
					: ITest(std::move(inImage), parentApp),	m_convolutionKernels(convolutionKernels), m_writeImagePath(writeImagePath),
					m_outImageDim(outImageDim), m_outImageLayers(outImageLayers), m_outImageFormat(outImageFormat),
					m_alphaSemantic(alphaSemantic), m_referenceAlpha(referenceAlpha), m_alphaBinCount(alphaBinCount)
				{}

				bool run() override
				{
					const auto& inImageExtent = m_inImage->getCreationParameters().extent;
					const auto& inImageFormat = m_inImage->getCreationParameters().format;

					auto outImage = createCPUImage(m_outImageDim, m_outImageLayers, m_inImage->getCreationParameters().type, m_outImageFormat);
					if (!outImage)
					{
						m_parentApp->m_logger->log("Failed to create CPU image for output.", system::ILogger::ELL_ERROR);
						return false;
					}

					// enabled clamping so the test outputs don't look weird on Kaiser filters which ring
					using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle,asset::IdentityDither,void,true,BlitUtilities>;
					typename BlitFilter::state_type blitFilterState(m_convolutionKernels);
					
					const auto mipSize = m_inImage->getMipSize();

					blitFilterState.inOffsetBaseLayer = hlsl::uint32_t4(0,0,0,0);
					blitFilterState.inExtentLayerCount = hlsl::uint32_t4(mipSize.x,mipSize.y,mipSize.z,m_outImageLayers);
					blitFilterState.inImage = m_inImage.get();

					blitFilterState.outImage = outImage.get();

					blitFilterState.outOffsetBaseLayer = hlsl::uint32_t4();
					blitFilterState.outExtentLayerCount = hlsl::uint32_t4(m_outImageDim[0],m_outImageDim[1],m_outImageDim[2],m_outImageLayers);

					blitFilterState.alphaSemantic = m_alphaSemantic;
					blitFilterState.alphaBinCount = m_alphaBinCount;
					blitFilterState.alphaRefValue = m_referenceAlpha;

					blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
					auto scratch = std::make_unique<uint8_t[]>(blitFilterState.scratchMemoryByteSize);
					blitFilterState.scratchMemory = scratch.get();

					const auto lutOffsetInScratch = BlitFilter::getScratchOffset(&blitFilterState,BlitFilter::ESU_SCALED_KERNEL_PHASED_LUT);
					if (!blit_utils_t::computeScaledKernelPhasedLUT(
						blitFilterState.scratchMemory+lutOffsetInScratch,
						blitFilterState.inExtentLayerCount,
						blitFilterState.outExtentLayerCount,
						blitFilterState.inImage->getCreationParameters().type,
						m_convolutionKernels
					))
					{
						m_parentApp->m_logger->log("Failed to compute the LUT for blitting",ILogger::ELL_ERROR);
						return false;
					}

					if (!BlitFilter::execute(core::execution::par_unseq,&blitFilterState))
					{
						m_parentApp->m_logger->log("Failed to blit",ILogger::ELL_ERROR);
						return false;
					}

					writeImage(outImage.get(),m_writeImagePath);

					return true;
			}

			private:
				const convolution_kernels_t				m_convolutionKernels;
				const char*								m_writeImagePath;
				const hlsl::uint32_t3					m_outImageDim;
				const uint32_t							m_outImageLayers;
				const E_FORMAT							m_outImageFormat;
				const IBlitUtilities::E_ALPHA_SEMANTIC	m_alphaSemantic;
				const float								m_referenceAlpha;
				const uint32_t							m_alphaBinCount;
		};

		template <typename Dither = IdentityDither, typename Normalization = void, bool Clamp = false>
		class CSwizzleAndConvertTest : public ITest
		{
			public:
				CSwizzleAndConvertTest(core::smart_refctd_ptr<asset::ICPUImage>&& inImage,
					BlitFilterTestApp* parentApp,
					const asset::E_FORMAT outFormat,
					hlsl::uint32_t4 inOffsetBaseLayer,
					hlsl::uint32_t4 outOffsetBaseLayer,
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

					auto outImage = createCPUImage({inImageExtent.width,inImageExtent.height,inImageExtent.depth}, m_inImage->getCreationParameters().arrayLayers, m_inImage->getCreationParameters().type, m_outFormat);
					if (!outImage)
						return false;

					using convert_filter_t = asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, asset::DefaultSwizzle, Dither, Normalization, Clamp>;

					typename convert_filter_t::state_type filterState = {};
					filterState.extentLayerCount = core::vectorSIMDu32(
						inImageExtent.width-m_inOffsetBaseLayer.x,
						inImageExtent.height-m_inOffsetBaseLayer.y,
						inImageExtent.depth- m_inOffsetBaseLayer.z,
						m_inImage->getCreationParameters().arrayLayers-m_inOffsetBaseLayer.w
					);
					assert((static_cast<core::vectorSIMDi32>(filterState.extentLayerCount) > core::vectorSIMDi32(0)).all());

					filterState.inOffsetBaseLayer = reinterpret_cast<const core::vectorSIMDu32&>(m_inOffsetBaseLayer);
					filterState.outOffsetBaseLayer = reinterpret_cast<const core::vectorSIMDu32&>(m_outOffsetBaseLayer);
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

					writeImage(outImage.get(),m_writeImagePath);

					return true;
				}

			private:
				asset::E_FORMAT m_outFormat = asset::EF_UNKNOWN;
				hlsl::uint32_t4 m_inOffsetBaseLayer;
				hlsl::uint32_t4 m_outOffsetBaseLayer;
				asset::ICPUImageView::SComponentMapping m_swizzle;
				const char* m_writeImagePath;
		};

		template <typename BlitUtilities>
		class CComputeBlitTest : public ITest
		{
				using blit_utils_t = BlitUtilities;
				using convolution_kernels_t = typename blit_utils_t::convolution_kernels_t;

			public:
				CComputeBlitTest(
					BlitFilterTestApp*						parentApp,
					const char*								outputName,
					smart_refctd_ptr<ICPUImage>&&			inImage,
					const hlsl::uint32_t3&					outImageDim,
					const convolution_kernels_t&			convolutionKernels,
					const IBlitUtilities::E_ALPHA_SEMANTIC	alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED,
					const float								referenceAlpha = 0.f,
					const uint32_t							alphaBinCount = IBlitUtilities::DefaultAlphaBinCount
				) : ITest(std::move(inImage), parentApp), m_outputName(outputName), m_convolutionKernels(convolutionKernels),
					m_outImageDim(outImageDim), m_alphaSemantic(alphaSemantic), m_referenceAlpha(referenceAlpha), m_alphaBinCount(alphaBinCount)
				{
				}

				bool run() override
				{
					assert(m_inImage->getCreationParameters().mipLevels == 1);

					// GPU clamps when storing to a texture, so the CPU needs to as well
					using BlitFilter = CBlitImageFilter<VoidSwizzle,IdentityDither,void,true,blit_utils_t>;

					const auto inCreationParams = m_inImage->getCreationParameters();
					const auto layerCount = inCreationParams.arrayLayers;

					auto* logger = m_parentApp->m_logger.get();

					auto computeAlphaCoverage = [&](ICPUImage* image) -> void
					{
						constexpr uint32_t mipLevel = 0u;

						const auto format = image->getCreationParameters().format;
						const auto extent = image->getCreationParameters().extent;
						for (auto layer=0; layer<layerCount; ++layer)
						{
							uint64_t alphaTestPassCount = 0u;
							for (uint32_t z=0u; z<extent.depth; ++z)
							for (uint32_t y=0u; y<extent.height; ++y)
							for (uint32_t x=0u; x<extent.width; ++x)
							{
								const core::vectorSIMDu32 texCoord(x,y,z,layer);
								core::vectorSIMDu32 dummy;
								const void* encodedPixel = image->getTexelBlockData(mipLevel,texCoord,dummy);

								double decodedPixel[4];
								asset::decodePixelsRuntime(format, &encodedPixel, decodedPixel, dummy.x, dummy.y);

								if (decodedPixel[3] > m_referenceAlpha)
									++alphaTestPassCount;
							}
							const float alphaCoverage = float(alphaTestPassCount) / float(extent.width * extent.height * extent.depth);
							logger->log("CPU alpha coverage: %f with reference value %f", ILogger::ELL_INFO, alphaCoverage, m_referenceAlpha);
						}
					};

					const auto type = inCreationParams.type;
					const E_FORMAT inImageFormat = inCreationParams.format;
					const E_FORMAT outImageFormat = inImageFormat;

					// CPU
					{
						auto outImageCPU = createCPUImage(m_outImageDim,layerCount,type,outImageFormat);
						if (!outImageCPU)
							return false;

						typename BlitFilter::state_type blitFilterState = typename BlitFilter::state_type(m_convolutionKernels);

						const auto mipSize = m_inImage->getMipSize();

						blitFilterState.inOffsetBaseLayer = hlsl::uint32_t4(0,0,0,0);
						blitFilterState.inExtentLayerCount = hlsl::uint32_t4(mipSize.x,mipSize.y,mipSize.z,layerCount);
						blitFilterState.inImage = m_inImage.get();
						blitFilterState.outImage = outImageCPU.get();

						blitFilterState.outOffsetBaseLayer = hlsl::uint32_t4();
						blitFilterState.outExtentLayerCount = hlsl::uint32_t4(m_outImageDim[0],m_outImageDim[1],m_outImageDim[2],layerCount);

						blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
						blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
						blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
						blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

						blitFilterState.alphaSemantic = m_alphaSemantic;
						blitFilterState.alphaBinCount = m_alphaBinCount;
						blitFilterState.alphaRefValue = m_referenceAlpha;

						blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
						blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

						const auto lutOffsetInScratch = BlitFilter::getScratchOffset(&blitFilterState, BlitFilter::ESU_SCALED_KERNEL_PHASED_LUT);
						if (!BlitFilter::blit_utils_t::computeScaledKernelPhasedLUT(
								blitFilterState.scratchMemory+lutOffsetInScratch,
								blitFilterState.inExtentLayerCount,
								blitFilterState.outExtentLayerCount,
								type,
								m_convolutionKernels
							))
							logger->log("Failed to compute the LUT for blitting\n", ILogger::ELL_ERROR);

						logger->log("CPU begin..");
						if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
							logger->log("Failed to blit\n", ILogger::ELL_ERROR);
						logger->log("CPU end..");

						if (m_alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
							computeAlphaCoverage(outImageCPU.get());

						_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);

						writeImage(outImageCPU.get(),"cpu_blit_ref_"+m_outputName+".dds");
					}
					// GPU
					{
						auto* device = m_parentApp->m_device.get();
						auto* computeQueue = m_parentApp->getComputeQueue();

						auto* utils = m_parentApp->m_utils.get();
						// timeline semaphore
						auto semaphore = device->createSemaphore(0);

						assert(m_inImage->getCreationParameters().mipLevels==1);

						//
						IImageViewBase::E_TYPE viewType;
						switch (type)
						{
							case IImage::E_TYPE::ET_1D:
								viewType = IImageViewBase::E_TYPE::ET_1D_ARRAY;
								break;
							case IImage::E_TYPE::ET_3D:
								viewType = IImageViewBase::E_TYPE::ET_3D;
								break;
							default:
								viewType = IImageViewBase::E_TYPE::ET_2D_ARRAY;
								break;
						}

						// start capturing
						m_parentApp->m_api->startCapture();

						// just use the asset converter to make the image view as well
						smart_refctd_ptr<IGPUImageView> inImageView;
						{
							auto* queue = m_parentApp->getTransferUpQueue();
							// intialize command buffers
							constexpr auto MultiBuffering = 2;
							std::array<smart_refctd_ptr<IGPUCommandBuffer>, MultiBuffering> commandBuffers;
							std::array<IQueue::SSubmitInfo::SCommandBufferInfo, MultiBuffering> commandBufferInfos;
							{
								auto pool = device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
								pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{commandBuffers.data(),MultiBuffering},smart_refctd_ptr<ILogger>(logger));
								//
								for (uint32_t i = 0u; i < MultiBuffering; ++i)
								{
									commandBuffers[i]->setObjectDebugName(("Command Buffer #" + std::to_string(i)).c_str());
									commandBufferInfos[i].cmdbuf = commandBuffers[i].get();
								}
							}

							// test creation of image views, also it seems more ergonomic this way
							smart_refctd_ptr<ICPUImageView> cpuImageView;
							{
								ICPUImageView::SCreationParams params = {};
								params.image = m_inImage;
								params.viewType = viewType;
								params.format = inImageFormat;
								cpuImageView = ICPUImageView::create(std::move(params));
							}
							video::IGPUImageView::SCreationParams creationParams = {};

							//
							auto converter = CAssetConverter::create({.device=device});
							// We don't want to generate mip-maps for these images, because compute blitting is what does it and we want to test it here! 
							struct SInputs final : CAssetConverter::SInputs
							{
								inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
								{
									return 1;
								}
								inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
								{
									return 0b0u;
								}
							} inputs = {};
							inputs.readCache = converter.get();
							inputs.logger = logger;
							std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = { &cpuImageView.get(),1 };
							auto reservation = converter->reserve(inputs);
							// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
							inImageView = reservation.getGPUObjects<ICPUImageView>().front().value;
							if (!inImageView)
							{
								logger->log("Cannot create a GPU image for the input CPU images!",ILogger::ELL_ERROR);
								return false;
							}
							inImageView->setObjectDebugName((m_outputName+" Input View").c_str());

							// scratch command buffers for asset converter transfer commands
							SIntendedSubmitInfo transfer =
							{
								.queue = queue,
								.waitSemaphores = {},
								.prevCommandBuffers = {},
								.scratchCommandBuffers = commandBufferInfos,
								.scratchSemaphore = {
									.semaphore = semaphore.get(),
									.value = 0,
									// because of layout transitions
									.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
								}
							};
							// as per the `SIntendedSubmitInfo` one commandbuffer must be begun
							commandBuffers[0]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
							// make sure we
							struct SConvertParams : CAssetConverter::SConvertParams
							{
								inline uint32_t getFinalOwnerQueueFamily(const IGPUImage* image, const core::blake3_hash_t& createdFrom, const uint8_t mipLevel) override
								{
									return m_finalFamily;
								}
								// keep the default final layout deduction rules

								uint32_t m_finalFamily;
							};
							SConvertParams params = {};
							params.transfer = &transfer;
							params.utilities = utils;
							params.m_finalFamily = computeQueue->getFamilyIndex();
							auto result = reservation.convert(params);
							if (!result.blocking() && result.copy()!=IQueue::RESULT::SUCCESS)
							{
								logger->log("Failed to upload CPU image data to GPU!",ILogger::ELL_ERROR);
								return false;
							}
						}

						// Create resources needed to do the blit
						auto blitFilter = make_smart_refctd_ptr<CComputeBlit>(smart_refctd_ptr<ILogicalDevice>(device));

						// create the outputs
						uint32_t normalizationScratchSize = 0;
						smart_refctd_ptr<IGPUImageView> outImageView;
						{
							const auto outImageViewFormat = blitFilter->getOutputViewFormat(outImageFormat);
							if (outImageViewFormat==EF_UNKNOWN)
							{
								logger->log("Cannot encode into this format, even manually!",ILogger::ELL_ERROR);
								return false;
							}
							const bool manualEncoding = outImageViewFormat!=outImageFormat;

							smart_refctd_ptr<IGPUImage> outImage;
							{
								IGPUImage::SCreationParams creationParams = {};
								creationParams.type = type;
								creationParams.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
								creationParams.format = outImageFormat;
								creationParams.extent = { m_outImageDim.x, m_outImageDim.y, m_outImageDim.z };
								creationParams.mipLevels = inCreationParams.mipLevels;
								creationParams.arrayLayers = layerCount;
								if (manualEncoding)
									creationParams.flags = core::bitflag(IGPUImage::ECF_MUTABLE_FORMAT_BIT)|IGPUImage::ECF_EXTENDED_USAGE_BIT;
								creationParams.usage = IGPUImage::EUF_STORAGE_BIT|video::IGPUImage::EUF_TRANSFER_SRC_BIT;
								creationParams.viewFormats.set(outImageFormat,true);
								creationParams.viewFormats.set(outImageViewFormat,true);
								outImage = device->createImage(std::move(creationParams));
								if (!outImage || !device->allocate(outImage->getMemoryReqs(),outImage.get()).isValid())
								{
									logger->log("Failed to create output GPU image!",ILogger::ELL_ERROR);
									return false;
								}
								outImage->setObjectDebugName((m_outputName + " Output").c_str());
							}

							IGPUImageView::SCreationParams params = {};
							params.image = std::move(outImage);
							params.viewType = viewType;
							params.format = outImageViewFormat;
							outImageView = device->createImageView(std::move(params));
							
							if (m_alphaSemantic==IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
							{
/*
								normalizationInFormat = CComputeBlit::getCoverageAdjustmentIntermediateFormat(outImageFormat);

								if (normalizationInFormat != outImageFormat)
								{
									video::IGPUImage::SCreationParams creationParams;
									creationParams = outImageGPU->getCreationParameters();
									creationParams.format = normalizationInFormat;
									creationParams.usage = IGPUImage::EUF_STORAGE_BIT | IGPUImage::EUF_SAMPLED_BIT);
									normalizationInImage = device->createImage(std::move(creationParams));
									auto memReqs = normalizationInImage->getMemoryReqs();
									memReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
									device->allocate(memReqs, normalizationInImage.get());
									transitionImageLayout(core::smart_refctd_ptr(normalizationInImage), asset::IImage::EL_GENERAL); // First we do the blit which requires storage image so starting layout is GENERAL

									video::IGPUImageView::SCreationParams viewCreationParams = {};
									viewCreationParams.image = normalizationInImage;
									viewCreationParams.viewType = getImageViewTypeFromImageType_GPU(inImageGPU->getCreationParameters().type);
									viewCreationParams.format = normalizationInImage->getCreationParameters().format;

									normalizationInImageView = device->createImageView(std::move(viewCreationParams));
								}
*/

								// TODO: create coverage adjustment scratch
							}
						}

						const hlsl::uint32_t3 inExtent(inCreationParams.extent.width,inCreationParams.extent.height,inCreationParams.extent.depth);

						// create scaledKernelPhasedLUT and its view
						smart_refctd_ptr<IGPUBufferView> scaledKernelPhasedLUTView;
						{
							const auto lutOffset = normalizationScratchSize;
							const auto lutSize = blit_utils_t::getScaledKernelPhasedLUTSize(inExtent,m_outImageDim,type,m_convolutionKernels);

							auto lutMemory = std::make_unique<uint8_t[]>(lutSize);
							if (!blit_utils_t::computeScaledKernelPhasedLUT(lutMemory.get(),inExtent,m_outImageDim,type,m_convolutionKernels))
							{
								logger->log("Failed to compute scaled kernel phased LUT for the GPU case!",ILogger::ELL_ERROR);
								return false;
							}

							IGPUBuffer::SCreationParams creationParams = {};
							// `samplerBuffer`, lut upload and scratch clear command, BDA
							creationParams.usage = IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT|IGPUBuffer::EUF_TRANSFER_DST_BIT|IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
							creationParams.size = normalizationScratchSize+lutSize;
							auto scaledKernelPhasedLUT = device->createBuffer(std::move(creationParams));
							if (!device->allocate(scaledKernelPhasedLUT->getMemoryReqs(),scaledKernelPhasedLUT.get(),IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT).isValid())
							{
								logger->log("Failed to create the Phase LUT and coverage buffer!",ILogger::ELL_ERROR);
								return false;
							}
/*
							// fill it up with data
							SBufferRange<IGPUBuffer> bufferRange = {};
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

							scaledKernelPhasedLUTView = device->createBufferView(scaledKernelPhasedLUT.get(), bufferViewFormat, 0ull, scaledKernelPhasedLUT->getSize());
*/
						}

#if 0
						core::smart_refctd_ptr<video::IGPUImageView> normalizationInImageView = outImageView;
						core::smart_refctd_ptr<video::IGPUImage> normalizationInImage = outImageGPU;
						auto normalizationInFormat = outImageFormat;


						// create scratch buffer
						core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer = nullptr;
						{
							const size_t scratchSize = blitFilter->getCoverageAdjustmentScratchSize(m_alphaSemantic, inImageType, m_alphaBinCount, layersToBlit);
							if (scratchSize > 0)
							{
								video::IGPUBuffer::SCreationParams creationParams = {};
								creationParams.size = scratchSize;
								creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);

								coverageAdjustmentScratchBuffer = device->createBuffer(std::move(creationParams));
								auto memReqs = coverageAdjustmentScratchBuffer->getMemoryReqs();
								memReqs.memoryTypeBits &= m_parentApp->m_physicalDevice->getDeviceLocalMemoryTypeBits();
								device->allocate(memReqs, coverageAdjustmentScratchBuffer.get());

								asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
								bufferRange.offset = 0ull;
								bufferRange.size = coverageAdjustmentScratchBuffer->getSize();
								bufferRange.buffer = coverageAdjustmentScratchBuffer;

								core::vector<uint32_t> fillValues(scratchSize / sizeof(uint32_t), 0u);
								m_parentApp->utilities->updateBufferRangeViaStagingBufferAutoSubmit(bufferRange, fillValues.data(), m_parentApp->queue);
							}
						}


						auto blitDSLayout = blitFilter->getDefaultBlitDescriptorSetLayout(m_alphaSemantic);
						auto kernelWeightsDSLayout = blitFilter->getDefaultKernelWeightsDescriptorSetLayout();
						auto blitPipelineLayout = blitFilter->getDefaultBlitPipelineLayout(m_alphaSemantic);

						video::IGPUDescriptorSetLayout* blitDSLayouts_raw[] = { blitDSLayout.get(), kernelWeightsDSLayout.get() };
						uint32_t dsCounts[] = { 2, 1 };
						auto descriptorPool = device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, blitDSLayouts_raw, blitDSLayouts_raw + 2ull, dsCounts);

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

						blitFilter->blit<BlitUtilities>(
							m_parentApp->queue, m_alphaSemantic,
							blitDS.get(), alphaTestPipeline.get(),
							blitDS.get(), blitWeightsDS.get(), blitPipeline.get(),
							normalizationDS.get(), normalizationPipeline.get(),
							inExtent, inImageType, inImageFormat, normalizationInImage, m_convolutionKernels,
							layersToBlit,
							coverageAdjustmentScratchBuffer, m_referenceAlpha,
							m_alphaBinCount, BlitWorkgroupSize);

						if (m_alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
						{
							auto outCPUImageView = ext::ScreenShot::createScreenShot(
								device.get(),
								m_parentApp->getTransferQueue(),
								nullptr,
								outImageView.get(),
								asset::EAF_NONE,
								IImage::EL_GENERAL
							);

							m_parentApp->m_logger->log("GPU alpha coverage: %f", system::ILogger::ELL_DEBUG, computeAlphaCoverage(m_referenceAlpha, outCPUImageView->getCreationParameters().image.get()));
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
#endif
					}

					m_parentApp->m_api->endCapture();
#if 0
//					core::vector<uint8_t> gpuOutput(static_cast<uint64_t>(m_outImageDim[0]) * m_outImageDim[1] * m_outImageDim[2] * asset::getTexelOrBlockBytesize(outImageFormat) * layerCount);

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
#endif
					return true;
				}

			private:
				const std::string									m_outputName;
				const typename blit_utils_t::convolution_kernels_t	m_convolutionKernels;
				const hlsl::uint32_t3								m_outImageDim;
				const IBlitUtilities::E_ALPHA_SEMANTIC				m_alphaSemantic;
				const float											m_referenceAlpha = 0.f;
				const uint32_t										m_alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount;
		};

		class CRegionBlockFunctorFilterTest : public ITest
		{
			public:
				CRegionBlockFunctorFilterTest(smart_refctd_ptr<ICPUImage>&& inImage, BlitFilterTestApp* parentApp, const char* writeImagePath)
					: ITest(std::move(inImage), parentApp), m_writeImagePath(writeImagePath)
				{}

				bool run() override
				{
					auto outImage = smart_refctd_ptr_static_cast<ICPUImage>(m_inImage->clone());
					if (!outImage)
						return false;

					// this is our per-block function
					const auto format = m_inImage->getCreationParameters().format;
					const auto regions = m_inImage->getRegions();
					const auto region = regions.begin();
					TexelBlockInfo blockInfo(format);
					const auto strides = region->getByteStrides(blockInfo);
					uint8_t* src = reinterpret_cast<uint8_t*>(m_inImage->getBuffer()->getPointer());
					uint8_t* dst = reinterpret_cast<uint8_t*>(outImage->getBuffer()->getPointer());
					auto copyFromLevel0 = [src, dst, &blockInfo, region, strides](uint64_t dstByteOffset, vectorSIMDu32 coord)
					{
						const uint64_t srcByteOffset = region->getByteOffset(coord, strides);
						memcpy(dst+dstByteOffset, src+srcByteOffset, blockInfo.getBlockByteSize());
					};

					using region_block_filter_t = asset::CRegionBlockFunctorFilter<decltype(copyFromLevel0), false>;

					region_block_filter_t::CState filterState(copyFromLevel0,outImage.get(),regions.data()+1);

					for (uint32_t i=1; i<outImage->getCreationParameters().mipLevels; ++i)
					{
						filterState.regionIterator = outImage->getRegions().data()+i;
						if (!region_block_filter_t::execute(&filterState))
						{
							m_parentApp->m_logger->log("CRegionBlockFunctorFilter failed for mip level %u", system::ILogger::ELL_ERROR, i);
							return false;
						}
					}

					writeImage(outImage.get(), m_writeImagePath);

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

			assetManager = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));


			constexpr bool TestCPUBlitFilter = true;
			constexpr bool TestSwizzleAndConvertFilter = false;
			constexpr bool TestGPUBlitFilter = true;
			constexpr bool TestRegionBlockFunctorFilter = false;

			auto loadImage = [this](const char* path) -> smart_refctd_ptr<ICPUImage>
			{
				// to prevent the images hanging around in the cache and taking up RAM
				constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
				IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
				auto imageBundle = assetManager->getAsset(path, loadParams);
				auto imageContents = imageBundle.getContents();

				if (imageContents.empty())
				{
					m_logger->log("Failed to load image at path %s", system::ILogger::ELL_ERROR, path);
					return nullptr;
				}

				auto asset = *imageContents.begin();

				smart_refctd_ptr<ICPUImage> result;
				{
					if (asset->getAssetType() == IAsset::ET_IMAGE_VIEW)
						result = smart_refctd_ptr_static_cast<ICPUImageView>(asset)->getCreationParameters().image;
					else if (asset->getAssetType() == IAsset::ET_IMAGE)
						result = std::move(smart_refctd_ptr_static_cast<ICPUImage>(asset));
					else
						assert(!"Invalid code path.");
				}

				return result;
			};

			auto runTests = [this](const std::span<std::unique_ptr<ITest>> tests)
			{
				auto i = 0;
				for (auto& test : tests)
				{
					assert(test);
					if (!test->run())
						m_logger->log("Test #%u failed.", system::ILogger::ELL_ERROR, i);
					else
						m_logger->log("Test #%u passed.", system::ILogger::ELL_INFO, i);
					i++;
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
						const hlsl::uint32_t3 outImageDim(inExtent.width/2,inExtent.height/4,1);
						const auto outImageFormat = asset::EF_R8G8B8A8_SRGB;

						using BlitUtilities = CBlitUtilities<CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>>;

						auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>({inExtent.width,inExtent.height,inExtent.depth},outImageDim);

						tests[0] = std::make_unique<CBlitImageFilterTest<BlitUtilities>>
						(
							this,
							std::move(inImage),
							outImageDim,
							1,
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

						using BlitUtilities = CBlitUtilities<CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>>;
						
						const auto& inExtent = inImage->getCreationParameters().extent;
						const hlsl::uint32_t3 outImageDim(inExtent.width*2, inExtent.height*4, 1);
						auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SKaiserFunction>>({inExtent.width,inExtent.height,inExtent.depth},outImageDim);

						const auto outImageFormat = asset::EF_R32G32B32A32_SFLOAT;
						tests[1] = std::make_unique<CBlitImageFilterTest<BlitUtilities>>
						(
							this,
							std::move(inImage),
							outImageDim,
							1,
							outImageFormat,
							"CBlitImageFilter_1.exr",
							convolutionKernels
						);
					}
				}

				runTests(tests);
			}

			if (TestSwizzleAndConvertFilter)
			{
				m_logger->log("CSwizzleAndConvertImageFilter",ILogger::ELL_INFO);

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
							hlsl::uint32_t4(0, 0, 0, 0),
							hlsl::uint32_t4(0, 0, 0, 0),
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
							hlsl::uint32_t4(64, 64, 0, 0),
							hlsl::uint32_t4(64, 0, 0, 0),
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
							hlsl::uint32_t4(0, 0, 0, 0),
							hlsl::uint32_t4(0, 0, 0, 0),
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
							hlsl::uint32_t4(0, 0, 0, 0),
							hlsl::uint32_t4(0, 0, 0, 0),
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
							hlsl::uint32_t4(0, 0, 0, 0),
							hlsl::uint32_t4(0, 0, 0, 0),
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
							hlsl::uint32_t4(0, 0, 0, 0),
							hlsl::uint32_t4(0, 0, 0, 0),
							asset::ICPUImageView::SComponentMapping(),
							"CSwizzleAndConvertImageFilter_5.png"
						);
					}
				}

				runTests(tests);
			}

			if (TestGPUBlitFilter)
			{
				m_logger->log("CComputeBlit", system::ILogger::ELL_INFO);

				constexpr uint32_t TestCount = 6;
				std::unique_ptr<ITest> tests[TestCount] = { nullptr };

				// Test 0: Resize 1D image with Mitchell
				{
					const hlsl::uint32_t3 inImageDim(59u,1u,1u);
					const auto layerCount = 10;
					auto inImage = createCPUImage(inImageDim,layerCount,IImage::ET_1D,EF_R32_SFLOAT,true);
					assert(inImage);

					auto reconstructionX = asset::CWeightFunction1D<asset::SMitchellFunction<>>();
					reconstructionX.stretchAndScale(0.35f);

					auto resamplingX = asset::CWeightFunction1D<asset::SMitchellFunction<>>();
					resamplingX.stretchAndScale(0.35f);

					using LutDataType = hlsl::float16_t;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType>;

					const hlsl::uint32_t3 outImageDim(800u, 1u, 1u);
					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(inImageDim, outImageDim, std::move(reconstructionX), std::move(resamplingX));

					tests[0] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						this,
						"mitchell_1d",
						std::move(inImage),
						outImageDim,
						convolutionKernels
					);
				}

				// Test 1: Resize 2D image with Kaiser
				{
					const char* path = "../../media/colorexr.exr";
					auto inImage = loadImage(path);
					if (inImage)
					{
						using LutDataType = float;
						using BlitUtilities = CBlitUtilities<
							CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>,
							CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>,
							CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>>,
							LutDataType
						>;
						
						const auto& inExtent = inImage->getCreationParameters().extent;
						const hlsl::uint32_t3 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
						auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SKaiserFunction>>({inExtent.width,inExtent.height,inExtent.depth},outImageDim);

						tests[1] = std::make_unique<CComputeBlitTest<BlitUtilities>>
						(
							this,
							"kaiser_2d",
							std::move(inImage),
							outImageDim,
							convolutionKernels
						);
					}
				}

				// Test 2: Resize 3D image with Box
				{
					const uint32_t layerCount = 1;
					const hlsl::uint32_t3 inImageDim(2,3,4);
					const IImage::E_TYPE inImageType = IImage::ET_3D;
					const E_FORMAT inImageFormat = EF_R32G32B32A32_SFLOAT;
					auto inImage = createCPUImage(inImageDim,layerCount,inImageType,inImageFormat,true);
					assert(inImage);

					auto reconstructionX = asset::CWeightFunction1D<asset::SBoxFunction>();
					reconstructionX.stretchAndScale(0.35f);
					auto resamplingX = asset::CWeightFunction1D<asset::SBoxFunction>();
					resamplingX.stretchAndScale(0.35f);

					auto reconstructionY = asset::CWeightFunction1D<asset::SBoxFunction>();
					reconstructionY.stretchAndScale(9.f/16.f);
					auto resamplingY = asset::CWeightFunction1D<asset::SBoxFunction>();
					resamplingY.stretchAndScale(9.f/16.f);

					using LutDataType = hlsl::float16_t;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>>,
						LutDataType
					>;

					const hlsl::uint32_t3 outImageDim(3, 4, 2);
					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SBoxFunction>>(inImageDim, outImageDim, std::move(reconstructionX), std::move(resamplingX), std::move(reconstructionY), std::move(resamplingY));

					tests[2] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						this,
						"box_3d",
						std::move(inImage),
						outImageDim,
						convolutionKernels
					);
				}

				// Test 3: Resize 2D image with alpha coverage adjustment
				{
					// We should find a better image for testing coverage adjustment
					// WARNING: The output of this will turn pixels with Alpha 1.0 to pixels with Alpha slightly > referenceAlpha !!!!
					// This is simply how coverage adjustment works!
					const char* path = "../../media/colorexr.exr";
					auto inImage = loadImage(path);
					if (inImage)
					{
						const auto& inExtent = inImage->getCreationParameters().extent;
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

						const hlsl::uint32_t3 outImageDim(inExtent.width/3,inExtent.height/7,inExtent.depth);
						auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>({inExtent.width,inExtent.height,inExtent.depth},outImageDim);

						tests[3] = std::make_unique<CComputeBlitTest<BlitUtilities>>
						(
							this,
							"coverage_2d",
							std::move(inImage),
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
					const hlsl::uint32_t3 inImageDim(257,129,63);
					const IImage::E_TYPE inImageType = IImage::ET_3D;
					const E_FORMAT inImageFormat = EF_B10G11R11_UFLOAT_PACK32;
					auto inImage = createCPUImage(inImageDim, layerCount, inImageType, inImageFormat, true);
					assert(inImage);

					using LutDataType = hlsl::float16_t;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType
					>;

					const hlsl::uint32_t3 outImageDim(256,128,64);
					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(inImageDim, outImageDim);

					tests[4] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						this,
						"b10g11r11_3d",
						std::move(inImage),
						outImageDim,
						convolutionKernels
					);
				}

				// Test 5: A 2D image with atypical dimensions and alpha coverage adjustment
				{
					const auto layerCount = 7;
					const hlsl::uint32_t3 inImageDim(511,1024,1);
					const IImage::E_TYPE inImageType = IImage::ET_2D;
					const E_FORMAT inImageFormat = EF_R16G16B16A16_SNORM;
					auto inImage = createCPUImage(inImageDim, layerCount, inImageType, inImageFormat, true);
					assert(inImage);

					using LutDataType = float;
					using BlitUtilities = CBlitUtilities<
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						CDefaultChannelIndependentWeightFunction1D<CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction<>>, CWeightFunction1D<SMitchellFunction<>>>>,
						LutDataType
					>;

					const hlsl::uint32_t3 outImageDim(512, 257, 1);
					auto convolutionKernels = BlitUtilities::getConvolutionKernels<CWeightFunction1D<SMitchellFunction<>>>(inImageDim, outImageDim);

					const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
					const float referenceAlpha = 0.5f;
					const auto alphaBinCount = 4096;
					tests[5] = std::make_unique<CComputeBlitTest<BlitUtilities>>
					(
						this,
						"rand_coverage_2d",
						std::move(inImage),
						outImageDim,
						convolutionKernels,
						alphaSemantic,
						referenceAlpha,
						alphaBinCount
					);
				}

				runTests(tests);
			}

			if (TestRegionBlockFunctorFilter)
			{
				m_logger->log("CRegionBlockFunctorFilter",ILogger::ELL_INFO);

				constexpr uint32_t TestCount = 1;
				std::unique_ptr<ITest> tests[TestCount] = { nullptr };

				// Test 0: Copy the first NxM texels of the 0th mip level to the ith mip level, where N and M are dimensions of the ith mip level.
				{
					auto inImage = loadImage("../../media/GLI/kueken7_rgba_dxt5_unorm.dds");
					if (inImage)
						tests[0] = std::make_unique<CRegionBlockFunctorFilterTest>(std::move(inImage), this, "CRegionBlockFunctorFilter_0.dds");
				}

				runTests(tests);
			}

			return true;
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

			using flags_t = IQueue::FAMILY_FLAGS;
			// IGPUObjectFromAssetConverter requires the queue to support graphics as well.
			retval.push_back({
				.requiredFlags = flags_t::COMPUTE_BIT|flags_t::TRANSFER_BIT|flags_t::GRAPHICS_BIT,
				.disallowedFlags = flags_t::NONE,
				.queueCount = 1,
				.maxImageTransferGranularity = {1,1,1}
			});

			return retval;
		}

	private:
		smart_refctd_ptr<IAssetManager> assetManager;
		IQueue* queue;
};

NBL_MAIN_FUNC(BlitFilterTestApp)