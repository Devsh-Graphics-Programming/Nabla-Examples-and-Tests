// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "utils.h"

// Constants
const unsigned int channelCountOverride = 3;

// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class FFTBloomApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	// Compute Pipelines
	smart_refctd_ptr<IGPUComputePipeline> m_firstAxisFFTPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_lastAxisFFT_convolution_lastAxisIFFTPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_firstAxisIFFTPipeline;

	// Descriptor Sets
	smart_refctd_ptr<IGPUComputePipeline> m_firstAxisFFTDescriptorSet;
	smart_refctd_ptr<IGPUComputePipeline> m_lastAxisFFT_convolution_lastAxisIFFTDescriptorSet;
	smart_refctd_ptr<IGPUComputePipeline> m_lastAxisFFTDescriptorSet;

	// Utils (might be useful, they stay for now)
	smart_refctd_ptr<nbl::video::IUtilities> m_utils;

	// Resources
	smart_refctd_ptr<IGPUImage> srcImage;
	smart_refctd_ptr<IGPUImageView> srcImageView;
	smart_refctd_ptr<IGPUImage> kerImage;
	smart_refctd_ptr<IGPUImageView> kerImageView;
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> outImgView;
	smart_refctd_ptr<IGPUImageView> kernelNormalizedSpectrums[channelCountOverride];

	// Used to store intermediate results
	smart_refctd_ptr<nbl::video::IGPUBuffer> m_rowMajorBuffer;
	smart_refctd_ptr<nbl::video::IGPUBuffer> m_colMajorBuffer;

	// These are Buffer Device Addresses
	uint64_t m_rowMajorBufferAddress;
	uint64_t m_colMajorBufferAddress;

	// Some parameters
	float bloomScale = 1.f;
	float useHalfFloats = false;
	
	// Other parameter-dependent variables
	asset::VkExtent3D marginSrcDim;

	// This example really lets the advantages of a timeline semaphore shine through!
	smart_refctd_ptr<ISemaphore> m_timeline;
	uint64_t m_iteration = 0;
	constexpr static inline uint64_t MaxIterations = 1;

	smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings)
	{
		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };
		return m_device->createPipelineLayout({ &pcRange,1 }, m_device->createDescriptorSetLayout(bindings));
	}

	inline void updateDescriptorSetFirstAxisFFT(IGPUDescriptorSet* set, smart_refctd_ptr<IGPUImageView> inputImageDescriptor, ISampler::E_TEXTURE_CLAMP textureWrap)
	{
		IGPUSampler::SParams params =
		{
			{
				textureWrap,
				textureWrap,
				textureWrap,
				ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				ISampler::ETF_LINEAR,
				ISampler::ETF_LINEAR,
				ISampler::ESMM_LINEAR,
				8u,
				0u,
				ISampler::ECO_ALWAYS
			}
		};
		auto sampler = m_device->createSampler(std::move(params));

		IGPUDescriptorSet::SDescriptorInfo info;
		IGPUDescriptorSet::SWriteDescriptorSet write;

		write.dstSet = set;
		write.binding = 0u;
		write.arrayElement = 0u;
		write.count = 1u;
		write.info = &info;

		info.desc = inputImageDescriptor;
		info.info.combinedImageSampler.sampler = sampler;
		info.info.combinedImageSampler.imageLayout = IImage::LAYOUT::UNDEFINED;

		m_device->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	inline void updateDescriptorSetConvolution(IGPUDescriptorSet* set, const smart_refctd_ptr<IGPUImageView>* kernelNormalizedSpectrumImageDescriptors)
	{
		IGPUDescriptorSet::SDescriptorInfo pInfos[channelCountOverride];
		IGPUDescriptorSet::SWriteDescriptorSet write;

		write.dstSet = set;
		write.binding = 0;
		write.arrayElement = 0u;
		write.count = channelCountOverride;
		write.info = pInfos;

		for (uint32_t i = 0u; i < channelCountOverride; i++)
		{
			auto& info = pInfos[i];
			info.desc = kernelNormalizedSpectrumImageDescriptors[i];
			info.info.combinedImageSampler.imageLayout = IImage::LAYOUT::UNDEFINED;
			info.info.combinedImageSampler.sampler = nullptr;
		}

		m_device->updateDescriptorSets(1, &write, 0u, nullptr);
	}

	inline void updateDescriptorSetFirstAxisIFFT(IGPUDescriptorSet* set, smart_refctd_ptr<IGPUImageView> outputImageDescriptor)
	{
		IGPUDescriptorSet::SDescriptorInfo info;
		IGPUDescriptorSet::SWriteDescriptorSet write;

		write.dstSet = set;
		write.binding = 0;
		write.arrayElement = 0u;
		write.count = channelCountOverride;
		write.info = &info;

		info.desc = outputImageDescriptor;
		info.info.combinedImageSampler.imageLayout = IImage::LAYOUT::UNDEFINED;
		info.info.combinedImageSampler.sampler = nullptr;

		m_device->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	FFTBloomApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// we stuff all our work here because its a "single shot" app
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// Load source and kernel images
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto srcImageBundle = m_assetMgr->getAsset("../media/colorexr.exr", lp);
			auto kerImageBundle = m_assetMgr->getAsset("../media/kernels/physical_flare_256.exr", lp);
			const auto srcImages = srcImageBundle.getContents();
			const auto kerImages = kerImageBundle.getContents();
			if (srcImages.empty() or kerImages.empty())
				return logFail("Could not load shader!");

			srcImage = IAsset::castDown<IGPUImage>(srcImages[0]);
			kerImage = IAsset::castDown<IGPUImage>(kerImages[0]);

			// The down-cast should not fail!
			assert(srcImage);
			assert(kerImage);
		}

		// Create views for these images
		{
			IGPUImageView::SCreationParams srcImgViewInfo;
			srcImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			srcImgViewInfo.image = srcImage;
			srcImgViewInfo.viewType = IGPUImageView::ET_2D;
			srcImgViewInfo.format = srcImgViewInfo.image->getCreationParameters().format;
			srcImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
			srcImgViewInfo.subresourceRange.baseMipLevel = 0;
			srcImgViewInfo.subresourceRange.levelCount = 1;
			srcImgViewInfo.subresourceRange.baseArrayLayer = 0;
			srcImgViewInfo.subresourceRange.layerCount = 1;
			srcImageView = m_device->createImageView(std::move(srcImgViewInfo));

			IGPUImageView::SCreationParams kerImgViewInfo;
			kerImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			kerImgViewInfo.image = kerImage;
			kerImgViewInfo.viewType = IGPUImageView::ET_2D;
			kerImgViewInfo.format = kerImgViewInfo.image->getCreationParameters().format;
			kerImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
			kerImgViewInfo.subresourceRange.baseMipLevel = 0;
			kerImgViewInfo.subresourceRange.levelCount = kerImgViewInfo.image->getCreationParameters().mipLevels;
			kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
			kerImgViewInfo.subresourceRange.layerCount = 1;
			kerImageView = m_device->createImageView(std::move(kerImgViewInfo));
		}

		// Create Out Image
		{
			auto dstImgViewInfo = srcImageView->getCreationParameters();

			IGPUImage::SCreationParams dstImgInfo(dstImgViewInfo.image->getCreationParameters());
			outImg = m_device->createImage(std::move(dstImgInfo));

			dstImgViewInfo.image = outImg;
			outImgView = m_device->createImageView(IGPUImageView::SCreationParams(dstImgViewInfo));

			auto memReqs = outImg->getMemoryReqs();
			memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpuMem = m_device->allocate(memReqs, outImg.get());
		}

		// agree on formats
		const E_FORMAT srcFormat = srcImageView->getCreationParameters().format;
		// TODO: this might be pointless?
		uint32_t srcNumChannels = getFormatChannelCount(srcFormat);
		uint32_t kerNumChannels = getFormatChannelCount(kerImageView->getCreationParameters().format);
		//! OVERRIDE (we dont need alpha)
		srcNumChannels = channelCountOverride;
		kerNumChannels = channelCountOverride;
		assert(srcNumChannels == kerNumChannels); // Just to make sure, because the other case is not handled in this example

		// Compute (kernel) padding size
		const float bloomRelativeScale = 0.25f;
		const auto kerDim = kerImageView->getCreationParameters().image->getCreationParameters().extent;
		const auto srcDim = srcImageView->getCreationParameters().image->getCreationParameters().extent;
		bloomScale = core::min(float(srcDim.width) / float(kerDim.width), float(srcDim.height) / float(kerDim.height)) * bloomRelativeScale;
		if (bloomScale > 1.f)
			std::cout << "WARNING: Bloom Kernel will Clip and loose sharpness, increase resolution of bloom kernel!" << std::endl;
		marginSrcDim = srcDim;
		// Add padding to marginSrcDim
		for (auto i = 0u; i < 3u; i++)
		{
			const auto coord = (&kerDim.width)[i];
			if (coord > 1u)
				(&marginSrcDim.width)[i] += core::max(coord * bloomScale, 1u) - 1u;
		}
		
		// Create intermediate buffers
		{
			IGPUBuffer::SCreationParams deviceLocalBufferParams = {};

			IQueue* const queue = getComputeQueue();
			uint32_t queueFamilyIndex = queue->getFamilyIndex();

			deviceLocalBufferParams.queueFamilyIndexCount = 1;
			deviceLocalBufferParams.queueFamilyIndices = &queueFamilyIndex;
			deviceLocalBufferParams.size = getOutputBufferSize(marginSrcDim, 3, useHalfFloats);
			deviceLocalBufferParams.usage = nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;

			m_rowMajorBuffer = m_device->createBuffer(std::move(deviceLocalBufferParams));
			deviceLocalBufferParams = m_rowMajorBuffer->getCreationParams();
			m_colMajorBuffer = m_device->createBuffer(std::move(deviceLocalBufferParams));

			auto rowMemReqs = m_rowMajorBuffer->getMemoryReqs();
			auto colMemReqs = m_colMajorBuffer->getMemoryReqs();
			rowMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			colMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpuRowMem = m_device->allocate(rowMemReqs, m_rowMajorBuffer.get(), IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);
			auto gpuColMem = m_device->allocate(colMemReqs, m_colMajorBuffer.get(), IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);

			m_rowMajorBufferAddress = m_rowMajorBuffer.get()->getDeviceAddress();
			m_colMajorBufferAddress = m_colMajorBuffer.get()->getDeviceAddress();
		}

		// Create pipeline layouts
		smart_refctd_ptr<IGPUPipelineLayout> m_firstAxisFFTPipelineLayout;
		{
			IGPUDescriptorSetLayout::SBinding bnd[] =
			{
				{
					0u,
					IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::ESS_COMPUTE,
					1u,
					nullptr
				}
			};
			m_firstAxisFFTPipelineLayout = createPipelineLayout({bnd, 1});
		}

		smart_refctd_ptr<IGPUPipelineLayout> m_lastAxisFFT_convolution_lastAxisIFFTPipelineLayout;
		{
			IGPUSampler::SParams params;
			params.MipmapMode = ISampler::ESMM_NEAREST;
			params.AnisotropicFilter = 0u;
			params.CompareEnable = false;
			params.CompareFunc = ISampler::ECO_ALWAYS;
			auto sampler = m_device->createSampler(std::move(params));
			smart_refctd_ptr<IGPUSampler> samplers[channelCountOverride];
			std::fill_n(samplers, channelCountOverride, sampler);

			IGPUDescriptorSetLayout::SBinding bnd[] =
			{
				{
					0u,
					IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::ESS_COMPUTE,
					channelCountOverride,
					samplers
				}
			};

			m_lastAxisFFT_convolution_lastAxisIFFTPipelineLayout = createPipelineLayout({ bnd, 1 });
		}

		smart_refctd_ptr<IGPUPipelineLayout> m_firstAxisIFFTPipelineLayout; 
		{
			IGPUDescriptorSetLayout::SBinding bnd[] =
			{
				{
					0u,
					IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::ESS_COMPUTE,
					1,
					nullptr
				}
			};
			
			m_lastAxisFFT_convolution_lastAxisIFFTPipelineLayout = createPipelineLayout({ bnd, 1 });
		}

		// Kernel FFT precomp
		{
			const asset::VkExtent3D paddedKerDim = padDimensions(kerDim);

			// create kernel spectrums
			auto createKernelSpectrum = [&]() -> auto
				{
					video::IGPUImage::SCreationParams imageParams;
					imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					imageParams.type = asset::IImage::ET_2D;
					imageParams.format = useHalfFloats ? EF_R16G16B16_SFLOAT : EF_R32G32B32_SFLOAT;
					imageParams.extent = { paddedKerDim.width,paddedKerDim.height,1u };
					imageParams.mipLevels = 1u;
					imageParams.arrayLayers = 1u;
					imageParams.samples = asset::IImage::ESCF_1_BIT;

					auto kernelImg = m_device->createImage(std::move(imageParams));

					auto memReqs = kernelImg->getMemoryReqs();
					memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					auto gpuMem = m_device->allocate(memReqs, kernelImg.get());

					video::IGPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = kernelImg;
					viewParams.viewType = video::IGPUImageView::ET_2D;
					viewParams.format = useHalfFloats ? EF_R16G16B16_SFLOAT : EF_R32G32B32_SFLOAT;
					viewParams.components = {};
					viewParams.subresourceRange = {};
					viewParams.subresourceRange.levelCount = 1u;
					viewParams.subresourceRange.layerCount = 1u;
					return m_device->createImageView(std::move(viewParams));
				};
			for (uint32_t i = 0u; i < channelCountOverride; i++)
				kernelNormalizedSpectrums[i] = createKernelSpectrum();

			// Invoke a workgroup per scanline
			FFTClass::Parameters_t fftPushConstants[2];
			FFTClass::DispatchInfo_t fftDispatchInfo[2];
			const ISampler::E_TEXTURE_CLAMP fftPadding[2] = { ISampler::ETC_CLAMP_TO_BORDER,ISampler::ETC_CLAMP_TO_BORDER };
			const auto passes = FFTClass::buildParameters(false, srcNumChannels, kerDim, fftPushConstants, fftDispatchInfo, fftPadding);
			assert(passes == 2u);
			// last axis FFT pipeline
			core::smart_refctd_ptr<IGPUComputePipeline> fftPipeline_SSBOInput(core::make_smart_refctd_ptr<FFTClass>(driver, 0x1u << fftPushConstants[1].getLog2FFTSize(), useHalfFloats)->getDefaultPipeline());

			// descriptor sets
			core::smart_refctd_ptr<IGPUDescriptorSet> fftDescriptorSet_Ker_FFT[2] =
			{
				driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(imageFirstFFTPipelineLayout->getDescriptorSetLayout(0u))),
				driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipeline_SSBOInput->getLayout()->getDescriptorSetLayout(0u)))
			};
			updateDescriptorSet(fftDescriptorSet_Ker_FFT[0].get(), kerImageView, ISampler::ETC_CLAMP_TO_BORDER, fftOutputBuffer_0);
			FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT[1].get(), fftOutputBuffer_0, fftOutputBuffer_1);

			// Normalization of FFT spectrum
			struct NormalizationPushConstants
			{
				ext::FFT::uvec4 stride;
				ext::FFT::uvec4 bitreverse_shift;
			};
			auto fftPipelineLayout_KernelNormalization = [&]() -> auto
				{
					IGPUDescriptorSetLayout::SBinding bnd[] =
					{
						{
							0u,
							EDT_STORAGE_BUFFER,
							1u,
							ISpecializedShader::ESS_COMPUTE,
							nullptr
						},
						{
							1u,
							EDT_STORAGE_IMAGE,
							channelCountOverride,
							ISpecializedShader::ESS_COMPUTE,
							nullptr
						},
					};
					SPushConstantRange pc_rng;
					pc_rng.offset = 0u;
					pc_rng.size = sizeof(NormalizationPushConstants);
					pc_rng.stageFlags = ISpecializedShader::ESS_COMPUTE;
					return driver->createPipelineLayout(
						&pc_rng, &pc_rng + 1u,
						driver->createDescriptorSetLayout(bnd, bnd + 2), nullptr, nullptr, nullptr
					);
				}();
				auto fftDescriptorSet_KernelNormalization = [&]() -> auto
					{
						auto dset = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_KernelNormalization->getDescriptorSetLayout(0u)));

						video::IGPUDescriptorSet::SDescriptorInfo pInfos[1 + channelCountOverride];
						video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[2];

						for (auto i = 0; i < 2; i++)
						{
							pWrites[i].dstSet = dset.get();
							pWrites[i].arrayElement = 0u;
							pWrites[i].count = 1u;
							pWrites[i].info = pInfos + i;
						}

						// In Buffer 
						pWrites[0].binding = 0;
						pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
						pWrites[0].count = 1;
						pInfos[0].desc = fftOutputBuffer_1;
						pInfos[0].buffer.size = fftOutputBuffer_1->getSize();
						pInfos[0].buffer.offset = 0u;

						// Out Buffer 
						pWrites[1].binding = 1;
						pWrites[1].descriptorType = asset::EDT_STORAGE_IMAGE;
						pWrites[1].count = channelCountOverride;
						for (uint32_t i = 0u; i < channelCountOverride; i++)
						{
							auto& info = pInfos[1u + i];
							info.desc = kernelNormalizedSpectrums[i];
							//info.image.imageLayout = ;
							info.image.sampler = nullptr;
						}

						driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
						return dset;
					}();

					// Ker Image First Axis FFT
					{
						auto fftPipeline_ImageInput = driver->createComputePipeline(nullptr, core::smart_refctd_ptr(imageFirstFFTPipelineLayout), createShader(driver, 0x1u << fftPushConstants[0].getLog2FFTSize(), useHalfFloats, "../image_first_fft.comp", bloomScale));
						driver->bindComputePipeline(fftPipeline_ImageInput.get());
						driver->bindDescriptorSets(EPBP_COMPUTE, imageFirstFFTPipelineLayout.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT[0].get(), nullptr);
						FFTClass::dispatchHelper(driver, imageFirstFFTPipelineLayout.get(), fftPushConstants[0], fftDispatchInfo[0]);
					}

					// Ker Image Last Axis FFT
					driver->bindComputePipeline(fftPipeline_SSBOInput.get());
					driver->bindDescriptorSets(EPBP_COMPUTE, fftPipeline_SSBOInput->getLayout(), 0u, 1u, &fftDescriptorSet_Ker_FFT[1].get(), nullptr);
					FFTClass::dispatchHelper(driver, fftPipeline_SSBOInput->getLayout(), fftPushConstants[1], fftDispatchInfo[1]);

					// Ker Normalization
					auto fftPipeline_KernelNormalization = driver->createComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_KernelNormalization), createShader(driver, 0xdeadbeefu, useHalfFloats, "../normalization.comp"));
					driver->bindComputePipeline(fftPipeline_KernelNormalization.get());
					driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_KernelNormalization.get(), 0u, 1u, &fftDescriptorSet_KernelNormalization.get(), nullptr);
					{
						NormalizationPushConstants normalizationPC;
						normalizationPC.stride = fftPushConstants[1].output_strides;
						normalizationPC.bitreverse_shift.x = 32 - core::findMSB(paddedKerDim.width);
						normalizationPC.bitreverse_shift.y = 32 - core::findMSB(paddedKerDim.height);
						normalizationPC.bitreverse_shift.z = 0;
						driver->pushConstants(fftPipelineLayout_KernelNormalization.get(), ICPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(normalizationPC), &normalizationPC);
					}
					{
						const uint32_t dispatchSizeX = (paddedKerDim.width - 1u) / 16u + 1u;
						const uint32_t dispatchSizeY = (paddedKerDim.height - 1u) / 16u + 1u;
						driver->dispatch(dispatchSizeX, dispatchSizeY, kerNumChannels);
						FFTClass::defaultBarrier();
					}
		}




























		// this time we load a shader directly from a file
		smart_refctd_ptr<IGPUShader> shader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			// The down-cast should not fail!
			assert(source);

			// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
			shader = m_device->createShader(source.get());
			if (!shader)
				return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
		}

		// The StreamingTransientDataBuffers are actually composed on top of another useful utility called `CAsyncSingleBufferSubAllocator`
		// The difference is that the streaming ones are made on top of ranges of `IGPUBuffer`s backed by mappable memory, whereas the
		// `CAsyncSingleBufferSubAllocator` just allows you suballocate subranges of any `IGPUBuffer` range with deferred/latched frees.
		constexpr uint32_t DownstreamBufferSize = sizeof(output_t) << 23;
		constexpr uint32_t UpstreamBufferSize = sizeof(input_t) << 23;

		m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger), DownstreamBufferSize, UpstreamBufferSize);
		if (!m_utils)
			return logFail("Failed to create Utilities!");
		m_upStreamingBuffer = m_utils->getDefaultUpStreamingBuffer();
		m_downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();
		m_upStreamingBufferAddress = m_upStreamingBuffer->getBuffer()->getDeviceAddress();
		m_downStreamingBufferAddress = m_downStreamingBuffer->getBuffer()->getDeviceAddress();

		// Create device-local buffer
		
		{
			const uint32_t scalarElementCount = 2 * complexElementCount;
			IGPUBuffer::SCreationParams deviceLocalBufferParams = {};
			
			IQueue* const queue = getComputeQueue();
			uint32_t queueFamilyIndex = queue->getFamilyIndex();
			
			deviceLocalBufferParams.queueFamilyIndexCount = 1;
			deviceLocalBufferParams.queueFamilyIndices = &queueFamilyIndex;
			deviceLocalBufferParams.size = sizeof(input_t) * scalarElementCount;
			deviceLocalBufferParams.usage = nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
			
			m_deviceLocalBuffer = m_device->createBuffer(std::move(deviceLocalBufferParams));
			auto mreqs = m_deviceLocalBuffer->getMemoryReqs();
			mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpubufMem = m_device->allocate(mreqs, m_deviceLocalBuffer.get(), IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);

			m_deviceLocalBufferAddress = m_deviceLocalBuffer.get()->getDeviceAddress();
		}
		

		// People love Reflection but I prefer Shader Sources instead!
		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };

		{
			auto layout = m_device->createPipelineLayout({ &pcRange,1 });
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = layout.get();
			params.shader.shader = shader.get();
			params.shader.requireFullSubgroups = true;
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
				return logFail("Failed to create compute pipeline!\n");
		}

		const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
		// The ranges of non-coherent mapped memory you flush or invalidate need to be aligned. You'll often see a value of 64 reported by devices
		// which just happens to coincide with a CPU cache line size. So we ask our streaming buffers during allocation to give us properly aligned offsets.
		// Sidenote: For SSBOs, UBOs, BufferViews, Vertex Buffer Bindings, Acceleration Structure BDAs, Shader Binding Tables, Descriptor Buffers, etc.
		// there is also a requirement to bind buffers at offsets which have a certain alignment. Memory binding to Buffers and Images also has those.
		// We'll align to max of coherent atom size even if the memory is coherent,
		// and we also need to take into account BDA shader loads need to be aligned to the type being loaded.
		m_alignment = core::max(deviceLimits.nonCoherentAtomSize, alignof(float));

		// We'll allow subsequent iterations to overlap each other on the GPU, the only limiting factors are
		// the amount of memory in the streaming buffers and the number of commandpools we can use simultaenously.
		constexpr auto MaxConcurrency = 64;

		// Since this time we don't throw the Command Pools away and we'll reset them instead, we don't create the pools with the transient flag
		m_poolCache = ICommandPoolCache::create(core::smart_refctd_ptr(m_device), getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::NONE, MaxConcurrency);

		// In contrast to fences, we just need one semaphore to rule all dispatches
		m_timeline = m_device->createSemaphore(m_iteration);

		return true;
	}

	// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
	bool keepRunning() override { return m_iteration < MaxIterations; }

	// Finally the first actual work-loop
	void workLoopBody() override
	{
		IQueue* const queue = getComputeQueue();

		// Note that I'm using the sample struct with methods that have identical code which compiles as both C++ and HLSL
		auto rng = nbl::hlsl::Xoroshiro64StarStar::construct({ m_iteration ^ 0xdeadbeefu,std::hash<string>()(_NBL_APP_NAME_) });

		const uint32_t scalarElementCount = 2 * complexElementCount;
		const uint32_t inputSize = sizeof(input_t) * scalarElementCount;

		// The allocators can do multiple allocations at once for efficiency
		const uint32_t AllocationCount = 1;

		// It comes with a certain drawback that you need to remember to initialize your "yet unallocated" offsets to the Invalid value
		// this is to allow a set of allocations to fail, and you to re-try after doing something to free up space without repacking args.
		auto inputOffset = m_upStreamingBuffer->invalid_value;

		// We always just wait till an allocation becomes possible (during allocation previous "latched" frees get their latch conditions polled)
		// Freeing of Streaming Buffer Allocations can and should be deferred until an associated polled event signals done (more on that later).
		std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
		// note that the API takes a time-point not a duration, because there are multiple waits and preemptions possible, so the durations wouldn't add up properly
		m_upStreamingBuffer->multi_allocate(waitTill, AllocationCount, &inputOffset, &inputSize, &m_alignment);

		// Generate our data in-place on the allocated staging buffer
		{	
			auto* const inputPtr = reinterpret_cast<input_t*>(reinterpret_cast<uint8_t*>(m_upStreamingBuffer->getBufferPointer()) + inputOffset);
			std::cout << "Begin array CPU\n";
			for (auto j = 0; j < complexElementCount; j++)
			{
				//Random array
				/*
				float x = rng() / float(nbl::hlsl::numeric_limits<decltype(rng())>::max), y = 0;//= rng() / float(nbl::hlsl::numeric_limits<decltype(rng())>::max);
				*/
				// FFT( (1,0), (0,0), (0,0),... ) = (1,0), (1,0), (1,0),...
				/*
				float x = j > 0 ? 0.f : 1.f;
				float y = 0;
				*/
				// FFT( (c,0), (c,0), (c,0),... ) = (Nc,0), (0,0), (0,0),...
				
				float x = 2.f;
				float y = 0.f;
				
				inputPtr[2 * j] = x;
				inputPtr[2 * j + 1] = y;
				std::cout << "(" << x << ", " << y << "), ";
			}
			std::cout << "\nEnd array CPU\n";
			// Always remember to flush!
			if (m_upStreamingBuffer->needsManualFlushOrInvalidate())
			{
				const auto bound = m_upStreamingBuffer->getBuffer()->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange range(bound.memory, bound.offset + inputOffset, inputSize);
				m_device->flushMappedMemoryRanges(1, &range);
			}
		}

		// Obtain our command pool once one gets recycled
		uint32_t poolIx;
		do
		{
			poolIx = m_poolCache->acquirePool();
		} while (poolIx == ICommandPoolCache::invalid_index);

		// finally allocate our output range
		const uint32_t outputSize = inputSize;

		auto outputOffset = m_downStreamingBuffer->invalid_value;
		m_downStreamingBuffer->multi_allocate(waitTill, AllocationCount, &outputOffset, &outputSize, &m_alignment);

		smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		{
			m_poolCache->getPool(poolIx)->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf,1 }, core::smart_refctd_ptr(m_logger));
			// lets record, its still a one time submit because we have to re-record with different push constants each time
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->bindComputePipeline(m_pipeline.get());
			// This is the new fun part, pushing constants
			const PushConstantData pc = {
				.inputAddress = m_deviceLocalBufferAddress,
				.outputAddress = m_deviceLocalBufferAddress,
				.dataElementCount = scalarElementCount
			};
			IGPUCommandBuffer::SBufferCopy copyInfo = {};
			copyInfo.srcOffset = 0;
			copyInfo.dstOffset = 0;
			copyInfo.size = m_deviceLocalBuffer->getSize();
			cmdbuf->copyBuffer(m_upStreamingBuffer->getBuffer(), m_deviceLocalBuffer.get(), 1, &copyInfo);
			cmdbuf->pushConstants(m_pipeline->getLayout(), IShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
			// Good old trick to get rounded up divisions, in case you're not familiar
			cmdbuf->dispatch(1, 1, 1);

			// Pipeline barrier: wait for FFT shader to be done before copying to downstream buffer 
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo = {};
			decltype(pipelineBarrierInfo)::buffer_barrier_t barrier = {};
			pipelineBarrierInfo.bufBarriers = {&barrier, 1u};

			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;

			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);
			cmdbuf->copyBuffer(m_deviceLocalBuffer.get(), m_downStreamingBuffer->getBuffer(), 1, &copyInfo);
			cmdbuf->end();
		}


		const auto savedIterNum = m_iteration++;
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
			{
				.cmdbuf = cmdbuf.get()
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
			{
				.semaphore = m_timeline.get(),
				.value = m_iteration,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
			};
			// Generally speaking we don't need to wait on any semaphore because in this example every dispatch gets its own clean piece of memory to use
			// from the point of view of the GPU. Implicit domain operations between Host and Device happen upon a submit and a semaphore/fence signal operation,
			// this ensures we can touch the input and get accurate values from the output memory using the CPU before and after respectively, each submit becoming PENDING.
			// If we actually cared about this submit seeing the memory accesses of a previous dispatch we could add a semaphore wait
			const IQueue::SSubmitInfo submitInfo = {
				.waitSemaphores = {},
				.commandBuffers = {&cmdbufInfo,1},
				.signalSemaphores = {&signalInfo,1}
			};

			queue->startCapture();
			queue->submit({ &submitInfo,1 });
			queue->endCapture();
		}

		// We let all latches know what semaphore and counter value has to be passed for the functors to execute
		const ISemaphore::SWaitInfo futureWait = { m_timeline.get(),m_iteration };

		// We can also actually latch our Command Pool reset and its return to the pool of free pools!
		m_poolCache->releasePool(futureWait, poolIx);

		// As promised, we can defer an upstreaming buffer deallocation until a fence is signalled
		// You can also attach an additional optional IReferenceCounted derived object to hold onto until deallocation.
		m_upStreamingBuffer->multi_deallocate(AllocationCount, &inputOffset, &inputSize, futureWait);

		// Now a new and even more advanced usage of the latched events, we make our own refcounted object with a custom destructor and latch that like we did the commandbuffer.
		// Instead of making our own and duplicating logic, we'll use one from IUtilities meant for down-staging memory.
		// Its nice because it will also remember to invalidate our memory mapping if its not coherent.
		auto latchedConsumer = make_smart_refctd_ptr<IUtilities::CDownstreamingDataConsumer>(
			IDeviceMemoryAllocation::MemoryRange(outputOffset, outputSize),
			// Note the use of capture by-value [=] and not by-reference [&] because this lambda will be called asynchronously whenever the event signals
			[=](const size_t dstOffset, const void* bufSrc, const size_t size)->void
			{
				// The unused variable is used for letting the consumer know the subsection of the output we've managed to download
				// But here we're sure we can get the whole thing in one go because we allocated the whole range ourselves.
				assert(dstOffset == 0 && size == outputSize);

				std::cout << "Begin array GPU\n";
				output_t* const data = reinterpret_cast<output_t*>(const_cast<void*>(bufSrc));
				for (auto i = 0u; i < complexElementCount; i++) {
					std::cout << "(" << data[2 * i] << ", " << data[2 * i + 1] << "), ";
				}

				std::cout << "\nEnd array GPU\n";
			},
			// Its also necessary to hold onto the commandbuffer, even though we take care to not reset the parent pool, because if it
			// hits its destructor, our automated reference counting will drop all references to objects used in the recorded commands.
			// It could also be latched in the upstreaming deallocate, because its the same fence.
			std::move(cmdbuf), m_downStreamingBuffer
		);
		// We put a function we want to execute 
		m_downStreamingBuffer->multi_deallocate(AllocationCount, &outputOffset, &outputSize, futureWait, &latchedConsumer.get());
	}

	bool onAppTerminated() override
	{
		// Need to make sure that there are no events outstanding if we want all lambdas to eventually execute before `onAppTerminated`
		// (the destructors of the Command Pool Cache and Streaming buffers will still wait for all lambda events to drain)
		while (m_downStreamingBuffer->cull_frees()) {}
		return device_base_t::onAppTerminated();
	}
};


NBL_MAIN_FUNC(StreamingAndBufferDeviceAddressApp)