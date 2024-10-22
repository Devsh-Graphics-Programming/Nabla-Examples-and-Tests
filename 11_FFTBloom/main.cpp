// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"


// TODO: Copy HelloSwapchain to set up a swapchain, draw to a surface
// TODO: Dynamically modify bloom size with mouse scroll/ +- keys
//		 Notes: To do so must check required size against buffer size (expand buffer and recompile kernel if it starts being small), also after making kernel larger then small again
//		        probably no need to shrink buffer and recompile kernel but would be nice to add that as well)
// TODO: Clean up example after FFT ext

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "utils.h"

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
	smart_refctd_ptr<IGPUDescriptorSet> m_firstAxisFFTDescriptorSet;
	smart_refctd_ptr<IGPUDescriptorSet> m_lastAxisFFT_convolution_lastAxisIFFTDescriptorSet;
	smart_refctd_ptr<IGPUDescriptorSet> m_firstAxisIFFTDescriptorSet;

	// Utils (might be useful, they stay for now)
	smart_refctd_ptr<nbl::video::IUtilities> m_utils;

	// Resources
	smart_refctd_ptr<IGPUImage> m_srcImage;
	smart_refctd_ptr<IGPUImageView> m_srcImageView;
	smart_refctd_ptr<IGPUImage> m_kerImage;
	smart_refctd_ptr<IGPUImageView> m_kerImageView;
	smart_refctd_ptr<IGPUImage> m_outImg;
	smart_refctd_ptr<IGPUImageView> m_outImgView;
	smart_refctd_ptr<IGPUImageView> m_kernelNormalizedSpectrums[CHANNELS];

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

	// We only hold onto one cmdbuffer at a time
	smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_computeCmdBuf;

	// Sync primitives
	smart_refctd_ptr<ISemaphore> m_timeline;
	uint64_t semaphorValue = 0;
	// For image uploads
	SIntendedSubmitInfo m_graphicsIntendedSubmit;
	smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
	SIntendedSubmitInfo m_computeIntendedSubmit;

	// Only use one queue
	IQueue* m_queue;

	// Termination cond
	bool m_keepRunning = true;

	smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings)
	{
		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };
		return m_device->createPipelineLayout({ &pcRange,1 }, m_device->createDescriptorSetLayout(bindings));
	}

	inline void updateDescriptorSetFirstAxisFFT(IGPUDescriptorSet* set, smart_refctd_ptr<IGPUImageView> inputImageDescriptor)
	{
		IGPUDescriptorSet::SDescriptorInfo info;
		IGPUDescriptorSet::SWriteDescriptorSet write;

		write.dstSet = set;
		write.binding = 0u;
		write.arrayElement = 0u;
		write.count = 1u;
		write.info = &info;

		info.desc = inputImageDescriptor;
		info.info.combinedImageSampler.sampler = nullptr;
		info.info.combinedImageSampler.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

		m_device->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	inline void updateDescriptorSetConvolutionAndNormalization(IGPUDescriptorSet* set, const smart_refctd_ptr<IGPUImageView>* kernelNormalizedSpectrumImageDescriptors, IImage::LAYOUT layout)
	{
		IGPUDescriptorSet::SDescriptorInfo pInfos[CHANNELS];
		IGPUDescriptorSet::SWriteDescriptorSet write;

		write.dstSet = set;
		write.binding = 0;
		write.arrayElement = 0u;
		write.count = CHANNELS;
		write.info = pInfos;

		for (uint32_t i = 0u; i < CHANNELS; i++)
		{
			auto& info = pInfos[i];
			info.desc = kernelNormalizedSpectrumImageDescriptors[i];
			info.info.combinedImageSampler.imageLayout = layout;
			info.info.combinedImageSampler.sampler = nullptr;
		}

		m_device->updateDescriptorSets(1, &write, 0u, nullptr);
	}

	inline void updateDescriptorSetConvolution(IGPUDescriptorSet* set, const smart_refctd_ptr<IGPUImageView>* kernelNormalizedSpectrumImageDescriptors)
	{
		updateDescriptorSetConvolutionAndNormalization(set, kernelNormalizedSpectrumImageDescriptors, IImage::LAYOUT::READ_ONLY_OPTIMAL);
	}

	inline void updateDescriptorSetNormalization(IGPUDescriptorSet* set, const smart_refctd_ptr<IGPUImageView>* kernelNormalizedSpectrumImageDescriptors)
	{
		updateDescriptorSetConvolutionAndNormalization(set, kernelNormalizedSpectrumImageDescriptors, IImage::LAYOUT::GENERAL);
	}

	inline void updateDescriptorSetFirstAxisIFFT(IGPUDescriptorSet* set, smart_refctd_ptr<IGPUImageView> outputImageDescriptor)
	{
		IGPUDescriptorSet::SDescriptorInfo info;
		IGPUDescriptorSet::SWriteDescriptorSet write;

		write.dstSet = set;
		write.binding = 0;
		write.arrayElement = 0u;
		write.count = 1;
		write.info = &info;

		info.desc = outputImageDescriptor;
		info.info.combinedImageSampler.imageLayout = IImage::LAYOUT::GENERAL;
		info.info.combinedImageSampler.sampler = nullptr;

		m_device->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	inline core::smart_refctd_ptr<video::IGPUShader> createShader(
		const char* includeMainName,
		uint32_t workgroupSize,
		uint32_t elementsPerThread,
		float kernelScale = 1.f)
	{

		const char* sourceFmt =
		R"===(
		#define _NBL_HLSL_WORKGROUP_SIZE_ %u
		#define ELEMENTS_PER_THREAD %u
		%s
		
		#define KERNEL_SCALE %f
 
		#include "%s"

		)===";

		const size_t extraSize = 4u + 4u + 26u + 128u;

		auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt) + extraSize + 1u);
		snprintf(
			reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), sourceFmt,
			workgroupSize,
			elementsPerThread,
			useHalfFloats ? "#define USE_HALF_PRECISION" : "",
			kernelScale,
			includeMainName
		);

		auto CPUShader = core::make_smart_refctd_ptr<ICPUShader>(std::move(shader), IShader::E_SHADER_STAGE::ESS_COMPUTE, IShader::E_CONTENT_TYPE::ECT_HLSL, includeMainName);
		assert(CPUShader);
		return m_device->createShader(CPUShader.get());
	}

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	FFTBloomApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// Setup semaphores
		m_timeline = m_device->createSemaphore(semaphorValue);
		// We can't use the same sepahore for uploads so we signal a different semaphore if we need to
		m_scratchSemaphore = m_device->createSemaphore(0);

		// Get compute queue
		m_queue = getComputeQueue();
		uint32_t queueFamilyIndex = m_queue->getFamilyIndex();

		// Create a resettable command buffer
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queueFamilyIndex, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_computeCmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		// Needs to be open for utilities
		m_computeCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		// Load source and kernel images
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto srcImageBundle = m_assetMgr->getAsset("../../media/colorexr.exr", lp);
			auto kerImageBundle = m_assetMgr->getAsset("../../media/kernels/physical_flare_256.exr", lp);
			const auto srcImages = srcImageBundle.getContents();
			const auto kerImages = kerImageBundle.getContents();
			if (srcImages.empty() or kerImages.empty())
				return logFail("Could not load image or kernel!");
			const auto srcImageCPU = IAsset::castDown<ICPUImage>(srcImages[0]);
			const auto kerImageCPU = IAsset::castDown<ICPUImage>(kerImages[0]);

			// Using asset converter
			smart_refctd_ptr<nbl::video::CAssetConverter> converter = nbl::video::CAssetConverter::create({ .device = m_device.get(),.optimizer = {} });
			CAssetConverter::SInputs inputs = {};
			inputs.logger = m_logger.get();
			nbl::asset::ICPUImage* CPUImages[2] = { srcImageCPU.get(), kerImageCPU.get() };
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { CPUImages, 2};
			
			// Need to provide patches
			CAssetConverter::patch_t<ICPUImage> srcImagePatch(srcImageCPU.get());
			srcImagePatch.usageFlags |= IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT;
			CAssetConverter::patch_t<ICPUImage> kerImagePatch(kerImageCPU.get());
			kerImagePatch.usageFlags |= IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT;
			CAssetConverter::patch_t<ICPUImage> patches[2] = { std::move(srcImagePatch), std::move(kerImagePatch)};

			std::get<CAssetConverter::SInputs::patch_span_t<ICPUImage>>(inputs.patches) = { patches, 2 };
			auto reservation = converter->reserve(inputs);
			const auto GPUImages = reservation.getGPUObjects<ICPUImage>();

			m_srcImage = GPUImages[0].value;
			m_kerImage = GPUImages[1].value;

			// The down-cast should not fail!
			assert(m_srcImage);
			assert(m_kerImage);

			// Required size for uploads
			auto srcImageDims = m_srcImage->getCreationParameters().extent;
			auto kerImageDims = m_kerImage->getCreationParameters().extent;
			uint32_t srcImageSize = srcImageDims.height * srcImageDims.width * srcImageDims.depth * CHANNELS * sizeof(float32_t);
			uint32_t kerImageSize = kerImageDims.height * kerImageDims.width * kerImageDims.depth * CHANNELS * sizeof(float32_t);

			m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger), srcImageSize, srcImageSize + kerImageSize);
			{
				uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
				uint32_t maxFreeBlock = m_utils->getDefaultUpStreamingBuffer()->max_size();
				const uint32_t allocationAlignment = 64u;
				const uint32_t allocationSize = srcImageSize + kerImageSize;
				m_utils->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
			}

			// Now convert uploads

			// Get graphics queue for image transfer
			auto graphicsQueue = getQueue(IQueue::FAMILY_FLAGS::GRAPHICS_BIT);
			m_graphicsIntendedSubmit.queue = graphicsQueue;
			// Set up submit for image transfers
			// wait for nothing before upload
			m_graphicsIntendedSubmit.waitSemaphores = {};
			m_graphicsIntendedSubmit.prevCommandBuffers = {};
			// fill later
			m_graphicsIntendedSubmit.scratchCommandBuffers = {};
			m_graphicsIntendedSubmit.scratchSemaphore = {
				.semaphore = m_scratchSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};

			// Set up the same for mipmap generation
			m_computeIntendedSubmit.queue = m_queue;
			// Set up submit for image transfers
			// wait for nothing
			m_computeIntendedSubmit.waitSemaphores = { };
			m_computeIntendedSubmit.prevCommandBuffers = {};
			// fill later
			m_computeIntendedSubmit.scratchCommandBuffers = {};
			// Signal to timeline semaphore although it's not needed because it's pipeline barriered
			m_computeIntendedSubmit.scratchSemaphore = {
				.semaphore = m_timeline.get(),
				.value = ++semaphorValue,
				.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
			};

			// now convert
			IQueue::SSubmitInfo::SCommandBufferInfo computeCmdbufInfo = { m_computeCmdBuf.get() };
			m_computeIntendedSubmit.scratchCommandBuffers = { &computeCmdbufInfo,1 };
			
			// Create a one-time command buffer for graphics submit
			smart_refctd_ptr<IGPUCommandBuffer> graphicsCmdBuf;
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(graphicsQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &graphicsCmdBuf))
				return logFail("Failed to create Command Buffers!\n");

			IQueue::SSubmitInfo::SCommandBufferInfo graphicsCmdbufInfo = { graphicsCmdBuf.get() };
			m_graphicsIntendedSubmit.scratchCommandBuffers = { &graphicsCmdbufInfo,1 };

			// TODO: FIXME ImGUI needs to know what Queue will have ownership of the image AFTER its uploaded (need to know the family of the graphics queue)
			// right now, the transfer queue will stay the owner after upload
			CAssetConverter::SConvertParams params = {};
			params.transfer = &m_graphicsIntendedSubmit;
			params.compute = &m_computeIntendedSubmit;
			params.utilities = m_utils.get();
			auto result = reservation.convert(params);
			// block immediately
			if (result.copy() != IQueue::RESULT::SUCCESS)
				return false;
		}

		// Create views for these images
		{
			IGPUImageView::SCreationParams srcImgViewInfo;
			srcImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			srcImgViewInfo.image = m_srcImage;
			srcImgViewInfo.viewType = IGPUImageView::ET_2D;
			srcImgViewInfo.format = srcImgViewInfo.image->getCreationParameters().format;
			srcImgViewInfo.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
			srcImgViewInfo.subresourceRange.baseMipLevel = 0;
			srcImgViewInfo.subresourceRange.levelCount = 1;
			srcImgViewInfo.subresourceRange.baseArrayLayer = 0;
			srcImgViewInfo.subresourceRange.layerCount = 1;
			m_srcImageView = m_device->createImageView(std::move(srcImgViewInfo));

			IGPUImageView::SCreationParams kerImgViewInfo;
			kerImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			kerImgViewInfo.image = m_kerImage;
			kerImgViewInfo.viewType = IGPUImageView::ET_2D;
			kerImgViewInfo.format = kerImgViewInfo.image->getCreationParameters().format;
			kerImgViewInfo.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
			kerImgViewInfo.subresourceRange.baseMipLevel = 0;
			kerImgViewInfo.subresourceRange.levelCount = kerImgViewInfo.image->getCreationParameters().mipLevels;
			kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
			kerImgViewInfo.subresourceRange.layerCount = 1;
			m_kerImageView = m_device->createImageView(std::move(kerImgViewInfo));
		}

		// Create Out Image
		{
			auto dstImgViewInfo = m_srcImageView->getCreationParameters();

			IGPUImage::SCreationParams dstImgInfo(dstImgViewInfo.image->getCreationParameters());
			// Specify we want this to be a storage image, + transfer for readback (blit when we have swapchain up)
			dstImgInfo.usage = IImage::EUF_STORAGE_BIT | IImage::EUF_TRANSFER_SRC_BIT;
			m_outImg = m_device->createImage(std::move(dstImgInfo));

			auto memReqs = m_outImg->getMemoryReqs();
			memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpuMem = m_device->allocate(memReqs, m_outImg.get());

			dstImgViewInfo.image = m_outImg;
			m_outImgView = m_device->createImageView(IGPUImageView::SCreationParams(dstImgViewInfo));
		}

		// agree on formats
		const E_FORMAT srcFormat = m_srcImageView->getCreationParameters().format;
		// TODO: this might be pointless?
		uint32_t srcNumChannels = getFormatChannelCount(srcFormat);
		uint32_t kerNumChannels = getFormatChannelCount(m_kerImageView->getCreationParameters().format);
		//! OVERRIDE (we dont need alpha)
		srcNumChannels = CHANNELS;
		kerNumChannels = CHANNELS;
		assert(srcNumChannels == kerNumChannels); // Just to make sure, because the other case is not handled in this example

		// Compute (kernel) padding size

		// Kernel pixel to image pixel conversion ratio
		const float bloomRelativeScale = 0.25f;
		const auto kerDim = m_kerImageView->getCreationParameters().image->getCreationParameters().extent;
		const auto srcDim = m_srcImageView->getCreationParameters().image->getCreationParameters().extent;
		bloomScale = core::min(float(srcDim.width) / float(kerDim.width), float(srcDim.height) / float(kerDim.height)) * bloomRelativeScale;
		if (bloomScale > 1.f)
			std::cout << "WARNING: Bloom Kernel will Clip and loose sharpness, increase resolution of bloom kernel!" << std::endl;
		marginSrcDim = srcDim;
		// Add padding to marginSrcDim
		for (auto i = 0u; i < 3u; i++)
		{
			const auto coord = (&kerDim.width)[i];
			if (coord > 1u)
				(&marginSrcDim.width)[i] += ceil(core::max(coord * bloomScale, 1u)) - 1u;
		}
		
		// Create intermediate buffers
		{
			IGPUBuffer::SCreationParams deviceLocalBufferParams = {};

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
		auto createSampler = [&](ISampler::E_TEXTURE_CLAMP textureWrap) -> smart_refctd_ptr<IGPUSampler>
		{
			IGPUSampler::SParams params =
			{
				textureWrap,
				textureWrap,
				textureWrap,
				ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				ISampler::ETF_LINEAR,
				ISampler::ETF_LINEAR,
				ISampler::ESMM_LINEAR,
				3u,
				0u,
				ISampler::ECO_ALWAYS
			};
			return m_device->createSampler(std::move(params));
		};

		smart_refctd_ptr<IGPUPipelineLayout> imageFirstAxisFFTPipelineLayout;
		{
			auto sampler = createSampler(ISampler::E_TEXTURE_CLAMP::ETC_MIRROR);
			IGPUDescriptorSetLayout::SBinding bnd =
			{
				IDescriptorSetLayoutBase::SBindingBase(),
				0u,
				IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				IShader::E_SHADER_STAGE::ESS_COMPUTE,
				1u,
				&sampler
			};

			imageFirstAxisFFTPipelineLayout = createPipelineLayout({ &bnd, 1 });
		}

		smart_refctd_ptr<IGPUPipelineLayout> lastAxisFFT_convolution_lastAxisIFFTPipelineLayout;
		{
			auto sampler = createSampler(ISampler::E_TEXTURE_CLAMP::ETC_MIRROR);
			smart_refctd_ptr<IGPUSampler> samplers[CHANNELS];
			std::fill_n(samplers, CHANNELS, sampler);
			IGPUDescriptorSetLayout::SBinding bnd =
			{
				IDescriptorSetLayoutBase::SBindingBase(),
				0u,
				IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				IShader::E_SHADER_STAGE::ESS_COMPUTE,
				CHANNELS,
				samplers
			};

			lastAxisFFT_convolution_lastAxisIFFTPipelineLayout = createPipelineLayout({ &bnd, 1 });
		}

		smart_refctd_ptr<IGPUPipelineLayout> imageFirstAxisIFFTPipelineLayout;
		{
			IGPUDescriptorSetLayout::SBinding bnd =
			{
				IDescriptorSetLayoutBase::SBindingBase(),
				0u,
				IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				IShader::E_SHADER_STAGE::ESS_COMPUTE,
				1,
				nullptr
			};

			imageFirstAxisIFFTPipelineLayout = createPipelineLayout({ &bnd, 1 });
		}

		// Kernel second axis FFT has no descriptor sets so we just create another pipeline with the same layout
		// TODO: To avoid duplicated layouts we could make samplers dynamic in the first axis FFT. Also if we don't hardcode (by #defining) some stuff in the first axis FFT
		//		 (once FFT ext is back) we can also avoid having duplicated pipelines (like the old Bloom example, which had a single pipeline for forward FFT along an axis)
		//       and setting stuff via shader push constants (such as which axis to perform FFT on and the size of output image).

		// -------------------------------------- KERNEL FFT PRECOMP ----------------------------------------------------------------
		{
			// Pipeline Layouts
			smart_refctd_ptr<IGPUPipelineLayout> kernelFirstAxisFFTPipelineLayout;
			{
				auto sampler = createSampler(ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER);
				IGPUDescriptorSetLayout::SBinding bnd =
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					0u,
					IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					1u,
					&sampler
				};

				kernelFirstAxisFFTPipelineLayout = createPipelineLayout({ &bnd, 1 });
			}

			smart_refctd_ptr<IGPUPipelineLayout> kernelNormalizationPipelineLayout;
			{
				IGPUDescriptorSetLayout::SBinding bnd =
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					0u,
					IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					CHANNELS,
					nullptr
				};

				kernelNormalizationPipelineLayout = createPipelineLayout({ &bnd, 1 });
			}


			const asset::VkExtent3D paddedKerDim = padDimensions(kerDim);

			// create kernel spectrums
			auto createKernelSpectrum = [&]() -> auto
				{
					video::IGPUImage::SCreationParams imageParams;
					imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					imageParams.type = asset::IImage::ET_2D;
					imageParams.format = useHalfFloats ? EF_R16G16_SFLOAT : EF_R32G32_SFLOAT;
					imageParams.extent = { paddedKerDim.width,paddedKerDim.height,1u };
					imageParams.mipLevels = 1u;
					imageParams.arrayLayers = 1u;
					imageParams.samples = asset::IImage::ESCF_1_BIT;
					imageParams.usage = IImage::EUF_STORAGE_BIT | IImage::EUF_SAMPLED_BIT;

					auto kernelImg = m_device->createImage(std::move(imageParams));

					auto memReqs = kernelImg->getMemoryReqs();
					memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					auto gpuMem = m_device->allocate(memReqs, kernelImg.get());

					video::IGPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = kernelImg;
					viewParams.viewType = video::IGPUImageView::ET_2D;
					viewParams.format = useHalfFloats ? EF_R16G16_SFLOAT : EF_R32G32_SFLOAT;
					viewParams.components = {};
					viewParams.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
					viewParams.subresourceRange.baseMipLevel = 0;
					viewParams.subresourceRange.levelCount = 1;
					viewParams.subresourceRange.baseArrayLayer = 0;
					viewParams.subresourceRange.layerCount = 1;
					return m_device->createImageView(std::move(viewParams));
				};

			for (uint32_t i = 0u; i < CHANNELS; i++)
				m_kernelNormalizedSpectrums[i] = createKernelSpectrum();

			// Invoke a workgroup per two vertical scanlines. Kernel is square and runs first in the y-direction.
			// That means we have to create a shader that does an FFT of size `paddedKerDim.height = paddedKerDim.width` (length of each column, already padded to PoT), 
			// and call `paddedKerDim.width / 2` workgroups to run it. We also have to keep in mind `paddedKerDim.y = WorkgroupSize * ElementsPerInvocation`. 
			// We prefer to go with 2 elements per invocation and max out WorkgroupSize when possible.
			// This is because we use PreloadedAccessors which reduce global memory accesses at the cost of decreasing occupancy with increasing ElementsPerInvocation

			// Create descriptor sets
			const IGPUDescriptorSetLayout* kernelDSLayouts[2] = { kernelFirstAxisFFTPipelineLayout->getDescriptorSetLayout(0), kernelNormalizationPipelineLayout->getDescriptorSetLayout(0) };
			smart_refctd_ptr<IDescriptorPool> kernelFFTDSPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { kernelDSLayouts, 2 });
			smart_refctd_ptr<IGPUDescriptorSet> kernelFFTDescriptorSets[2];
			uint32_t dsCreated = kernelFFTDSPool->createDescriptorSets({kernelDSLayouts, 2}, kernelFFTDescriptorSets);

			// Cba to handle errors it's an example
			if (dsCreated != 2)
				return logFail("Failed to create Descriptor Sets!\n");

			// Write descriptor sets
			updateDescriptorSetFirstAxisFFT(kernelFFTDescriptorSets[0].get(), m_kerImageView);
			updateDescriptorSetNormalization(kernelFFTDescriptorSets[1].get(), m_kernelNormalizedSpectrums);

			// Compute required WorkgroupSize and ElementsPerThread for FFT
			// Remember we assume kernel is square!
			uint32_t maxWorkgroupSize = *m_device->getPhysicalDevice()->getLimits().maxWorkgroupSize;
			uint32_t elementsPerThread = 1, workgroupSize;
			do
			{
				elementsPerThread <<= 1;
				workgroupSize = paddedKerDim.width / elementsPerThread;
			} 
			while (workgroupSize > maxWorkgroupSize);

			// Create shaders
			smart_refctd_ptr<IGPUShader> shaders[3];
			shaders[0] = createShader("app_resources/kernel_fft_first_axis.hlsl", workgroupSize, elementsPerThread, bloomScale);
			shaders[1] = createShader("app_resources/kernel_fft_second_axis.hlsl", workgroupSize, elementsPerThread, bloomScale);
			shaders[2] = createShader("app_resources/kernel_spectrum_normalize.hlsl", workgroupSize, elementsPerThread, bloomScale);

			// -------------------------------------------

			// Create compute pipelines - First axis FFT -> Second axis FFT -> Normalization
			IGPUComputePipeline::SCreationParams params[3];
			// First axis FFT
			params[0].layout = kernelFirstAxisFFTPipelineLayout.get();	
			// Second axis FFT -  since no descriptor sets are used, just keep the same pipeline layout to avoid having yet another pipeline layout creation step
			params[1].layout = kernelFirstAxisFFTPipelineLayout.get();
			// Normalization
			params[2].layout = kernelNormalizationPipelineLayout.get();
			// Common
			for (auto i = 0u; i < 3; i++) {
				params[i].shader.entryPoint = "main";
				params[i].shader.shader = shaders[i].get();
				params[i].shader.requireFullSubgroups = true;
			}
			
			smart_refctd_ptr<IGPUComputePipeline> pipelines[3];
			bool success = m_device->createComputePipelines(nullptr, { params, 3 }, pipelines);
			if(! success)
				return logFail("Failed to create Compute Pipelines!\n");

			// Push Constants - only need to specify BDAs here
			PushConstantData pushConstants;
			pushConstants.colMajorBufferAddress = m_colMajorBufferAddress;
			pushConstants.rowMajorBufferAddress = m_rowMajorBufferAddress;

			// First Axis FFT
			m_computeCmdBuf->bindComputePipeline(pipelines[0].get());
			m_computeCmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, pipelines[0]->getLayout(), 0, 1, &kernelFFTDescriptorSets[0].get());
			m_computeCmdBuf->pushConstants(pipelines[0]->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pushConstants), &pushConstants);
			// One workgroup per 2 columns
			m_computeCmdBuf->dispatch(paddedKerDim.width / 2, 1, 1);

			// Pipeline barrier: wait for first axis FFT before second axis can begin
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo = {};
			decltype(pipelineBarrierInfo)::buffer_barrier_t bufBarrier = {};
			pipelineBarrierInfo.bufBarriers = { &bufBarrier, 1u };
			
			// First axis FFT writes to colMajorBuffer
			bufBarrier.range.buffer = m_colMajorBuffer;

			// Wait for first compute write (first axis FFT) before next compute read (second axis FFT)
			bufBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			bufBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
			bufBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			bufBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;

			m_computeCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);

			// Now do second axis FFT - no need to push the same constants again I think
			m_computeCmdBuf->bindComputePipeline(pipelines[1].get());
			// Same number of workgroups - this time because we only saved half the rows
			m_computeCmdBuf->dispatch(paddedKerDim.width / 2, 1, 1);

			// Recycle the pipelineBarrierInfo since it's identical, just change buffer access: Second axis FFT writes to rowMajorBuffer
			bufBarrier.range.buffer = m_rowMajorBuffer;

			// Also set kernel channel images to GENERAL for writing
			decltype(pipelineBarrierInfo)::image_barrier_t imgBarriers[CHANNELS];
			pipelineBarrierInfo.imgBarriers = { imgBarriers, CHANNELS };
			for (auto i = 0u; i < CHANNELS; i++)
			{
				imgBarriers[i].image = m_kernelNormalizedSpectrums[i]->getCreationParameters().image.get();
				imgBarriers[i].subresourceRange.aspectMask =IImage::EAF_COLOR_BIT;
				imgBarriers[i].subresourceRange.levelCount = 1u;
				imgBarriers[i].subresourceRange.layerCount = 1u;
				imgBarriers[i].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				imgBarriers[i].barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
				imgBarriers[i].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				imgBarriers[i].barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
				imgBarriers[i].oldLayout = IImage::LAYOUT::UNDEFINED;
				imgBarriers[i].newLayout = IImage::LAYOUT::GENERAL;
			}

			m_computeCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);

			//Finally, normalize kernel Image - same number of workgroups
			m_computeCmdBuf->bindComputePipeline(pipelines[2].get());
			m_computeCmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, pipelines[2]->getLayout(), 0, 1, &kernelFFTDescriptorSets[1].get());
			m_computeCmdBuf->dispatch(paddedKerDim.width / 2, 1, 1);
			m_computeCmdBuf->end();

			// Submit to queue and add sync point
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
				{
					.cmdbuf = m_computeCmdBuf.get()
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
				{
					.semaphore = m_timeline.get(),
					.value = ++semaphorValue,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				};

				// No need to wait on transfer since it's done in same queue (pipeline barriered)
				const IQueue::SSubmitInfo submitInfo = {
					.waitSemaphores = {},
					.commandBuffers = {&cmdbufInfo,1},
					.signalSemaphores = {&signalInfo,1}
				};

				//queue->startCapture();
				m_queue->submit({ &submitInfo,1 });
				//queue->endCapture();
			}
		}
		// ----------------------------------------- KERNEL PRECOMP END -------------------------------------------------

		// Now create the pipelines for the image FFT
		uint32_t maxWorkgroupSize = *m_device->getPhysicalDevice()->getLimits().maxWorkgroupSize;
		uint32_t elementsPerThread = 1, workgroupSize;
		do
		{
			elementsPerThread <<= 1;
			// First axis FFT is along Y axis
			workgroupSize = core::roundUpToPoT(marginSrcDim.height) / elementsPerThread;
		} while (workgroupSize > maxWorkgroupSize);

		smart_refctd_ptr<IGPUShader> shaders[3];
		shaders[0] = createShader("app_resources/image_fft_first_axis.hlsl", workgroupSize, elementsPerThread);
		// IFFT along first axis has same dimensions as FFT
		shaders[2] = createShader("app_resources/image_ifft_first_axis.hlsl", workgroupSize, elementsPerThread);
		
		// Second axis FFT might have different dimensions
		elementsPerThread = 1;
		do
		{
			elementsPerThread <<= 1;
			// First axis FFT is along Y axis
			workgroupSize = core::roundUpToPoT(marginSrcDim.width) / elementsPerThread;
		} while (workgroupSize > maxWorkgroupSize);
		shaders[1] = createShader("app_resources/fft_convolve_ifft.hlsl", workgroupSize, elementsPerThread);

		// Create compute pipelines - First axis FFT -> Second axis FFT -> Normalization
		IGPUComputePipeline::SCreationParams params[3];
		// First axis FFT
		params[0].layout = imageFirstAxisFFTPipelineLayout.get();
		// Second axis FFT + Conv + IFFT
		params[1].layout = lastAxisFFT_convolution_lastAxisIFFTPipelineLayout.get();
		// First axis IFFT
		params[2].layout = imageFirstAxisIFFTPipelineLayout.get();
		// Common
		for (auto i = 0u; i < 3; i++) {
			params[i].shader.entryPoint = "main";
			params[i].shader.shader = shaders[i].get();
			params[i].shader.requireFullSubgroups = true;
		}

		smart_refctd_ptr<IGPUComputePipeline> pipelines[3];
		bool success = m_device->createComputePipelines(nullptr, { params, 3 }, pipelines);
		if (!success)
			return logFail("Failed to create Compute Pipelines!\n");

		m_firstAxisFFTPipeline = pipelines[0];
		m_lastAxisFFT_convolution_lastAxisIFFTPipeline = pipelines[1];
		m_firstAxisIFFTPipeline = pipelines[2];

		// Create descriptor sets
		const IGPUDescriptorSetLayout* DSLayouts[3] = { pipelines[0]->getLayout()->getDescriptorSetLayout(0), pipelines[1]->getLayout()->getDescriptorSetLayout(0) , pipelines[2]->getLayout()->getDescriptorSetLayout(0) };
		smart_refctd_ptr<IDescriptorPool> DSPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { DSLayouts, 3 });
		smart_refctd_ptr<IGPUDescriptorSet> descriptorSets[3];
		uint32_t dsCreated = DSPool->createDescriptorSets({ DSLayouts, 3 }, descriptorSets);

		// Cba to handle errors it's an example
		if (dsCreated != 3)
			return logFail("Failed to create Descriptor Sets!\n");

		m_firstAxisFFTDescriptorSet = descriptorSets[0];
		m_lastAxisFFT_convolution_lastAxisIFFTDescriptorSet = descriptorSets[1];
		m_firstAxisIFFTDescriptorSet = descriptorSets[2];

		// Write descriptor sets
		updateDescriptorSetFirstAxisFFT(m_firstAxisFFTDescriptorSet.get(), m_srcImageView);
		updateDescriptorSetConvolution(m_lastAxisFFT_convolution_lastAxisIFFTDescriptorSet.get(), m_kernelNormalizedSpectrums);
		updateDescriptorSetFirstAxisIFFT(m_firstAxisIFFTDescriptorSet.get(), m_outImgView);
	}

	bool keepRunning() override { return m_keepRunning; }

	// Right now it's one shot app, but I put this code here for easy refactoring when it refactors into it being a live app
	void workLoopBody() override
	{
		uint32_t queueFamilyIndex = m_queue->getFamilyIndex();

		// RESET COMMAND BUFFER OR SOMETHING

		m_computeCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		// Pipeline barrier: transition kernel spectrum images into read only, and outImage into general
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo imagePipelineBarrierInfo = {};
		decltype(imagePipelineBarrierInfo)::image_barrier_t imgBarriers[CHANNELS + 1];
		imagePipelineBarrierInfo.imgBarriers = { imgBarriers, CHANNELS + 1};

		// outImage just needs a layout transition before it can be written to
		imgBarriers[0].image = m_outImg.get();
		imgBarriers[0].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		imgBarriers[0].barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
		imgBarriers[0].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		imgBarriers[0].barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
		imgBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
		imgBarriers[0].newLayout = IImage::LAYOUT::GENERAL;

		// We need to wait on kernel spectrums to be written to before we can read them
		for (auto i = 0u; i < CHANNELS; i++)
		{
			imgBarriers[i + 1].image = m_kernelNormalizedSpectrums[i]->getCreationParameters().image.get();
			imgBarriers[i + 1].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarriers[i + 1].barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			imgBarriers[i + 1].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarriers[i + 1].barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			imgBarriers[i + 1].oldLayout = IImage::LAYOUT::GENERAL;
			imgBarriers[i + 1].newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		}

		m_computeCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), imagePipelineBarrierInfo);
		
		// Prepare for first axis FFT
		// Push Constants - only need to specify BDAs here
		PushConstantData pushConstants;
		pushConstants.colMajorBufferAddress = m_colMajorBufferAddress;
		pushConstants.rowMajorBufferAddress = m_rowMajorBufferAddress;
		pushConstants.dataElementCount = m_srcImage->getCreationParameters().extent.width;
		// Compute kernel half pixel size
		const auto& kernelImgExtent = m_kernelNormalizedSpectrums[0]->getCreationParameters().image->getCreationParameters().extent;
		float32_t2 kernelHalfPixelSize{ 0.5f,0.5f };
		kernelHalfPixelSize.x /= kernelImgExtent.width;
		kernelHalfPixelSize.y /= kernelImgExtent.height;
		pushConstants.kernelHalfPixelSize = kernelHalfPixelSize;


		m_computeCmdBuf->bindComputePipeline(m_firstAxisFFTPipeline.get());
		m_computeCmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_firstAxisFFTPipeline->getLayout(), 0, 1, &m_firstAxisFFTDescriptorSet.get());
		m_computeCmdBuf->pushConstants(m_firstAxisFFTPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pushConstants), &pushConstants);
		// One workgroup per 2 columns
		m_computeCmdBuf->dispatch(marginSrcDim.width / 2, 1, 1);

		// Pipeline Barrier: Wait for colMajorBuffer to be written to before reading it from next shader
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo bufferPipelineBarrierInfo = {};
		decltype(bufferPipelineBarrierInfo)::buffer_barrier_t bufBarrier = {};
		bufferPipelineBarrierInfo.bufBarriers = { &bufBarrier, 1u };

		// First axis FFT writes to colMajorBuffer
		bufBarrier.range.buffer = m_colMajorBuffer;

		// Wait for first compute write (first axis FFT) before next compute read (second axis FFT)
		bufBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		bufBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
		bufBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		bufBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;

		m_computeCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), bufferPipelineBarrierInfo);
		// Now comes Second axis FFT + Conv + IFFT
		m_computeCmdBuf->bindComputePipeline(m_lastAxisFFT_convolution_lastAxisIFFTPipeline.get());
		m_computeCmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_lastAxisFFT_convolution_lastAxisIFFTPipeline->getLayout(), 0, 1, &m_lastAxisFFT_convolution_lastAxisIFFTDescriptorSet.get());
		// We need to pass the log of number of workgroups as a push constant now
		uint32_t numWorkgroups = core::roundUpToPoT(marginSrcDim.height) / 2;
		uint32_t workgroupsLog2 = std::bit_width(numWorkgroups) - 1;
		m_computeCmdBuf->pushConstants(m_lastAxisFFT_convolution_lastAxisIFFTPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, offsetof(PushConstantData, numWorkgroupsLog2), sizeof(pushConstants.numWorkgroupsLog2), &workgroupsLog2);

		m_computeCmdBuf->dispatch(numWorkgroups, 1, 1);

		// Recycle pipeline barrier, only have to change which buffer we need to wait to be written to
		bufBarrier.range.buffer = m_rowMajorBuffer;
		m_computeCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), bufferPipelineBarrierInfo);

		// Finally run the IFFT on the first axis
		m_computeCmdBuf->bindComputePipeline(m_firstAxisIFFTPipeline.get());
		m_computeCmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_firstAxisIFFTPipeline->getLayout(), 0, 1, &m_firstAxisIFFTDescriptorSet.get());
		// One workgroup per 2 columns
		m_computeCmdBuf->dispatch(marginSrcDim.width / 2, 1, 1);
		m_computeCmdBuf->end();

		// Submit to queue and add sync point
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
			{
				.cmdbuf = m_computeCmdBuf.get()
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
			{
				.semaphore = m_timeline.get(),
				.value = ++semaphorValue,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
			};

			const IQueue::SSubmitInfo submitInfo = {
				.waitSemaphores = {},
				.commandBuffers = {&cmdbufInfo,1},
				.signalSemaphores = {&signalInfo,1}
			};

			//queue->startCapture();
			m_queue->submit({ &submitInfo,1 });
			//queue->endCapture();
		}

		// Kill after one iteration for now
		m_keepRunning = false;
	}

	bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}
};


NBL_MAIN_FUNC(FFTBloomApp)