// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "SimpleWindowedApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace ui;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

// Defaults that match this example's image
constexpr uint32_t WIN_W = 1280;
constexpr uint32_t WIN_H = 720;

class FFTBloomApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;

	// Persistent compute Pipelines
	smart_refctd_ptr<IGPUComputePipeline> m_firstAxisFFTPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_lastAxisFFT_convolution_lastAxisIFFTPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_firstAxisIFFTPipeline;

	// Universal descriptor set
	smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet;

	// Utils
	smart_refctd_ptr<IUtilities> m_utils;

	// Resources
	smart_refctd_ptr<IGPUImageView> m_srcImageView;
	smart_refctd_ptr<IGPUImageView> m_kerImageView;
	smart_refctd_ptr<IGPUImageView> m_outImgView;
	smart_refctd_ptr<IGPUImageView> m_kernelNormalizedSpectrums;

	// Used to store intermediate results
	smart_refctd_ptr<IGPUBuffer> m_rowMajorBuffer;
	smart_refctd_ptr<IGPUBuffer> m_colMajorBuffer;

	// These are Buffer Device Addresses
	uint64_t m_rowMajorBufferAddress;
	uint64_t m_colMajorBufferAddress;

	bool m_useHalfFloats = false;
	
	// Other parameter-dependent variables
	asset::VkExtent3D m_marginSrcDim;
	uint16_t m_imageFirstAxisFFTWorkgroupSize;

	// Shader Cache
	smart_refctd_ptr<IShaderCompiler::CCache> m_readCache;
	smart_refctd_ptr<IShaderCompiler::CCache> m_writeCache;

	// Only use one queue
	IQueue* m_queue;

	// Windowed App members
	constexpr static inline uint32_t MaxFramesInFlight = 3u;
	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_timeline;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	// -------------------------------- WINDOWED APP OVERRIDES ---------------------------------------------------
	inline bool isComputeOnly() const override { return false; }

	virtual video::SPhysicalDeviceLimits getRequiredDeviceLimits() const override
	{
		auto retval = device_base_t::getRequiredDeviceLimits();
		retval.shaderFloat16 = m_useHalfFloats;
		return retval;
	}

	inline core::vector<SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
				params.width = WIN_W;
				params.height = WIN_H;
				params.x = 32;
				params.y = 32;
				params.flags = IWindow::ECF_BORDERLESS | IWindow::ECF_HIDDEN;
				params.windowCaption = "FFT Bloom Demo";
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}
			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
		}

		if (m_surface)
			return { {m_surface->getSurface()/*,EQF_NONE*/} };

		return {};
	}

	// -------------------------------- END WINDOWED APP OVERRIDES ---------------------------------------------------

	inline void updateDescriptorSet(smart_refctd_ptr<IGPUImageView> imageDescriptor, smart_refctd_ptr<IGPUImageView> storageImageDescriptor, smart_refctd_ptr<IGPUImageView> textureArrayDescriptor = nullptr)
	{
		IGPUDescriptorSet::SDescriptorInfo infos[3] = {};
		IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {};

		for (auto i = 0u; i < 3; i++) {
			writes[i].dstSet = m_descriptorSet.get();
			writes[i].arrayElement = 0u;
			writes[i].count = 1u;
			writes[i].info = &infos[i];
		}

		infos[0].desc = imageDescriptor;
		infos[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		// Read image at binding 0
		writes[0].binding = 0u;

		// Binding 2 skipped since it's the sampler which we never need to change

		infos[1].desc = storageImageDescriptor;
		infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
		// Storage image at binding 1
		writes[1].binding = 2u;

		// If nullptr give it SOME value so validation layer doesn't complain even though we don't use it
		infos[2].desc = textureArrayDescriptor ? textureArrayDescriptor : imageDescriptor;
		infos[2].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		// Texture array for reading at binding 3
		writes[2].binding = 3u;

		m_device->updateDescriptorSets(writes, std::span<IGPUDescriptorSet::SCopyDescriptorSet>());
	}


	struct SShaderConstevalParameters
	{
		struct SShaderConstevalParametersCreateInfo
		{
			bool useHalfFloats = false;
			uint16_t elementsPerInvocationLog2 = 0;
			uint16_t workgroupSizeLog2 = 0;
			uint16_t numWorkgroupsLog2 = 0;
			uint16_t previousElementsPerInvocationLog2 = 0;
			uint16_t previousWorkgroupSizeLog2 = 0;
			float32_t2 kernelHalfPixelSize = { 0.5f, 0.5f };
		};

		SShaderConstevalParameters(SShaderConstevalParametersCreateInfo& info) : 
			scalar_t(info.useHalfFloats ? "float16_t" : "float32_t"), elementsPerInvocationLog2(info.elementsPerInvocationLog2), 
			workgroupSizeLog2(info.workgroupSizeLog2), numWorkgroupsLog2(info.numWorkgroupsLog2), numWorkgroups(uint32_t(1) << info.numWorkgroupsLog2), 
			previousElementsPerInvocationLog2(info.previousElementsPerInvocationLog2), previousWorkgroupSizeLog2(info.previousWorkgroupSizeLog2), 
			previousWorkgroupSize(uint16_t(1) << info.previousWorkgroupSizeLog2), kernelHalfPixelSize(info.kernelHalfPixelSize)
		{
			const uint32_t totalSize = uint32_t(1) << (elementsPerInvocationLog2 + workgroupSizeLog2);
			totalSizeReciprocal = 1.f / float32_t(totalSize);
		}

		std::string scalar_t;
		uint16_t elementsPerInvocationLog2;
		uint16_t workgroupSizeLog2;
		uint16_t numWorkgroupsLog2;
		uint32_t numWorkgroups;
		uint16_t previousElementsPerInvocationLog2;
		uint16_t previousWorkgroupSizeLog2;
		uint16_t previousWorkgroupSize;
		float32_t2 kernelHalfPixelSize;
		float32_t totalSizeReciprocal;
	};

	inline core::smart_refctd_ptr<video::IGPUShader> createShader(const char* includeMainName, const SShaderConstevalParameters& shaderConstants)
	{
		// The annoying "const static member field must be initialized outside of struct" bug strikes again
		std::ostringstream kernelHalfPixelSizeStream;
		kernelHalfPixelSizeStream << "{" << shaderConstants.kernelHalfPixelSize.x << "," << shaderConstants.kernelHalfPixelSize.y << "}";
		std::string kernelHalfPixelSizeString = kernelHalfPixelSizeStream.str();

		const auto prelude = [&]()->std::string
			{
				std::ostringstream tmp;
				tmp << R"===(
				#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
				struct ShaderConstevalParameters
				{
					using scalar_t = )===" << shaderConstants.scalar_t << R"===(;

					NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocationLog2 = )===" << shaderConstants.elementsPerInvocationLog2 << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = )===" << shaderConstants.workgroupSizeLog2 << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE uint16_t NumWorkgroupsLog2 = )===" << shaderConstants.numWorkgroupsLog2 << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE uint32_t NumWorkgroups = )===" << shaderConstants.numWorkgroups << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE uint16_t PreviousElementsPerInvocationLog2 = )===" << shaderConstants.previousElementsPerInvocationLog2 << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE uint16_t PreviousWorkgroupSizeLog2 = )===" << shaderConstants.previousWorkgroupSizeLog2 << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE uint16_t PreviousWorkgroupSize = )===" << shaderConstants.previousWorkgroupSize << R"===(;
					NBL_CONSTEXPR_STATIC_INLINE float32_t2 KernelHalfPixelSize;
					NBL_CONSTEXPR_STATIC_INLINE float32_t TotalSizeReciprocal = )===" << shaderConstants.totalSizeReciprocal << R"===(;
				};
				NBL_CONSTEXPR_STATIC_INLINE float32_t2 ShaderConstevalParameters::KernelHalfPixelSize = )===" << kernelHalfPixelSizeString << R"===(;
				)===";
				return tmp.str();
			}();



		auto CPUShader = core::make_smart_refctd_ptr<ICPUShader>((prelude+"\n#include \"" + includeMainName + "\"\n").c_str(),
																IShader::E_SHADER_STAGE::ESS_COMPUTE, 
																IShader::E_CONTENT_TYPE::ECT_HLSL, 
																includeMainName);
		assert(CPUShader);

		#ifndef _NBL_DEBUG
		ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
		auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
		return m_device->createShader({ CPUShader.get(), opt.get(), m_readCache.get(), m_writeCache.get()});
		#else 
		return m_device->createShader({ CPUShader.get(), nullptr, m_readCache.get(), m_writeCache.get() });
		#endif
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
		m_timeline = m_device->createSemaphore(m_realFrameIx);
		// We can't use the same sepahore for uploads so we signal a different semaphore
		smart_refctd_ptr<ISemaphore> scratchSemaphore = m_device->createSemaphore(0);

		// Get graphics queue - these queues can do compute + blit 
		// In the real world you might do queue ownership transfers and have compute-dedicated queues - but here we KISS
		m_queue = getGraphicsQueue();
		uint32_t queueFamilyIndex = m_queue->getFamilyIndex();

		// Create command buffers for managing frames in flight + 1 extra
		smart_refctd_ptr<IGPUCommandBuffer> assConvCmdBuf;
		{
			std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight + 1> commandBuffers;

			smart_refctd_ptr<video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queueFamilyIndex, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, commandBuffers))
				return logFail("Failed to create Command Buffers!\n");

			for (auto i = 0u; i < MaxFramesInFlight; i++)
				m_cmdBufs[i] = std::move(commandBuffers[i]);

			assConvCmdBuf = std::move(commandBuffers[MaxFramesInFlight]);
		}

		// Use asset converter to upload images to GPU, while at the same time creating our universal descriptor set and pipeline layout
		smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
		{
			// Load source and kernel images
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto srcImageBundle = m_assetMgr->getAsset("../../media/colorexr.exr", lp);
			auto kerImageBundle = m_assetMgr->getAsset("../../media/kernels/physical_flare_256.exr", lp);
			const auto srcImages = srcImageBundle.getContents();
			const auto kerImages = kerImageBundle.getContents();
			if (srcImages.empty() or kerImages.empty())
				return logFail("Could not load image or kernel!");
			auto srcImageCPU = IAsset::castDown<ICPUImage>(srcImages[0]);
			auto kerImageCPU = IAsset::castDown<ICPUImage>(kerImages[0]);
			{
				auto kerImageExtent = kerImageCPU->getCreationParameters().extent;
				if (kerImageExtent.width != kerImageExtent.height || (kerImageExtent.width & (kerImageExtent.width - 1)))
					return logFail("Kernel Image must be square, with side length a power of two!");
			}

			const auto srcImageFormat = srcImageCPU->getCreationParameters().format;
			const auto kerImageFormat = kerImageCPU->getCreationParameters().format;

			// Create views for these images
			ICPUImageView::SCreationParams viewParams[2] =
			{
				{
					.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
					.image = std::move(srcImageCPU),
					.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
					.format = srcImageFormat,
				},
				{
					.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
					.image = std::move(kerImageCPU),
					.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
					.format = kerImageFormat,
			}
			};
			const auto srcImageViewCPU = ICPUImageView::create(std::move(viewParams[0]));
			const auto kerImageViewCPU = ICPUImageView::create(std::move(viewParams[1]));

			// Create a CPU Descriptor Set
			ICPUDescriptorSetLayout::SBinding bnd[4] =
			{
				// Kernel FFT and Image FFT read from a single Image
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					0u,
					IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					1u,
					nullptr
				},
				// Sampler: First Axis FFT (image) and convolution use a mirror-sampler
				// Could be static for each pipeline but since it's shared it implies no descriptor changes between pipelines
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					1u,
					IDescriptor::E_TYPE::ET_SAMPLER,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					1u,
					nullptr
				},
				// Storage Image: Normalization binds a texture array, First Axis IFFT binds a single image
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					2u,
					IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					1,
					nullptr
				},
				// Convolution binds a texture array. Trying to have this in same binding slot as image would be cool but we lose the ability to write the descriptor set only once
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					3u,
					IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					1,
					nullptr
				}
			};
			auto descriptorSetLayoutCPU = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bnd);
			// Create a CPU pipeline layout so we also create that one here
			const asset::SPushConstantRange pcRange[1] = { {IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(PushConstantData)} };
			auto pipelineLayoutCPU = make_smart_refctd_ptr <ICPUPipelineLayout>(pcRange, std::move(descriptorSetLayoutCPU), nullptr, nullptr, nullptr);

			// Create a Descriptor Set and fill it out
			// Reassigning because it's been moved out of
			descriptorSetLayoutCPU = smart_refctd_ptr<ICPUDescriptorSetLayout>(pipelineLayoutCPU->getDescriptorSetLayout(0));
			auto descriptorSetCPU = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(descriptorSetLayoutCPU));

			// Create a sampler
			ICPUSampler::SParams samplerCreationParams =
			{
				ISampler::ETC_MIRROR,
				ISampler::ETC_MIRROR,
				ISampler::ETC_MIRROR,
				ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				ISampler::ETF_LINEAR,
				ISampler::ETF_LINEAR,
				ISampler::ESMM_LINEAR,
				3u,
				0u,
				ISampler::ECO_ALWAYS
			};

			// Set descriptor set values for automatic upload
			
			auto& firstSampledImageDescriptorInfo = descriptorSetCPU->getDescriptorInfos(ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t(0u), IDescriptor::E_TYPE::ET_SAMPLED_IMAGE).front();
			auto& secondSampledImageDescriptorInfo = descriptorSetCPU->getDescriptorInfos(ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t(3u), IDescriptor::E_TYPE::ET_SAMPLED_IMAGE).front();
			
			firstSampledImageDescriptorInfo.desc = kerImageViewCPU;
			firstSampledImageDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			
			secondSampledImageDescriptorInfo.desc = srcImageViewCPU;
			secondSampledImageDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			

			auto samplerCPU = make_smart_refctd_ptr<ICPUSampler>(samplerCreationParams);

			auto& samplerDescriptorInfo = descriptorSetCPU->getDescriptorInfos(ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t(1u), IDescriptor::E_TYPE::ET_SAMPLER).front();
			samplerDescriptorInfo.desc = samplerCPU;



			// Using asset converter
			smart_refctd_ptr<video::CAssetConverter> converter = video::CAssetConverter::create({ .device = m_device.get(),.optimizer = {} });
			// We don't want to generate mip-maps for these images (YET), to ensure that we must override the default callbacks.
			struct SInputs final : CAssetConverter::SInputs
			{
				inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return image->getCreationParameters().mipLevels;
				}
				inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return 0b0u;
				}
			} inputs = {};
			inputs.logger = m_logger.get();
			asset::ICPUImageView* CPUImageViews[2] = { kerImageViewCPU.get(), srcImageViewCPU.get() };

			std::get<CAssetConverter::SInputs::asset_span_t<ICPUPipelineLayout>>(inputs.assets) = { &pipelineLayoutCPU.get(), 1};
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = { CPUImageViews, 2 };
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = { &descriptorSetCPU.get(), 1 };

			auto reservation = converter->reserve(inputs);

			// Retrieve GPU uploads
			const auto pipelineLayoutGPU = reservation.getGPUObjects<ICPUPipelineLayout>();
			pipelineLayout = pipelineLayoutGPU.front().value;

			const auto imagesGPU = reservation.getGPUObjects<ICPUImageView>();
			m_kerImageView = imagesGPU[0].value;
			m_srcImageView = imagesGPU[1].value;

			const auto descriptorSetGPU = reservation.getGPUObjects<ICPUDescriptorSet>();
			m_descriptorSet = descriptorSetGPU.front().value;

			// Give them debug names
			m_srcImageView->setObjectDebugName("Source image view");
			m_srcImageView->getCreationParameters().image->setObjectDebugName("Source Image");
			m_kerImageView->setObjectDebugName("Bloom kernel image view");
			m_kerImageView->getCreationParameters().image->setObjectDebugName("Bloom kernel Image");

			// The down-cast should not fail!
			assert(m_srcImageView);
			assert(m_kerImageView);

			// Going to need an IUtils to perform uploads/downloads
			m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger));

			// Now convert uploads
			// Get graphics queue for image transfer
			// For image uploads
			SIntendedSubmitInfo intendedSubmit;

			intendedSubmit.queue = m_queue;
			// Set up submit for image transfers
			// wait for nothing before upload
			intendedSubmit.waitSemaphores = {};
			intendedSubmit.prevCommandBuffers = {};
			// fill later
			intendedSubmit.scratchCommandBuffers = {};
			intendedSubmit.scratchSemaphore = {
				.semaphore = scratchSemaphore.get(),
				.value = 0,
				// because of layout transitions
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
			};

			// Needs to be open for utilities
			assConvCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			IQueue::SSubmitInfo::SCommandBufferInfo assConvCmdBufInfo = { assConvCmdBuf.get() };
			intendedSubmit.scratchCommandBuffers = { &assConvCmdBufInfo,1 };

			CAssetConverter::SConvertParams params = {};
			params.transfer = &intendedSubmit;
			params.utilities = m_utils.get();
			auto result = reservation.convert(params);
			// block immediately
			if (result.copy() != IQueue::RESULT::SUCCESS)
				return false;
		}

		// Create a swapchain
		ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface(),.sharedParams = {.presentMode = ISurface::EPM_IMMEDIATE}};
		if (!swapchainParams.deduceFormat(m_physicalDevice))
			return logFail("Could not choose a Surface Format for the Swapchain!");

		// Initialize surface
		auto graphicsQueue = getGraphicsQueue();
		if (!m_surface || !m_surface->init(graphicsQueue, std::make_unique<ISimpleManagedSurface::ISwapchainResources>(), swapchainParams.sharedParams))
			return logFail("Could not create Window & Surface or initialize the Surface!");

		// Set window size to match input image
		auto srcImgExtent = m_srcImageView->getCreationParameters().image->getCreationParameters().extent;
		m_winMgr->setWindowSize(m_window.get(), srcImgExtent.width, srcImgExtent.height);
		m_surface->recreateSwapchain();

		m_winMgr->show(m_window.get());

		// Create Out Image
		{
			auto dstImgViewInfo = m_srcImageView->getCreationParameters();

			IGPUImage::SCreationParams dstImgInfo(dstImgViewInfo.image->getCreationParameters());
			// Specify we want this to be a storage image, + transfer for readback (blit when we have swapchain up)
			dstImgInfo.usage = IImage::EUF_STORAGE_BIT | IImage::EUF_TRANSFER_SRC_BIT;
			dstImgInfo.format = m_useHalfFloats ? EF_R16G16B16A16_SFLOAT : EF_R32G32B32A32_SFLOAT;
			auto outImg = m_device->createImage(std::move(dstImgInfo));

			outImg->setObjectDebugName("Convolved Image");

			auto memReqs = outImg->getMemoryReqs();
			memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpuMem = m_device->allocate(memReqs, outImg.get());

			dstImgViewInfo.image = outImg;
			dstImgViewInfo.subUsages = IImage::EUF_STORAGE_BIT | IImage::EUF_TRANSFER_SRC_BIT;
			dstImgViewInfo.format = m_useHalfFloats ? EF_R16G16B16A16_SFLOAT : EF_R32G32B32A32_SFLOAT;
			m_outImgView = m_device->createImageView(IGPUImageView::SCreationParams(dstImgViewInfo));

			m_outImgView->setObjectDebugName("Convolved Image View");
		}

		// agree on formats
		const E_FORMAT srcFormat = m_srcImageView->getCreationParameters().format;
		
		//! OVERRIDE (we dont need alpha)
		uint32_t srcNumChannels = Channels;
		uint32_t kerNumChannels = Channels;

		// Kernel pixel to image pixel conversion ratio
		const float bloomRelativeScale = 0.25f;
		const auto kerDim = m_kerImageView->getCreationParameters().image->getCreationParameters().extent;
		const auto srcDim = m_srcImageView->getCreationParameters().image->getCreationParameters().extent;
		auto bloomScale = core::min(float(srcDim.width) / float(kerDim.width), float(srcDim.height) / float(kerDim.height)) * bloomRelativeScale;
		assert(bloomScale <= 1.f);

		m_marginSrcDim = srcDim;
		
		// Add padding to m_marginSrcDim
		for (auto i = 0u; i < 3u; i++)
		{
			const auto coord = (&kerDim.width)[i];
			(&m_marginSrcDim.width)[i] += core::max(coord, 1u) - 1u;
		}
		
		
		// Create intermediate buffers
		{
			IGPUBuffer::SCreationParams deviceLocalBufferParams = {};

			deviceLocalBufferParams.queueFamilyIndexCount = 1;
			deviceLocalBufferParams.queueFamilyIndices = &queueFamilyIndex;
			// Axis on which we perform first FFT is the only one that needs to be padded in memory - in this case it's the y-axis
			hlsl::vector <uint16_t, 1> firstAxis(uint16_t(1));
			// We only need enough memory to hold (half, since it's real) of the original-sized image with padding for the kernel (and the padding up to PoT) only on the first axis.
			// That's because (in this case, since it's a 2D convolution) the "middle" step (that which does the last FFT -> convolves -> first IFFT) doesn't store
			// anything and the "padding" can be seen as virtual.
			hlsl::vector <uint32_t, 3> paddedSrcDimensions(srcDim.width, m_marginSrcDim.height, srcDim.depth);
			deviceLocalBufferParams.size = fft::getOutputBufferSize<3, 1>(paddedSrcDimensions, 3, firstAxis, true, m_useHalfFloats);
			deviceLocalBufferParams.usage = asset::IBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;

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
		// Create cache for shader compilation

		m_readCache = nullptr;
		m_writeCache = core::make_smart_refctd_ptr<IShaderCompiler::CCache>();
		// Keep caches separate for debug runs
		#ifndef _NBL_DEBUG
		auto shaderCachePath = localOutputCWD / "cache.bin";
		#else
		auto shaderCachePath = localOutputCWD / "cache_d.bin";
        #endif

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
								m_readCache = IShaderCompiler::CCache::deserialize(contents);
						}
					}
				}
				else
					m_logger->log("Failed Opening Shader Cache File.", ILogger::ELL_ERROR);
			}

		}

		// Kernel second axis FFT has no descriptor sets so we just create another pipeline with the same layout
		// TODO: To avoid duplicated layouts we could make samplers dynamic in the first axis FFT. Also if we don't hardcode (by #defining) some stuff in the first axis FFT
		//		 (once FFT ext is back) we can also avoid having duplicated pipelines (like the old Bloom example, which had a single pipeline for forward FFT along an axis)
		//       and setting stuff via shader push constants (such as which axis to perform FFT on and the size of output image).

		// -------------------------------------- KERNEL FFT PRECOMP ----------------------------------------------------------------
		{
			// create kernel spectrums
			auto createKernelSpectrum = [&]() -> auto
				{
					video::IGPUImage::SCreationParams imageParams;
					imageParams.flags = static_cast<video::IGPUImage::E_CREATE_FLAGS>(0u);
					imageParams.type = asset::IImage::ET_2D;
					imageParams.format = m_useHalfFloats ? EF_R16G16_SFLOAT : EF_R32G32_SFLOAT;
					imageParams.extent = { kerDim.width,kerDim.height / 2 + 1, 1u };
					imageParams.mipLevels = 1u;
					imageParams.arrayLayers = Channels;
					imageParams.samples = asset::IImage::ESCF_1_BIT;
					imageParams.usage = IImage::EUF_STORAGE_BIT | IImage::EUF_SAMPLED_BIT;

					auto kernelImg = m_device->createImage(std::move(imageParams));

					auto memReqs = kernelImg->getMemoryReqs();
					memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					auto gpuMem = m_device->allocate(memReqs, kernelImg.get());

					video::IGPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = kernelImg;
					viewParams.viewType = video::IGPUImageView::ET_2D_ARRAY;
					viewParams.format = m_useHalfFloats ? EF_R16G16_SFLOAT : EF_R32G32_SFLOAT;
					viewParams.subresourceRange.layerCount = Channels;
					return m_device->createImageView(std::move(viewParams));
				};

			m_kernelNormalizedSpectrums = createKernelSpectrum();

			// Give them names
			m_kernelNormalizedSpectrums->setObjectDebugName("Kernel spectrum array view");
			m_kernelNormalizedSpectrums->getCreationParameters().image->setObjectDebugName("Kernel spectrum array");

			// Provide sampler so it's bound already
			updateDescriptorSet(m_kerImageView, m_kernelNormalizedSpectrums);

			// Invoke a workgroup per two vertical scanlines. Kernel is square and runs first in the y-direction.
			// That means we have to create a shader that does an FFT of size `kerDim.height = kerDim.width` (length of each column, already padded to PoT), 
			// and call `kerDim.width / 2` workgroups to run it. We also have to keep in mind `kerDim.y = WorkgroupSize * ElementsPerInvocation`. 
			// We prefer to go with 2 elements per invocation and max out WorkgroupSize when possible.
			// This is because we use PreloadedAccessors which reduce global memory accesses at the cost of decreasing occupancy with increasing ElementsPerInvocation

			// Compute required WorkgroupSize and ElementsPerThread for FFT
			// Remember we assume kernel is square!

			auto [elementsPerInvocationLog2, workgroupSizeLog2] = workgroup::fft::optimalFFTParameters(m_device->getPhysicalDevice()->getLimits().maxWorkgroupSize[0], kerDim.width);
			// Normalization shader needs this info
			uint16_t secondAxisFFTHalfLengthLog2 = elementsPerInvocationLog2 + workgroupSizeLog2 - 1;
			// Create shaders
			smart_refctd_ptr<IGPUShader> shaders[3];
			uint16_t2 kernelDimensions = { kerDim.width, kerDim.height };
			SShaderConstevalParameters::SShaderConstevalParametersCreateInfo shaderConstevalInfo = { .useHalfFloats = m_useHalfFloats, .elementsPerInvocationLog2 = elementsPerInvocationLog2, .workgroupSizeLog2 = workgroupSizeLog2, .numWorkgroupsLog2 = secondAxisFFTHalfLengthLog2, .previousWorkgroupSizeLog2 = workgroupSizeLog2 };
			SShaderConstevalParameters shaderConstevalParameters(shaderConstevalInfo);
			shaders[0] = createShader("app_resources/kernel_fft_first_axis.hlsl", shaderConstevalParameters);
			shaders[1] = createShader("app_resources/kernel_fft_second_axis.hlsl", shaderConstevalParameters);
			shaders[2] = createShader("app_resources/kernel_spectrum_normalize.hlsl", shaderConstevalParameters);

			// Create compute pipelines - First axis FFT -> Second axis FFT -> Normalization
			IGPUComputePipeline::SCreationParams params[3] = {};
			for (auto i = 0u; i < 3; i++)
			{
				params[i].layout = pipelineLayout.get();
				params[i].shader.entryPoint = "main";
				params[i].shader.shader = shaders[i].get();
				// Normalization doesn't require full subgroups
				params[i].shader.requireFullSubgroups = bool(2-i);
			}
			
			smart_refctd_ptr<IGPUComputePipeline> pipelines[3];
			if(!m_device->createComputePipelines(nullptr, { params, 3 }, pipelines))
				return logFail("Failed to create Compute Pipelines!\n");

			// Push Constants - only need to specify BDAs here
			PushConstantData pushConstants;
			pushConstants.colMajorBufferAddress = m_colMajorBufferAddress;
			pushConstants.rowMajorBufferAddress = m_rowMajorBufferAddress;

			// Create a command buffer for this submit only
			smart_refctd_ptr<IGPUCommandBuffer> kernelPrecompCmdBuf;
			{
				smart_refctd_ptr<video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queueFamilyIndex, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, {&kernelPrecompCmdBuf, 1u}))
					return logFail("Failed to create Command Buffers!\n");
			}

			kernelPrecompCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			// First Axis FFT
			kernelPrecompCmdBuf->bindComputePipeline(pipelines[0].get());
			kernelPrecompCmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, pipelines[0]->getLayout(), 0, 1, &m_descriptorSet.get());
			kernelPrecompCmdBuf->pushConstants(pipelines[0]->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pushConstants), &pushConstants);
			// One workgroup per 2 columns
			kernelPrecompCmdBuf->dispatch(kerDim.width / 2, 1, 1);

			// Pipeline barrier: wait for first axis FFT before second axis can begin
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo = {};
			decltype(pipelineBarrierInfo)::buffer_barrier_t bufBarrier = {};
			pipelineBarrierInfo.bufBarriers = { &bufBarrier, 1u };

			// First axis FFT writes to colMajorBuffer
			bufBarrier.range.buffer = m_colMajorBuffer;

			// Wait for first compute write (first axis FFT) before next compute read (second axis FFT)
			bufBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			bufBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			bufBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			bufBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;

			// Also set kernel channel image array to GENERAL for writing
			decltype(pipelineBarrierInfo)::image_barrier_t imgBarrier = {};
			pipelineBarrierInfo.imgBarriers = { &imgBarrier, 1 };

			imgBarrier.image = m_kernelNormalizedSpectrums->getCreationParameters().image.get();
			imgBarrier.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
			imgBarrier.subresourceRange.levelCount = 1u;
			imgBarrier.subresourceRange.layerCount = Channels;
			imgBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
			imgBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
			imgBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			imgBarrier.oldLayout = IImage::LAYOUT::UNDEFINED;
			imgBarrier.newLayout = IImage::LAYOUT::GENERAL;

			kernelPrecompCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);

			// Now do second axis FFT
			kernelPrecompCmdBuf->bindComputePipeline(pipelines[1].get());
			// Same number of workgroups - this time because we only saved half the rows
			kernelPrecompCmdBuf->dispatch(kerDim.width / 2, 1, 1);

			// Wait on second axis FFT to write the kernel image before running normalization step
			// Normalization needs to access the power value stored in the rowmajorbuffer
			bufBarrier.range.buffer = m_rowMajorBuffer;

			// No layout transition now
			imgBarrier.oldLayout = IImage::LAYOUT::UNDEFINED;
			imgBarrier.newLayout = IImage::LAYOUT::UNDEFINED;

			// Wait on second axis FFT write ...
			imgBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			// ... before normalization read
			imgBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;

			kernelPrecompCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);

			//Finally, normalize kernel Image - same number of workgroups
			kernelPrecompCmdBuf->bindComputePipeline(pipelines[2].get());
			// Hardcoded 8x8 workgroup seems to be optimal for tex access
			const auto& kernelSpectraExtent = m_kernelNormalizedSpectrums->getCreationParameters().image->getCreationParameters().extent;
			// Assumed PoT. +1 in Y dispatch to account for Nyquist row
			kernelPrecompCmdBuf->dispatch(kernelSpectraExtent.width / 8, kernelSpectraExtent.height / 8 + 1, 1);

			// Pipeline barrier: transition kernel spectrum images into read only, and outImage into general
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo imagePipelineBarrierInfo = {};
			decltype(imagePipelineBarrierInfo)::image_barrier_t imgBarriers[2] = {};
			imagePipelineBarrierInfo.imgBarriers = { imgBarriers, 2 };

			// outImage just needs a layout transition before it can be written to
			// Masks left empty because we will wait on device idle at the end of app initialization anyway
			imgBarriers[0].image = m_outImgView->getCreationParameters().image.get();
			imgBarriers[0].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
			imgBarriers[0].barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
			imgBarriers[0].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::NONE;
			imgBarriers[0].barrier.dep.dstAccessMask = ACCESS_FLAGS::NONE;
			imgBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
			imgBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
			imgBarriers[0].subresourceRange = { IGPUImage::EAF_COLOR_BIT, 0u, 1u, 0u, 1 };

			// Transition kernel spectrums so that convolution shader can access them later
			imgBarriers[1].image = m_kernelNormalizedSpectrums->getCreationParameters().image.get();
			imgBarriers[1].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarriers[1].barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			imgBarriers[1].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::NONE;
			imgBarriers[1].barrier.dep.dstAccessMask = ACCESS_FLAGS::NONE;
			imgBarriers[1].oldLayout = IImage::LAYOUT::GENERAL;
			imgBarriers[1].newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			imgBarriers[1].subresourceRange = { IGPUImage::EAF_COLOR_BIT, 0u, 1u, 0u, Channels };

			kernelPrecompCmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), imagePipelineBarrierInfo);

			kernelPrecompCmdBuf->end();

			// Submit to queue and add sync point
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
				{
					.cmdbuf = kernelPrecompCmdBuf.get()
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
				{
					.semaphore = m_timeline.get(),
					.value = 1,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				};

				// Could check whether queue used for upload is different than the compute one, but oh well
				IQueue::SSubmitInfo::SSemaphoreInfo transferSemaphore = {
					.semaphore = scratchSemaphore.get(),
					.value = 1,
					// because of layout transitions
				.	stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				};
				const IQueue::SSubmitInfo submitInfo = {
					.waitSemaphores = {&transferSemaphore, 1},
					.commandBuffers = {&cmdbufInfo,1},
					.signalSemaphores = {&signalInfo,1}
				};

				m_api->startCapture();
				m_queue->submit({ &submitInfo,1 });
				m_api->endCapture();
			}
		}
		// ----------------------------------------- KERNEL PRECOMP END -------------------------------------------------

		// Now create the pipelines for the image FFT

		// Second axis FFT launches an amount of workgroups equal to half of the length of the first axis FFT. Second pass FFT needs the log2 of this number baked in as a constant.
		uint16_t firstAxisFFTHalfLengthLog2;
		uint16_t firstAxisFFTElementsPerInvocationLog2;
		uint16_t firstAxisFFTWorkgroupSizeLog2;
		smart_refctd_ptr<IGPUShader> shaders[3];
		{
			auto [elementsPerInvocationLog2, workgroupSizeLog2] = workgroup::fft::optimalFFTParameters(m_device->getPhysicalDevice()->getLimits().maxWorkgroupSize[0], m_marginSrcDim.height);
			SShaderConstevalParameters::SShaderConstevalParametersCreateInfo shaderConstevalInfo = { .useHalfFloats = m_useHalfFloats, .elementsPerInvocationLog2 = elementsPerInvocationLog2, .workgroupSizeLog2 = workgroupSizeLog2 };
			SShaderConstevalParameters shaderConstevalParameters(shaderConstevalInfo);
			shaders[0] = createShader("app_resources/image_fft_first_axis.hlsl", shaderConstevalParameters);
			// IFFT along first axis has same dimensions as FFT
			shaders[2] = createShader("app_resources/image_ifft_first_axis.hlsl", shaderConstevalParameters);
			firstAxisFFTHalfLengthLog2 = elementsPerInvocationLog2 + workgroupSizeLog2 - 1;
			firstAxisFFTElementsPerInvocationLog2 = elementsPerInvocationLog2;
			firstAxisFFTWorkgroupSizeLog2 = workgroupSizeLog2;
			m_imageFirstAxisFFTWorkgroupSize = uint16_t(1) << workgroupSizeLog2;
		}

		// Second axis FFT might have different dimensions
		{
			auto [elementsPerInvocationLog2, workgroupSizeLog2] = workgroup::fft::optimalFFTParameters(m_device->getPhysicalDevice()->getLimits().maxWorkgroupSize[0], m_marginSrcDim.width);
			// Compute kernel half pixel size
			const auto& kernelSpectraExtent = m_kernelNormalizedSpectrums->getCreationParameters().image->getCreationParameters().extent;
			float32_t2 kernelHalfPixelSize{ 0.5f,0.5f };
			kernelHalfPixelSize.x /= kernelSpectraExtent.width;
			kernelHalfPixelSize.y /= kernelSpectraExtent.height;
			SShaderConstevalParameters::SShaderConstevalParametersCreateInfo shaderConstevalInfo =
			{
				.useHalfFloats = m_useHalfFloats,
				.elementsPerInvocationLog2 = elementsPerInvocationLog2, 
				.workgroupSizeLog2 = workgroupSizeLog2, 
				.numWorkgroupsLog2 = firstAxisFFTHalfLengthLog2, 
				.previousElementsPerInvocationLog2 = firstAxisFFTElementsPerInvocationLog2,
				.previousWorkgroupSizeLog2 = firstAxisFFTWorkgroupSizeLog2, 
				.kernelHalfPixelSize = kernelHalfPixelSize
			};
			SShaderConstevalParameters shaderConstevalParameters(shaderConstevalInfo);
			shaders[1] = createShader("app_resources/fft_convolve_ifft.hlsl", shaderConstevalParameters);
		}

		// Create compute pipelines - First axis FFT -> Second axis FFT -> Normalization
		IGPUComputePipeline::SCreationParams params[3] = {};
		for (auto i = 0u; i < 3; i++) {
			params[i].layout = pipelineLayout.get();
			params[i].shader.entryPoint = "main";
			params[i].shader.shader = shaders[i].get();
			params[i].shader.requireFullSubgroups = true;
		}

		smart_refctd_ptr<IGPUComputePipeline> pipelines[3];
		if (!m_device->createComputePipelines(nullptr, { params, 3 }, pipelines))
			return logFail("Failed to create Compute Pipelines!\n");

		m_firstAxisFFTPipeline = pipelines[0];
		m_lastAxisFFT_convolution_lastAxisIFFTPipeline = pipelines[1];
		m_firstAxisIFFTPipeline = pipelines[2];

		// Dump cache to disk since we won't be doing any more compilations - for now
		{
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
						auto serializedCache = m_writeCache->serialize();
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
		// Block and wait until kernel FFT is done before we drop the pipelines.
		// One could instead opt to create a latch that does nothing but capture the pipelines and gets called when semaphore is signalled.
		// IMPORTANT: This wait offsets our frames in flight math by 1, so it's important to remember it
		const ISemaphore::SWaitInfo waitInfo = { m_timeline.get(), 1 };

		m_device->blockForSemaphores({ &waitInfo, 1 });

		// Before leaving, update descriptor set with values needed by image transform
		// Write descriptor set for kernel FFT computation
		updateDescriptorSet(m_srcImageView, m_outImgView, m_kernelNormalizedSpectrums);

		return true;
	}

	bool keepRunning() override 
	{
		if (m_surface->irrecoverable())
			return false;

		return true;
	}

	// Right now it's one shot app, but I put this code here for easy refactoring when it refactors into it being a live app
	void workLoopBody() override
	{
		// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
		const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
		// We block for semaphores for 2 reasons here:
			// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
			// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
		if (m_realFrameIx >= framesInFlight)
		{
			const ISemaphore::SWaitInfo cbDonePending[] =
			{
				{
					.semaphore = m_timeline.get(),
					.value = m_realFrameIx + 2 - framesInFlight // There is a +2 here instead of +1 to account for the kernel precomp value increase
				}
			};
			if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
				return;
		}

		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		m_currentImageAcquire = m_surface->acquireNextImage();
		if (!m_currentImageAcquire)
			return;

		// Acquire and reset command buffer
		auto* const cmdBuf = m_cmdBufs[resourceIx].get();
		cmdBuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		// Compute to blit barrier: Ensure compute pass is done before blitting from image
		const IGPUImage::SSubresourceRange whole2DColorImage =
		{
			.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		};
		
		// -------------------------------------- DRAW BEGIN -------------------------------------------

		// Prepare for first axis FFT
		// Push Constants - only need to specify BDAs here
		const auto& imageExtent = m_srcImageView->getCreationParameters().image->getCreationParameters().extent;
		const int32_t paddingAlongColumns = int32_t(core::roundUpToPoT(m_marginSrcDim.height) - imageExtent.height) / 2;
		const int32_t paddingAlongRows = int32_t(core::roundUpToPoT(m_marginSrcDim.width) - imageExtent.width) / 2;
		const int32_t halfPaddingAlongRows = paddingAlongRows / 2;

		PushConstantData pushConstants;
		pushConstants.colMajorBufferAddress = m_colMajorBufferAddress;
		pushConstants.rowMajorBufferAddress = m_rowMajorBufferAddress;
		pushConstants.imageRowLength = int32_t(imageExtent.width);
		pushConstants.imageHalfRowLength = int32_t(imageExtent.width) / 2;
		pushConstants.imageColumnLength = int32_t(imageExtent.height);
		pushConstants.padding = paddingAlongColumns;
		pushConstants.halfPadding = halfPaddingAlongRows;

		float32_t2 imageHalfPixelSize = { 0.5f, 0.5f };
		imageHalfPixelSize.x /= imageExtent.width;
		imageHalfPixelSize.y /= imageExtent.height;
		pushConstants.imageHalfPixelSize = imageHalfPixelSize;
		pushConstants.imagePixelSize = 2.f * imageHalfPixelSize;
		pushConstants.imageTwoPixelSize_x = 4.f * imageHalfPixelSize.x;
		pushConstants.imageWorkgroupSizePixelSize_y = m_imageFirstAxisFFTWorkgroupSize * pushConstants.imagePixelSize.y;

		// Interpolate between dirac delta and kernel based on current time
		auto epochNanoseconds = clock_t::now().time_since_epoch().count();
		pushConstants.interpolatingFactor = cos(epochNanoseconds / 1000000000.f) * cos(epochNanoseconds / 1000000000.f);

		cmdBuf->bindComputePipeline(m_firstAxisFFTPipeline.get());
		cmdBuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_firstAxisFFTPipeline->getLayout(), 0, 1, &m_descriptorSet.get());
		cmdBuf->pushConstants(m_firstAxisFFTPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pushConstants), &pushConstants);
		// One workgroup per 2 columns
		auto srcDim = m_srcImageView->getCreationParameters().image->getCreationParameters().extent;
		cmdBuf->dispatch(srcDim.width / 2, 1, 1);

		// Pipeline Barrier: Wait for colMajorBuffer to be written to before reading it from next shader
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo bufferPipelineBarrierInfo = {};
		decltype(bufferPipelineBarrierInfo)::buffer_barrier_t bufBarrier = {};
		bufferPipelineBarrierInfo.bufBarriers = { &bufBarrier, 1u };

		// First axis FFT writes to colMajorBuffer
		bufBarrier.range.buffer = m_colMajorBuffer;

		// Wait for first compute write (first axis FFT) before next compute read (second axis FFT)
		bufBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		bufBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
		bufBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		bufBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;

		cmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), bufferPipelineBarrierInfo);
		// Now comes Second axis FFT + Conv + IFFT
		cmdBuf->bindComputePipeline(m_lastAxisFFT_convolution_lastAxisIFFTPipeline.get());
		// Update padding for run along rows
		cmdBuf->pushConstants(m_firstAxisFFTPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, offsetof(PushConstantData, padding), sizeof(paddingAlongRows), &paddingAlongRows);
		// One workgroup per row in the lower half of the DFT
		cmdBuf->dispatch(core::roundUpToPoT(m_marginSrcDim.height) / 2, 1, 1);

		// Recycle pipeline barrier, only have to change which buffer we need to wait to be written to
		bufBarrier.range.buffer = m_rowMajorBuffer;
		cmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), bufferPipelineBarrierInfo);

		// Finally run the IFFT on the first axis
		cmdBuf->bindComputePipeline(m_firstAxisIFFTPipeline.get());
		// Update padding for run along columns
		cmdBuf->pushConstants(m_firstAxisFFTPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, offsetof(PushConstantData, padding), sizeof(paddingAlongColumns), &paddingAlongColumns);
		// One workgroup per 2 columns
		cmdBuf->dispatch(srcDim.width / 2, 1, 1);

		// -------------------------------------- DRAW END ----------------------------------------

		// BLIT
		{
			auto swapImg = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);

			using image_memory_barrier_t = IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>;
			image_memory_barrier_t imgComputeToBlitBarrier = {
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						.srcAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT | ACCESS_FLAGS::STORAGE_READ_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
			},
			.image = m_outImgView->getCreationParameters().image.get(),
			.subresourceRange = whole2DColorImage,
			.oldLayout = IImage::LAYOUT::UNDEFINED,
			.newLayout = IImage::LAYOUT::UNDEFINED
			};

			// special case, the swapchain is a NONE stage with NONE accesses
			image_memory_barrier_t swapchainAcquireToBlitBarrier = {
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
							.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
						}
				// no ownership transfer and don't care about contents
			},
			.image = swapImg,
			.subresourceRange = whole2DColorImage,
			.oldLayout = IImage::LAYOUT::UNDEFINED, // don't care about old contents
			.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL
			};

			image_memory_barrier_t imgBarriers[] = {imgComputeToBlitBarrier, swapchainAcquireToBlitBarrier};
			cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {},.bufBarriers = {},.imgBarriers = imgBarriers });

			auto outImg = m_outImgView->getCreationParameters().image.get();
			auto outImgExtent = outImg->getCreationParameters().extent;

			const IGPUCommandBuffer::SImageBlit regions[] = { {
				.srcMinCoord = {0,0,0},
				.srcMaxCoord = {outImgExtent.width,outImgExtent.height,1},
				.dstMinCoord = {0,0,0},
				.dstMaxCoord = {outImgExtent.width,outImgExtent.height,1},
				.layerCount = 1,
				.srcBaseLayer = 0,
				.dstBaseLayer = 0,
				.srcMipLevel = 0,
				.dstMipLevel = 0,
				.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT
			} };
			cmdBuf->blitImage(outImg, IGPUImage::LAYOUT::GENERAL, swapImg, IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL, regions, IGPUSampler::ETF_NEAREST);

			auto& swapImageBarrier = imgBarriers[1];
			swapImageBarrier.barrier.dep = swapImageBarrier.barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::NONE, ACCESS_FLAGS::NONE);
			swapImageBarrier.oldLayout = imgBarriers[1].newLayout;
			swapImageBarrier.newLayout = IGPUImage::LAYOUT::PRESENT_SRC;
			cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {},.bufBarriers = {},.imgBarriers = {&swapImageBarrier,1} });
		}

		cmdBuf->end();

		// Submit to queue and add sync point
		{
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
			{
				{
					.semaphore = m_timeline.get(),
					.value = ++m_realFrameIx + 1, // The +1 here is to account for the first submit on this semaphore on kernel precomp
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS // because of the layout transition of the swapchain image
				}
			};
			{
				{
					const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
					{
						{.cmdbuf = cmdBuf }
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

					m_api->startCapture();
					if (m_queue->submit(infos) != IQueue::RESULT::SUCCESS)
						--m_realFrameIx;
					m_api->endCapture();
				}
			}

			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}
	}

	bool onAppTerminated() override
	{
		// Wait for all work to be done
		m_device->waitIdle();

		// Copied from 64_FFT which I think copies from ex 07
		// Create a buffer for download
		const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
		uint32_t alignment = core::max(deviceLimits.nonCoherentAtomSize, alignof(float));
		auto srcImageDims = m_srcImageView->getCreationParameters().image->getCreationParameters().extent;
		const uint32_t srcImageSize = srcImageDims.height * srcImageDims.width * srcImageDims.depth * (Channels + 1) * sizeof(float32_t);

		auto downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();

		auto outputOffset = downStreamingBuffer->invalid_value;
		std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
		const uint32_t AllocationCount = 1;
		downStreamingBuffer->multi_allocate(waitTill, AllocationCount, &outputOffset, &srcImageSize, &alignment);
		
		// Since all work is done grab any command buffer
		auto cmdBuf = m_cmdBufs[0];

		// Send download commands to GPU
		{
			cmdBuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			// Pipeline barrier: transition outImg to transfer source optimal
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo imagePipelineBarrierInfo = {};
			decltype(imagePipelineBarrierInfo)::image_barrier_t imgBarrier;
			imagePipelineBarrierInfo.imgBarriers = { &imgBarrier, 1 };

			imgBarrier.image = m_outImgView->getCreationParameters().image.get();
			imgBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			imgBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			imgBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			imgBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT;
			imgBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			imgBarrier.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			imgBarrier.subresourceRange = { IGPUImage::EAF_COLOR_BIT, 0u, 1u, 0u, 1 };

			cmdBuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), imagePipelineBarrierInfo);
			IImage::SBufferCopy copy;
			copy.imageExtent = m_outImgView->getCreationParameters().image->getCreationParameters().extent;
			copy.imageSubresource = { IImage::EAF_COLOR_BIT, 0u, 0u, 1u };
			cmdBuf->copyImageToBuffer(m_outImgView->getCreationParameters().image.get(), IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, downStreamingBuffer->getBuffer(), 1, &copy);
			cmdBuf->end();
		}
		// Submit
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
			{
				.cmdbuf = cmdBuf.get()
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
			{
				.semaphore = m_timeline.get(),
				.value = m_realFrameIx + 1,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT
			};

			const IQueue::SSubmitInfo submitInfo = {
				.commandBuffers = {&cmdbufInfo,1},
				.signalSemaphores = {&signalInfo,1}
			};

			m_queue->submit({ &submitInfo,1 });
		}

		// We let all latches know what semaphore and counter value has to be passed for the functors to execute
		const ISemaphore::SWaitInfo futureWait = { m_timeline.get(),m_realFrameIx + 1 };

		// Now a new and even more advanced usage of the latched events, we make our own refcounted object with a custom destructor and latch that like we did the commandbuffer.
		// Instead of making our own and duplicating logic, we'll use one from IUtilities meant for down-staging memory.
		// Its nice because it will also remember to invalidate our memory mapping if its not coherent.
		auto latchedConsumer = make_smart_refctd_ptr<IUtilities::CDownstreamingDataConsumer>(
			IDeviceMemoryAllocation::MemoryRange(outputOffset, srcImageSize),
			// Note the use of capture by-value [=] and not by-reference [&] because this lambda will be called asynchronously whenever the event signals
			[=](const size_t dstOffset, const void* bufSrc, const size_t size)->void
			{
				// image view
				core::smart_refctd_ptr<ICPUImageView> imageView;
				{
					// create image
					ICPUImage::SCreationParams imgParams;
					imgParams.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u); // no flags
					imgParams.type = ICPUImage::ET_2D;
					imgParams.format = m_outImgView->getCreationParameters().image->getCreationParameters().format;
					imgParams.extent = m_outImgView->getCreationParameters().image->getCreationParameters().extent;
					imgParams.mipLevels = 1u;
					imgParams.arrayLayers = 1u;
					imgParams.samples = ICPUImage::ESCF_1_BIT;

					auto image = ICPUImage::create(std::move(imgParams));
					{
						// set up regions
						auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy> >(1u);
						{
							auto& region = regions->front();
							region.bufferOffset = 0u;
							region.bufferRowLength = 0;
							region.bufferImageHeight = 0;
							region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
							region.imageSubresource.mipLevel = 0u;
							region.imageSubresource.baseArrayLayer = 0u;
							region.imageSubresource.layerCount = 1u;
							region.imageOffset = { 0u,0u,0u };
							region.imageExtent = imgParams.extent;
						}
						// the cpu is not touching the data yet because the custom CPUBuffer is adopting the memory (no copy)
						auto* data = reinterpret_cast<uint8_t*>(downStreamingBuffer->getBufferPointer()) + outputOffset;
						//.size = srcImageSize, .data = data, .memoryResource = core::getNullMemoryResource()
						ICPUBuffer::SCreationParams cpuBufferAliasCreationParams = { .data = data, .memoryResource = core::getNullMemoryResource()}; // Don't free on exit, we're not taking ownership
						cpuBufferAliasCreationParams.size = srcImageSize;
						auto cpuBufferAlias = ICPUBuffer::create(std::move(cpuBufferAliasCreationParams), core::adopt_memory);
						image->setBufferAndRegions(std::move(cpuBufferAlias), regions);
					}

					// create image view
					ICPUImageView::SCreationParams imgViewParams;
					imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
					imgViewParams.format = image->getCreationParameters().format;
					imgViewParams.image = std::move(image);
					imgViewParams.viewType = ICPUImageView::ET_2D;
					imgViewParams.subresourceRange = { IImage::EAF_COLOR_BIT,0u,1u,0u,1u };
					imageView = ICPUImageView::create(std::move(imgViewParams));
				}

				// save as .EXR image
				{
					IAssetWriter::SAssetWriteParams wp(imageView.get());
					m_assetMgr->writeAsset((localOutputCWD / "convolved.exr").string(), wp);
				}
			},
			// Its also necessary to hold onto the commandbuffer, even though we take care to not reset the parent pool, because if it
			// hits its destructor, our automated reference counting will drop all references to objects used in the recorded commands.
			// It could also be latched in the upstreaming deallocate, because its the same fence.
			std::move(cmdBuf), downStreamingBuffer
		);
		// We put a function we want to execute 
		downStreamingBuffer->multi_deallocate(AllocationCount, &outputOffset, &srcImageSize, futureWait, &latchedConsumer.get());

		// Need to make sure that there are no events outstanding if we want all lambdas to eventually execute before `onAppTerminated`
		// (the destructors of the Command Pool Cache and Streaming buffers will still wait for all lambda events to drain)
		while (downStreamingBuffer->cull_frees()) {}
		return device_base_t::onAppTerminated();
	}
};


NBL_MAIN_FUNC(FFTBloomApp)