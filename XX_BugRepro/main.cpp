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

// Defaults that match this example's image
constexpr uint32_t WIN_W = 1280;
constexpr uint32_t WIN_H = 720;

class BugReproApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	// Utils
	smart_refctd_ptr<IUtilities> m_utils;

	// Resources
	smart_refctd_ptr<IGPUImageView> m_imageView;

	// For asset converter usage
	IQueue* m_queue;

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	BugReproApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		
		// Setup semaphore for asset converter
		smart_refctd_ptr<ISemaphore> scratchSemaphore = m_device->createSemaphore(0);

		// Get graphics queue - these queues can do compute + blit 
		// In the real world you might do queue ownership transfers and have compute-dedicated queues - but here we KISS
		m_queue = getComputeQueue();
		uint32_t queueFamilyIndex = m_queue->getFamilyIndex();

		// Create command buffer for asset converter
		smart_refctd_ptr<IGPUCommandBuffer> assConvCmdBuf;
		{
			smart_refctd_ptr<video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queueFamilyIndex, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, {&assConvCmdBuf, 1}))
				return logFail("Failed to create Command Buffers!\n");
		}

		// Use asset converter to upload image to GPU and create a descriptor set
		smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
		{
			// Load source and kernel images
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto imageBundle = m_assetMgr->getAsset("../../media/colorexr.exr", lp);
			const auto images = imageBundle.getContents();
			if (images.empty())
				return logFail("Could not load image or kernel!");
			auto imageCPU = IAsset::castDown<ICPUImage>(images[0]);
			const auto imageFormat = imageCPU->getCreationParameters().format;
			// Create views for the image
			ICPUImageView::SCreationParams viewParams[1] =
			{
				{
					.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
					.image = std::move(imageCPU),
					.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
					.format = imageFormat,
				}
			};
			const auto imageViewCPU = ICPUImageView::create(std::move(viewParams[0]));

			// Create a CPU Descriptor Set
			ICPUDescriptorSetLayout::SBinding bnd[1] =
			{
				// Example of a binding of type SAMPLED_IMAGE
				{
					IDescriptorSetLayoutBase::SBindingBase(),
					0u,
					IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
					IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					IShader::E_SHADER_STAGE::ESS_COMPUTE,
					1u,
					nullptr
				}
			};
			auto descriptorSetLayoutCPU = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bnd);
			auto descriptorSetCPU = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(descriptorSetLayoutCPU));
			
			// "Write" CPU image view to CPU descriptor set so that its GPU version gets written to the GPU DS automatically
			// Blocked by https://github.com/Devsh-Graphics-Programming/Nabla/issues/798 - will throw a validation error

			auto& sampledImageDescriptorInfo = descriptorSetCPU->getDescriptorInfos(ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t(0u), IDescriptor::E_TYPE::ET_SAMPLED_IMAGE).front();

			sampledImageDescriptorInfo.desc = imageViewCPU;
			sampledImageDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

			// Using asset converter
			smart_refctd_ptr<video::CAssetConverter> converter = video::CAssetConverter::create({ .device = m_device.get(),.optimizer = {} });
			// We don't want to generate mip-maps for the uploaded image, to ensure that we must override the default callbacks.
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

			// Creation + DS write
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = { &descriptorSetCPU.get(), 1 };
			auto reservation = converter->reserve(inputs);

			// Going to need an IUtils to perform uploads/downloads
			m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger));

			// Now convert uploads
			
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

		return true;
	}

	bool keepRunning() override 
	{
		return false;
	}

	void workLoopBody() override
	{

	}

	bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}
};


NBL_MAIN_FUNC(BugReproApp)