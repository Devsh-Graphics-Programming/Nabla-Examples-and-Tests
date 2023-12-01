// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


// Here we showcase the use of Graphics Queue only 
class HelloGraphicsQueueApp final : public examples::MonoDeviceApplication
{
	using device_base_t = examples::MonoDeviceApplication;

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	HelloGraphicsQueueApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// This time we will load images and compute their histograms and output them as CSV
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;


		video::IGPUQueue* queue = getGraphicsQueue();

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::ECF_TRANSIENT_BIT);
			if (!m_device->createCommandBuffers(cmdpool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		// Imaage Creation parameters
		nbl::video::IGPUImage::SCreationParams imageParams;
		imageParams.format = EF_B8G8R8A8_UNORM;
		imageParams.type = nbl::video::IGPUImage::ET_2D;
		imageParams.extent.width = 256;
		imageParams.extent.height = 256;
		imageParams.extent.depth = 1u;
		imageParams.mipLevels = 1u;
		imageParams.arrayLayers = 1u;
		imageParams.samples = nbl::video::IGPUImage::ESCF_1_BIT;
		imageParams.flags = nbl::asset::IImage::E_CREATE_FLAGS::ECF_NONE;
		imageParams.usage = nbl::asset::IImage::EUF_STORAGE_BIT | nbl::asset::IImage::EUF_TRANSFER_DST_BIT | nbl::asset::IImage::EUF_TRANSFER_SRC_BIT;
		imageParams.initialLayout = nbl::video::IGPUImage::EL_UNDEFINED;
		imageParams.tiling = nbl::video::IGPUImage::ET_OPTIMAL;

		//Create Image
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImage> image = m_device->createImage(std::move(imageParams));
		
		// Allocate memory to image
		auto imageMemReqs = image->getMemoryReqs();
		imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		m_device->allocate(imageMemReqs, image.get());

		// ImageView creation params
		nbl::video::IGPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.format = image->getCreationParameters().format;
		imgViewParams.image = image;
		imgViewParams.viewType = IGPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(IImage::EAF_COLOR_BIT),0u,1u,0u,1u };

		// Create ImageView
		smart_refctd_ptr<nbl::video::IGPUImageView> imageView = m_device->createImageView(std::move(imgViewParams));

		// Buffer creation params
		video::IGPUBuffer::SCreationParams bufferParams;
		bufferParams.size = image->getMemoryReqs().size;
		bufferParams.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT;

		// Create buffer
		smart_refctd_ptr<nbl::video::IGPUBuffer> buffer = m_device->createBuffer(std::move(bufferParams));

		// Allocate memory to buffer
		auto bufferMemReqs = buffer->getMemoryReqs();
		bufferMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		m_device->allocate(bufferMemReqs, buffer.get());


		// Clear color values
		float redColor[] = { 1.0f, 0.0f, 0.0f, 1.0f };
		SClearColorValue clearColor;
		memcpy(clearColor.float32, redColor, sizeof(redColor));


		asset::IImage::SBufferCopy copyRegion;
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageOffset = { 0, 0, 0 };
		copyRegion.imageExtent = { 256, 256, 1 };

		asset::SMemoryBarrier memoryBarrier;
		memoryBarrier.srcAccessMask = asset::E_ACCESS_FLAGS::EAF_MEMORY_READ_BIT;
		memoryBarrier.dstAccessMask = asset::E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;


		cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		// clear the image to a given colour
		cmdbuf->clearColorImage(image.get(), nbl::video::IGPUImage::EL_GENERAL, &clearColor, 1, &imgViewParams.subresourceRange);

		// Before copyImageToBuffer
		cmdbuf->pipelineBarrier(
			EPSF_TRANSFER_BIT, 
			EPSF_TRANSFER_BIT, 
			EDF_NONE,
			1u, &memoryBarrier,
			0u, nullptr, 
			0u, nullptr  
		);

		// copy the resulting image to a buffer
		cmdbuf->copyImageToBuffer(image.get(), asset::IImage::EL_GENERAL, buffer.get(), 1, &copyRegion);
		cmdbuf->end();

		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));


		ICPUImage::SCreationParams par;
		par.type = asset::ICPUImage::ET_2D;
		par.samples = asset::ICPUImage::ESCF_1_BIT;
		par.format = EF_B8G8R8A8_UNORM;
		par.extent.width = 256;
		par.extent.height = 256;
		par.extent.depth = 1u;
		par.mipLevels = 1u;
		par.arrayLayers = 1u;
		par.flags = nbl::asset::IImage::E_CREATE_FLAGS::ECF_NONE;
		par.usage = nbl::asset::IImage::EUF_STORAGE_BIT | nbl::asset::IImage::EUF_TRANSFER_DST_BIT | nbl::asset::IImage::EUF_TRANSFER_SRC_BIT;


		smart_refctd_ptr<nbl::asset::ICPUImage> cpuImage = asset::ICPUImage::create(par);

		ICPUImageView::SCreationParams imageViewParams;
		imageViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imageViewParams.format = cpuImage->getCreationParameters().format;;
		imageViewParams.image = core::smart_refctd_ptr<ICPUImage>(cpuImage);
		imageViewParams.viewType = ICPUImageView::ET_2D;
		imageViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		smart_refctd_ptr<nbl::asset::ICPUImageView> imgView = ICPUImageView::create(std::move(imageViewParams));

		
//		IAssetWriter::SAssetWriteParams wp(imgView.get());
//		wp.workingDirectory = "";
//		assetManager->writeAsset("jpgWriteSuccessful.png", wp);


		// read the buffer back, create an ICPUImage with an adopted buffer over its contents
		// save the image to PNG,JPG and whatever other non-EXR format we support writing

		return true;
	}

	//
	void workLoopBody() override {}

	//
	bool keepRunning() override { return false; }

private:

	video::IGPUQueue* getGraphicsQueue() const
	{
		const auto familyProperties = m_device->getPhysicalDevice()->getQueueFamilyProperties();
		for (auto i = 0u; i < familyProperties.size(); i++)
			if (familyProperties[i].queueFlags.hasFlags(video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT))
				return m_device->getQueue(i, 0);

		return nullptr;
	}
};


NBL_MAIN_FUNC(HelloGraphicsQueueApp)