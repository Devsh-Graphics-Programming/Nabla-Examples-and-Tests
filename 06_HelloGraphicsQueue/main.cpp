// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// Here we showcase the use of Graphics Queue only 
// Steps we take in this example:
// - create two images:
//		- one called 'bigImg' with size of 512x512 and initially cleared to red colour
//		- one called 'smallImg' with size of 256x256 and initially cleared to blue colour
// - blit the smallImg into center of the bigImg
// - blit the bigImg back into smallImg
// - save the smallImg to disk
// 
// all without using IUtilities.

class HelloGraphicsQueueApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::MonoDeviceApplication;
	using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore.
	HelloGraphicsQueueApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		IGPUQueue* const queue = getQueue(IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT | IPhysicalDevice::E_QUEUE_FLAGS::EQF_TRANSFER_BIT);

		constexpr VkExtent2D bigImgExtent = { 512u, 512u };
		constexpr VkExtent2D smallImgExtent = { 256u, 256u };

		const auto bigImg = createImage(bigImgExtent);
		if (!bigImg)
			return false;
		const size_t bigImgByteSize = bigImg->getImageDataSizeInBytes();

		const auto smallImg = createImage(smallImgExtent);
		if (!smallImg)
			return false;
		const size_t smallImgByteSize = smallImg->getImageDataSizeInBytes();

		// Create a buffer large enough to back the output 256^2 image, blited image will be saved to this buffer.

		// A `nbl::video::DeviceMemoryAllocator` is an interface to implement anything that can dish out free memory range to bind to back a `nbl::video::IGPUBuffer` or a `nbl::video::IGPUImage`
		// The Logical Device itself implements the interface and behaves as the most simple allocator, it will create a new `nbl::video::IDeviceMemoryAllocation` every single time.
		// We will cover allocators and suballocation in a later example.
		nbl::video::IDeviceMemoryAllocator::SMemoryOffset outputBufferAllocation = {};
		smart_refctd_ptr<IGPUBuffer> outputImageBuffer = nullptr;
		{
			IGPUBuffer::SCreationParams gpuBufCreationParams;
			gpuBufCreationParams.size = smallImgByteSize;
			// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
			gpuBufCreationParams.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT;
			outputImageBuffer = m_device->createBuffer(std::move(gpuBufCreationParams));
			if (!outputImageBuffer)
				return logFail("Failed to create a GPU Buffer of size %d!\n", gpuBufCreationParams.size);

			// Naming objects is cool because not only errors (such as Vulkan Validation Layers) will show their names, but RenderDoc captures too.
			outputImageBuffer->setObjectDebugName("Output Image Buffer");

			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputImageBuffer->getMemoryReqs();
			// you can simply constrain the memory requirements by AND-ing the type bits of the host visible memory types
			reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

			// There are actually two `allocate` overloads, one which allocates memory if you already know the type you want.
			// And this one which is a utility which tries to allocate from every type that matches your requirements in some order of preference.
			// The other of preference (iteration over compatible types) can be controlled by the method's template parameter,
			// the default is from lowest index to highest, but skipping over incompatible types.
			outputBufferAllocation = m_device->allocate(reqs, outputImageBuffer.get(), nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
			if (!outputBufferAllocation.isValid())
				return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

			// Note that we performed a Dedicated Allocation above, so there's no need to bind the memory anymore (since the allocator knows the dedication, it can already bind).
			// This is a carryover from having an OpenGL backend, where you couldn't have a memory allocation separate from the resource, so all allocations had to be "dedicated".
			// In Vulkan dedicated allocations are the most performant and still make sense as long as you won't blow the 4096 allocation limit on windows.
			// You should always use dedicated allocations for images used for swapchains, framebuffer attachments (esp transient), as well as objects used in CUDA/DirectX interop.
			assert(outputImageBuffer->getBoundMemory() == outputBufferAllocation.memory.get());
		}

		const IImage::SSubresourceLayers subresourceLayers = constructDefaultInitializedSubresourceLayers();

		IImage::SBufferCopy bufferCopy;
		bufferCopy.bufferImageHeight = smallImgExtent.height;
		bufferCopy.bufferRowLength = smallImgExtent.width;
		bufferCopy.bufferOffset = 0u;
		bufferCopy.imageExtent = { smallImgExtent.width, smallImgExtent.height, 1u };
		bufferCopy.imageSubresource = subresourceLayers;

		SImageBlit firstImgBlit{};
		{
			asset::VkOffset3D dstOffset0;
			dstOffset0.x = bigImgExtent.width / 2u - smallImgExtent.width / 2u;
			dstOffset0.y = bigImgExtent.height / 2u - smallImgExtent.height / 2u;
			dstOffset0.z = 0u;

			asset::VkOffset3D dstOffset1;
			dstOffset1.x = dstOffset0.x + smallImgExtent.width;
			dstOffset1.y = dstOffset0.y + smallImgExtent.height;
			dstOffset1.z = 1u;

			firstImgBlit.srcSubresource = subresourceLayers;
			firstImgBlit.srcOffsets[0] = { 0u, 0u, 0u };
			firstImgBlit.srcOffsets[1] = { smallImgExtent.width, smallImgExtent.height, 1u };
			firstImgBlit.dstSubresource = subresourceLayers;
			firstImgBlit.dstOffsets[0] = dstOffset0;
			firstImgBlit.dstOffsets[1] = dstOffset1;
		}

		SImageBlit secondImgBlit{};
		{
			secondImgBlit.srcSubresource = subresourceLayers;
			secondImgBlit.srcOffsets[0] = { 0u, 0u, 0u };
			secondImgBlit.srcOffsets[1] = { bigImgExtent.width, bigImgExtent.height, 1u };
			secondImgBlit.dstSubresource = subresourceLayers;
			secondImgBlit.dstOffsets[0] = { 0u, 0u, 0u };
			secondImgBlit.dstOffsets[1] = { smallImgExtent.width, smallImgExtent.height, 1u };
		}

		core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		{
			core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags = static_cast<IGPUCommandPool::E_CREATE_FLAGS>(IGPUCommandPool::E_CREATE_FLAGS::ECF_TRANSIENT_BIT);
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), flags);
			if (!m_device->createCommandBuffers(cmdpool.get(), IGPUCommandBuffer::EL_PRIMARY, 1, &cmdbuf))
				return false;
		}

		IGPUQueue::SSubmitInfo submitInfo{};
		submitInfo.commandBufferCount = 1;
		submitInfo.commandBuffers = &cmdbuf.get();

		IGPUCommandBuffer::SImageMemoryBarrier imgLayoutTransitionBarrier = constructDefaultInitializedImageBarrier();
		imgLayoutTransitionBarrier.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		imgLayoutTransitionBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;

		IGPUCommandBuffer::SImageMemoryBarrier imgClearBarrier = constructDefaultInitializedImageBarrier();
		imgClearBarrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		imgClearBarrier.barrier.srcAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;

		constexpr SClearColorValue red = { .float32{1.0f, 0.0f, 0.0f, 1.0f} };
		constexpr SClearColorValue blue = { .float32{0.0f, 0.0f, 1.0f, 1.0f} };

		smart_refctd_ptr<IGPUFence> done = m_device->createFence(IGPUFence::ECF_UNSIGNALED);

		// Start recording.
		cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		// In order to use images, we need to change their layout from asset::IImage::EL_UNDEFINED.
		// Here it is done with use of barriers.
		IGPUCommandBuffer::SImageMemoryBarrier bigImgLayoutTransitionBarrier = imgLayoutTransitionBarrier;
		bigImgLayoutTransitionBarrier.image = bigImg;

		IGPUCommandBuffer::SImageMemoryBarrier smallImgLayoutTransitionBarrier = imgLayoutTransitionBarrier;
		smallImgLayoutTransitionBarrier.image = smallImg;

		// Transit layouts of both images.
		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_HOST_BIT | E_PIPELINE_STAGE_FLAGS::EPSF_ALL_COMMANDS_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u,
			&bigImgLayoutTransitionBarrier
		);

		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_HOST_BIT | E_PIPELINE_STAGE_FLAGS::EPSF_ALL_COMMANDS_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u,
			&smallImgLayoutTransitionBarrier
		);

		// Clear the image to a given colour.

		auto subResourceRange = constructDefaultInitializedSubresourceRange();
		cmdbuf->clearColorImage(bigImg.get(), IImage::E_LAYOUT::EL_TRANSFER_DST_OPTIMAL, &red, 1u, &subResourceRange);
		cmdbuf->clearColorImage(smallImg.get(), IImage::E_LAYOUT::EL_TRANSFER_DST_OPTIMAL, &blue, 1u, &subResourceRange);

		IGPUCommandBuffer::SImageMemoryBarrier bigImgClearBarrier = imgClearBarrier;
		bigImgClearBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;
		bigImgClearBarrier.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		bigImgClearBarrier.image = bigImg;

		IGPUCommandBuffer::SImageMemoryBarrier smallImgClearBarrier = imgClearBarrier;
		smallImgClearBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_READ_BIT;
		smallImgClearBarrier.newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		smallImgClearBarrier.image = smallImg;

		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &bigImgClearBarrier
		);

		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &smallImgClearBarrier
		);

		// Now blit the smallImg into the center of the bigImg.
		cmdbuf->blitImage(smallImg.get(), IImage::E_LAYOUT::EL_TRANSFER_SRC_OPTIMAL, bigImg.get(), IImage::E_LAYOUT::EL_TRANSFER_DST_OPTIMAL, 1u, &firstImgBlit, ISampler::E_TEXTURE_FILTER::ETF_NEAREST);

		IGPUCommandBuffer::SImageMemoryBarrier bigImgBlitBarrier = constructDefaultInitializedImageBarrier();
		bigImgBlitBarrier.barrier.srcAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;
		bigImgBlitBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_READ_BIT;
		bigImgBlitBarrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		bigImgBlitBarrier.newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		bigImgBlitBarrier.image = bigImg;

		IGPUCommandBuffer::SImageMemoryBarrier smallImgBlitBarrier = constructDefaultInitializedImageBarrier();
		smallImgBlitBarrier.barrier.srcAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_READ_BIT;
		smallImgBlitBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;
		smallImgBlitBarrier.oldLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		smallImgBlitBarrier.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		smallImgBlitBarrier.image = smallImg;

		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &bigImgBlitBarrier
		);

		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &smallImgBlitBarrier
		);

		// Blit whole bigImage into the smallImage, force downsampling with linear filtering.
		cmdbuf->blitImage(bigImg.get(), IImage::E_LAYOUT::EL_TRANSFER_SRC_OPTIMAL, smallImg.get(), IImage::E_LAYOUT::EL_TRANSFER_DST_OPTIMAL, 1u, &secondImgBlit, ISampler::E_TEXTURE_FILTER::ETF_LINEAR);

		IGPUCommandBuffer::SImageMemoryBarrier smallImgSecondBlitBarrier = constructDefaultInitializedImageBarrier();
		smallImgSecondBlitBarrier.barrier.srcAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;
		smallImgSecondBlitBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_READ_BIT;
		smallImgSecondBlitBarrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		smallImgSecondBlitBarrier.newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		smallImgSecondBlitBarrier.image = smallImg;

		cmdbuf->pipelineBarrier(
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_PIPELINE_STAGE_FLAGS::EPSF_TRANSFER_BIT,
			E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &smallImgSecondBlitBarrier
		);

		// Copy the resulting image to a buffer.
		cmdbuf->copyImageToBuffer(smallImg.get(), IImage::E_LAYOUT::EL_TRANSFER_SRC_OPTIMAL, outputImageBuffer.get(), 1u, &bufferCopy);

		cmdbuf->end();

		queue->startCapture();
		queue->submit(1u, &submitInfo, done.get());
		queue->endCapture();
		m_device->blockForFences(1u, &done.get());

		// Read the buffer back, create an ICPUImage with an adopted buffer over its contents.

		// Map memory, so contents of `outputImageBuffer` will be host visible.
		const IDeviceMemoryAllocation::MappedMemoryRange memoryRange(outputBufferAllocation.memory.get(), 0ull, outputBufferAllocation.memory->getAllocationSize());
		auto imageBufferMemPtr = m_device->mapMemory(memoryRange, IDeviceMemoryAllocation::EMCAF_READ);
		if (!imageBufferMemPtr)
			return logFail("Failed to map the Device Memory!\n");

		// If the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches.
		if (!outputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memoryRange);

		const auto ouputImageCreationParams = smallImg->getCreationParameters();
		// While JPG/PNG/BMP/EXR Loaders create ICPUImages because they cannot disambiguate colorspaces,
		// 2D_ARRAY vs 2D and even sometimes formats (looking at your PNG normalmaps!),
		// the writers are always meant to be fed by ICPUImageViews.
		ICPUImageView::SCreationParams params = {};
		{
			params.image = ICPUImage::create(ouputImageCreationParams);
			{
				auto cpuView = core::make_smart_refctd_ptr<CDummyCPUBuffer>(smallImgByteSize, imageBufferMemPtr, core::adopt_memory_t());
				ICPUImage::SBufferCopy region = {};
				region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				region.imageSubresource.layerCount = 1;
				region.imageExtent = ouputImageCreationParams.extent;
				params.image->setBufferAndRegions(std::move(cpuView),core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1,region));
			}
			// Only DDS and KTX support exporting layered views.
			params.viewType = ICPUImageView::ET_2D;
			params.format = ouputImageCreationParams.format;
			params.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			params.subresourceRange.layerCount = 1;
		}
		auto cpuImageView = ICPUImageView::create(std::move(params));
		asset::IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
		m_assetMgr->writeAsset("blit.png", writeParams);

		m_device->unmapMemory(outputBufferAllocation.memory.get());

		return true;
	}

	//
	void workLoopBody() override {}

	//
	bool keepRunning() override {return false;}

protected:
	core::vector<queue_req_t> getQueueRequirements() const override
	{
		using flags_t = IPhysicalDevice::E_QUEUE_FLAGS;
		return {{.requiredFlags=flags_t::EQF_GRAPHICS_BIT,.disallowedFlags=flags_t::EQF_NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}}};
	}

private:
	IImage::SSubresourceRange constructDefaultInitializedSubresourceRange() const
	{
		IImage::SSubresourceRange res{};
		res.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		res.baseMipLevel = 0u;
		res.levelCount = 1u;
		res.baseArrayLayer = 0u;
		res.layerCount = 1u;

		return res;
	}

	IImage::SSubresourceLayers constructDefaultInitializedSubresourceLayers() const
	{
		IImage::SSubresourceLayers res;
		res.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		res.mipLevel = 0u;
		res.baseArrayLayer = 0u;
		res.layerCount = 1u;

		return res;
	}

	IGPUCommandBuffer::SImageMemoryBarrier constructDefaultInitializedImageBarrier() const
	{
		IImage::SSubresourceRange imgSubresourceRange{};
		imgSubresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		imgSubresourceRange.baseMipLevel = 0u;
		imgSubresourceRange.levelCount = 1u;
		imgSubresourceRange.baseArrayLayer = 0u;
		imgSubresourceRange.layerCount = 1u;

		IGPUCommandBuffer::SImageMemoryBarrier res;
		res.barrier.srcAccessMask = E_ACCESS_FLAGS::EAF_NONE;
		res.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_NONE;
		res.oldLayout = asset::IImage::EL_UNDEFINED;
		res.newLayout = asset::IImage::EL_UNDEFINED;
		res.srcQueueFamilyIndex = 0u;
		res.dstQueueFamilyIndex = 0u;
		res.image = nullptr;
		res.subresourceRange = constructDefaultInitializedSubresourceRange();

		return res;
	}

	IGPUImage::SCreationParams constructImageCreationParams(const VkExtent2D& extent) const
	{
		IGPUImage::SCreationParams res{};
		res.type = IImage::E_TYPE::ET_2D;
		res.extent.height = 512u;
		res.extent.width = 512u;
		res.extent.depth = 1u;
		res.format = asset::E_FORMAT::EF_R8G8B8A8_SRGB;
		res.mipLevels = 1u;
		res.flags = IImage::ECF_NONE;
		res.arrayLayers = 1u;
		res.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
		res.tiling = video::IGPUImage::ET_OPTIMAL;
		res.usage = asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
		res.queueFamilyIndexCount = 0u;
		res.queueFamilyIndices = nullptr;
		res.initialLayout = asset::IImage::EL_UNDEFINED;

		return res;
	}

	core::smart_refctd_ptr<IGPUImage> createImage(const VkExtent2D& extent)
	{
		IGPUImage::SCreationParams imgParams{};
		imgParams.type = IImage::E_TYPE::ET_2D;
		imgParams.extent.height = extent.height;
		imgParams.extent.width = extent.width;
		imgParams.extent.depth = 1u;
		imgParams.format = asset::E_FORMAT::EF_R8G8B8A8_SRGB;
		imgParams.mipLevels = 1u;
		imgParams.flags = IImage::ECF_NONE;
		imgParams.arrayLayers = 1u;
		imgParams.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
		imgParams.tiling = video::IGPUImage::ET_OPTIMAL;
		imgParams.usage = asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
		imgParams.queueFamilyIndexCount = 0u;
		imgParams.queueFamilyIndices = nullptr;
		imgParams.initialLayout = asset::IImage::EL_UNDEFINED;

		auto img = m_device->createImage(std::move(imgParams));

		const size_t imageByteSize = img->getImageDataSizeInBytes();
		nbl::video::IGPUBuffer::SCreationParams params = {};
		params.size = imageByteSize;
		params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
		smart_refctd_ptr<IGPUBuffer> imageBuffer = m_device->createBuffer(std::move(params));
		if (!imageBuffer)
		{
			logFail("Failed to create a GPU Buffer of size %d!\n", params.size);
			return nullptr;
		}
		auto allocation = m_device->allocate(imageBuffer->getMemoryReqs(), imageBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);

		ILogicalDevice::SBindImageMemoryInfo bindMemInfo;
		bindMemInfo.image = img.get();
		bindMemInfo.memory = allocation.memory.get();
		bindMemInfo.offset = 0ull;

		m_device->bindImageMemory(1u, &bindMemInfo);

		return img;
	}
};


NBL_MAIN_FUNC(HelloGraphicsQueueApp)
