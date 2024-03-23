// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

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
class HelloGraphicsQueueApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = application_templates::MonoDeviceApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore.
		HelloGraphicsQueueApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(core::smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(core::smart_refctd_ptr(system)))
				return false;

			// Requesting queue with graphics and transfer capabilities. Transfer capablility will be needed for image to buffer copy operation.
			IQueue* const queue = getQueue(IQueue::FAMILY_FLAGS::GRAPHICS_BIT|IQueue::FAMILY_FLAGS::TRANSFER_BIT);

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
			nbl::video::IDeviceMemoryAllocator::SAllocation outputBufferAllocation = {};
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

				outputBufferAllocation = m_device->allocate(reqs, outputImageBuffer.get(), nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
				if (!outputBufferAllocation.isValid())
					return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
			}

			const IImage::SSubresourceLayers subresourceLayers = constructDefaultInitializedSubresourceLayers();

			IImage::SBufferCopy bufferCopy;
			bufferCopy.bufferImageHeight = smallImgExtent.height;
			bufferCopy.bufferRowLength = smallImgExtent.width;
			bufferCopy.bufferOffset = 0u;
			bufferCopy.imageExtent = { smallImgExtent.width, smallImgExtent.height, 1u };
			bufferCopy.imageSubresource = subresourceLayers;

			IGPUCommandBuffer::SImageBlit firstImgBlit{};
			{
				asset::VkOffset3D dstOffset0;
				dstOffset0.x = bigImgExtent.width / 2u - smallImgExtent.width / 2u;
				dstOffset0.y = bigImgExtent.height / 2u - smallImgExtent.height / 2u;
				dstOffset0.z = 0u;

				asset::VkOffset3D dstOffset1;
				dstOffset1.x = dstOffset0.x + smallImgExtent.width;
				dstOffset1.y = dstOffset0.y + smallImgExtent.height;
				dstOffset1.z = 1u;

				firstImgBlit.srcMinCoord = { 0u, 0u, 0u };
				firstImgBlit.srcMaxCoord = { smallImgExtent.width, smallImgExtent.height, 1u };
				firstImgBlit.dstMinCoord = dstOffset0;
				firstImgBlit.dstMaxCoord = dstOffset1;
				firstImgBlit.layerCount = subresourceLayers.layerCount;
				firstImgBlit.srcBaseLayer = subresourceLayers.baseArrayLayer;
				firstImgBlit.dstBaseLayer = subresourceLayers.baseArrayLayer;
				firstImgBlit.srcMipLevel = subresourceLayers.mipLevel;
				firstImgBlit.dstMipLevel = subresourceLayers.mipLevel;
				firstImgBlit.aspectMask = subresourceLayers.aspectMask.value;
			}

			IGPUCommandBuffer::SImageBlit secondImgBlit{};
			{
				secondImgBlit.srcMinCoord = { 0u, 0u, 0u };
				secondImgBlit.srcMaxCoord = { bigImgExtent.width, bigImgExtent.height, 1u };
				secondImgBlit.dstMinCoord = { 0u, 0u, 0u };
				secondImgBlit.dstMaxCoord = { smallImgExtent.width, smallImgExtent.height, 1u };
				secondImgBlit.layerCount = subresourceLayers.layerCount;
				secondImgBlit.srcBaseLayer = subresourceLayers.baseArrayLayer;
				secondImgBlit.dstBaseLayer = subresourceLayers.baseArrayLayer;
				secondImgBlit.srcMipLevel = subresourceLayers.mipLevel;
				secondImgBlit.dstMipLevel = subresourceLayers.mipLevel;
				secondImgBlit.aspectMask = subresourceLayers.aspectMask.value;
			}

			core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags = static_cast<IGPUCommandPool::CREATE_FLAGS>(IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), flags);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmdbuf))
					return false;
			}

			smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(0);
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {{
				.semaphore = progress.get(),
				.value = 1,
				// wait for the Copy Image to Buffer to finish before we signal
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT
			}};

			IQueue::SSubmitInfo submitInfos[1];
			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfos[1] = {cmdbuf.get()};
			submitInfos[0].commandBuffers = cmdbufInfos;
			submitInfos[0].signalSemaphores = signals;
		

			IGPUCommandBuffer::SImageMemoryBarrier imgLayoutTransitionBarrier = constructDefaultInitializedImageBarrier();
			imgLayoutTransitionBarrier.newLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
			imgLayoutTransitionBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;

			auto generateNextBarrier = [](
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>& prevoiusBarrier,
				IImage::LAYOUT newLayout,
				asset::ACCESS_FLAGS dstAccessMask) -> IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>
				{
					IGPUCommandBuffer::SImageMemoryBarrier ret = prevoiusBarrier;
					ret.oldLayout = prevoiusBarrier.newLayout;
					ret.newLayout = newLayout;
					ret.barrier.dep.srcAccessMask = prevoiusBarrier.barrier.dep.dstAccessMask;
					ret.barrier.dep.dstAccessMask = dstAccessMask;

					return ret;
				};

			// Using `float32` union member since format of used images is E_FORMAT::EF_R8G8B8A8_SRGB,
			// which is a fixed-point format and those are encoded/decoded from/to float.
			constexpr IGPUCommandBuffer::SClearColorValue red = { .float32{1.0f, 0.0f, 0.0f, 1.0f} };
			constexpr IGPUCommandBuffer::SClearColorValue blue = { .float32{0.0f, 0.0f, 1.0f, 1.0f} };

			// Start recording.
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			// In order to use images, we need to change their layout from asset::IImage::EL_UNDEFINED.
			// Here it is done with use of barriers.
			std::array<IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>, 2u> imgLayoutTransitionBarriers = {
				imgLayoutTransitionBarrier,
				imgLayoutTransitionBarrier
			};
			imgLayoutTransitionBarriers[0].image = bigImg.get();
			imgLayoutTransitionBarriers[1].image = smallImg.get();

			imgLayoutTransitionBarriers[0].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT | PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
			imgLayoutTransitionBarriers[0].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			imgLayoutTransitionBarriers[1].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT | PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
			imgLayoutTransitionBarriers[1].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;

			// Transit layouts of both images.
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.imgBarriers = imgLayoutTransitionBarriers });

			// Clear the image to a given colour.
			auto subResourceRange = constructDefaultInitializedSubresourceRange();
			cmdbuf->clearColorImage(bigImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, &red, 1u, &subResourceRange);
			cmdbuf->clearColorImage(smallImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, &blue, 1u, &subResourceRange);

			std::array<IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>, 2u> imgClearBarriers = {
				generateNextBarrier(imgLayoutTransitionBarriers[0], IImage::LAYOUT::TRANSFER_DST_OPTIMAL, ACCESS_FLAGS::MEMORY_WRITE_BITS),
				generateNextBarrier(imgLayoutTransitionBarriers[1], IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, ACCESS_FLAGS::MEMORY_READ_BITS)
			};

			imgClearBarriers[0].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			imgClearBarriers[0].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			imgClearBarriers[1].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			imgClearBarriers[1].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo imgClearBarriersDepInfo = { .imgBarriers = imgClearBarriers };
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, imgClearBarriersDepInfo);

			// Now blit the smallImg into the center of the bigImg.
			cmdbuf->blitImage(smallImg.get(), IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, bigImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, {&firstImgBlit,1}, ISampler::E_TEXTURE_FILTER::ETF_NEAREST);

			std::array<IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>, 2u> imgBlitBarriers = {
				generateNextBarrier(imgClearBarriers[0], IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, ACCESS_FLAGS::MEMORY_READ_BITS),
				generateNextBarrier(imgClearBarriers[1], IImage::LAYOUT::TRANSFER_DST_OPTIMAL, ACCESS_FLAGS::MEMORY_WRITE_BITS)
			};

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo imgBlitBarriersDepInfo = { .imgBarriers = imgBlitBarriers };
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, imgBlitBarriersDepInfo);

			// Blit whole bigImage into the smallImage, force downsampling with linear filtering.
			cmdbuf->blitImage(bigImg.get(), IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, smallImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, {&secondImgBlit,1}, ISampler::E_TEXTURE_FILTER::ETF_LINEAR);

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> smallImgSecondBlitBarrier[1] = { constructDefaultInitializedImageBarrier() };
			smallImgSecondBlitBarrier[0].barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
			smallImgSecondBlitBarrier[0].barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;
			smallImgSecondBlitBarrier[0].oldLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
			smallImgSecondBlitBarrier[0].newLayout = asset::IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			smallImgSecondBlitBarrier[0].image = smallImg.get();
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo smallImgSecondBlitBarrierDepInfo;
			smallImgSecondBlitBarrierDepInfo.imgBarriers = smallImgSecondBlitBarrier;

			smallImgSecondBlitBarrier[0].barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			smallImgSecondBlitBarrier[0].barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;

			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, smallImgSecondBlitBarrierDepInfo);

			// Copy the resulting image to a buffer.
			cmdbuf->copyImageToBuffer(smallImg.get(), IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, outputImageBuffer.get(), 1u, &bufferCopy);

			cmdbuf->end();

			queue->startCapture();
			queue->submit(submitInfos);
			queue->endCapture();
		
			// The signalling and waiting on the semaphore make all writes from the Semaphore Signal Stages visible and available to the Host
			const ISemaphore::SWaitInfo waitInfos[1] = { {
					.semaphore = progress.get(),
					.value = 1
			} };
			m_device->blockForSemaphores(waitInfos);

			// Read the buffer back, create an ICPUImage with an adopted buffer over its contents.
			// TODO: Use the Screenshot Extension!

			// Map memory, so contents of `outputImageBuffer` will be host visible.
			const ILogicalDevice::MappedMemoryRange memoryRange(outputBufferAllocation.memory.get(), 0ull, outputBufferAllocation.memory->getAllocationSize());
			auto imageBufferMemPtr = outputBufferAllocation.memory->map({ 0ull,outputBufferAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ);
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

				// ICPUImage isn't really a representation of a GPU Image in itself, more of a recipe for creating one from a series of ICPUBuffer to ICPUImage copies.
				// This means that an ICPUImage has no internal storage or memory bound for its texels and rather references separate ICPUBuffer ranges to provide its contents,
				// which also means it can be sparsely(with gaps) specified.
				params.image = ICPUImage::create(ouputImageCreationParams);
				{
					// CDummyCPUBuffer is used for creating ICPUBuffer over an already existing memory, without any memcopy operations 
					// or taking over the memory ownership. CDummyCPUBuffer cannot free its memory.
					auto cpuOutputImageBuffer = core::make_smart_refctd_ptr<CDummyCPUBuffer>(smallImgByteSize, imageBufferMemPtr, core::adopt_memory_t());
					ICPUImage::SBufferCopy region = {};
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					region.imageSubresource.layerCount = 1;
					region.imageExtent = ouputImageCreationParams.extent;

					// 
					params.image->setBufferAndRegions(std::move(cpuOutputImageBuffer),core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1,region));
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

			// Even if you forgot to unmap, it would unmap itself when `outputBufferAllocation.memory` 
			// gets dropped by its last reference and its destructor runs.
			outputBufferAllocation.memory->unmap();

			return true;
		}

		//
		void workLoopBody() override {}

		//
		bool keepRunning() override {return false;}

	protected:
		core::vector<queue_req_t> getQueueRequirements() const override
		{
			using flags_t = IQueue::FAMILY_FLAGS;
			return {{.requiredFlags=flags_t::GRAPHICS_BIT|flags_t::TRANSFER_BIT,.disallowedFlags=flags_t::NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}}};
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

		IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> constructDefaultInitializedImageBarrier() const
		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> res = {};
			res.subresourceRange = constructDefaultInitializedSubresourceRange();
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
			imgParams.tiling = video::IGPUImage::TILING::OPTIMAL;
			imgParams.usage = asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
			imgParams.queueFamilyIndexCount = 0u;
			imgParams.queueFamilyIndices = nullptr;
			imgParams.preinitialized = false;

			auto img = m_device->createImage(std::move(imgParams));

			// Dedicated allocation for an image, no need to bind memory later.
			auto allocation = m_device->allocate(img->getMemoryReqs(), img.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
			IDeviceMemoryAllocator::SAllocateInfo;
			if (!allocation.isValid())
			{
				logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
				return nullptr;
			}

			return img;
		}
};


NBL_MAIN_FUNC(HelloGraphicsQueueApp)
