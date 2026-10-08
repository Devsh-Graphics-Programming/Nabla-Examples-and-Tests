#ifndef _NBL_THIS_EXAMPLE_APP_SWAPCHAIN_RESOURCES_HPP_
#define _NBL_THIS_EXAMPLE_APP_SWAPCHAIN_RESOURCES_HPP_

#include "app/AppTypes.hpp"

class CSwapchainResources final : public ISmoothResizeSurface::ISwapchainResources
{
public:
	constexpr static inline IQueue::FAMILY_FLAGS RequiredQueueFlags = IQueue::FAMILY_FLAGS::GRAPHICS_BIT;

	inline uint8_t getLastImageIndex() const
	{
		return m_lastImageIndex;
	}

protected:
	inline core::bitflag<asset::PIPELINE_STAGE_FLAGS> getTripleBufferPresentStages() const override
	{
		return asset::PIPELINE_STAGE_FLAGS::BLIT_BIT;
	}

	inline bool tripleBufferPresent(
		IGPUCommandBuffer* cmdbuf,
		const ISmoothResizeSurface::SPresentSource& source,
		const uint8_t imageIndex,
		const uint32_t qFamToAcquireSrcFrom) override
	{
		bool success = true;
		auto acquiredImage = getImage(imageIndex);
		m_lastImageIndex = imageIndex;

		const bool needToAcquireSrcOwnership = qFamToAcquireSrcFrom != IQueue::FamilyIgnored;
		assert(!source.image->getCachedCreationParams().isConcurrentSharing() || !needToAcquireSrcOwnership);

		const auto blitDstLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = {};

		using image_barrier_t = decltype(depInfo.imgBarriers)::element_type;
		const image_barrier_t preBarriers[2] = {
			{
				.barrier = {
					.dep = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
						.srcAccessMask = asset::ACCESS_FLAGS::NONE,
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				},
				.image = acquiredImage,
				.subresourceRange = {
					.aspectMask = IGPUImage::EAF_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				},
				.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
				.newLayout = blitDstLayout
			},
			{
				.barrier = {
					.dep = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
						.srcAccessMask = asset::ACCESS_FLAGS::NONE,
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT
					},
					.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
					.otherQueueFamilyIndex = qFamToAcquireSrcFrom
				},
				.image = source.image,
				.subresourceRange = TripleBufferUsedSubresourceRange
			}
		};

		depInfo.imgBarriers = { preBarriers, needToAcquireSrcOwnership ? 2ull : 1ull };
		success &= cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);

		{
			const auto srcOffset = source.rect.offset;
			const auto srcExtent = source.rect.extent;
			const auto dstExtent = acquiredImage->getCreationParameters().extent;
			const IGPUCommandBuffer::SImageBlit regions[1] = { {
				.srcMinCoord = { static_cast<uint32_t>(srcOffset.x), static_cast<uint32_t>(srcOffset.y), 0 },
				.srcMaxCoord = { srcExtent.width, srcExtent.height, 1 },
				.dstMinCoord = { 0, 0, 0 },
				.dstMaxCoord = { dstExtent.width, dstExtent.height, 1 },
				.layerCount = acquiredImage->getCreationParameters().arrayLayers,
				.srcBaseLayer = 0,
				.dstBaseLayer = 0,
				.srcMipLevel = 0
			} };
			success &= cmdbuf->blitImage(
				source.image,
				IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
				acquiredImage,
				blitDstLayout,
				regions,
				IGPUSampler::ETF_LINEAR);
		}

		const image_barrier_t postBarrier[1] = {
			{
				.barrier = {
					.dep = preBarriers[0].barrier.dep.nextBarrier(asset::PIPELINE_STAGE_FLAGS::NONE, asset::ACCESS_FLAGS::NONE)
				},
				.image = preBarriers[0].image,
				.subresourceRange = preBarriers[0].subresourceRange,
				.oldLayout = blitDstLayout,
				.newLayout = IGPUImage::LAYOUT::PRESENT_SRC
			}
		};
		depInfo.imgBarriers = postBarrier;
		success &= cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);

		return success;
	}

private:
	uint8_t m_lastImageIndex = 0u;
};

#endif
