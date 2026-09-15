#include "app/App.hpp"
#include "app/AppRenderPassUtilities.hpp"

bool App::waitForInflightFrameSlot()
{
	const uint32_t framesInFlight = getFramesInFlight();
	if (m_realFrameIx < framesInFlight)
		return true;

	const ISemaphore::SWaitInfo cmdbufDonePending[] = {
		{
			.semaphore = m_semaphore.get(),
			.value = m_realFrameIx + 1 - framesInFlight
		}
	};
	return m_device->blockForSemaphores(cmdbufDonePending) == ISemaphore::WAIT_RESULT::SUCCESS;
}

uint32_t App::getFramesInFlight() const
{
	return core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
}

std::optional<SFrameSubmissionContext> App::tryBuildFrameSubmissionContext()
{
	const auto currentSwapchainExtent = m_surface->getCurrentExtent();
	if (currentSwapchainExtent.width * currentSwapchainExtent.height <= 0)
		return std::nullopt;

	SFrameSubmissionContext frameContext = {};
	frameContext.resourceIx = m_realFrameIx % MaxFramesInFlight;
	frameContext.renderArea = makeRenderArea(currentSwapchainExtent.width, currentSwapchainExtent.height);
	frameContext.frame = m_tripleBuffers[frameContext.resourceIx].get();
	frameContext.cmdbuf = m_cmdBufs[frameContext.resourceIx].get();
	frameContext.blitWaitValue = m_blitWaitValues.data() + frameContext.resourceIx;
	return frameContext;
}

bool App::recordFramePasses(const SFrameSubmissionContext& frameContext)
{
	auto* cmdbuf = frameContext.cmdbuf;
	bool success = cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
	success = success && cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
	success = success && cmdbuf->beginDebugMarker("UIApp Frame");

	updateSceneDebugInstances();
	if (m_viewports.useWindow)
	{
		for (uint32_t bindingIx = 0u; bindingIx < m_viewports.windowBindings.size(); ++bindingIx)
			success = success && recordSceneFramebufferPass(cmdbuf, m_viewports.windowBindings[bindingIx], bindingIx);
	}
	else
	{
		success = success && recordSceneFramebufferPass(cmdbuf, m_viewports.windowBindings[m_viewports.activeRenderWindowIx], m_viewports.activeRenderWindowIx);
	}

	success = success && recordUiRenderPass(cmdbuf, frameContext.resourceIx);

	const auto blitQueueFamily = m_surface->getAssignedQueue()->getFamilyIndex();
	const bool needOwnershipRelease = cmdbuf->getQueueFamilyIndex() != blitQueueFamily &&
		!frameContext.frame->getCachedCreationParams().isConcurrentSharing();
	if (needOwnershipRelease)
	{
		const IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barrier[] = { {
			.barrier = {
				.dep = {
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
					.srcAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS | asset::ACCESS_FLAGS::MEMORY_WRITE_BITS,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
					.dstAccessMask = asset::ACCESS_FLAGS::NONE
				},
				.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
				.otherQueueFamilyIndex = blitQueueFamily
			},
			.image = frameContext.frame,
			.subresourceRange = TripleBufferUsedSubresourceRange
		} };
		const IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = { .imgBarriers = barrier };
		success = success && cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);
	}

	return success && cmdbuf->end();
}

bool App::submitAndPresentFrame(const SFrameSubmissionContext& frameContext)
{
	const IQueue::SSubmitInfo::SSemaphoreInfo rendered = {
		.semaphore = m_semaphore.get(),
		.value = m_realFrameIx + 1u,
		.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
	};
	const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = { { .cmdbuf = frameContext.cmdbuf } };
	auto swapchainLock = m_surface->pseudoAcquire(frameContext.blitWaitValue);
	const IQueue::SSubmitInfo::SSemaphoreInfo blitted = {
		.semaphore = m_surface->getPresentSemaphore(),
		.value = frameContext.blitWaitValue->load(),
		.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
	};
	const IQueue::SSubmitInfo submitInfos[1] = {
		{
			.waitSemaphores = { &blitted, 1 },
			.commandBuffers = cmdbufs,
			.signalSemaphores = { &rendered, 1 }
		}
	};

	updateGUIDescriptorSet();
	if (getGraphicsQueue()->submit(submitInfos) != IQueue::RESULT::SUCCESS)
		return false;

	++m_realFrameIx;
	const uint64_t renderedFrameIx = m_realFrameIx - 1u;
	handleFrameCaptureRequests(frameContext.frame, renderedFrameIx);

	const ISmoothResizeSurface::SPresentInfo presentInfo = {
		{
			.source = { .image = frameContext.frame, .rect = frameContext.renderArea },
			.waitSemaphore = rendered.semaphore,
			.waitValue = rendered.value,
			.pPresentSemaphoreWaitValue = frameContext.blitWaitValue,
		},
		frameContext.cmdbuf->getQueueFamilyIndex()
	};
	m_surface->present(std::move(swapchainLock), presentInfo);
	return true;
}
