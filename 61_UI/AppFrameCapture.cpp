#include "app/App.hpp"

void App::captureRenderedFrame(IGPUImage* frame, const uint64_t renderedFrameIx, const nbl::system::path& outPath, const char* tag)
{
	if (!m_device || !m_assetMgr || !m_surface || !frame)
		return;

	m_logger->log("%s screenshot capture start (frame %llu).", ILogger::ELL_INFO, tag, static_cast<unsigned long long>(renderedFrameIx));
	const ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx };
	if (m_device->blockForSemaphores({ &waitInfo, &waitInfo + 1 }) != ISemaphore::WAIT_RESULT::SUCCESS)
	{
		m_logger->log("%s screenshot failed: wait for render finished.", ILogger::ELL_ERROR, tag);
		return;
	}

	auto viewParams = IGPUImageView::SCreationParams{
		.subUsages = IGPUImage::EUF_TRANSFER_SRC_BIT,
		.image = core::smart_refctd_ptr<IGPUImage>(frame),
		.viewType = IGPUImageView::ET_2D,
		.format = frame->getCreationParameters().format
	};
	viewParams.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = 1u;
	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = 1u;
	auto frameView = m_device->createImageView(std::move(viewParams));
	if (!frameView)
	{
		m_logger->log("%s screenshot failed: could not create frame view.", ILogger::ELL_ERROR, tag);
		return;
	}

	m_logger->log("%s screenshot capture: calling createScreenShot.", ILogger::ELL_INFO, tag);
	const bool ok = ext::ScreenShot::createScreenShot(
		m_device.get(),
		getGraphicsQueue(),
		nullptr,
		frameView.get(),
		m_assetMgr.get(),
		outPath,
		asset::IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
		asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT);

	if (ok)
		m_logger->log("%s screenshot saved to \"%s\".", ILogger::ELL_INFO, tag, outPath.string().c_str());
	else
		m_logger->log("%s screenshot failed to save.", ILogger::ELL_ERROR, tag);
}

void App::handleFrameCaptureRequests(IGPUImage* frame, const uint64_t renderedFrameIx)
{
	if (m_cliRuntime.ciMode && !m_cliRuntime.ciScreenshotDone)
	{
		++m_cliRuntime.ciFrameCounter;
		if (m_cliRuntime.ciFrameCounter >= SCameraAppRuntimeDefaults::CiFramesBeforeCapture)
		{
			m_cliRuntime.ciScreenshotDone = true;
			if (!m_cliRuntime.disableScreenshotsCli)
				captureRenderedFrame(frame, renderedFrameIx, m_cliRuntime.ciScreenshotPath, "CI");
		}
	}

	if (m_cliRuntime.disableScreenshotsCli || !m_scriptedInput.enabled)
		return;

	while (m_scriptedInput.nextCaptureIndex < m_scriptedInput.timeline.captureFrames.size() &&
		m_scriptedInput.timeline.captureFrames[m_scriptedInput.nextCaptureIndex] == renderedFrameIx)
	{
		const auto outPath = m_scriptedInput.captureOutputDir /
			(m_scriptedInput.capturePrefix + "_" + std::to_string(renderedFrameIx) + ".png");
		captureRenderedFrame(frame, renderedFrameIx, outPath, "Script");
		++m_scriptedInput.nextCaptureIndex;
	}
}
