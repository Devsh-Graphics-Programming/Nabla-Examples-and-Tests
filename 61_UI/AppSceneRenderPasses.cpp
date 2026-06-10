#include "app/App.hpp"
#include "app/AppRenderPassUtilities.hpp"

bool App::recordSceneFramebufferPass(IGPUCommandBuffer* cmdbuf, SWindowControlBinding& binding, const uint32_t)
{
	if (!cmdbuf || !binding.sceneFramebuffer)
		return true;

	const auto& framebufferParams = binding.sceneFramebuffer->getCreationParameters();
	const auto renderArea = makeRenderArea(framebufferParams.width, framebufferParams.height);
	const IGPUCommandBuffer::SRenderpassBeginInfo renderPassInfo = {
		.framebuffer = binding.sceneFramebuffer.get(),
		.colorClearValues = &SCameraAppRenderDefaults::SceneClearColor,
		.depthStencilClearValues = &SCameraAppRenderDefaults::SceneClearDepth,
		.renderArea = renderArea
	};

	bool success = cmdbuf->beginRenderPass(renderPassInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
	const auto finalize = [&]() -> bool
	{
		success = success && cmdbuf->endRenderPass();
		return success;
	};

	const auto viewport = makeFramebufferViewport(framebufferParams.width, framebufferParams.height);
	success = success && cmdbuf->setViewport(0u, 1u, &viewport);
	success = success && cmdbuf->setScissor(0u, 1u, &renderArea);

	if (m_spaceEnvironment.pipeline && m_spaceEnvironment.descriptorSet)
	{
		auto* pipelineLayout = m_spaceEnvironment.pipeline->getLayout();
		const IGPUDescriptorSet* descriptorSets[] = { m_spaceEnvironment.descriptorSet.get() };
		SpaceEnvPushConstants pushConstants = {};
		pushConstants.invProj = hlsl::inverse(binding.projectionMatrix);
		pushConstants.invViewRot = buildInverseViewRotation(binding.viewMatrix);
		pushConstants.orthoMode = binding.isOrthographicProjection ? 1u : 0u;

		success = success && cmdbuf->bindGraphicsPipeline(m_spaceEnvironment.pipeline.get());
		success = success && cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, pipelineLayout, 0u, 1u, descriptorSets);
		success = success && cmdbuf->pushConstants(pipelineLayout, IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0u, sizeof(pushConstants), &pushConstants);
		success = success && nbl::ext::FullScreenTriangle::recordDrawCall(cmdbuf);
	}

	const auto viewParams = CSimpleDebugRenderer::SViewParams(binding.viewMatrix, binding.viewProjMatrix);
	m_debugScene.renderer->render(cmdbuf, viewParams);
	return finalize();
}

bool App::recordUiRenderPass(IGPUCommandBuffer* cmdbuf, const uint32_t resourceIx)
{
	if (!cmdbuf)
		return false;

	const auto uiClearColor = SCameraAppFrameRuntimeDefaults::UiClearColor;
	const auto renderArea = makeRenderArea(m_window->getWidth(), m_window->getHeight());
	const IGPUCommandBuffer::SRenderpassBeginInfo renderPassInfo = {
		.framebuffer = m_framebuffers[resourceIx].get(),
		.colorClearValues = &uiClearColor,
		.depthStencilClearValues = nullptr,
		.renderArea = renderArea
	};

	bool success = cmdbuf->beginRenderPass(renderPassInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
	const auto viewport = makeFramebufferViewport(m_window->getWidth(), m_window->getHeight());
	success = success && cmdbuf->setViewport(0u, 1u, &viewport);

	auto* pipeline = m_ui.manager->getPipeline();
	const auto uiParams = m_ui.manager->getCreationParameters();
	const nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

	success = success && cmdbuf->bindGraphicsPipeline(pipeline);
	success = success && cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());

	if (!keepRunning())
		return false;

	success = success && m_ui.manager->render(cmdbuf, waitInfo);
	success = success && cmdbuf->endRenderPass();
	return success;
}
