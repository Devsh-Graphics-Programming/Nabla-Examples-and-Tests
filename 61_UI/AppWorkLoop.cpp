#include "app/App.hpp"

void App::workLoopBody()
{
			paceScriptedVisualDebugFrame();

			// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
			const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
			// We block for semaphores for 2 reasons here:
			// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
			// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
			if (m_realFrameIx >= framesInFlight)
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[] = {
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - framesInFlight
					}
				};
				if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			// Predict size of next render, and bail if nothing to do
			const auto currentSwapchainExtent = m_surface->getCurrentExtent();
			if (currentSwapchainExtent.width * currentSwapchainExtent.height <= 0)
				return;
			// The extent of the swapchain might change between now and `present` but the blit should adapt nicely
			const VkRect2D currentRenderArea = { .offset = {0,0},.extent = currentSwapchainExtent };

			// You explicitly should not use `getAcquireCount()` see the comment on `m_realFrameIx`
			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			// We will be using this command buffer to produce the frame
			auto frame = m_tripleBuffers[resourceIx].get();
			auto cmdbuf = m_cmdBufs[resourceIx].get();

			// update CPU stuff - input bindings, events, UI state
			update();

			bool willSubmit = true;
			{
				willSubmit &= cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
				willSubmit &= cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				willSubmit &= cmdbuf->beginDebugMarker("UIApp Frame");

				auto renderScene = [&](SWindowControlBinding& binding, const uint32_t bindingIx)
				{
					if (!binding.sceneFramebuffer)
						return;

					const auto& fbParams = binding.sceneFramebuffer->getCreationParameters();
					const VkRect2D renderArea = { .offset = {0,0}, .extent = {fbParams.width, fbParams.height} };
					const IGPUCommandBuffer::SRenderpassBeginInfo info = {
						.framebuffer = binding.sceneFramebuffer.get(),
						.colorClearValues = &SceneClearColor,
						.depthStencilClearValues = &SceneClearDepth,
						.renderArea = renderArea
					};

					willSubmit &= cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
					{
						asset::SViewport viewport = {};
						viewport.minDepth = 1.f;
						viewport.maxDepth = 0.f;
						viewport.x = 0u;
						viewport.y = 0u;
						viewport.width = fbParams.width;
						viewport.height = fbParams.height;

						willSubmit &= cmdbuf->setViewport(0u, 1u, &viewport);
						willSubmit &= cmdbuf->setScissor(0u, 1u, &renderArea);

						if (m_spaceEnvPipeline && m_spaceEnvDescriptorSet)
						{
							auto* pipelineLayout = m_spaceEnvPipeline->getLayout();
							const IGPUDescriptorSet* descriptorSets[] = { m_spaceEnvDescriptorSet.get() };
							SpaceEnvPushConstants pc = {};
							pc.invProj = hlsl::inverse(binding.projectionMatrix);
							pc.invViewRot = hlsl::transpose(getMatrix3x4As4x4(binding.viewMatrix));
							pc.invViewRot[0].w = 0.0f;
							pc.invViewRot[1].w = 0.0f;
							pc.invViewRot[2].w = 0.0f;
							pc.invViewRot[3] = float32_t4(0.0f, 0.0f, 0.0f, 1.0f);
							pc.orthoMode = binding.isOrthographicProjection ? 1u : 0u;

							willSubmit &= cmdbuf->bindGraphicsPipeline(m_spaceEnvPipeline.get());
							willSubmit &= cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, pipelineLayout, 0u, 1u, descriptorSets);
							willSubmit &= cmdbuf->pushConstants(pipelineLayout, IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0u, sizeof(pc), &pc);
							willSubmit &= nbl::ext::FullScreenTriangle::recordDrawCall(cmdbuf);
						}

						const auto viewParams = CSimpleDebugRenderer::SViewParams(binding.viewMatrix, binding.viewProjMatrix);
						m_renderer->render(cmdbuf, viewParams);
						const bool drawScriptFrustum = m_scriptedInput.enabled && m_scriptedInput.visualDebug;
						if (m_drawFrustum && drawScriptFrustum)
						{
							const auto findSourceBindingIxForPlanar = [&](const uint32_t planarIx) -> std::optional<uint32_t>
							{
								if (activeRenderWindowIx < windowBindings.size())
								{
									const auto& activeBinding = windowBindings[activeRenderWindowIx];
									if (activeBinding.activePlanarIx == planarIx && activeBinding.boundProjectionIx.has_value())
										return activeRenderWindowIx;
								}

								for (uint32_t i = 0u; i < windowBindings.size(); ++i)
								{
									const auto& candidate = windowBindings[i];
									if (candidate.activePlanarIx != planarIx)
										continue;
									if (!candidate.boundProjectionIx.has_value())
										continue;
									return i;
								}
								return std::nullopt;
							};

							std::optional<uint32_t> sourceBindingIx = std::nullopt;
							if (boundPlanarCameraIxToManipulate.has_value())
								sourceBindingIx = findSourceBindingIxForPlanar(boundPlanarCameraIxToManipulate.value());
							if (!sourceBindingIx.has_value() && activeRenderWindowIx < windowBindings.size())
							{
								const auto& activeBinding = windowBindings[activeRenderWindowIx];
								if (activeBinding.boundProjectionIx.has_value() && activeBinding.activePlanarIx < m_planarProjections.size())
									sourceBindingIx = activeRenderWindowIx;
							}

							if (sourceBindingIx.has_value())
							{
								const auto& sourceBinding = windowBindings[sourceBindingIx.value()];
								const bool sameCameraAsView = binding.activePlanarIx == sourceBinding.activePlanarIx;
								const bool sameWindow = bindingIx == sourceBindingIx.value();
								if (!sameCameraAsView && !sameWindow)
								{
									ext::frustum::CDrawFrustum::DrawParameters drawParams = {};
									drawParams.commandBuffer = cmdbuf;
									drawParams.viewProjectionMatrix = binding.viewProjMatrix;
									drawParams.lineWidth = 1.0f;

									const float32_t4 color = float32_t4(1.0f, 0.95f, 0.25f, 1.0f);
									willSubmit &= m_drawFrustum->renderSingle(drawParams, hlsl::inverse(sourceBinding.viewProjMatrix), color);
								}
							}
						}

					}
					willSubmit &= cmdbuf->endRenderPass();
				};

				if (m_renderer && !m_renderer->m_instances.empty())
				{
					auto& instance = m_renderer->m_instances[0];
					instance.world = m_model;
					const auto geomCount = m_renderer->getGeometries().size();
					if (geomCount)
					{
						if (gcIndex >= geomCount)
							gcIndex = 0;
						instance.packedGeo = m_renderer->getGeometries().data() + gcIndex;
					}

					const uint32_t gridInstanceIx = 1u;
					if (m_gridGeometryIx.has_value() && m_renderer->m_instances.size() > gridInstanceIx)
					{
						const auto gridIx = m_gridGeometryIx.value();
						if (gridIx < geomCount)
						{
							auto& gridInstance = m_renderer->m_instances[gridInstanceIx];
							gridInstance.packedGeo = m_renderer->getGeometries().data() + gridIx;

							constexpr float gridExtent = 32.0f;
							float32_t3x4 gridWorld = float32_t3x4(1.0f);
							gridWorld[0] = float32_t4(gridExtent, 0.0f, 0.0f, -0.5f * gridExtent);
							gridWorld[1] = float32_t4(0.0f, 1.0f, 0.0f, -0.5f);
							gridWorld[2] = float32_t4(0.0f, 0.0f, gridExtent, -0.5f * gridExtent);
							gridInstance.world = gridWorld;
						}
					}

					const uint32_t followInstanceIx = 1u + (m_gridGeometryIx.has_value() ? 1u : 0u);
					if (m_renderer->m_instances.size() > followInstanceIx)
					{
						auto& followInstance = m_renderer->m_instances[followInstanceIx];
						if (m_followTargetVisible && m_followTargetGeometryIx.has_value() && m_followTargetGeometryIx.value() < geomCount)
						{
							followInstance.packedGeo = m_renderer->getGeometries().data() + m_followTargetGeometryIx.value();
							followInstance.world = computeFollowTargetMarkerWorld();
						}
						else
						{
							followInstance.packedGeo = nullptr;
							followInstance.world = float32_t3x4(1.0f);
						}
					}
				}

				if (useWindow)
					for (uint32_t i = 0u; i < windowBindings.size(); ++i)
						renderScene(windowBindings[i], i);
				else
					renderScene(windowBindings[activeRenderWindowIx], activeRenderWindowIx);
				
				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
				const IGPUCommandBuffer::SRenderpassBeginInfo info = {
					.framebuffer = m_framebuffers[resourceIx].get(),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};

				// UI renderpass
				willSubmit &= cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				{
					asset::SViewport viewport;
					{
						viewport.minDepth = 1.f;
						viewport.maxDepth = 0.f;
						viewport.x = 0u;
						viewport.y = 0u;
						viewport.width = m_window->getWidth();
						viewport.height = m_window->getHeight();
					}

					willSubmit &= cmdbuf->setViewport(0u, 1u, &viewport);

					const VkRect2D currentRenderArea =
					{
						.offset = {0,0},
						.extent = {m_window->getWidth(),m_window->getHeight()}
					};

					IQueue::SSubmitInfo::SCommandBufferInfo commandBuffersInfo[] = { {.cmdbuf = cmdbuf } };

					const IGPUCommandBuffer::SRenderpassBeginInfo info =
					{
						.framebuffer = m_framebuffers[resourceIx].get(),
						.colorClearValues = &clearValue,
						.depthStencilClearValues = nullptr,
						.renderArea = currentRenderArea
					};

					nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };
					const auto uiParams = m_ui.manager->getCreationParameters();
					auto* pipeline = m_ui.manager->getPipeline();

					cmdbuf->bindGraphicsPipeline(pipeline);
					cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get()); // note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx

					if (!keepRunning())
						return;

					willSubmit &= m_ui.manager->render(cmdbuf, waitInfo);
				}
				willSubmit &= cmdbuf->endRenderPass();

				// If the Rendering and Blit/Present Queues don't come from the same family we need to transfer ownership, because we need to preserve contents between them.
				auto blitQueueFamily = m_surface->getAssignedQueue()->getFamilyIndex();
				// Also should crash/error if concurrent sharing enabled but would-be-user-queue is not in the share set, but oh well.
				const bool needOwnershipRelease = cmdbuf->getQueueFamilyIndex() != blitQueueFamily && !frame->getCachedCreationParams().isConcurrentSharing();
				if (needOwnershipRelease)
				{
					const IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barrier[] = { {
						.barrier = {
							.dep = {
							// Normally I'd put `COLOR_ATTACHMENT` on the masks, but we want this to happen after Layout Transition :(
							// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
							.srcAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS | asset::ACCESS_FLAGS::MEMORY_WRITE_BITS,
							// For a Queue Family Ownership Release the destination access masks are irrelevant
							// and source stage mask can be NONE as long as the semaphore signals ALL_COMMANDS_BIT
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
							.dstAccessMask = asset::ACCESS_FLAGS::NONE
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
						.otherQueueFamilyIndex = blitQueueFamily
					},
					.image = frame,
					.subresourceRange = TripleBufferUsedSubresourceRange
						// there will be no layout transition, already done by the Renderpass End
					} };
					const IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = { .imgBarriers = barrier };
					willSubmit &= cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);
				}
			}
			willSubmit &= cmdbuf->end();

			// submit and present under a mutex ASAP
			if (willSubmit)
			{
				// We will signal a semaphore in the rendering queue, and await it with the presentation/blit queue
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered = 
				{
					.semaphore = m_semaphore.get(),
					.value = m_realFrameIx + 1,
					// Normally I'd put `COLOR_ATTACHMENT` on the masks, but we want to signal after Layout Transitions and optional Ownership Release
					// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
					.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				};
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = 
				{ {
					.cmdbuf = cmdbuf
				} };
				// We need to wait on previous triple buffer blits/presents from our source image to complete
				auto* pBlitWaitValue = m_blitWaitValues.data() + resourceIx;
				auto swapchainLock = m_surface->pseudoAcquire(pBlitWaitValue);
				const IQueue::SSubmitInfo::SSemaphoreInfo blitted = 
				{
					.semaphore = m_surface->getPresentSemaphore(),
					.value = pBlitWaitValue->load(),
					// Normally I'd put `BLIT` on the masks, but we want to wait before Implicit Layout Transitions and optional Implicit Ownership Acquire
					// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
					.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				};
				const IQueue::SSubmitInfo submitInfos[1] = 
				{
					{
						.waitSemaphores = {&blitted,1},
						.commandBuffers = cmdbufs,
						.signalSemaphores = {&rendered,1}
					}
				};

				updateGUIDescriptorSet();

				if (getGraphicsQueue()->submit(submitInfos) != IQueue::RESULT::SUCCESS)
					return;

				m_realFrameIx++;

				const uint64_t renderedFrameIx = m_realFrameIx - 1u;
				auto captureScreenshot = [&](const nbl::system::path& outPath, const char* tag) -> void
				{
					if (!m_device || !m_assetMgr || !m_surface)
						return;

					m_logger->log("%s screenshot capture start (frame %llu).", ILogger::ELL_INFO, tag, static_cast<unsigned long long>(renderedFrameIx));
					const ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx };
					if (m_device->blockForSemaphores({ &waitInfo, &waitInfo + 1 }) != ISemaphore::WAIT_RESULT::SUCCESS)
					{
						m_logger->log("%s screenshot failed: wait for render finished.", ILogger::ELL_ERROR, tag);
						return;
					}

					if (!frame)
					{
						m_logger->log("%s screenshot failed: missing frame image.", ILogger::ELL_ERROR, tag);
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
				};

				// only present if there's successful content to show
				const ISmoothResizeSurface::SPresentInfo presentInfo = {
					{
						.source = {.image = frame,.rect = currentRenderArea},
						.waitSemaphore = rendered.semaphore,
						.waitValue = rendered.value,
						.pPresentSemaphoreWaitValue = pBlitWaitValue,
					},
					// The Graphics Queue will be the the most recent owner just before it releases ownership
					cmdbuf->getQueueFamilyIndex()
				};
				if (m_ciMode && !m_ciScreenshotDone)
				{
					++m_ciFrameCounter;
					if (m_ciFrameCounter >= CiFramesBeforeCapture)
					{
						m_ciScreenshotDone = true;
						if (!m_disableScreenshotsCli)
							captureScreenshot(m_ciScreenshotPath, "CI");
					}
				}

				if (!m_disableScreenshotsCli && m_scriptedInput.enabled && !m_scriptedInput.timeline.captureFrames.empty())
				{
					while (m_scriptedInput.nextCaptureIndex < m_scriptedInput.timeline.captureFrames.size() &&
						m_scriptedInput.timeline.captureFrames[m_scriptedInput.nextCaptureIndex] == renderedFrameIx)
					{
						const auto outPath = m_scriptedInput.captureOutputDir /
							(m_scriptedInput.capturePrefix + "_" + std::to_string(renderedFrameIx) + ".png");
						captureScreenshot(outPath, "Script");
						++m_scriptedInput.nextCaptureIndex;
					}
				}

				m_surface->present(std::move(swapchainLock), presentInfo);
			}
			firstFrame = false;

}


