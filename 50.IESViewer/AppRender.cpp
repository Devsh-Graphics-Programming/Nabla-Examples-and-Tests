// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "app_resources/common.hlsl"

bool IESViewer::recreate3DPlotFramebuffers(uint32_t width, uint32_t height)
{
#ifdef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
    return true;
#else
    if (width == 0u || height == 0u)
        return false;

    if (width == m_plot3DWidth && height == m_plot3DHeight)
        return true;

    m_device->waitIdle();
    m_plot3DWidth = width;
    m_plot3DHeight = height;

    auto* scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
    auto renderpass = smart_refctd_ptr<IGPURenderpass>(scRes->getRenderpass());

    for (uint32_t i = 0u; i < m_frameBuffers3D.size(); ++i)
    {
        auto& fb3D = m_frameBuffers3D[i];
        auto ixs = std::to_string(i);

        auto color = createImageView(width, height, EF_R8G8B8A8_SRGB, "[3D Plot]: framebuffer[" + ixs + "].color attachement", IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT, IImage::EAF_COLOR_BIT);
        if (!color)
            return false;

        auto depth = createImageView(width, height, EF_D16_UNORM, "[3D Plot]: framebuffer[" + ixs + "].depth attachement", IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT, IGPUImage::EAF_DEPTH_BIT);
        if (!depth)
            return false;

        fb3D = m_device->createFramebuffer
        (
            { {
                .renderpass = renderpass,
                .depthStencilAttachments = &depth.get(),
                .colorAttachments = &color.get(),
                .width = width,
                .height = height
            } }
        );
        if (!fb3D)
            return false;
    }

    if (ui.it && ui.descriptor)
    {
        std::array<IGPUDescriptorSet::SDescriptorInfo, 1u + 2u * device_base_t::MaxFramesInFlight> infos;
        for (auto& it : infos)
            it.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

        auto* ix = infos.data();
        ix->desc = smart_refctd_ptr<IGPUImageView>(ui.it->getFontAtlasView());
        ++ix;
        for (uint8_t i = 0u; i < device_base_t::MaxFramesInFlight; ++i, ++ix)
            ix->desc = m_frameBuffers2D[i]->getCreationParameters().colorAttachments[0u];
        for (uint8_t i = 0u; i < device_base_t::MaxFramesInFlight; ++i, ++ix)
            ix->desc = m_frameBuffers3D[i]->getCreationParameters().colorAttachments[0u];

        const auto texturesBinding = ui.it->getCreationParameters().resources.texturesInfo.bindingIx;
        auto writes = std::to_array({ IGPUDescriptorSet::SWriteDescriptorSet{
            .dstSet = ui.descriptor->getDescriptorSet(),
            .binding = texturesBinding,
            .arrayElement = ext::imgui::UI::FontAtlasTexId,
            .count = static_cast<uint32_t>(infos.size()),
            .info = infos.data()
        } });

        if (!m_device->updateDescriptorSets(writes, {}))
            return false;
    }

    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(m_cameraFovDeg), float(width) / float(height), 0.1f, 10000.0f);
    camera.setProjectionMatrix(projectionMatrix);

    return true;
#endif
}

IQueue::SSubmitInfo::SSemaphoreInfo IESViewer::renderFrame(const std::chrono::microseconds nextPresentationTimestamp)
{
    const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;
    auto* const cb = m_cmdBuffers.data()[resourceIx].get();

    auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());

    const bool windowFocused = m_window->hasInputFocus() || m_window->hasMouseFocus();
    if (!windowFocused && m_cameraControlEnabled)
        m_cameraControlEnabled = false;
    const bool wantCameraControl = m_cameraControlEnabled && windowFocused;

    const uint32_t windowWidth = m_window->getWidth();
    const uint32_t windowHeight = m_window->getHeight();
    if (windowWidth == 0u || windowHeight == 0u || m_window->isMinimized())
        return {};

    if (m_cameraControlApplied != wantCameraControl)
    {
        m_cameraControlApplied = wantCameraControl;
        const float moveSpeed = wantCameraControl ? m_cameraMoveSpeed : 0.0f;
        const float rotateSpeed = wantCameraControl ? m_cameraRotateSpeed : 0.0f;
        camera.setMoveSpeed(moveSpeed);
        camera.setRotateSpeed(rotateSpeed);
    }



#ifdef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
    IGPUFramebuffer* const fb2D = nullptr;
    auto* const fb3D = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex);
#else
    const uint32_t desired3DWidth = windowWidth;
    const uint32_t desired3DHeight = windowHeight;
    if (!recreate3DPlotFramebuffers(desired3DWidth, desired3DHeight))
        return {};

    auto* const fb2D = m_frameBuffers2D[resourceIx].get();
    auto* const fb3D = m_frameBuffers3D[resourceIx].get();
#endif 

    cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
    cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

    m_inputSystem->getDefaultMouse(&mouse);
    m_inputSystem->getDefaultKeyboard(&keyboard);
    {
        struct
        {
            std::vector<SMouseEvent> mouse{}; std::vector<SKeyboardEvent> keyboard{};
        } captured;

        camera.beginInputProcessing(nextPresentationTimestamp);
        if (windowFocused)
        {
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
            {
                if (wantCameraControl)
                    camera.mouseProcess(events);
                processMouse(events);
                for (const auto& e : events)
                    captured.mouse.emplace_back(e);
            }, m_logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
            {
                camera.keyboardProcess(events);
                processKeyboard(events);
                for (const auto& e : events)
                    captured.keyboard.emplace_back(e);
            }, m_logger.get());
        }
        else
        {
            mouse.consumeEvents([&](const IMouseEventChannel::range_t&) -> void {}, m_logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t&) -> void {}, m_logger.get());
        }
        camera.endInputProcessing(nextPresentationTimestamp);

        {
            const float maxRadius = m_plotRadius * 0.98f;
            const float clampRadius = maxRadius * 0.999f;
            auto pos = camera.getPosition();
            const float dist = core::length(pos)[0];
            if (dist > maxRadius)
            {
                auto forward = camera.getTarget() - pos;
                pos.makeSafe3D();
                pos = core::normalize(pos) * clampRadius;
                camera.setPosition(pos);
                camera.setTarget(pos + forward);
            }
        }

        auto* cursorControl = m_window->getCursorControl();
        const auto cursorPosition = cursorControl->getPosition();
        const int32_t windowX = m_window->getX();
        const int32_t windowY = m_window->getY();
        const int32_t windowW = static_cast<int32_t>(m_window->getWidth());
        const int32_t windowH = static_cast<int32_t>(m_window->getHeight());
        const bool cursorInsideWindow =
            cursorPosition.x >= windowX && cursorPosition.x < windowX + windowW &&
            cursorPosition.y >= windowY && cursorPosition.y < windowY + windowH;
        cursorControl->setVisible(!(cursorInsideWindow || m_cameraControlApplied));
        ext::imgui::UI::SUpdateParameters params =
        {
            .mousePosition = float32_t2(cursorPosition.x,cursorPosition.y) - float32_t2(m_window->getX(),m_window->getY()),
            .displaySize = {m_window->getWidth(),m_window->getHeight()},
            .mouseEvents = captured.mouse,
            .keyboardEvents = captured.keyboard
        };

#ifndef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
        ui.it->update(params);
#endif
    }

    if (m_cameraControlApplied)
    {
        if (auto* cursor = m_window->getCursorControl())
            cursor->setRelativePosition(m_window.get(), {0.5f, 0.5f});
    }

    auto& ies = m_assets[m_activeAssetIx];
    const auto* profile = ies.getProfile();
	const auto& accessor = profile->getAccessor();
    const auto pc = nbl::hlsl::this_example::ies::CdcPC 
	{
        .hAnglesBDA = ies.buffers.hAngles->getDeviceAddress(),
        .vAnglesBDA = ies.buffers.vAngles->getDeviceAddress(),
        .dataBDA = ies.buffers.data->getDeviceAddress(),
		.txtInfoBDA = ies.buffers.textureInfo.buffer->getDeviceAddress(),
		.mode = mode.view,
		.texIx = (uint32_t)m_activeAssetIx,
        .hAnglesCount = accessor.hAnglesCount(),
        .vAnglesCount = accessor.vAnglesCount(),
        .zAngleDegreeRotation = ies.zDegree,
		.properties = accessor.getProperties()
	};

    for (auto& buffer : { ies.buffers.data, ies.buffers.hAngles, ies.buffers.vAngles }) // flush request for sanity
    {
        auto bound = buffer->getBoundMemory();
        if (bound.memory->haveToMakeVisible())
        {
            const ILogicalDevice::MappedMemoryRange range(bound.memory, bound.offset, buffer->getSize());
            m_device->flushMappedMemoryRanges(1, &range);
        }
    }

    auto* const descriptor = m_descriptors[0].get();
    auto* image = ies.getActiveImage(IES::EM_OCTAHEDRAL_MAP);

    bool needCompute = true;
    if (m_activeAssetIx < m_candelaDirty.size())
        needCompute = m_candelaDirty[m_activeAssetIx];

    if (needCompute)
    {
        cb->beginDebugMarker("IES::compute");
        IES::barrier<IImage::LAYOUT::GENERAL>(cb, image);
        auto* layout = m_computePipeline->getLayout();
        cb->bindComputePipeline(m_computePipeline.get());
        cb->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout, 0, 1, &descriptor);
        cb->pushConstants(layout, layout->getPushConstantRanges().begin()->stageFlags, offsetof(hlsl::this_example::ies::PushConstants, cdc), sizeof(pc), &pc);
        const auto xGroups = (ies.getProfile()->getAccessor().properties.optimalIESResolution.x - 1u) / WORKGROUP_DIMENSION + 1u;
        cb->dispatch(xGroups, xGroups, 1);
        IES::barrier<IImage::LAYOUT::READ_ONLY_OPTIMAL>(cb, image);
        cb->endDebugMarker();
        if (m_activeAssetIx < m_candelaDirty.size())
            m_candelaDirty[m_activeAssetIx] = false;
    }

    // Graphics
    {
#ifdef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
        asset::VkExtent3D extent = { m_window->getWidth(), m_window->getHeight() };
#else
        auto extent = fb2D->getCreationParameters().colorAttachments[0u]->getCreationParameters().image->getCreationParameters().extent;
#endif
        const uint32_t plotHeight = extent.height / 2u;

        asset::SViewport viewport;
        {
            viewport.minDepth = 1.f;
            viewport.maxDepth = 0.f;
            viewport.x = 0u;
            viewport.y = 0u;
            viewport.width = extent.width;
            viewport.height = extent.height;
        }
        cb->setViewport(0u, 1u, &viewport);

        VkRect2D scissor =
        {
            .offset = { 0, 0 },
            .extent = { extent.width, extent.height },
        };
        cb->setScissor(0u, 1u, &scissor);

        VkRect2D currentRenderArea =
        {
            .offset = {0,0},
            .extent = {extent.width,extent.height}
        };

        const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
        const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
        IGPUCommandBuffer::SRenderpassBeginInfo info =
        {
            .framebuffer = fb2D,
            .colorClearValues = &clearValue,
            .depthStencilClearValues = &depthValue,
            .renderArea = currentRenderArea
        };

#ifndef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
        cb->beginDebugMarker("IES::graphics 2D plot");
        cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
        {
            auto* layout = m_graphicsPipeline->getLayout();
            cb->bindGraphicsPipeline(m_graphicsPipeline.get());
            cb->bindDescriptorSets(EPBP_GRAPHICS, layout, 0, 1, &descriptor);
            asset::SViewport viewport2D = viewport;
            viewport2D.width = static_cast<float>(extent.width);
            viewport2D.height = static_cast<float>(plotHeight);
            VkRect2D scissor2D = scissor;
            scissor2D.extent = { extent.width, plotHeight };

            auto pc2D = pc;
            pc2D.mode = mode.view;
            cb->setViewport(0u, 1u, &viewport2D);
            cb->setScissor(0u, 1u, &scissor2D);
            cb->pushConstants(layout, layout->getPushConstantRanges().begin()->stageFlags, 0, sizeof(pc2D), &pc2D);
            ext::FullScreenTriangle::recordDrawCall(cb);

            if (m_showOctaMapPreview)
            {
                viewport2D.y = static_cast<float>(plotHeight);
                scissor2D.offset.y = static_cast<int32_t>(plotHeight);
                pc2D.mode = IES::EM_OCTAHEDRAL_MAP;
                cb->setViewport(0u, 1u, &viewport2D);
                cb->setScissor(0u, 1u, &scissor2D);
                cb->pushConstants(layout, layout->getPushConstantRanges().begin()->stageFlags, 0, sizeof(pc2D), &pc2D);
                ext::FullScreenTriangle::recordDrawCall(cb);
            }
        }
        cb->endRenderPass();
        cb->endDebugMarker();
#endif

        const IGPUCommandBuffer::SClearColorValue d3clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
        auto info3D = info;
        info3D.colorClearValues = &d3clearValue; // tmp
        info3D.depthStencilClearValues = &depthValue;
        info3D.framebuffer = fb3D;
        auto extent3D = fb3D->getCreationParameters().colorAttachments[0u]->getCreationParameters().image->getCreationParameters().extent;
        viewport.width = extent3D.width;
        viewport.height = extent3D.height;
        cb->setViewport(0u, 1u, &viewport);
        scissor.extent = { extent3D.width, extent3D.height };
        cb->setScissor(0u, 1u, &scissor);
        currentRenderArea.extent = { extent3D.width, extent3D.height };
        info3D.renderArea = currentRenderArea;
        cb->beginDebugMarker("IES::graphics 3D plot");
        cb->beginRenderPass(info3D, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
        {
            float32_t3x4 viewMatrix;
            float32_t4x4 viewProjMatrix;
            // TODO: get rid of legacy matrices
            {
                memcpy(&viewMatrix, camera.getViewMatrix().pointer(), sizeof(viewMatrix));
                memcpy(&viewProjMatrix, camera.getConcatenatedMatrix().pointer(), sizeof(viewProjMatrix));
            }
            const auto viewParams = CSimpleIESRenderer::SViewParams(viewMatrix, viewProjMatrix);
            const auto iesParams = CSimpleIESRenderer::SIESParams({ .radius = m_plotRadius, .ds = m_descriptors[0u].get(), .texID = (uint16_t)m_activeAssetIx, .mode = mode.sphere.value, .wireframe = m_wireframeEnabled });

            // tear down scene every frame
            m_renderer->m_instances[0].packedGeo = m_renderer->getGeometries().data() + m_activeAssetIx;
            m_renderer->render(cb, viewParams, iesParams);
        }
        cb->endRenderPass();
        cb->endDebugMarker();

#ifndef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
        cb->beginDebugMarker("IES::graphics ImGUI");

        viewport.width = m_window->getWidth(); viewport.height = m_window->getHeight();
        cb->setViewport(0u, 1u, &viewport);
        scissor.extent = { m_window->getWidth(), m_window->getHeight() };
        cb->setScissor(0u, 1u, &scissor);
        currentRenderArea.extent = { m_window->getWidth(),m_window->getHeight() };
        
        info.framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex);
        info.renderArea = currentRenderArea;

        cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
        {
            auto* imgui = ui.it.get();
            auto* pipeline = imgui->getPipeline();
            cb->bindGraphicsPipeline(pipeline);
            const auto* ds = ui.descriptor->getDescriptorSet();
            cb->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), imgui->getCreationParameters().resources.texturesInfo.setIx, 1u, &ds);
            const ISemaphore::SWaitInfo wait = { .semaphore = m_semaphore.get(),.value = m_realFrameIx + 1u };
            if (!imgui->render(cb, wait))
            {
                m_logger->log("TODO: need to present acquired image before bailing because its already acquired.", ILogger::ELL_ERROR);
                return {};
            }
        }
        cb->endRenderPass();
        cb->endDebugMarker();
#endif
        cb->end();
    }

    IQueue::SSubmitInfo::SSemaphoreInfo retval =
    {
        .semaphore = m_semaphore.get(),
        .value = ++m_realFrameIx,
        .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
    };
    const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
    {
        {.cmdbuf = cb }
    };
    const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
    {
        {
            .semaphore = device_base_t::getCurrentAcquire().semaphore,
            .value = device_base_t::getCurrentAcquire().acquireCount,
            .stageMask = PIPELINE_STAGE_FLAGS::NONE
        }
    };
    const IQueue::SSubmitInfo infos[] =
    {
        {
            .waitSemaphores = acquired,
            .commandBuffers = commandBuffers,
            .signalSemaphores = {&retval,1}
        }
    };

    if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
    {
        retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
        m_realFrameIx--;
    }

    std::string caption = "[Nabla Engine] IES Viewer";
    {
        m_window->setCaption(caption);
    }
    return retval;
}

const video::IGPURenderpass::SCreationParams::SSubpassDependency* IESViewer::getDefaultSubpassDependencies() const
{
    // Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
    const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
    {
        // wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
        {
            .srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
            .dstSubpass = 0,
            .memoryBarrier = {
            // last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
            .srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
            // don't want any writes to be available, we'll clear 
            .srcAccessMask = ACCESS_FLAGS::NONE,
            // destination needs to wait as early as possible
            // TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
            .dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
            // because depth and color get cleared first no read mask
            .dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
        }
        // leave view offsets and flags default
        },
        // color from ATTACHMENT_OPTIMAL to PRESENT_SRC
        {
            .srcSubpass = 0,
            .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
            .memoryBarrier = {
            // last place where the color can get modified, depth is implicitly earlier
            .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
            // only write ops, reads can't be made available
            .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
            // spec says nothing is needed when presentation is the destination
        }
        // leave view offsets and flags default
        },
        IGPURenderpass::SCreationParams::DependenciesEnd
    };
    return dependencies;
}
