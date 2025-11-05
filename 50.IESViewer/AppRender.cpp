// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "app_resources/common.hlsl"

IQueue::SSubmitInfo::SSemaphoreInfo IESViewer::renderFrame(const std::chrono::microseconds nextPresentationTimestamp)
{
    const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;
    auto* const cb = m_cmdBuffers.data()[resourceIx].get();

    auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());

#ifdef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
    IGPUFramebuffer* const fb2D = nullptr;
    auto* const fb3D = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex);
#else
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
        mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); processMouse(events); for (const auto& e : events) captured.mouse.emplace_back(e); }, m_logger.get());
        keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); processKeyboard(events); for (const auto& e : events) captured.keyboard.emplace_back(e); }, m_logger.get());
        camera.endInputProcessing(nextPresentationTimestamp);

        const auto cursorPosition = m_window->getCursorControl()->getPosition();
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

    auto& ies = m_assets[m_activeAssetIx];
    const auto* profile = ies.getProfile();
    PushConstants pc;
    {
        pc.vAnglesBDA = ies.buffers.vAngles->getDeviceAddress();
        pc.hAnglesBDA = ies.buffers.hAngles->getDeviceAddress();
        pc.dataBDA = ies.buffers.data->getDeviceAddress();

        pc.maxIValue = profile->getMaxCandelaValue();
        pc.vAnglesCount = profile->getVertAngles().size();
        pc.hAnglesCount = profile->getHoriAngles().size();
        pc.dataCount = profile->getData().size();

        pc.zAngleDegreeRotation = ies.zDegree;
        pc.mode = ies.mode;
        pc.texIx = m_activeAssetIx;
    }

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
    auto* image = ies.getActiveImage();

    // Compute
    {
        cb->beginDebugMarker("IES::compute");
        IES::barrier<IImage::LAYOUT::GENERAL>(cb, image);
        auto* layout = m_computePipeline->getLayout();
        cb->bindComputePipeline(m_computePipeline.get());
        cb->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout, 0, 1, &descriptor);
        cb->pushConstants(layout, layout->getPushConstantRanges().begin()->stageFlags, 0, sizeof(pc), &pc);
        const auto xGroups = (ies.getProfile()->getOptimalIESResolution().x - 1u) / WORKGROUP_DIMENSION + 1u;
        cb->dispatch(xGroups, xGroups, 1);
        cb->endDebugMarker();
    }

    // Graphics
    {
        IES::barrier<IImage::LAYOUT::READ_ONLY_OPTIMAL>(cb, image);

#ifdef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
        asset::VkExtent3D extent = { m_window->getWidth(), m_window->getHeight() };
#else
        auto extent = fb2D->getCreationParameters().colorAttachments[0u]->getCreationParameters().image->getCreationParameters().extent;
#endif

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
            cb->pushConstants(layout, layout->getPushConstantRanges().begin()->stageFlags, 0, sizeof(pc), &pc);
            ext::FullScreenTriangle::recordDrawCall(cb);
        }
        cb->endRenderPass();
        cb->endDebugMarker();
#endif

        const IGPUCommandBuffer::SClearColorValue d3clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
        auto info3D = info;
        info3D.colorClearValues = &d3clearValue; // tmp
        info3D.depthStencilClearValues = &depthValue;
        info3D.framebuffer = fb3D;
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

            auto resolution = profile->getOptimalIESResolution();
            const auto iesParams = CSimpleIESRenderer::SIESParams({ .radius = 100.f, .resX = resolution.x, .resY = resolution.y, .ds = m_descriptors[0u].get(), .texID = (uint32_t)m_activeAssetIx });

            // tear down scene every frame
            m_renderer->m_instances[0].packedGeo = m_renderer->getGeometries().data() + m_activeAssetIx;
            m_renderer->render(cb, viewParams, iesParams);
        }
        cb->endRenderPass();
        cb->endDebugMarker();

#ifndef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
        cb->beginDebugMarker("IES::graphics ImGUI");

        viewport.width = m_window->getWidth(); viewport.height = m_window->getHeight();
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