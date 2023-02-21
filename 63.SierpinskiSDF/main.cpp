// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <cstdio>
#include <iostream>
#include <CommonAPI.h>
#include <nabla.h>

#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../common/Camera.hpp"

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;

/*
    Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

#include "nbl/nblpack.h" //! Designed for use with interface blocks declared with `layout (row_major, std140)`
struct NBL_CAMERA_VECTORS {
    nbl::core::vectorSIMDf cameraPosition;
    nbl::core::vectorSIMDf cameraTarget;
};
#include "nbl/nblunpack.h"

class SierpinskiSDF : public ApplicationBase 
{
    constexpr static uint32_t WIN_W = 1280;
    constexpr static uint32_t WIN_H = 720;
    constexpr static uint32_t SC_IMG_COUNT = 3u;
    constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
    constexpr static size_t NBL_FRAMES_TO_AVERAGE = 100ull;

    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

    public:
        nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
        nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
        nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
        nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
        nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
        nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
        nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
        nbl::video::IPhysicalDevice *physicalDevice;
        std::array<nbl::video::IGPUQueue *, CommonAPI::InitOutput::MaxQueuesCount>
        queues = {nullptr, nullptr, nullptr, nullptr};
        nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
        nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
        nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> fbo;
        std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>,CommonAPI::InitOutput::MaxFramesInFlight>,CommonAPI::InitOutput::MaxQueuesCount> commandPools; // TODO: Multibuffer and reset the commandpools
        nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
        nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
        nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
        nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
        nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

        nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

        core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer;
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuRenderpassIndependentPipeline;
        core::smart_refctd_ptr<IGPUGraphicsPipeline> gpuGraphicsPipeline;

        core::smart_refctd_ptr<IGPUBuffer> gpuubo;
        core::smart_refctd_ptr<IGPUDescriptorSet> gpuDescriptorSet1;
   
        core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
        core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
        core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
        core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

        nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

        CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> mouse;
        CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> keyboard;
        Camera camera;

        int resourceIx;
        uint32_t acquiredNextFBO = {};

        std::chrono::system_clock::time_point lastTime;
        bool frameDataFilled = false;
        size_t frame_count = 0ull;
        double time_sum = 0;
        double dtList[NBL_FRAMES_TO_AVERAGE] = {};

        auto createDescriptorPool(const uint32_t textureCount) 
        {
            constexpr uint32_t maxItemCount = 256u;
            {
                nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
                poolSize.count = textureCount;
                poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
                return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
            }
        }

        void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow> &&wnd) override 
        {
            window = std::move(wnd);
        }

        void setSystem(core::smart_refctd_ptr<nbl::system::ISystem> &&s) override 
        {
            system = std::move(s);
        }
        
        nbl::ui::IWindow *getWindow() override 
        {
            return window.get();
        
        }
           
        video::IAPIConnection *getAPIConnection() override 
        {
            return apiConnection.get();
        }


        video::ILogicalDevice *getLogicalDevice() override 
        {
            return logicalDevice.get();
        }

        video::IGPURenderpass *getRenderpass() override 
        {
            return renderpass.get(); 
        }

        void setSurface(core::smart_refctd_ptr<video::ISurface> &&s) override 
        {
            surface = std::move(s);
        }

        void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>> &f) override 
        {
            for (int i = 0; i < f.size(); i++) 
                fbo->begin()[i] = core::smart_refctd_ptr(f[i]);
        }

        void setSwapchain(core::smart_refctd_ptr<video::ISwapchain> &&s) override 
        {
            swapchain = std::move(s);
        }

        uint32_t getSwapchainImageCount() override 
        {
            return swapchain->getImageCount();
        }

        virtual nbl::asset::E_FORMAT getDepthFormat() override 
        {
            return nbl::asset::EF_D32_SFLOAT;
        }

        APP_CONSTRUCTOR(SierpinskiSDF)

        void onAppInitialized_impl() override 
        {
            const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
            CommonAPI::InitParams initParams;
            initParams.window = core::smart_refctd_ptr(window);
            initParams.apiType = video::EAT_VULKAN;
            initParams.appName = {"63.SierpinskiSDFApp"};
            initParams.framesInFlight = FRAMES_IN_FLIGHT;
            initParams.windowWidth = WIN_W;
            initParams.windowHeight = WIN_H;
            initParams.swapchainImageCount = SC_IMG_COUNT;
            initParams.swapchainImageUsage = swapchainImageUsage;
            initParams.depthFormat = nbl::asset::EF_D32_SFLOAT;
            auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

            window = std::move(initParams.window);
            windowCb = std::move(initParams.windowCb);
            apiConnection = std::move(initOutput.apiConnection);
            surface = std::move(initOutput.surface);
            utilities = std::move(initOutput.utilities);
            logicalDevice = std::move(initOutput.logicalDevice);
            physicalDevice = initOutput.physicalDevice;
            queues = std::move(initOutput.queues);
            renderpass = std::move(initOutput.renderToSwapchainRenderpass);
            commandPools = std::move(initOutput.commandPools);
            system = std::move(initOutput.system);
            assetManager = std::move(initOutput.assetManager);
            cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
            logger = std::move(initOutput.logger);
            inputSystem = std::move(initOutput.inputSystem);
            m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

            CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, WIN_W, WIN_H, swapchain);
            assert(swapchain);

            fbo = CommonAPI::createFBOWithSwapchainImages(swapchain->getImageCount(), WIN_W, WIN_H, logicalDevice, swapchain, renderpass, nbl::asset::EF_D32_SFLOAT);

            auto descriptorPool = createDescriptorPool(1u);

            const size_t ds1UboBinding = 0;

            IGPUDescriptorSetLayout::SBinding gpuUboBinding;
            gpuUboBinding.count = 1u;
            gpuUboBinding.binding = ds1UboBinding;
            gpuUboBinding.stageFlags = static_cast<asset::ICPUShader::E_SHADER_STAGE>(asset::ICPUShader::ESS_VERTEX | asset::ICPUShader::ESS_FRAGMENT);
            gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

            auto gpuDs1Layout = logicalDevice->createDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
            {
                IGPUBuffer::SCreationParams creationParams = {};
                creationParams.usage = core::bitflag(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
                creationParams.size = sizeof(NBL_CAMERA_VECTORS);
                gpuubo = logicalDevice->createBuffer(std::move(creationParams));

                IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuubo->getMemoryReqs();
                memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
                logicalDevice->allocate(memReq, gpuubo.get());
            }

            gpuDescriptorSet1 = logicalDevice->createDescriptorSet(descriptorPool.get(), gpuDs1Layout);
            {
                video::IGPUDescriptorSet::SWriteDescriptorSet write;
                write.dstSet = gpuDescriptorSet1.get();
                write.binding = ds1UboBinding;
                write.count = 1u;
                write.arrayElement = 0u;
                write.descriptorType = asset::EDT_UNIFORM_BUFFER;
                video::IGPUDescriptorSet::SDescriptorInfo info;
                {
                    info.desc = gpuubo;
                    info.buffer.offset = 0ull;
                    info.buffer.size = sizeof(SBasicViewParameters);
                }
                write.info = &info;
                logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
            }

            auto fstProtoPipeline = ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams, 0u);
            auto constants = std::get<asset::SPushConstantRange>(fstProtoPipeline);
            auto gpuPipelineLayout = logicalDevice->createPipelineLayout(&constants, &constants + 1, nullptr, core::smart_refctd_ptr(gpuDs1Layout), nullptr, nullptr);

            auto fragmentShaderBundle = assetManager->getAsset("../shader.frag", {});
            {
                bool status = !fragmentShaderBundle.getContents().empty();
                assert(status);
            }

            auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(fragmentShaderBundle.getContents().begin()[0]);
            smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
            {
                auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
                if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);

                gpuFragmentShader = (*gpu_array)[0];
            }
            gpuRenderpassIndependentPipeline = ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));

            nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
            graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(gpuRenderpassIndependentPipeline.get());
            graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
            gpuGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

            const auto &graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
            for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++) 
            {
                logicalDevice->createCommandBuffers(graphicsCommandPools[i].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers + i);
                imageAcquire[i] = logicalDevice->createSemaphore();
                renderFinished[i] = logicalDevice->createSemaphore();
            }

            matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.01f, 5000.0f);
            camera = Camera(core::vectorSIMDf(0.f, 0.f, 6.f), core::vectorSIMDf(0.f, 0.f, -1.f), projectionMatrix, 10.f, 1.f);
        }

        void workLoopBody() override 
        {
            ++resourceIx;

            if (resourceIx >= FRAMES_IN_FLIGHT)
                resourceIx = 0;

            auto &commandBuffer = commandBuffers[resourceIx];
            auto &fence = frameComplete[resourceIx];

            if (fence) 
            {
                logicalDevice->blockForFences(1u, &fence.get());
                logicalDevice->resetFences(1u, &fence.get());
            } 
            else
                fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

            auto renderStart = std::chrono::system_clock::now();
            const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
            lastTime = renderStart;
            { // Calculate Simple Moving Average for FrameTime
                time_sum -= dtList[frame_count];
                time_sum += renderDt;
                dtList[frame_count] = renderDt;
                frame_count++;
                if (frame_count >= NBL_FRAMES_TO_AVERAGE)
                {
                    frameDataFilled = true;
                    frame_count = 0;
                }

            }
            const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

            auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
            auto nextPresentationTime = renderStart + averageFrameTimeDuration;
            auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

            inputSystem->getDefaultMouse(&mouse);
            inputSystem->getDefaultKeyboard(&keyboard);

            camera.beginInputProcessing(nextPresentationTimeStamp);
            mouse.consumeEvents([&](const ui::IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
            keyboard.consumeEvents([&](const ui::IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
            camera.endInputProcessing(nextPresentationTimeStamp);

            commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
            commandBuffer->begin(IGPUCommandBuffer::EU_NONE);

            NBL_CAMERA_VECTORS uboData;
            uboData.cameraPosition = camera.getPosition();
            uboData.cameraTarget = camera.getTarget();
            commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), &uboData);

            asset::SViewport viewport;
            viewport.minDepth = 1.f;
            viewport.maxDepth = 0.f;
            viewport.x = 0u;
            viewport.y = 0u;
            viewport.width = WIN_W;
            viewport.height = WIN_H;
            commandBuffer->setViewport(0u, 1u, &viewport);
            VkRect2D scissor;
            scissor.offset = {0u, 0u};
            scissor.extent = {WIN_W, WIN_H};
            commandBuffer->setScissor(0u, 1u, &scissor);

            swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

            nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
            {
                VkRect2D area;
                area.offset = {0, 0};
                area.extent = {WIN_W, WIN_H};
                asset::SClearValue clear[2] = {};
                clear[0].color.float32[0] = 0.f;
                clear[0].color.float32[1] = 0.f;
                clear[0].color.float32[2] = 0.f;
                clear[0].color.float32[3] = 1.f;
                clear[1].depthStencil.depth = 0.f;

                beginInfo.clearValueCount = 2u;
                beginInfo.framebuffer = fbo->begin()[acquiredNextFBO];
                beginInfo.renderpass = renderpass;
                beginInfo.renderArea = area;
                beginInfo.clearValues = clear;
            }

            commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), 0u);
            ext::FullScreenTriangle::recordDrawCalls(gpuGraphicsPipeline, 0u, swapchain->getPreTransform(), commandBuffer.get());
            commandBuffer->endRenderPass();
            commandBuffer->end();

            CommonAPI::Submit(logicalDevice.get(), commandBuffer.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
            CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
        }

        bool keepRunning() override 
        { 
            return windowCb->isWindowOpen(); 
        }
};

NBL_COMMON_API_MAIN(SierpinskiSDF)