// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/CentralLimitBoxBlur/CBlurPerformer.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;
using BlurClass = ext::CentralLimitBoxBlur::CBlurPerformer;

#define FATAL_LOG(x, ...) {logger->log(##x, system::ILogger::ELL_ERROR, __VA_ARGS__); exit(-1);}

class BlurTestApp : public ApplicationBase
{
    constexpr static inline uint32_t FRAMES_IN_FLIGHT = 5u;
    constexpr static inline uint32_t SC_IMG_COUNT = 3u;
    constexpr static inline uint64_t MAX_TIMEOUT = 99999999999999ull;
    constexpr static inline uint32_t WIN_W = 1024u;
    constexpr static inline uint32_t WIN_H = 1024u;

public:
	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
        CommonAPI::InitWithDefaultExt(initOutput, video::EAT_VULKAN, "Blur", FRAMES_IN_FLIGHT, WIN_W, WIN_H, SC_IMG_COUNT, asset::IImage::EUF_COLOR_ATTACHMENT_BIT);

		system = std::move(initOutput.system);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		commandPools = std::move(initOutput.commandPools);
        renderpass = std::move(initOutput.renderpass);
        fbos = std::move(initOutput.fbo);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

        const char* inImagePath = "../cube_face.png";
        core::smart_refctd_ptr<IGPUImage> inImage = nullptr;
        {
            asset::IAssetLoader::SAssetLoadParams loadParams(0, nullptr,
                static_cast<nbl::asset::IAssetLoader::E_CACHING_FLAGS>(nbl::asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & nbl::asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL));

            auto cpuImageBundle = assetManager->getAsset(inImagePath, loadParams);
            auto cpuImageContents = cpuImageBundle.getContents();
            if (cpuImageContents.empty())
                FATAL_LOG("Failed to load the image at path: %s\n", inImagePath);

            auto inImageCPU = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuImageContents.begin());
            inImageCPU->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);

            cpu2gpuParams.beginCommandBuffers();
            auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&inImageCPU, &inImageCPU + 1ull, cpu2gpuParams);
            cpu2gpuParams.waitForCreationToComplete();
            if (!gpuArray || gpuArray->size() < 1ull || (!(*gpuArray)[0]))
                FATAL_LOG("Cannot convert the input CPU image to GPU image\n");

            inImage = gpuArray->begin()[0];

            IGPUImageView::SCreationParams viewCreationParams;
            viewCreationParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
            viewCreationParams.image = inImage;
            viewCreationParams.viewType = IGPUImageView::ET_2D;
            viewCreationParams.format = viewCreationParams.image->getCreationParameters().format;
            viewCreationParams.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
            viewCreationParams.subresourceRange.baseMipLevel = 0;
            viewCreationParams.subresourceRange.levelCount = 1;
            viewCreationParams.subresourceRange.baseArrayLayer = 0;
            viewCreationParams.subresourceRange.layerCount = 1;
            m_inImageView = logicalDevice->createImageView(std::move(viewCreationParams));
        }

        const asset::VkExtent3D blurDSFactor = {2u, 2u, 1u};

        const auto& inDim = m_inImageView->getCreationParameters().image->getCreationParameters().extent;
        asset::VkExtent3D outDim;
        for (uint32_t i = 0; i < 3u; ++i)
            (&outDim.width)[i] = ((&inDim.width)[i]) / ((&blurDSFactor.width)[i]);

        {
            IGPUImage::SCreationParams imageCreationParams = m_inImageView->getCreationParameters().image->getCreationParameters();
            imageCreationParams.extent = outDim;
            imageCreationParams.format = asset::EF_R16G16B16A16_SFLOAT;
            imageCreationParams.usage = static_cast<IGPUImage::E_USAGE_FLAGS>(IGPUImage::EUF_STORAGE_BIT | IGPUImage::EUF_SAMPLED_BIT);
            imageCreationParams.mipLevels = 1; // Asset converter blows up the mipLevels all the way to 10, so reset it back to 1.
            m_outImage = logicalDevice->createImage(std::move(imageCreationParams));

            auto memReqs = m_outImage->getMemoryReqs();
            memReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            logicalDevice->allocate(memReqs, m_outImage.get());

            // transition layout to GENERAL
            {
                core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
                logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE][0].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

                auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

                video::IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
                barrier.oldLayout = EIL_UNDEFINED;
                barrier.newLayout = EIL_GENERAL;
                barrier.srcQueueFamilyIndex = ~0u;
                barrier.dstQueueFamilyIndex = ~0u;
                barrier.image = m_outImage;
                barrier.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
                barrier.subresourceRange.levelCount = m_outImage->getCreationParameters().mipLevels;
                barrier.subresourceRange.layerCount = m_outImage->getCreationParameters().arrayLayers;

                cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
                cmdbuf->end();

                video::IGPUQueue::SSubmitInfo submitInfo = {};
                submitInfo.commandBufferCount = 1u;
                submitInfo.commandBuffers = &cmdbuf.get();
                queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());
                logicalDevice->blockForFences(1u, &fence.get());
            }

            IGPUImageView::SCreationParams viewCreationParams = m_inImageView->getCreationParameters();
            viewCreationParams.image = m_outImage;
            viewCreationParams.format = viewCreationParams.image->getCreationParameters().format;
            m_outImageView = logicalDevice->createImageView(std::move(viewCreationParams));
        }

        const auto scratchBufferSize = BlurClass::getPassOutputBufferSize(outDim, asset::getFormatChannelCount(inImage->getCreationParameters().format));
        {
            IGPUBuffer::SCreationParams creationParams = {};
            creationParams.size = scratchBufferSize;
            creationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            m_scratchBuffer = logicalDevice->createBuffer(creationParams);

            auto memReqs = m_scratchBuffer->getMemoryReqs();
            memReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            logicalDevice->allocate(memReqs, m_scratchBuffer.get());
        }

        const auto pcRange = BlurClass::getDefaultPushConstantRanges();

        core::smart_refctd_ptr<IGPUPipelineLayout> pipelineLayoutHorizontal = nullptr;
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayoutHorizontal = nullptr;
        {
            const uint32_t count = 2u;
            IGPUDescriptorSetLayout::SBinding binding[count] =
            {
                {
                    0u,
                    EDT_COMBINED_IMAGE_SAMPLER,
                    1u,
                    IShader::ESS_COMPUTE,
                    nullptr
                },
                {
                    1u,
                    EDT_STORAGE_BUFFER,
                    1u,
                    IShader::ESS_COMPUTE,
                    nullptr
                }
            };
            dsLayoutHorizontal = logicalDevice->createDescriptorSetLayout(binding, binding + count);
            pipelineLayoutHorizontal = logicalDevice->createPipelineLayout(pcRange.begin(), pcRange.end(), core::smart_refctd_ptr(dsLayoutHorizontal));
        }

        core::smart_refctd_ptr<IGPUPipelineLayout> pipelineLayoutVertical = nullptr;
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayoutVertical = nullptr;
        {
            const uint32_t count = 2u;
            IGPUDescriptorSetLayout::SBinding binding[count] =
            {
                {
                    0u,
                    EDT_STORAGE_BUFFER,
                    1u,
                    IShader::ESS_COMPUTE,
                    nullptr
                },
                {
                    1u,
                    EDT_STORAGE_IMAGE,
                    1u,
                    IShader::ESS_COMPUTE,
                    nullptr
                }
            };
            dsLayoutVertical = logicalDevice->createDescriptorSetLayout(binding, binding + count);
            pipelineLayoutVertical = logicalDevice->createPipelineLayout(pcRange.begin(), pcRange.end(), core::smart_refctd_ptr(dsLayoutVertical));
        }

        IGPUDescriptorSetLayout* dsLayouts[] = {dsLayoutHorizontal.get(), dsLayoutVertical.get()};
        const uint32_t dsCount[] = {1, 1};
        auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, dsLayouts, dsLayouts + 2ull, dsCount);

        const bool useHalfStorage = false;

        m_dsHorizontal = logicalDevice->createDescriptorSet(dsPool.get(), std::move(dsLayoutHorizontal));
        m_dsVertical = logicalDevice->createDescriptorSet(dsPool.get(), std::move(dsLayoutVertical));
        m_pipelineHorizontal = logicalDevice->createComputePipeline(nullptr, std::move(pipelineLayoutHorizontal), createShader("../BlurPassHorizontal.comp", outDim.width, useHalfStorage));
        m_pipelineVertical = logicalDevice->createComputePipeline(nullptr, std::move(pipelineLayoutVertical), createShader("../BlurPassVertical.comp", outDim.height, useHalfStorage));

        const uint32_t channelCount = getFormatChannelCount(m_inImageView->getCreationParameters().format);

        const float initalBlurRadius = 0.01f;
        const ISampler::E_TEXTURE_CLAMP blurWrapMode[2] = { ISampler::ETC_MIRROR, ISampler::ETC_MIRROR };
        const ISampler::E_TEXTURE_BORDER_COLOR blurBorderColors[2] = { ISampler::ETBC_FLOAT_OPAQUE_WHITE, ISampler::ETBC_FLOAT_OPAQUE_WHITE };
        
        const auto passCount = BlurClass::buildParameters(channelCount, outDim, m_pushConstants, m_dispatchInfo, initalBlurRadius, blurWrapMode, blurBorderColors);
        assert(passCount == 2);

        {
            IGPUSampler::SParams params =
            {
                {
                    // These wrapping params don't really matter for this example
                    ISampler::ETC_CLAMP_TO_EDGE,
                    ISampler::ETC_CLAMP_TO_EDGE,
                    ISampler::ETC_CLAMP_TO_EDGE,

                    ISampler::ETBC_FLOAT_OPAQUE_BLACK,
                    ISampler::ETF_LINEAR,
                    ISampler::ETF_LINEAR,
                    ISampler::ESMM_LINEAR,
                    8u,
                    0u,
                    ISampler::ECO_ALWAYS
                }
            };
            m_sampler = logicalDevice->createSampler(std::move(params));

            constexpr auto MaxDescriptorCount = 2;

            IGPUDescriptorSet::SWriteDescriptorSet writes[MaxDescriptorCount];
            IGPUDescriptorSet::SDescriptorInfo infos[MaxDescriptorCount];

            const auto& bindings = m_dsHorizontal->getLayout()->getBindings();
            const auto bindingCount = bindings.size();
            assert(bindingCount <= MaxDescriptorCount);

            for (auto i = 0; i < bindings.size(); ++i)
            {
                writes[i].dstSet = m_dsHorizontal.get();
                writes[i].binding = i;
                writes[i].arrayElement = 0u;
                writes[i].count = 1u;
                writes[i].info = &infos[i];
                writes[i].descriptorType = bindings.begin()[i].type;
            }

            infos[0].desc = m_inImageView;
            infos[0].image.sampler = m_sampler;
            infos[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;

            infos[1].desc = m_scratchBuffer;
            infos[1].buffer = { 0ull, m_scratchBuffer->getSize() };

            logicalDevice->updateDescriptorSets(2, writes, 0, nullptr);
        }

        {
            constexpr auto MaxDescriptorCount = 2;

            IGPUDescriptorSet::SWriteDescriptorSet writes[MaxDescriptorCount];
            IGPUDescriptorSet::SDescriptorInfo infos[MaxDescriptorCount];

            const auto& bindings = m_dsVertical->getLayout()->getBindings();
            const auto bindingCount = bindings.size();
            assert(bindingCount <= MaxDescriptorCount);

            for (auto i = 0; i < bindings.size(); ++i)
            {
                writes[i].dstSet = m_dsVertical.get();
                writes[i].binding = i;
                writes[i].arrayElement = 0u;
                writes[i].count = 1u;
                writes[i].info = &infos[i];
                writes[i].descriptorType = bindings.begin()[i].type;
            }

            infos[0].desc = m_scratchBuffer;
            infos[0].buffer = { 0ull, m_scratchBuffer->getSize() };

            infos[1].desc = m_outImageView;
            infos[1].image.sampler = nullptr;
            infos[1].image.imageLayout = EIL_GENERAL;

            logicalDevice->updateDescriptorSets(2, writes, 0, nullptr);
        }

        logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE][0].get(), IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, m_cmdbufs);

        for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
        {
            m_imageAcquireSem[i] = logicalDevice->createSemaphore();
            m_renderFinishedSem[i] = logicalDevice->createSemaphore();
        }

        {
            auto fstFragContents = assetManager->getAsset("../FST.frag", {}).getContents();
            if (fstFragContents.empty())
                FATAL_LOG("Failed to load FST fragment shader.\n");

            core::smart_refctd_ptr<IGPUSpecializedShader> fstFragShader = nullptr;
            {
                auto* fstFragShaderCPU = static_cast<asset::ICPUSpecializedShader*>(fstFragContents.begin()->get());

                cpu2gpuParams.beginCommandBuffers();
                auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&fstFragShaderCPU, &fstFragShaderCPU + 1ull, cpu2gpuParams);
                cpu2gpuParams.waitForCreationToComplete();

                if (!gpuArray.get() || gpuArray->size() < 1u || !(*gpuArray)[0])
                    FATAL_LOG("Failed to create GPU FST fragment shader from CPU FST fragment shader.\n");

                fstFragShader = (*gpuArray)[0];
            }

            ISampler::SParams samplerParams = { nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ESMM_LINEAR, 0u, false, nbl::asset::ECO_ALWAYS };
            auto sampler = logicalDevice->createSampler(samplerParams);

            video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUShader::ESS_FRAGMENT, &sampler };
            auto fstDSLayout = logicalDevice->createDescriptorSetLayout(&binding, &binding + 1u);

            const uint32_t fstDSCount = 1u;
            auto fstDSPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &fstDSLayout.get(), &fstDSLayout.get() + 1ull, &fstDSCount);
            m_fstDS = logicalDevice->createDescriptorSet(fstDSPool.get(), core::smart_refctd_ptr(fstDSLayout));
            {
                IGPUDescriptorSet::SWriteDescriptorSet write;
                IGPUDescriptorSet::SDescriptorInfo info;

                write.dstSet = m_fstDS.get();
                write.binding = 0;
                write.descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
                write.arrayElement = 0u;
                write.count = 1u;
                write.info = &info;

                info.image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
                info.image.sampler = nullptr;
                info.desc = m_outImageView;
                logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
            }

            auto fstProtoPipeline = ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams, 0u);

            auto constants = std::get<SPushConstantRange>(fstProtoPipeline);
            m_fstPipelineLayout = logicalDevice->createPipelineLayout(&constants, &constants+1ull, nullptr, nullptr, nullptr, std::move(fstDSLayout));

            video::IGPUGraphicsPipeline::SCreationParams creationParams = {};
            creationParams.renderpassIndependent = ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(fstFragShader), core::smart_refctd_ptr(m_fstPipelineLayout));
            creationParams.renderpass = core::smart_refctd_ptr(renderpass);

            m_fstGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(creationParams));
        }
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{
        ++m_resourceIx;
        if (m_resourceIx >= FRAMES_IN_FLIGHT)
            m_resourceIx = 0;

        inputSystem->getDefaultKeyboard(&m_keyboard);
        m_keyboard.consumeEvents(
            [this](const ui::IKeyboardEventChannel::range_t& events)
            {
                for (auto eventIt = events.begin(); eventIt != events.end(); ++eventIt)
                {
                    const auto& ev = *eventIt;
                    if ((ev.keyCode == ui::EKC_Q) && (ev.action == ui::SKeyboardEvent::ECA_RELEASED))
                    {
                        m_appRunning = false;
                    }
                }
            },
            logger.get());

        inputSystem->getDefaultMouse(&m_mouse);
        m_mouse.consumeEvents([this](const ui::IMouseEventChannel::range_t& events)
            {
                for (auto eventIt = events.begin(); eventIt != events.end(); ++eventIt)
                {
                    const auto& ev = *eventIt;
                    if (ev.type == ui::SMouseEvent::EET_SCROLL)
                    {
                        logger->log("Mouse vertical scroll event detected: %d\n", system::ILogger::ELL_DEBUG, ev.scrollEvent.verticalScroll);
                    }
                }
            }, logger.get());

        auto& cmdbuf = m_cmdbufs[m_resourceIx];
        auto& fence = m_frameCompleteFences[m_resourceIx];

        if (fence)
        {
            logicalDevice->blockForFences(1, &fence.get());
            logicalDevice->resetFences(1, &fence.get());
        }
        else
            fence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);

        uint32_t acquiredNextFBO;
        swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquireSem[m_resourceIx].get(), nullptr, &acquiredNextFBO);

        cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

        cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipelineHorizontal->getLayout(), 0, 1, &m_dsHorizontal.get());
        cmdbuf->bindComputePipeline(m_pipelineHorizontal.get());
        cmdbuf->pushConstants(m_pipelineHorizontal->getLayout(), IShader::ESS_COMPUTE, 0, sizeof(BlurClass::Parameters_t), &m_pushConstants[0]);
        cmdbuf->dispatch(m_dispatchInfo[0].wg_count[0], m_dispatchInfo[0].wg_count[1], m_dispatchInfo[0].wg_count[2]);

        // memory barrier to ensure the completion of previous dispatch
        IGPUCommandBuffer::SBufferMemoryBarrier bufferMemoryBarrier = {};
        bufferMemoryBarrier.barrier.srcAccessMask = EAF_SHADER_WRITE_BIT;
        bufferMemoryBarrier.barrier.dstAccessMask = EAF_SHADER_READ_BIT;
        bufferMemoryBarrier.srcQueueFamilyIndex = ~0u;
        bufferMemoryBarrier.dstQueueFamilyIndex = ~0u;
        bufferMemoryBarrier.buffer = m_scratchBuffer;
        bufferMemoryBarrier.offset = 0;
        bufferMemoryBarrier.size = m_scratchBuffer->getSize();
        cmdbuf->pipelineBarrier(EPSF_COMPUTE_SHADER_BIT, EPSF_COMPUTE_SHADER_BIT, EDF_NONE, 0u, nullptr, 1u, &bufferMemoryBarrier, 0u, nullptr);

        cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipelineVertical->getLayout(), 0, 1, &m_dsVertical.get());
        cmdbuf->bindComputePipeline(m_pipelineVertical.get());
        cmdbuf->pushConstants(m_pipelineVertical->getLayout(), IShader::ESS_COMPUTE, 0, sizeof(BlurClass::Parameters_t), &m_pushConstants[1]);
        cmdbuf->dispatch(m_dispatchInfo[1].wg_count[0], m_dispatchInfo[1].wg_count[1], m_dispatchInfo[1].wg_count[2]);

        // memory barrier to ensure the previous dispatch has completed modifying the image/transitions the layout to SAMPLED_BIT
        IGPUCommandBuffer::SImageMemoryBarrier imageMemoryBarrier = {};
        imageMemoryBarrier.barrier.srcAccessMask = EAF_SHADER_WRITE_BIT;
        imageMemoryBarrier.barrier.dstAccessMask = EAF_SHADER_READ_BIT;
        imageMemoryBarrier.oldLayout = EIL_GENERAL;
        imageMemoryBarrier.newLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
        imageMemoryBarrier.srcQueueFamilyIndex = ~0u;
        imageMemoryBarrier.dstQueueFamilyIndex = ~0u;
        imageMemoryBarrier.image = m_outImage;
        imageMemoryBarrier.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
        imageMemoryBarrier.subresourceRange.layerCount = m_outImage->getCreationParameters().arrayLayers;
        imageMemoryBarrier.subresourceRange.levelCount = m_outImage->getCreationParameters().mipLevels;
        cmdbuf->pipelineBarrier(EPSF_COMPUTE_SHADER_BIT, EPSF_FRAGMENT_SHADER_BIT, EDF_BY_REGION_BIT, 0u, nullptr, 0u, nullptr, 1u, &imageMemoryBarrier);

        asset::SViewport viewport = {};
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
        cmdbuf->setViewport(0u, 1u, &viewport);

        VkRect2D scissor = { {0, 0}, {WIN_W, WIN_H} };
        cmdbuf->setScissor(0u, 1u, &scissor);

        video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        {
            VkRect2D area;
            area.offset = { 0,0 };
            area.extent = { WIN_W, WIN_H };
            asset::SClearValue clear[2] = {};
            clear[0].color.float32[0] = 1.f;
            clear[0].color.float32[1] = 1.f;
            clear[0].color.float32[2] = 1.f;
            clear[0].color.float32[3] = 1.f;

            beginInfo.clearValueCount = 1u;
            beginInfo.framebuffer = fbos[acquiredNextFBO];
            beginInfo.renderpass = renderpass;
            beginInfo.renderArea = area;
            beginInfo.clearValues = clear;
        }
        cmdbuf->beginRenderPass(&beginInfo, ESC_INLINE);

        cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_fstPipelineLayout.get(), 3u, 1u, &m_fstDS.get());
        cmdbuf->bindGraphicsPipeline(m_fstGraphicsPipeline.get());
        ext::FullScreenTriangle::recordDrawCalls(m_fstGraphicsPipeline, 0u, swapchain->getPreTransform(), cmdbuf.get());

        cmdbuf->endRenderPass();

        // transition outImage back to GENERAL so it can be written to by next frame's compute passes
        imageMemoryBarrier.barrier.srcAccessMask = EAF_COLOR_ATTACHMENT_WRITE_BIT;
        imageMemoryBarrier.barrier.dstAccessMask = static_cast<E_ACCESS_FLAGS>(0u);
        imageMemoryBarrier.oldLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
        imageMemoryBarrier.newLayout = EIL_GENERAL;
        cmdbuf->pipelineBarrier(EPSF_COLOR_ATTACHMENT_OUTPUT_BIT, EPSF_BOTTOM_OF_PIPE_BIT, EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &imageMemoryBarrier);

        cmdbuf->end();

        CommonAPI::Submit(
            logicalDevice.get(),
            swapchain.get(),
            cmdbuf.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS],
            m_imageAcquireSem[m_resourceIx].get(),
            m_renderFinishedSem[m_resourceIx].get(),
            fence.get());

        CommonAPI::Present(
            logicalDevice.get(),
            swapchain.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS],
            m_renderFinishedSem[m_resourceIx].get(),
            acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return m_appRunning && windowCb->isWindowOpen();
	}

private:
	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
	std::array<nbl::core::smart_refctd_ptr<video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

    CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> m_keyboard;
    CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> m_mouse;

    int32_t m_resourceIx = -1;
    bool m_appRunning = true;

    core::smart_refctd_ptr<IGPUCommandBuffer> m_cmdbufs[FRAMES_IN_FLIGHT] = {nullptr};
    core::smart_refctd_ptr<IGPUFence> m_frameCompleteFences[FRAMES_IN_FLIGHT] = {nullptr};
    core::smart_refctd_ptr<IGPUSemaphore> m_imageAcquireSem[FRAMES_IN_FLIGHT] = {nullptr};
    core::smart_refctd_ptr<IGPUSemaphore> m_renderFinishedSem[FRAMES_IN_FLIGHT] = { nullptr };

    core::smart_refctd_ptr<IGPUDescriptorSet> m_dsHorizontal = nullptr;
    core::smart_refctd_ptr<IGPUDescriptorSet> m_dsVertical = nullptr;
    core::smart_refctd_ptr<IGPUComputePipeline> m_pipelineHorizontal = nullptr;
    core::smart_refctd_ptr<IGPUComputePipeline> m_pipelineVertical = nullptr;

    core::smart_refctd_ptr<IGPUDescriptorSet> m_fstDS = nullptr;
    core::smart_refctd_ptr<IGPUPipelineLayout> m_fstPipelineLayout = nullptr;
    core::smart_refctd_ptr<IGPUGraphicsPipeline> m_fstGraphicsPipeline = nullptr;

    core::smart_refctd_ptr<IGPUSampler> m_sampler = nullptr;
    core::smart_refctd_ptr<IGPUImageView> m_inImageView = nullptr;
    core::smart_refctd_ptr<IGPUImage> m_outImage = nullptr;
    core::smart_refctd_ptr<IGPUImageView> m_outImageView = nullptr;
    core::smart_refctd_ptr<IGPUBuffer> m_scratchBuffer = nullptr;

    BlurClass::Parameters_t m_pushConstants[2];
    BlurClass::DispatchInfo_t m_dispatchInfo[2];

    inline core::smart_refctd_ptr<IGPUSpecializedShader> createShader(const char* shaderIncludePath, const uint32_t axisDim, const bool useHalfStorage)
    {
        std::ostringstream shaderSourceStream;
        shaderSourceStream
            << "#version 460 core\n"
            << "#define _NBL_GLSL_WORKGROUP_SIZE_ " << BlurClass::DEFAULT_WORKGROUP_SIZE << "\n" // Todo(achal): Get the workgroup size from outside
            << "#define _NBL_GLSL_EXT_BLUR_PASSES_PER_AXIS_ " << BlurClass::PASSES_PER_AXIS << "\n" // Todo(achal): Get this from outside?
            << "#define _NBL_GLSL_EXT_BLUR_AXIS_DIM_ " << axisDim << "\n"
            << "#define _NBL_GLSL_EXT_BLUR_HALF_STORAGE_ " << (useHalfStorage ? 1 : 0) << "\n"
            << "#include \"" << shaderIncludePath << "\"\n";

        auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, "CBlurPerformer::createSpecializedShader");
        auto gpuUnspecShader = logicalDevice->createShader(std::move(cpuShader));
        auto specShader = logicalDevice->createSpecializedShader(gpuUnspecShader.get(), {nullptr, nullptr, "main"});

        return specShader;
    }

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbos[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(BlurTestApp);
};

NBL_COMMON_API_MAIN(BlurTestApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }