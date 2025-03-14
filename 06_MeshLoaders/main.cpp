// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "CCamera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace ui;
/*
    Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

class MeshLoadersApp : public ApplicationBase
{
    constexpr static uint32_t WIN_W = 1280;
    constexpr static uint32_t WIN_H = 720;
    constexpr static uint32_t SC_IMG_COUNT = 3u;
    constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
    constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
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
    nbl::video::IPhysicalDevice* physicalDevice;
    std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
    nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
    nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
    nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> fbo;
    std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
    nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
    nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
    nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
    nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
    nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    
    video::IDeviceMemoryBacked::SDeviceMemoryRequirements ubomemreq;
    core::smart_refctd_ptr<video::IGPUBuffer> gpuubo;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> gpuds1;

    core::smart_refctd_ptr<video::IQueryPool> occlusionQueryPool;
    core::smart_refctd_ptr<video::IQueryPool> timestampQueryPool;

    asset::ICPUMesh* meshRaw = nullptr;
    const asset::COBJMetadata* metaOBJ = nullptr;

    core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
    Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

    using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
    std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
    const asset::ICPUMeshBuffer* firstMeshBuffer;
    const nbl::asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata;
    nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

    uint32_t ds1UboBinding = 0;
    int resourceIx;
    uint32_t acquiredNextFBO = {};
    std::chrono::steady_clock::time_point lastTime;
    bool frameDataFilled = false;
    size_t frame_count = 0ull;
    double time_sum = 0;
    double dtList[NBL_FRAMES_TO_AVERAGE] = {};

    video::CDumbPresentationOracle oracle;
    
    core::smart_refctd_ptr<video::IGPUBuffer> queryResultsBuffer;

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
            fbo->begin()[i] = core::smart_refctd_ptr(f[i]);
        }
    }
    void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
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

    void getAndLogQueryPoolResults()
    {
#ifdef QUERY_POOL_LOGS
        {
            uint64_t samples_passed[4] = {};
            auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT) | video::IQueryPool::EQRF_64_BIT;
            logicalDevice->getQueryPoolResults(occlusionQueryPool.get(), 0u, 2u, sizeof(samples_passed), samples_passed, sizeof(uint64_t) * 2, queryResultFlags);
            logger->log("[AVAIL+64] SamplesPassed[0] = %d, SamplesPassed[1] = %d, Result Available = %d, %d", system::ILogger::ELL_INFO, samples_passed[0], samples_passed[2], samples_passed[1], samples_passed[3]);
        }
        {
            uint64_t samples_passed[4] = {};
            auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT) | video::IQueryPool::EQRF_64_BIT | video::IQueryPool::EQRF_WAIT_BIT;
            logicalDevice->getQueryPoolResults(occlusionQueryPool.get(), 0u, 2u, sizeof(samples_passed), samples_passed, sizeof(uint64_t) * 2, queryResultFlags);
            logger->log("[WAIT+AVAIL+64] SamplesPassed[0] = %d, SamplesPassed[1] = %d, Result Available = %d, %d", system::ILogger::ELL_INFO, samples_passed[0], samples_passed[2], samples_passed[1], samples_passed[3]);
        }
        {
            uint32_t samples_passed[2] = {};
            auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT);
            logicalDevice->getQueryPoolResults(occlusionQueryPool.get(), 0u, 2u, sizeof(samples_passed), samples_passed, sizeof(uint32_t), queryResultFlags);
            logger->log("[WAIT] SamplesPassed[0] = %d, SamplesPassed[1] = %d", system::ILogger::ELL_INFO, samples_passed[0], samples_passed[1]);
        }
        {
            uint64_t timestamps[4] = {};
            auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT) | video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT | video::IQueryPool::EQRF_64_BIT;
            logicalDevice->getQueryPoolResults(timestampQueryPool.get(), 0u, 2u, sizeof(timestamps), timestamps, sizeof(uint64_t) * 2ull, queryResultFlags);
            float timePassed = (timestamps[2] - timestamps[0]) * physicalDevice->getLimits().timestampPeriodInNanoSeconds;
            logger->log("Time Passed (Seconds) = %f", system::ILogger::ELL_INFO, (timePassed * 1e-9));
            logger->log("Timestamps availablity: %d, %d", system::ILogger::ELL_INFO, timestamps[1], timestamps[3]);
        }
#endif
    }

    APP_CONSTRUCTOR(MeshLoadersApp)
    void onAppInitialized_impl() override
    {
        const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_TRANSFER_SRC_BIT);
        CommonAPI::InitParams initParams;
        initParams.window = core::smart_refctd_ptr(window);
        initParams.apiType = video::EAT_VULKAN;
        initParams.appName = { _NBL_APP_NAME_ };
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
        fbo = CommonAPI::createFBOWithSwapchainImages(
            swapchain->getImageCount(), WIN_W, WIN_H,
            logicalDevice, swapchain, renderpass,
            nbl::asset::EF_D32_SFLOAT
        );
        
        // Occlusion Query
        {
            video::IQueryPool::SCreationParams queryPoolCreationParams = {};
            queryPoolCreationParams.queryType = video::IQueryPool::EQT_OCCLUSION;
            queryPoolCreationParams.queryCount = 2u;
            occlusionQueryPool = logicalDevice->createQueryPool(std::move(queryPoolCreationParams));
        }

        // Timestamp Query
        video::IQueryPool::SCreationParams queryPoolCreationParams = {};
        {
            video::IQueryPool::SCreationParams queryPoolCreationParams = {};
            queryPoolCreationParams.queryType = video::IQueryPool::EQT_TIMESTAMP;
            queryPoolCreationParams.queryCount = 2u;
            timestampQueryPool = logicalDevice->createQueryPool(std::move(queryPoolCreationParams));
        }

        {
            // SAMPLES_PASSED_0 + AVAILABILIY_0 + SAMPLES_PASSED_1 + AVAILABILIY_1 (uint32_t)
            const size_t queriesSize = sizeof(uint32_t) * 4;
            video::IGPUBuffer::SCreationParams gpuuboCreationParams;
            gpuuboCreationParams.size = queriesSize;
            gpuuboCreationParams.usage = core::bitflag<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT)|asset::IBuffer::EUF_TRANSFER_DST_BIT|asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
            gpuuboCreationParams.queueFamilyIndexCount = 0u;
            gpuuboCreationParams.queueFamilyIndices = nullptr;

            queryResultsBuffer = logicalDevice->createBuffer(std::move(gpuuboCreationParams));
            auto memReqs = queryResultsBuffer->getMemoryReqs();
            memReqs.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
            auto queriesMem = logicalDevice->allocate(memReqs, queryResultsBuffer.get());

            queryResultsBuffer->setObjectDebugName("QueryResults");
        }

        nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
        {
            auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

            system::path archPath = sharedInputCWD / "sponza.zip";
            auto arch = system->openFileArchive(archPath);
            // test no alias loading (TODO: fix loading from absolute paths)
            system->mount(std::move(arch));
            asset::IAssetLoader::SAssetLoadParams loadParams;
            loadParams.workingDirectory = sharedInputCWD;
            loadParams.logger = logger.get();
            auto meshes_bundle = assetManager->getAsset((sharedInputCWD / "sponza.zip/sponza.obj").string(), loadParams);
            assert(!meshes_bundle.getContents().empty());

            metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");
        }

        // Fix FrontFace and BlendParams for meshBuffers
        for (size_t i = 0ull; i < meshRaw->getMeshBuffers().size(); ++i)
        {
            auto& meshBuffer = meshRaw->getMeshBuffers().begin()[i];
            meshBuffer->getPipeline()->getRasterizationParams().frontFaceIsCCW = false;
        }

        // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
        firstMeshBuffer = *meshRaw->getMeshBuffers().begin();
        pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

        // so we can create just one DS
        const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
        ds1UboBinding = ds1layout->getDescriptorRedirect(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER).getBinding(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ 0 }).data;

        size_t neededDS1UBOsz = 0ull;
        {
            for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
                if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::E_TYPE::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
                    neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);
        }

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
        {
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);

            gpuds1layout = (*gpu_array)[0];
        }

        core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool = nullptr;
        {
            video::IDescriptorPool::SCreateInfo createInfo = {};
            createInfo.maxSets = 1u;
            createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] = 1u;
            descriptorPool = logicalDevice->createDescriptorPool(std::move(createInfo));
        }

        video::IGPUBuffer::SCreationParams gpuuboCreationParams;
        gpuuboCreationParams.size = neededDS1UBOsz;
        gpuuboCreationParams.usage = core::bitflag<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
        gpuuboCreationParams.queueFamilyIndexCount = 0u;
        gpuuboCreationParams.queueFamilyIndices = nullptr;

        gpuubo = logicalDevice->createBuffer(std::move(gpuuboCreationParams));
        auto gpuuboMemReqs = gpuubo->getMemoryReqs();
        gpuuboMemReqs.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
        auto uboMemoryOffset = logicalDevice->allocate(gpuuboMemReqs, gpuubo.get(), video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);

        gpuds1 = descriptorPool->createDescriptorSet(std::move(gpuds1layout));

        {
            video::IGPUDescriptorSet::SWriteDescriptorSet write;
            write.dstSet = gpuds1.get();
            write.binding = ds1UboBinding;
            write.count = 1u;
            write.arrayElement = 0u;
            write.descriptorType = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
            video::IGPUDescriptorSet::SDescriptorInfo info;
            {
                info.desc = gpuubo;
                info.info.buffer.offset = 0ull;
                info.info.buffer.size = neededDS1UBOsz;
            }
            write.info = &info;
            logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
        }
        {
            cpu2gpuParams.beginCommandBuffers();

            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);
            
            cpu2gpuParams.waitForCreationToComplete(false);

            gpumesh = (*gpu_array)[0];
        }
       
        {
            for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
            {
                auto gpuIndependentPipeline = gpumesh->getMeshBuffers().begin()[i]->getPipeline();

                nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
                graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
                graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

                const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
                gpuPipelines[adress] = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
            }
        }

        core::vectorSIMDf cameraPosition(-250.0f,177.0f,1.69f);
        core::vectorSIMDf cameraTarget(50.0f,125.0f,-3.0f);
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), video::ISurface::getTransformedAspectRatio(swapchain->getPreTransform(), WIN_W, WIN_H), 0.1, 10000);
        camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 10.f, 1.f);
        lastTime = std::chrono::steady_clock::now();

        for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
            dtList[i] = 0.0;

        oracle.reportBeginFrameRecord();
        

       const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			logicalDevice->createCommandBuffers(graphicsCommandPools[i].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers+i);
            imageAcquire[i] = logicalDevice->createSemaphore();
            renderFinished[i] = logicalDevice->createSemaphore();
        }

        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
        uint32_t acquiredNextFBO = {};
        resourceIx = -1;
    }
    void onAppTerminated_impl() override
    {
        const auto& fboCreationParams = fbo->begin()[acquiredNextFBO]->getCreationParameters();
        auto gpuSourceImageView = fboCreationParams.attachments[0];

        bool status = ext::ScreenShot::createScreenShot(
            logicalDevice.get(),
            queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN],
            renderFinished[resourceIx].get(),
            gpuSourceImageView.get(),
            assetManager.get(),
            "ScreenShot.png",
            asset::IImage::EL_PRESENT_SRC,
            asset::EAF_NONE);

        assert(status);
        logicalDevice->waitIdle();
    }
    void workLoopBody() override
    {
        ++resourceIx;
        if (resourceIx >= FRAMES_IN_FLIGHT)
            resourceIx = 0;

        auto& commandBuffer = commandBuffers[resourceIx];
        auto& fence = frameComplete[resourceIx];
        if (fence)
            logicalDevice->blockForFences(1u, &fence.get());
        else
            fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(nbl::video::IGPUCommandBuffer::EU_NONE);

        const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);
        {
            inputSystem->getDefaultMouse(&mouse);
            inputSystem->getDefaultKeyboard(&keyboard);

            camera.beginInputProcessing(nextPresentationTimestamp);
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
            camera.endInputProcessing(nextPresentationTimestamp);
        }

        const auto& viewMatrix = camera.getViewMatrix();
        const auto& viewProjectionMatrix = matrix4SIMD::concatenateBFollowedByAPrecisely(
            video::ISurface::getSurfaceTransformationMatrix(swapchain->getPreTransform()),
            camera.getConcatenatedMatrix()
        );

        asset::SViewport viewport;
        viewport.minDepth = 1.f;
        viewport.maxDepth = 0.f;
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
        commandBuffer->setViewport(0u, 1u, &viewport);
        
        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = { WIN_W, WIN_H };
        commandBuffer->setScissor(0u, 1u, &scissor);

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        const size_t uboSize = gpuubo->getSize();
        core::vector<uint8_t> uboData(uboSize);
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::E_TYPE::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                {
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                } break;

                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
                {
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                } break;

                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                {
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                } break;
                }
            }
        }
        commandBuffer->updateBuffer(gpuubo.get(), 0ull, uboSize, uboData.data());
        
        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        {
            VkRect2D area;
            area.offset = { 0,0 };
            area.extent = { WIN_W, WIN_H };
            asset::SClearValue clear[2] = {};
            clear[0].color.float32[0] = 1.f;
            clear[0].color.float32[1] = 1.f;
            clear[0].color.float32[2] = 1.f;
            clear[0].color.float32[3] = 1.f;
            clear[1].depthStencil.depth = 0.f;

            beginInfo.clearValueCount = 2u;
            beginInfo.framebuffer = fbo->begin()[acquiredNextFBO];
            beginInfo.renderpass = renderpass;
            beginInfo.renderArea = area;
            beginInfo.clearValues = clear;
        }

        commandBuffer->resetQueryPool(occlusionQueryPool.get(), 0u, 2u);
        commandBuffer->resetQueryPool(timestampQueryPool.get(), 0u, 2u);
        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
        
        commandBuffer->writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT, timestampQueryPool.get(), 0u);
        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
        {
            if(i < 2)
                commandBuffer->beginQuery(occlusionQueryPool.get(), i);
            auto gpuMeshBuffer = gpumesh->getMeshBuffers().begin()[i];
            auto gpuGraphicsPipeline = gpuPipelines[reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];

            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();

            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr);
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), asset::IShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);

            if(i < 2)
                commandBuffer->endQuery(occlusionQueryPool.get(), i);
        }
        commandBuffer->writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT, timestampQueryPool.get(), 1u);

        commandBuffer->endRenderPass();

        auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT) | video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT;
        commandBuffer->copyQueryPoolResults(occlusionQueryPool.get(), 0, 2, queryResultsBuffer.get(), 0u, sizeof(uint32_t) * 2, queryResultFlags);

        commandBuffer->end();
        
        logicalDevice->resetFences(1, &fence.get());
        CommonAPI::Submit(
            logicalDevice.get(),
            commandBuffer.get(),
            queues[CommonAPI::InitOutput::EQT_COMPUTE],
            imageAcquire[resourceIx].get(),
            renderFinished[resourceIx].get(),
            fence.get());
        CommonAPI::Present(logicalDevice.get(), 
            swapchain.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);

        getAndLogQueryPoolResults();
    }
    bool keepRunning() override
    {
        return windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(MeshLoadersApp)
