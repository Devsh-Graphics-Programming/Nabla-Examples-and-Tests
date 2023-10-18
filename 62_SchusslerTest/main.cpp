// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <cstdio>
#include <iostream>
#include <nabla.h>

#include "nbl/asset/utils/CGeometryCreator.h"
#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;

/*
        Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

class SchusslerTestApp : public ApplicationBase {
  constexpr static uint32_t WIN_W = 1280;
  constexpr static uint32_t WIN_H = 720;
  constexpr static uint32_t SC_IMG_COUNT = 3u;
  constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;

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
  nbl::core::smart_refctd_dynamic_array<
      nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>
      fbo;
  std::array<
      std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>,
                 CommonAPI::InitOutput::MaxFramesInFlight>,
      CommonAPI::InitOutput::MaxQueuesCount>
      commandPools; // TODO: Multibuffer and reset the commandpools
  nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
  nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
  nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
  nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
  nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

  nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

  core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer;
  core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>
      gpuRenderpassIndependentPipeline;
  core::smart_refctd_ptr<IGPUBuffer> gpuubo;
  core::smart_refctd_ptr<IGPUDescriptorSet> gpuDescriptorSet1;
  core::smart_refctd_ptr<IGPUDescriptorSet> gpuDescriptorSet3;
  core::smart_refctd_ptr<IGPUGraphicsPipeline> gpuGraphicsPipeline;

  core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = {
      nullptr};
  core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] =
      {nullptr};
  core::smart_refctd_ptr<video::IGPUSemaphore>
      renderFinished[FRAMES_IN_FLIGHT] = {nullptr};
  core::smart_refctd_ptr<video::IGPUCommandBuffer>
      commandBuffers[FRAMES_IN_FLIGHT];

  nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

  CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> mouse;
  CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> keyboard;
  Camera camera;

  int resourceIx;
  uint32_t acquiredNextFBO = {};

  enum BRDFTestNumber : uint32_t {
    TEST_GGX = 1,
    TEST_BECKMANN,
    TEST_PHONG,
    TEST_AS,
    TEST_OREN_NAYAR,
    TEST_LAMBERT,
  };

  BRDFTestNumber currentTestNum = TEST_GGX;

  struct SPushConsts {
    struct VertStage {
      core::matrix4SIMD VP;
    } vertStage;
    struct FragStage {
      core::vectorSIMDf campos;
      BRDFTestNumber testNum;
      uint32_t pad[3];
    } fragStage;
  };

  void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow> &&wnd) override {
    window = std::move(wnd);
  }
  void setSystem(core::smart_refctd_ptr<nbl::system::ISystem> &&s) override {
    system = std::move(s);
  }
  nbl::ui::IWindow *getWindow() override { return window.get(); }
  video::IAPIConnection *getAPIConnection() override {
    return apiConnection.get();
  }
  video::ILogicalDevice *getLogicalDevice() override {
    return logicalDevice.get();
  }
  video::IGPURenderpass *getRenderpass() override { return renderpass.get(); }
  void setSurface(core::smart_refctd_ptr<video::ISurface> &&s) override {
    surface = std::move(s);
  }
  void setFBOs(
      std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>> &f) override {
    for (int i = 0; i < f.size(); i++) {
      fbo->begin()[i] = core::smart_refctd_ptr(f[i]);
    }
  }
  void setSwapchain(core::smart_refctd_ptr<video::ISwapchain> &&s) override {
    swapchain = std::move(s);
  }
  uint32_t getSwapchainImageCount() override {
    return swapchain->getImageCount();
  }
  virtual nbl::asset::E_FORMAT getDepthFormat() override {
    return nbl::asset::EF_D32_SFLOAT;
  }

  APP_CONSTRUCTOR(SchusslerTestApp)

  void onAppInitialized_impl() override {
    const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(
        asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
    CommonAPI::InitParams initParams;
    initParams.window = core::smart_refctd_ptr(window);
    initParams.apiType = video::EAT_VULKAN;
    initParams.appName = {_NBL_APP_NAME_};
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

    CommonAPI::createSwapchain(std::move(logicalDevice),
                               m_swapchainCreationParams, WIN_W, WIN_H,
                               swapchain);
    assert(swapchain);
    fbo = CommonAPI::createFBOWithSwapchainImages(
        swapchain->getImageCount(), WIN_W, WIN_H, logicalDevice, swapchain,
        renderpass, nbl::asset::EF_D32_SFLOAT);

    auto* geometryCreator = assetManager->getGeometryCreator();
    auto* quantNormalCache = assetManager->getMeshManipulator()
               ->getQuantNormalCache();
    constexpr uint32_t INSTANCE_COUNT = 25u;

    auto geometryObject = geometryCreator->createIcoSphere(0.5f, 2, true);
    
    asset::SPushConstantRange rng[2];
    rng[0].offset = 0u;
    rng[0].size = sizeof(SPushConsts::vertStage);
    rng[0].stageFlags = asset::IShader::ESS_VERTEX;
    rng[1].offset = offsetof(SPushConsts, fragStage);
    rng[1].size = sizeof(SPushConsts::fragStage);
    rng[1].stageFlags = asset::IShader::ESS_FRAGMENT;

    auto gpuPipelineLayout = logicalDevice->createPipelineLayout(
        rng, rng + 2, nullptr, nullptr, nullptr, nullptr);

    auto vertexShaderBundle = assetManager->getAsset("../shader.vert", {});
    {
      bool status = !vertexShaderBundle.getContents().empty();
      assert(status);
    }

    auto cpuVertexShader =
        core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(
            vertexShaderBundle.getContents().begin()[0]);
    smart_refctd_ptr<video::IGPUSpecializedShader> gpuVertexShader;
    {
      auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(
          &cpuVertexShader, &cpuVertexShader + 1, cpu2gpuParams);
      if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
        assert(false);

      gpuVertexShader = (*gpu_array)[0];
    }

    auto fragmentShaderBundle = assetManager->getAsset("../shader.frag", {});
    {
      bool status = !fragmentShaderBundle.getContents().empty();
      assert(status);
    }

    auto cpuFragmentShader =
        core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(
            fragmentShaderBundle.getContents().begin()[0]);
    smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
    {
      auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(
          &cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
      if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
        assert(false);

      gpuFragmentShader = (*gpu_array)[0];
    }

    core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuGShaders[] = {
        gpuVertexShader, gpuFragmentShader};
    auto gpuGShadersPointer =
        reinterpret_cast<video::IGPUSpecializedShader **>(gpuGShaders);

    asset::SBlendParams blendParams;
    asset::SRasterizationParams rasterParams;
    rasterParams.faceCullingMode = asset::EFCM_NONE;

    auto gpuPipeline = logicalDevice->createRenderpassIndependentPipeline(
        nullptr, std::move(gpuPipelineLayout), gpuGShadersPointer,
        gpuGShadersPointer + 2, geometryObject.inputParams, blendParams,
        geometryObject.assemblyParams, rasterParams);

    constexpr auto MAX_ATTR_BUF_BINDING_COUNT =
        video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
    constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;
    core::vector<asset::ICPUBuffer *> cpubuffers;
    cpubuffers.reserve(MAX_DATA_BUFFERS);
    for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++) {
      auto buf = geometryObject.bindings[i].buffer.get();
      if (buf)
        cpubuffers.push_back(buf);
    }
    auto cpuindexbuffer = geometryObject.indexBuffer.buffer.get();
    if (cpuindexbuffer)
      cpubuffers.push_back(cpuindexbuffer);

    cpu2gpuParams.beginCommandBuffers();
    auto gpubuffers = cpu2gpu.getGPUObjectsFromAssets(
        cpubuffers.data(), cpubuffers.data() + cpubuffers.size(),
        cpu2gpuParams);
    cpu2gpuParams.waitForCreationToComplete();

    asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
    for (auto i = 0, j = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++) {
      if (!geometryObject.bindings[i].buffer)
        continue;
      auto buffPair = gpubuffers->operator[](j++);
      bindings[i].offset = buffPair->getOffset();
      bindings[i].buffer =
          core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
    }
    if (cpuindexbuffer) {
      auto buffPair = gpubuffers->back();
      bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
      bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer =
          core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
    }

    gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(
        core::smart_refctd_ptr(gpuPipeline), nullptr, bindings,
        std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
    {
      gpuMeshBuffer->setIndexType(geometryObject.indexType);
      gpuMeshBuffer->setIndexCount(geometryObject.indexCount);
      gpuMeshBuffer->setBoundingBox(geometryObject.bbox);
      gpuMeshBuffer->setInstanceCount(INSTANCE_COUNT);
    }

    {
      nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
      graphicsPipelineParams.renderpassIndependent =
          core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(
              const_cast<video::IGPURenderpassIndependentPipeline *>(
                  gpuMeshBuffer->getPipeline()));
      graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
      gpuGraphicsPipeline = logicalDevice->createGraphicsPipeline(
          nullptr, std::move(graphicsPipelineParams));
    }

    const auto &graphicsCommandPools =
        commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
    for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++) {
      logicalDevice->createCommandBuffers(graphicsCommandPools[i].get(),
                                          video::IGPUCommandBuffer::EL_PRIMARY,
                                          1, commandBuffers + i);
      imageAcquire[i] = logicalDevice->createSemaphore();
      renderFinished[i] = logicalDevice->createSemaphore();
    }

    matrix4SIMD projectionMatrix =
        matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(
            core::radians(60.0f), float(WIN_W) / WIN_H, 0.01f, 5000.0f);
    camera = Camera(core::vectorSIMDf(0.f, 0.f, 6.f),
                    core::vectorSIMDf(0.f, 0.f, -1.f), projectionMatrix, 10.f,
                    1.f);
  }

  void workLoopBody() override {
    ++resourceIx;
    if (resourceIx >= FRAMES_IN_FLIGHT)
      resourceIx = 0;

    auto &commandBuffer = commandBuffers[resourceIx];
    auto &fence = frameComplete[resourceIx];

    if (fence) {
      logicalDevice->blockForFences(1u, &fence.get());
      logicalDevice->resetFences(1u, &fence.get());
    } else
      fence = logicalDevice->createFence(
          static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

    inputSystem->getDefaultKeyboard(&keyboard);
    keyboard.consumeEvents(
        [&](const ui::IKeyboardEventChannel::range_t &events) -> void {
          for (auto &event : events) {
            if (event.action == ui::SKeyboardEvent::ECA_PRESSED) {
              switch (event.keyCode) {
              case ui::EKC_1:
                currentTestNum = TEST_GGX;
                break;
              case ui::EKC_2:
                currentTestNum = TEST_BECKMANN;
                break;
              case ui::EKC_3:
                currentTestNum = TEST_PHONG;
                break;
              case ui::EKC_4:
                currentTestNum = TEST_AS;
                break;
              case ui::EKC_5:
                currentTestNum = TEST_OREN_NAYAR;
                break;
              case ui::EKC_6:
                currentTestNum = TEST_LAMBERT;
                break;
              }
            }
          }
        },
        logger.get());

    commandBuffer->reset(
        nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
    commandBuffer->begin(IGPUCommandBuffer::EU_NONE);

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

    swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(),
                                nullptr, &acquiredNextFBO);

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

    SPushConsts pc;
    pc.vertStage.VP = camera.getConcatenatedMatrix();
    pc.fragStage.campos = core::vectorSIMDf(&camera.getPosition().X);
    pc.fragStage.testNum = currentTestNum;
    commandBuffer->pushConstants(
        gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(),
        asset::IShader::ESS_VERTEX, 0u, sizeof(SPushConsts::vertStage),
        &pc.vertStage);

    commandBuffer->pushConstants(
        gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(),
        asset::IShader::ESS_FRAGMENT, offsetof(SPushConsts, fragStage),
        sizeof(SPushConsts::fragStage), &pc.fragStage);

    commandBuffer->drawMeshBuffer(gpuMeshBuffer.get());

    commandBuffer->endRenderPass();
    commandBuffer->end();

    CommonAPI::Submit(logicalDevice.get(), commandBuffer.get(),
                      queues[CommonAPI::InitOutput::EQT_GRAPHICS],
                      imageAcquire[resourceIx].get(),
                      renderFinished[resourceIx].get(), fence.get());
    CommonAPI::Present(logicalDevice.get(), swapchain.get(),
                       queues[CommonAPI::InitOutput::EQT_GRAPHICS],
                       renderFinished[resourceIx].get(), acquiredNextFBO);
  }

  bool keepRunning() override { return windowCb->isWindowOpen(); }

  void onAppTerminated_impl() override { logicalDevice->waitIdle(); }
};

NBL_COMMON_API_MAIN(SchusslerTestApp)