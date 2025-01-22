// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"

class RaytracingPipelineApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
  using device_base_t = examples::SimpleWindowedApplication;
  using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
  using clock_t = std::chrono::steady_clock;

  constexpr static inline uint32_t WIN_W = 1280, WIN_H = 720;
  constexpr static inline uint32_t MaxFramesInFlight = 3u;

  constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

  struct ShaderBindingTable
  {
    SStridedBufferRegion<IGPUBuffer> raygenGroupRegion;
    SStridedBufferRegion<IGPUBuffer> hitGroupsRegion;
    SStridedBufferRegion<IGPUBuffer> missGroupsRegion;
    SStridedBufferRegion<IGPUBuffer> callableGroupsRegion;
  };

  struct CameraView
  {
    float32_t3 position;
    float32_t3 target;
    float32_t3 upVector;
  };

public:
  inline RaytracingPipelineApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
    : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
  }

  inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
  {
    auto retval = device_base_t::getRequiredDeviceFeatures();
    retval.rayTracingPipeline = true;
    retval.accelerationStructure = true;
    retval.rayQuery = true;
    return retval;
  }

  inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
  {
    auto retval = device_base_t::getPreferredDeviceFeatures();
    retval.accelerationStructureHostCommands = true;
    return retval;
  }

  inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
  {
    if (!m_surface)
    {
      {
        auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
        IWindow::SCreationParams params = {};
        params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
        params.width = WIN_W;
        params.height = WIN_H;
        params.x = 32;
        params.y = 32;
        params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
        params.windowCaption = "RaytracingPipelineApp";
        params.callback = windowCallback;
        const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
      }

      auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
      const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
    }

    if (m_surface)
      return { {m_surface->getSurface()/*,EQF_NONE*/} };

    return {};
  }

  // so that we can use the same queue for asset converter and rendering
  inline core::vector<queue_req_t> getQueueRequirements() const override
  {
    auto reqs = device_base_t::getQueueRequirements();
    reqs.front().requiredFlags |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
    return reqs;
  }

  inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
  {
    m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

    if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
      return false;

    if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
      return false;

    const auto compileShader = [&]<typename... Args>(const std::string& filePath, const std::string& header = "") -> smart_refctd_ptr<IGPUShader>
      {
        IAssetLoader::SAssetLoadParams lparams = {};
        lparams.logger = m_logger.get();
        lparams.workingDirectory = "";
        auto bundle = m_assetMgr->getAsset(filePath, lparams);
        if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
        {
          m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
          exit(-1);
        }

        const auto assets = bundle.getContents();
        assert(assets.size() == 1);
        smart_refctd_ptr<ICPUShader> sourceRaw = IAsset::castDown<ICPUShader>(assets[0]);
        if (!sourceRaw)
          m_logger->log("Fail to load shader source", ILogger::ELL_ERROR, filePath);
        smart_refctd_ptr<ICPUShader> source = CHLSLCompiler::createOverridenCopy(
          sourceRaw.get(),
          "%s\n",
          header.c_str()
        );

        return m_device->createShader(source.get());
      };

    // shader
    const auto raygenShader = compileShader("app_resources/raytrace.rgen.hlsl");
    const auto closestHitShader = compileShader("app_resources/raytrace.rchit.hlsl");
    const auto anyHitShaderColorPayload = compileShader("app_resources/raytrace.rahit.hlsl", "#define USE_COLOR_PAYLOAD\n");
    const auto anyHitShaderShadowPayload = compileShader("app_resources/raytrace.rahit.hlsl", "#define USE_SHADOW_PAYLOAD\n");
    const auto missShader = compileShader("app_resources/raytrace.rmiss.hlsl");
    const auto shadowMissShader = compileShader("app_resources/raytraceShadow.rmiss.hlsl");

    m_semaphore = m_device->createSemaphore(m_realFrameIx);
    if (!m_semaphore)
      return logFail("Failed to Create a Semaphore!");

    ISwapchain::SCreationParams swapchainParams = { .surface = core::smart_refctd_ptr<nbl::video::ISurface>(m_surface->getSurface()) };
    if (!swapchainParams.deduceFormat(m_physicalDevice))
      return logFail("Could not choose a Surface Format for the Swapchain!");

    auto gQueue = getGraphicsQueue();
    if (!m_surface || !m_surface->init(gQueue, std::make_unique<ISimpleManagedSurface::ISwapchainResources>(), swapchainParams.sharedParams))
      return logFail("Could not create Window & Surface or initialize the Surface!");

    auto pool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

    m_converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });

    for (auto i = 0u; i < MaxFramesInFlight; i++)
    {
      if (!pool)
        return logFail("Couldn't create Command Pool!");
      if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
        return logFail("Couldn't create Command Buffer!");
    }

    m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
    m_surface->recreateSwapchain();

    // create output images
    m_hdrImage = m_device->createImage({
        {
          .type = IGPUImage::ET_2D,
          .samples = asset::ICPUImage::ESCF_1_BIT,
          .format = asset::EF_R16G16B16A16_SFLOAT,
          .extent = {WIN_W, WIN_H, 1},
          .mipLevels = 1,
          .arrayLayers = 1,
          .flags = IImage::ECF_NONE,
          .usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT
        }
      });

    if (!m_hdrImage || !m_device->allocate(m_hdrImage->getMemoryReqs(), m_hdrImage.get()).isValid())
      return logFail("Could not create HDR Image");

    auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
    auto* geometryCreator = assetManager->getGeometryCreator();

    auto cQueue = getComputeQueue();

    // create geometry objects
    if (!createGeometries(gQueue, geometryCreator))
      return logFail("Could not create geometries from geometry creator");

    if (!createAccelerationStructures(cQueue))
      return logFail("Could not create acceleration structures");


    // create pipelines
    {
      // descriptors
      const IGPUDescriptorSetLayout::SBinding bindings[] = {
        {
          .binding = 0,
          .type = asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
          .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
          .count = 1,
        },
        {
          .binding = 1,
          .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
          .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
          .count = 1,
        }
      };
      const auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

      const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayoutPtrs = { descriptorSetLayout.get() };
      m_renderPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs.begin(), dsLayoutPtrs.end()));
      if (!m_renderPool)
        return logFail("Could not create descriptor pool");
      m_renderDs = m_renderPool->createDescriptorSet(descriptorSetLayout);
      if (!m_renderDs)
        return logFail("Could not create descriptor set");

      const SPushConstantRange pcRange = {
        .stageFlags = IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
        .offset = 0u,
        .size = sizeof(SPushConstants),
      };
      const auto pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

      IGPURayTracingPipeline::SCreationParams params = {};


      const IGPUShader::SSpecInfo shaders[] = {
          {.shader = raygenShader.get()},
          {.shader = closestHitShader.get()},
          {.shader = anyHitShaderColorPayload.get()},
          {.shader = anyHitShaderShadowPayload.get()},
          {.shader = missShader.get()},
          {.shader = shadowMissShader.get()},
      };

      params.layout = pipelineLayout.get();
      params.shaders = std::span(shaders, std::size(shaders));
      params.cached.shaderGroups.raygenGroup = {
        .shaderIndex = 0,
      };
      params.cached.shaderGroups.hitGroups.push_back({ .closestHitShaderIndex = 1, .anyHitShaderIndex = 2 });
      params.cached.shaderGroups.hitGroups.push_back({ .closestHitShaderIndex = 1, .anyHitShaderIndex = 3 });
      params.cached.shaderGroups.missGroups.push_back({ .shaderIndex = 4 });
      params.cached.shaderGroups.missGroups.push_back({ .shaderIndex = 5 });
      params.cached.maxRecursionDepth = 2;
      if (!m_device->createRayTracingPipelines(nullptr, { &params, 1 }, &m_rayTracingPipeline))
        return logFail("Failed to create ray tracing pipeline");
      m_logger->log("Ray Tracing Pipeline Created!",system::ILogger::ELL_INFO);

      //create shader binding table
      if (!createShaderBindingTable(gQueue, m_rayTracingPipeline))
        return logFail("Could not create shader binding table");
    }


    // write descriptors
    IGPUDescriptorSet::SDescriptorInfo infos[2];
    infos[0].desc = m_gpuTlas;
    infos[1].desc = m_device->createImageView({
        .flags = IGPUImageView::ECF_NONE,
        .subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
        .image = m_hdrImage,
        .viewType = IGPUImageView::E_TYPE::ET_2D,
        .format = asset::EF_R16G16B16A16_SFLOAT
      });
    if (!infos[1].desc)
      return logFail("Failed to create image view");
    infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
    IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
        {.dstSet = m_renderDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
        {.dstSet = m_renderDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]}
    };
    m_device->updateDescriptorSets(std::span(writes, 2), {});

    // camera
    {
      core::vectorSIMDf cameraPosition(-5.81655884, 2.58630896, -4.23974705);
      core::vectorSIMDf cameraTarget(-0.349590302, -0.213266611, 0.317821503);
      matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 1000);
      m_camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);
    }

    m_winMgr->show(m_window.get());
    m_oracle.reportBeginFrameRecord();

    return true;
  }

  inline void workLoopBody() override
  {
    const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

    const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());

    if (m_realFrameIx >= framesInFlight)
    {
      const ISemaphore::SWaitInfo cbDonePending[] =
      {
          {
            .semaphore = m_semaphore.get(),
            .value = m_realFrameIx + 1 - framesInFlight
          }
      };
      if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
        return;
    }

    m_inputSystem->getDefaultMouse(&m_mouse);
    m_inputSystem->getDefaultKeyboard(&m_keyboard);

    auto updatePresentationTimestamp = [&]()
      {
        m_currentImageAcquire = m_surface->acquireNextImage();

        m_oracle.reportEndFrameRecord();
        const auto timestamp = m_oracle.getNextPresentationTimeStamp();
        m_oracle.reportBeginFrameRecord();

        return timestamp;
      };

    const auto nextPresentationTimestamp = updatePresentationTimestamp();

    if (!m_currentImageAcquire)
      return;

    static bool first = true;
    if (first)
    {
      m_api->startCapture();
      first = false;
    }

    auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
    cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
    cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
    cmdbuf->beginDebugMarker("RaytracingPipelineApp Frame");
    {
      m_camera.beginInputProcessing(nextPresentationTimestamp);
      m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
      {
        if (m_camera.mouseProcess(events)) 
        {
          m_frameAccumulationCounter = 0;
        }
      }, m_logger.get());
      m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
      {
        if (m_camera.keyboardProcess(events))
        {
          m_frameAccumulationCounter = 0;
        }
      }, m_logger.get());
      m_camera.endInputProcessing(nextPresentationTimestamp);

    }

    const auto viewMatrix = m_camera.getViewMatrix();
    const auto projectionMatrix = m_camera.getProjectionMatrix();
    const auto viewProjectionMatrix = m_camera.getConcatenatedMatrix();

    core::matrix3x4SIMD modelMatrix;
    modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
    modelMatrix.setRotation(quaternion(0, 0, 0));

    core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
    core::matrix4SIMD invModelViewProjectionMatrix;
    modelViewProjectionMatrix.getInverseTransform(invModelViewProjectionMatrix);

    auto* queue = getGraphicsQueue();

    {
      IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
      imageBarriers[0].barrier = {
         .dep = {
           .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
           .srcAccessMask = ACCESS_FLAGS::NONE,
           .dstStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
           .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
        }
      };
      imageBarriers[0].image = m_hdrImage.get();
      imageBarriers[0].subresourceRange = {
        .aspectMask = IImage::EAF_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = 1u,
        .baseArrayLayer = 0u,
        .layerCount = 1u
      };
      imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
      imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
      cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }

    // do ray query
    SPushConstants pc;
    pc.geometryInfoBuffer = m_geometryInfoBuffer->getDeviceAddress();
    pc.frameCounter = m_frameAccumulationCounter;
    const core::vector3df camPos = m_camera.getPosition().getAsVector3df();
    pc.camPos = { camPos.X, camPos.Y, camPos.Z };
    memcpy(&pc.invMVP, invModelViewProjectionMatrix.pointer(), sizeof(pc.invMVP));

    cmdbuf->bindRayTracingPipeline(m_rayTracingPipeline.get());
    cmdbuf->pushConstants(m_rayTracingPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING, 0, sizeof(SPushConstants), &pc);
    cmdbuf->bindDescriptorSets(EPBP_RAY_TRACING, m_rayTracingPipeline->getLayout(), 0, 1, &m_renderDs.get());
    cmdbuf->traceRays(m_shaderBindingTable.raygenGroupRegion, 
      m_shaderBindingTable.missGroupsRegion,
      m_shaderBindingTable.hitGroupsRegion,
      m_shaderBindingTable.callableGroupsRegion,
      WIN_W, WIN_H, 1);

    // blit
    {
      IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[2];
      imageBarriers[0].barrier = {
         .dep = {
           .srcStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
           .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
           .dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
           .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
        }
      };
      imageBarriers[0].image = m_hdrImage.get();
      imageBarriers[0].subresourceRange = {
        .aspectMask = IImage::EAF_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = 1u,
        .baseArrayLayer = 0u,
        .layerCount = 1u
      };
      imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
      imageBarriers[0].newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;

      imageBarriers[1].barrier = {
         .dep = {
           .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
           .srcAccessMask = ACCESS_FLAGS::NONE,
           .dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
           .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
        }
      };
      imageBarriers[1].image = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);
      imageBarriers[1].subresourceRange = {
        .aspectMask = IImage::EAF_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = 1u,
        .baseArrayLayer = 0u,
        .layerCount = 1u
      };
      imageBarriers[1].oldLayout = IImage::LAYOUT::UNDEFINED;
      imageBarriers[1].newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL;

      cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }

    {
      IGPUCommandBuffer::SImageBlit regions[] = { {
        .srcMinCoord = {0,0,0},
        .srcMaxCoord = {WIN_W,WIN_H,1},
        .dstMinCoord = {0,0,0},
        .dstMaxCoord = {WIN_W,WIN_H,1},
        .layerCount = 1,
        .srcBaseLayer = 0,
        .dstBaseLayer = 0,
        .srcMipLevel = 0,
        .dstMipLevel = 0,
        .aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT
      } };

      auto srcImg = m_hdrImage.get();
      auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
      auto dstImg = scRes->getImage(m_currentImageAcquire.imageIndex);

      cmdbuf->blitImage(srcImg, IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, dstImg, IImage::LAYOUT::TRANSFER_DST_OPTIMAL, regions, ISampler::ETF_NEAREST);
    }

    // TODO: transition to present
    {
      IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
      imageBarriers[0].barrier = {
         .dep = {
           .srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
           .srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
           .dstStageMask = PIPELINE_STAGE_FLAGS::NONE,
           .dstAccessMask = ACCESS_FLAGS::NONE
        }
      };
      imageBarriers[0].image = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);
      imageBarriers[0].subresourceRange = {
        .aspectMask = IImage::EAF_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = 1u,
        .baseArrayLayer = 0u,
        .layerCount = 1u
      };
      imageBarriers[0].oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
      imageBarriers[0].newLayout = IImage::LAYOUT::PRESENT_SRC;

      cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }

    cmdbuf->endDebugMarker();
    cmdbuf->end();

    {
      const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
      {
        {
          .semaphore = m_semaphore.get(),
          .value = ++m_realFrameIx,
          .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
        }
      };
      {
        {
          const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
          {
            {.cmdbuf = cmdbuf }
          };

          const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
          {
            {
              .semaphore = m_currentImageAcquire.semaphore,
              .value = m_currentImageAcquire.acquireCount,
              .stageMask = PIPELINE_STAGE_FLAGS::NONE
            }
          };
          const IQueue::SSubmitInfo infos[] =
          {
            {
              .waitSemaphores = acquired,
              .commandBuffers = commandBuffers,
              .signalSemaphores = rendered
            }
          };

          if (queue->submit(infos) == IQueue::RESULT::SUCCESS)
          {
            const nbl::video::ISemaphore::SWaitInfo waitInfos[] =
            { {
              .semaphore = m_semaphore.get(),
              .value = m_realFrameIx
            } };

            m_device->blockForSemaphores(waitInfos); // this is not solution, quick wa to not throw validation errors
          }
          else
            --m_realFrameIx;
        }
      }

      std::string caption = "[Nabla Engine] Ray Tracing Pipeline";
      {
        caption += ", displaying [all objects]";
        m_window->setCaption(caption);
      }
      m_surface->present(m_currentImageAcquire.imageIndex, rendered);
    }

    m_frameAccumulationCounter++;
  }

  inline bool keepRunning() override
  {
    if (m_surface->irrecoverable())
      return false;

    return true;
  }

  inline bool onAppTerminated() override
  {
    return device_base_t::onAppTerminated();
  }

private:
  uint32_t getWorkgroupCount(uint32_t dim, uint32_t size)
  {
    return (dim + size - 1) / size;
  }

  smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams& params)
  {
    smart_refctd_ptr<IGPUBuffer> buffer;
    buffer = m_device->createBuffer(std::move(params));
    auto bufReqs = buffer->getMemoryReqs();
    bufReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
    m_device->allocate(bufReqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

    return buffer;
  }

  smart_refctd_ptr<IGPUCommandBuffer> getSingleUseCommandBufferAndBegin(smart_refctd_ptr<IGPUCommandPool> pool)
  {
    smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
    if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
      return nullptr;

    cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
    cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

    return cmdbuf;
  }

  void cmdbufSubmitAndWait(smart_refctd_ptr<IGPUCommandBuffer> cmdbuf, CThreadSafeQueueAdapter* queue, uint64_t startValue)
  {
    cmdbuf->end();

    uint64_t finishedValue = startValue + 1;

    // submit builds
    {
      auto completed = m_device->createSemaphore(startValue);

      std::array<IQueue::SSubmitInfo::SSemaphoreInfo, 1u> signals;
      {
        auto& signal = signals.front();
        signal.value = finishedValue;
        signal.stageMask = bitflag(PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);
        signal.semaphore = completed.get();
      }

      const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
        .cmdbuf = cmdbuf.get()
      } };

      const IQueue::SSubmitInfo infos[] =
      {
        {
          .waitSemaphores = {},
          .commandBuffers = commandBuffers,
          .signalSemaphores = signals
        }
      };

      if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
      {
        m_logger->log("Failed to submit geometry transfer upload operations!", ILogger::ELL_ERROR);
        return;
      }

      const ISemaphore::SWaitInfo info[] =
      { {
        .semaphore = completed.get(),
        .value = finishedValue
      } };

      m_device->blockForSemaphores(info);
    }
  }

  bool createGeometries(video::CThreadSafeQueueAdapter* queue, const IGeometryCreator* gc)
  {
    auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
    if (!pool)
      return logFail("Couldn't create Command Pool for geometry creation!");

    const auto defaultMaterial = Material{
      .ambient = {0.1, 0.1, 0.1},
      .diffuse = {0.8, 0.3, 0.3},
      .specular = {0.8, 0.8, 0.8},
      .shininess = 1.0f,
      .illum = 2
    };

    auto getTranslationMatrix = [](float32_t x, float32_t y, float32_t z)
      {
        core::matrix3x4SIMD transform;
        transform.setTranslation(nbl::core::vectorSIMDf(x, y, z, 0));
        return transform;
      };
    
    core::matrix3x4SIMD planeTransform;
    planeTransform.setRotation(quaternion::fromAngleAxis(core::radians(-90.0f), vector3df_SIMD{1, 0, 0}));

    const auto cpuObjects = std::array{
      ReferenceObjectCpu {
        .meta = {.type = OT_RECTANGLE, .name = "Plane Mesh"},
        .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(10, 10)),
        .material = defaultMaterial,
        .transform = planeTransform,
      },
      ReferenceObjectCpu {
        .meta = {.type = OT_CUBE, .name = "Cube Mesh"},
        .data = gc->createCubeMesh(nbl::core::vector3df(1, 1, 1)),
        .material = defaultMaterial,
        .transform = getTranslationMatrix(0, 0.5f, 0),
      },
      ReferenceObjectCpu {
        .meta = {.type = OT_SPHERE, .name = "Sphere Mesh"},
        .data = gc->createSphereMesh(2, 16, 16),
        .material = {
          .ambient = {0.1, 0.1, 0.1},
          .diffuse = {0.2, 0.2, 0.8},
          .specular = {0.8, 0.8, 0.8},
          .shininess = 1.0f,
          .illum = 2
        },
        .transform = getTranslationMatrix(-5.0f, 1.0f, 0),
      },
      ReferenceObjectCpu {
        .meta = {.type = OT_SPHERE, .name = "Transparent Sphere Mesh"},
        .data = gc->createSphereMesh(2, 16, 16),
        .material = {
          .ambient = {0.1, 0.1, 0.1},
          .diffuse = {0.2, 0.8, 0.2},
          .specular = {0.8, 0.8, 0.8},
          .shininess = 1.0f,
          .dissolve = 0.2,
          .illum = 4
        },
        .transform = getTranslationMatrix(5.0f, 1.0f, 0),
      },
    };

    struct ScratchVIBindings
    {
      nbl::asset::SBufferBinding<ICPUBuffer> vertex, index;
    };
    std::array<ScratchVIBindings, std::size(cpuObjects)> scratchBuffers;

    for (uint32_t i = 0; i < cpuObjects.size(); i++)
    {
      const auto& cpuObject = cpuObjects[i];

      auto vBuffer = smart_refctd_ptr(cpuObject.data.bindings[0].buffer); // no offset
      auto vUsage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF |
        IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
      vBuffer->addUsageFlags(vUsage);
      vBuffer->setContentHash(vBuffer->computeContentHash());

      auto iBuffer = smart_refctd_ptr(cpuObject.data.indexBuffer.buffer); // no offset
      auto iUsage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF |
        IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

      if (cpuObject.data.indexType != EIT_UNKNOWN)
        if (iBuffer)
        {
          iBuffer->addUsageFlags(iUsage);
          iBuffer->setContentHash(iBuffer->computeContentHash());
        }

      scratchBuffers[i] = {
        .vertex = {.offset = 0, .buffer = vBuffer},
        .index = {.offset = 0, .buffer = iBuffer},
      };

    }

    auto cmdbuf = getSingleUseCommandBufferAndBegin(pool);
    cmdbuf->beginDebugMarker("Build geometry vertex and index buffers");

    CAssetConverter::SInputs inputs = {};
    inputs.logger = m_logger.get();
    std::array<ICPUBuffer*, std::size(cpuObjects) * 2u> tmpBuffers;
    {
      for (uint32_t i = 0; i < cpuObjects.size(); i++)
      {
        tmpBuffers[2 * i + 0] = scratchBuffers[i].vertex.buffer.get();
        tmpBuffers[2 * i + 1] = scratchBuffers[i].index.buffer.get();
      }

      std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = tmpBuffers;
    }

    auto reservation = m_converter->reserve(inputs);
    {
      auto prepass = [&]<typename asset_type_t>(const auto & references) -> bool
      {
        auto objects = reservation.getGPUObjects<asset_type_t>();
        uint32_t counter = {};
        for (auto& object : objects)
        {
          auto gpu = object.value;
          auto* reference = references[counter];

          if (reference)
          {
            if (!gpu)
            {
              m_logger->log("Failed to convert a CPU object to GPU!", ILogger::ELL_ERROR);
              return false;
            }
          }
          counter++;
        }
        return true;
      };

      prepass.template operator() < ICPUBuffer > (tmpBuffers);
    }

    auto geomInfoBuffer = ICPUBuffer::create({ std::size(cpuObjects) * sizeof(SGeomInfo) });
    SGeomInfo* geomInfos = reinterpret_cast<SGeomInfo*>(geomInfoBuffer->getPointer());

    m_gpuObjects.reserve(std::size(cpuObjects));
    // convert
    {
      // not sure if need this (probably not, originally for transition img view)
      auto semaphore = m_device->createSemaphore(0u);

      std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1> cmdbufs = {};
      cmdbufs.front().cmdbuf = cmdbuf.get();

      SIntendedSubmitInfo transfer = {};
      transfer.queue = queue;
      transfer.scratchCommandBuffers = cmdbufs;
      transfer.scratchSemaphore = {
        .semaphore = semaphore.get(),
        .value = 0u,
        .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
      };

      CAssetConverter::SConvertParams params = {};
      params.utilities = m_utils.get();
      params.transfer = &transfer;

      auto future = reservation.convert(params);
      if (future.copy() != IQueue::RESULT::SUCCESS)
      {
        m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
        return false;
      }

      auto&& buffers = reservation.getGPUObjects<ICPUBuffer>();
      for (uint32_t i = 0; i < cpuObjects.size(); i++)
      {
        auto& cpuObject = cpuObjects[i];

        m_gpuObjects.push_back(ReferenceObjectGpu{
          .meta = cpuObject.meta,
          .bindings = {
            .vertex = {.offset = 0, .buffer = buffers[2 * i + 0].value },
            .index = {.offset = 0, .buffer = buffers[2 * i + 1].value },
          },
          .vertexStride = cpuObject.data.inputParams.bindings[0].stride,
          .indexType = cpuObject.data.indexType,
          .indexCount = cpuObject.data.indexCount,
          .material = cpuObject.material,
          .transform = cpuObject.transform,
        });
      }

      for (uint32_t i = 0; i < m_gpuObjects.size(); i++)
      {
        const auto& gpuObject = m_gpuObjects[i];
        const uint64_t vertexBufferAddress = gpuObject.bindings.vertex.buffer->getDeviceAddress();
        geomInfos[i] = {
          .vertexBufferAddress = vertexBufferAddress,
          .indexBufferAddress = gpuObject.useIndex() ? gpuObject.bindings.index.buffer->getDeviceAddress() : vertexBufferAddress,
          .vertexStride = gpuObject.vertexStride,
          .indexType = gpuObject.indexType,
          .smoothNormals = s_smoothNormals[gpuObject.meta.type],
          .objType = gpuObject.meta.type,
          .material = gpuObject.material,
        };
      }
    }

    {
      IGPUBuffer::SCreationParams params;
      params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
      params.size = geomInfoBuffer->getSize();
      m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), geomInfos).move_into(m_geometryInfoBuffer);
    }

    return true;
  }

  bool createShaderBindingTable(video::CThreadSafeQueueAdapter* queue, const smart_refctd_ptr<video::IGPURayTracingPipeline>& pipeline)
  {
    const auto& limits = m_device->getPhysicalDevice()->getLimits();
    const auto handleSize = limits.shaderGroupHandleSize;
    const auto handleSizeAligned = nbl::core::alignUp(handleSize, limits.shaderGroupHandleAlignment);

    auto& raygenRegion = m_shaderBindingTable.raygenGroupRegion;
    auto& hitRegion = m_shaderBindingTable.hitGroupsRegion;
    auto& missRegion = m_shaderBindingTable.missGroupsRegion;
    auto& callableRegion = m_shaderBindingTable.callableGroupsRegion;

    raygenRegion = {
      .offset = 0,
      .stride = core::alignUp(handleSizeAligned, limits.shaderGroupBaseAlignment),
      .size = core::alignUp(handleSizeAligned, limits.shaderGroupBaseAlignment)
    };

    missRegion = {
      .offset = raygenRegion.size,
      .stride = handleSizeAligned,
      .size = core::alignUp(pipeline->getMissGroupCount(), limits.shaderGroupBaseAlignment),
    };

    hitRegion = {
      .offset = missRegion.offset + missRegion.size,
      .stride = handleSizeAligned,
      .size = core::alignUp(pipeline->getHitGroupCount(), limits.shaderGroupBaseAlignment),
    };

    callableRegion = {
      .offset = hitRegion.offset + hitRegion.size,
      .stride = handleSizeAligned,
      .size = core::alignUp(pipeline->getCallableGroupCount(), limits.shaderGroupBaseAlignment),
    };

    const auto bufferSize = raygenRegion.size + missRegion.size + hitRegion.size + callableRegion.size;

    ICPUBuffer::SCreationParams cpuBufferParams;
    cpuBufferParams.size = bufferSize;
    auto cpuBuffer = ICPUBuffer::create(std::move(cpuBufferParams));
    uint8_t* pData = reinterpret_cast<uint8_t*>(cpuBuffer->getPointer());
    
    // copy raygen region
    memcpy(pData, pipeline->getRaygenGroupShaderHandle().data(), handleSize);

    // copy miss region
    uint8_t* pMissData = pData + missRegion.offset;
    for (int32_t missIx = 0; missIx < pipeline->getMissGroupCount(); missIx++)
    {
      memcpy(pMissData, pipeline->getMissGroupShaderHandle(missIx).data(), handleSize);
      pMissData += missRegion.stride;
    }

    // copy hit region
    uint8_t* pHitData = pData + hitRegion.offset;
    for (int32_t hitIx = 0; hitIx < pipeline->getHitGroupCount(); hitIx++)
    {
      memcpy(pHitData, pipeline->getHitGroupShaderHandle(hitIx).data(), handleSize);
      pHitData += hitRegion.stride;
    }

    // copy callable region
    uint8_t* pCallableData = pData + callableRegion.offset;
    for (int32_t callableIx = 0; callableIx < pipeline->getCallableGroupCount(); callableIx++)
    {
      memcpy(pCallableData, pipeline->getCallableGroupShaderHandle(callableIx).data(), handleSize);
      pCallableData += callableRegion.stride;
    }

    {
      IGPUBuffer::SCreationParams params;
      params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT;
      params.size = bufferSize;
      m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), pData).move_into(raygenRegion.buffer);
      m_logger->log("Device address : %d", ILogger::ELL_INFO, raygenRegion.buffer->getDeviceAddress());
      missRegion.buffer = core::smart_refctd_ptr(raygenRegion.buffer);
      hitRegion.buffer = core::smart_refctd_ptr(raygenRegion.buffer);
      callableRegion.buffer = core::smart_refctd_ptr(raygenRegion.buffer);
    }

    return true;
  }

  bool createAccelerationStructures(video::CThreadSafeQueueAdapter* queue)
  {
    IQueryPool::SCreationParams qParams{ .queryCount = static_cast<uint32_t>(m_gpuObjects.size()), .queryType = IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE};
    smart_refctd_ptr<IQueryPool> queryPool = m_device->createQueryPool(std::move(qParams));

    auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
    if (!pool)
      return logFail("Couldn't create Command Pool for blas/tlas creation!");

    m_api->startCapture();
#ifdef TRY_BUILD_FOR_NGFX // NSight is "debugger-challenged" it can't capture anything not happenning "during a frame", so we need to trick it
    m_currentImageAcquire = m_surface->acquireNextImage();
    {
      const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = { {
        .semaphore = m_currentImageAcquire.semaphore,
        .value = m_currentImageAcquire.acquireCount,
        .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
      } };
      m_surface->present(m_currentImageAcquire.imageIndex, acquired);
    }
    m_currentImageAcquire = m_surface->acquireNextImage();
#endif
    size_t totalScratchSize = 0;

    // build bottom level ASes
    {
      core::vector<IGPUBottomLevelAccelerationStructure::DeviceBuildInfo> blasBuildInfos(m_gpuObjects.size());
      core::vector<uint32_t> primitiveCounts(m_gpuObjects.size());
      core::vector<IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer>> triangles(m_gpuObjects.size());
      core::vector<uint32_t> scratchSizes(m_gpuObjects.size());
      m_gpuBlasList.resize(m_gpuObjects.size());

      for (uint32_t i = 0; i < m_gpuObjects.size(); i++)
      {
        const auto& gpuObject = m_gpuObjects[i];

        const uint32_t vertexStride = gpuObject.vertexStride;
        const uint32_t numVertices = gpuObject.bindings.vertex.buffer->getSize() / vertexStride;
        if (gpuObject.useIndex())
          primitiveCounts[i] = gpuObject.indexCount / 3;
        else
          primitiveCounts[i] = numVertices / 3;

        triangles[i].vertexData[0] = gpuObject.bindings.vertex;
        triangles[i].indexData = gpuObject.useIndex() ? gpuObject.bindings.index : gpuObject.bindings.vertex;
        triangles[i].maxVertex = numVertices - 1;
        triangles[i].vertexStride = vertexStride;
        triangles[i].vertexFormat = EF_R32G32B32_SFLOAT;
        triangles[i].indexType = gpuObject.indexType;
        triangles[i].geometryFlags = IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::NO_DUPLICATE_ANY_HIT_INVOCATION_BIT;

        auto blasFlags = bitflag(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT) | IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_COMPACTION_BIT;
        if (m_physicalDevice->getProperties().limits.rayTracingPositionFetch)
          blasFlags |= IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_DATA_ACCESS_KHR;

        blasBuildInfos[i].buildFlags = blasFlags;
        blasBuildInfos[i].geometryCount = 1;	// only 1 geometry object per blas
        blasBuildInfos[i].srcAS = nullptr;
        blasBuildInfos[i].dstAS = nullptr;
        blasBuildInfos[i].triangles = &triangles[i];
        blasBuildInfos[i].scratch = {};

        ILogicalDevice::AccelerationStructureBuildSizes buildSizes;
        {
          const uint32_t maxPrimCount[1] = { primitiveCounts[i] };
          buildSizes = m_device->getAccelerationStructureBuildSizes(blasFlags, false, std::span{ &triangles[i], 1 }, maxPrimCount);
          if (!buildSizes)
            return logFail("Failed to get BLAS build sizes");
        }

        scratchSizes[i] = buildSizes.buildScratchSize;
        totalScratchSize += buildSizes.buildScratchSize;

        {
          IGPUBuffer::SCreationParams params;
          params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
          params.size = buildSizes.accelerationStructureSize;
          smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

          IGPUBottomLevelAccelerationStructure::SCreationParams blasParams;
          blasParams.bufferRange.buffer = asBuffer;
          blasParams.bufferRange.offset = 0u;
          blasParams.bufferRange.size = buildSizes.accelerationStructureSize;
          blasParams.flags = IGPUBottomLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
          m_gpuBlasList[i] = m_device->createBottomLevelAccelerationStructure(std::move(blasParams));
          if (!m_gpuBlasList[i])
            return logFail("Could not create BLAS");
        }
      }

      auto cmdbufBlas = getSingleUseCommandBufferAndBegin(pool);
      cmdbufBlas->beginDebugMarker("Build BLAS");

      cmdbufBlas->resetQueryPool(queryPool.get(), 0, m_gpuObjects.size());

      smart_refctd_ptr<IGPUBuffer> scratchBuffer;
      {
        IGPUBuffer::SCreationParams params;
        params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
        params.size = totalScratchSize;
        scratchBuffer = createBuffer(params);
      }

      uint32_t queryCount = 0;
      core::vector<IGPUBottomLevelAccelerationStructure::BuildRangeInfo> buildRangeInfos(m_gpuObjects.size());
      core::vector<IGPUBottomLevelAccelerationStructure::BuildRangeInfo*> pRangeInfos(m_gpuObjects.size());
      for (uint32_t i = 0; i < m_gpuObjects.size(); i++)
      {
        blasBuildInfos[i].dstAS = m_gpuBlasList[i].get();
        blasBuildInfos[i].scratch.buffer = scratchBuffer;
        blasBuildInfos[i].scratch.offset = (i == 0) ? 0u : blasBuildInfos[i - 1].scratch.offset + scratchSizes[i - 1];

        buildRangeInfos[i].primitiveCount = primitiveCounts[i];
        buildRangeInfos[i].primitiveByteOffset = 0u;
        buildRangeInfos[i].firstVertex = 0u;
        buildRangeInfos[i].transformByteOffset = 0u;

        pRangeInfos[i] = &buildRangeInfos[i];
      }

      if (!cmdbufBlas->buildAccelerationStructures(std::span(blasBuildInfos), pRangeInfos.data()))
        return logFail("Failed to build BLAS");

      {
        SMemoryBarrier memBarrier;
        memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
        memBarrier.srcAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
        memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
        memBarrier.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT;
        cmdbufBlas->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
      }


      core::vector<const IGPUAccelerationStructure*> ases(m_gpuObjects.size());
      for (uint32_t i = 0; i < m_gpuObjects.size(); i++)
        ases[i] = m_gpuBlasList[i].get();
      if (!cmdbufBlas->writeAccelerationStructureProperties(std::span(ases), IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE,
        queryPool.get(), queryCount++))
        return logFail("Failed to write acceleration structure properties!");

      cmdbufBlas->endDebugMarker();
      cmdbufSubmitAndWait(cmdbufBlas, queue, 39);
    }

    auto cmdbufCompact = getSingleUseCommandBufferAndBegin(pool);
    cmdbufCompact->beginDebugMarker("Compact BLAS");

    // compact blas
    {
      core::vector<size_t> asSizes(m_gpuObjects.size(), 0);
      if (!m_device->getQueryPoolResults(queryPool.get(), 0, m_gpuObjects.size(), asSizes.data(), sizeof(size_t), IQueryPool::WAIT_BIT))
        return logFail("Could not get query pool results for AS sizes");

      core::vector<smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>> cleanupBlas(m_gpuObjects.size());
      for (uint32_t i = 0; i < m_gpuObjects.size(); i++)
      {
        cleanupBlas[i] = m_gpuBlasList[i];
        {
          IGPUBuffer::SCreationParams params;
          params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
          params.size = asSizes[i];
          smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

          IGPUBottomLevelAccelerationStructure::SCreationParams blasParams;
          blasParams.bufferRange.buffer = asBuffer;
          blasParams.bufferRange.offset = 0u;
          blasParams.bufferRange.size = asSizes[i];
          blasParams.flags = IGPUBottomLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
          m_gpuBlasList[i] = m_device->createBottomLevelAccelerationStructure(std::move(blasParams));
          if (!m_gpuBlasList[i])
            return logFail("Could not create compacted BLAS");
        }

        IGPUBottomLevelAccelerationStructure::CopyInfo copyInfo;
        copyInfo.src = cleanupBlas[i].get();
        copyInfo.dst = m_gpuBlasList[i].get();
        copyInfo.mode = IGPUBottomLevelAccelerationStructure::COPY_MODE::COMPACT;
        if (!cmdbufCompact->copyAccelerationStructure(copyInfo))
          return logFail("Failed to copy AS to compact");
      }
    }

    cmdbufCompact->endDebugMarker();
    cmdbufSubmitAndWait(cmdbufCompact, queue, 40);

    auto cmdbufTlas = getSingleUseCommandBufferAndBegin(pool);
    cmdbufTlas->beginDebugMarker("Build TLAS");

    // build top level AS
    {
      const uint32_t instancesCount = m_gpuObjects.size();
      core::vector<IGPUTopLevelAccelerationStructure::DeviceStaticInstance> instances(m_gpuObjects.size());
      for (uint32_t i = 0; i < instancesCount; i++)
      {
        instances[i].base.blas.deviceAddress = m_gpuBlasList[i]->getReferenceForDeviceOperations().deviceAddress;
        instances[i].base.mask = 0xFF;
        instances[i].base.instanceCustomIndex = i;
        instances[i].base.instanceShaderBindingTableRecordOffset = 0;
        instances[i].base.flags = static_cast<uint32_t>(IGPUTopLevelAccelerationStructure::INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT);
        instances[i].transform = m_gpuObjects[i].transform;
      }

      {
        size_t bufSize = instancesCount * sizeof(IGPUTopLevelAccelerationStructure::DeviceStaticInstance);
        IGPUBuffer::SCreationParams params;
        params.usage = bitflag(IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT |
          IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
        params.size = bufSize;
        m_instanceBuffer = createBuffer(params);

        SBufferRange<IGPUBuffer> range = { .offset = 0u, .size = bufSize, .buffer = m_instanceBuffer };
        cmdbufTlas->updateBuffer(range, instances.data());
      }

      // make sure instances upload complete first
      {
        SMemoryBarrier memBarrier;
        memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
        memBarrier.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
        memBarrier.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
        cmdbufTlas->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
      }

      auto tlasFlags = bitflag(IGPUTopLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT);

      IGPUTopLevelAccelerationStructure::DeviceBuildInfo tlasBuildInfo;
      tlasBuildInfo.buildFlags = tlasFlags;
      tlasBuildInfo.srcAS = nullptr;
      tlasBuildInfo.dstAS = nullptr;
      tlasBuildInfo.instanceData.buffer = m_instanceBuffer;
      tlasBuildInfo.instanceData.offset = 0u;
      tlasBuildInfo.scratch = {};

      auto buildSizes = m_device->getAccelerationStructureBuildSizes(tlasFlags, false, instancesCount);
      if (!buildSizes)
        return logFail("Failed to get TLAS build sizes");

      {
        IGPUBuffer::SCreationParams params;
        params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
        params.size = buildSizes.accelerationStructureSize;
        smart_refctd_ptr<IGPUBuffer> asBuffer = createBuffer(params);

        IGPUTopLevelAccelerationStructure::SCreationParams tlasParams;
        tlasParams.bufferRange.buffer = asBuffer;
        tlasParams.bufferRange.offset = 0u;
        tlasParams.bufferRange.size = buildSizes.accelerationStructureSize;
        tlasParams.flags = IGPUTopLevelAccelerationStructure::SCreationParams::FLAGS::NONE;
        m_gpuTlas = m_device->createTopLevelAccelerationStructure(std::move(tlasParams));
        if (!m_gpuTlas)
          return logFail("Could not create TLAS");
      }

      smart_refctd_ptr<IGPUBuffer> scratchBuffer;
      {
        IGPUBuffer::SCreationParams params;
        params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
        params.size = buildSizes.buildScratchSize;
        scratchBuffer = createBuffer(params);
      }

      tlasBuildInfo.dstAS = m_gpuTlas.get();
      tlasBuildInfo.scratch.buffer = scratchBuffer;
      tlasBuildInfo.scratch.offset = 0u;

      IGPUTopLevelAccelerationStructure::BuildRangeInfo buildRangeInfo[1u];
      buildRangeInfo[0].instanceCount = instancesCount;
      buildRangeInfo[0].instanceByteOffset = 0u;
      IGPUTopLevelAccelerationStructure::BuildRangeInfo* pRangeInfos;
      pRangeInfos = &buildRangeInfo[0];

      if (!cmdbufTlas->buildAccelerationStructures({ &tlasBuildInfo, 1 }, pRangeInfos))
        return logFail("Failed to build TLAS");
    }

    cmdbufTlas->endDebugMarker();
    cmdbufSubmitAndWait(cmdbufTlas, queue, 45);

#ifdef TRY_BUILD_FOR_NGFX
    {
      const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = { {
        .semaphore = m_currentImageAcquire.semaphore,
        .value = m_currentImageAcquire.acquireCount,
        .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
      } };
      m_surface->present(m_currentImageAcquire.imageIndex, acquired);
    }
#endif
    m_api->endCapture();

    return true;
  }


  smart_refctd_ptr<IWindow> m_window;
  smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
  smart_refctd_ptr<ISemaphore> m_semaphore;
  uint64_t m_realFrameIx = 0;
  uint32_t m_frameAccumulationCounter = -1;
  std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
  ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

  core::smart_refctd_ptr<InputSystem> m_inputSystem;
  InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
  InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

  Camera m_camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
  CameraView m_oldCameraView;
  video::CDumbPresentationOracle m_oracle;

  std::vector<ReferenceObjectGpu> m_gpuObjects;

  std::vector<smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>> m_gpuBlasList;
  smart_refctd_ptr<IGPUTopLevelAccelerationStructure> m_gpuTlas;
  smart_refctd_ptr<IGPUBuffer> m_instanceBuffer;

  smart_refctd_ptr<IGPUBuffer> m_geometryInfoBuffer;
  ShaderBindingTable m_shaderBindingTable;
  smart_refctd_ptr<IGPUImage> m_hdrImage;

  smart_refctd_ptr<IGPURayTracingPipeline> m_rayTracingPipeline;
  smart_refctd_ptr<IGPUDescriptorSet> m_renderDs;
  smart_refctd_ptr<IDescriptorPool> m_renderPool;

  smart_refctd_ptr<CAssetConverter> m_converter;
  smart_refctd_ptr<IGPUBuffer> m_sbtBuffer;

  uint16_t gcIndex = {};

};

NBL_MAIN_FUNC(RaytracingPipelineApp)
