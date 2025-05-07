// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/builtin/hlsl/indirect_commands.hlsl"

class RaytracingPipelineApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
  using device_base_t = examples::SimpleWindowedApplication;
  using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
  using clock_t = std::chrono::steady_clock;

  constexpr static inline uint32_t WIN_W = 1280, WIN_H = 720;
  constexpr static inline uint32_t MaxFramesInFlight = 3u;
  constexpr static inline uint8_t MaxUITextureCount = 1u;
  constexpr static inline uint32_t NumberOfProceduralGeometries = 5;

  static constexpr const char* s_lightTypeNames[E_LIGHT_TYPE::ELT_COUNT] = {
    "Directional",
    "Point",
    "Spot"
  };

  constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

  struct ShaderBindingTable
  {
    SBufferRange<IGPUBuffer> raygenGroupRange;
    SBufferRange<IGPUBuffer> hitGroupsRange;
    uint32_t hitGroupsStride;
    SBufferRange<IGPUBuffer> missGroupsRange;
    uint32_t missGroupsStride;
    SBufferRange<IGPUBuffer> callableGroupsRange;
    uint32_t callableGroupsStride;
  };


public:
  inline RaytracingPipelineApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
    : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
  {
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

    smart_refctd_ptr<IShaderCompiler::CCache> shaderReadCache = nullptr;
    smart_refctd_ptr<IShaderCompiler::CCache> shaderWriteCache = core::make_smart_refctd_ptr<IShaderCompiler::CCache>();
    auto shaderCachePath = localOutputCWD / "main_pipeline_shader_cache.bin";

    {
        core::smart_refctd_ptr<system::IFile> shaderReadCacheFile;
        {
            system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
            m_system->createFile(future, shaderCachePath.c_str(), system::IFile::ECF_READ);
            if (future.wait())
            {
                future.acquire().move_into(shaderReadCacheFile);
                if (shaderReadCacheFile)
                {
                    const size_t size = shaderReadCacheFile->getSize();
                    if (size > 0ull)
                    {
                        std::vector<uint8_t> contents(size);
                        system::IFile::success_t succ;
                        shaderReadCacheFile->read(succ, contents.data(), 0, size);
                        if (succ)
                            shaderReadCache = IShaderCompiler::CCache::deserialize(contents);
                    }
                }
            }
            else
                m_logger->log("Failed Openning Shader Cache File.", ILogger::ELL_ERROR);
        }

    }

    // Load Custom Shader
    auto loadCompileAndCreateShader = [&](const std::string& relPath) -> smart_refctd_ptr<IGPUShader>
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            auto assetBundle = m_assetMgr->getAsset(relPath, lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return nullptr;

            // lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
            auto sourceRaw = IAsset::castDown<ICPUShader>(assets[0]);
            if (!sourceRaw)
                return nullptr;

            return m_device->createShader({ sourceRaw.get(), nullptr, shaderReadCache.get(), shaderWriteCache.get() });
        };

    // load shaders
    const auto raygenShader = loadCompileAndCreateShader("app_resources/raytrace.rgen.hlsl");
    const auto closestHitShader = loadCompileAndCreateShader("app_resources/raytrace.rchit.hlsl");
    const auto proceduralClosestHitShader = loadCompileAndCreateShader("app_resources/raytrace_procedural.rchit.hlsl");
    const auto intersectionHitShader = loadCompileAndCreateShader("app_resources/raytrace.rint.hlsl");
    const auto anyHitShaderColorPayload = loadCompileAndCreateShader("app_resources/raytrace.rahit.hlsl");
    const auto anyHitShaderShadowPayload = loadCompileAndCreateShader("app_resources/raytrace_shadow.rahit.hlsl");
    const auto missShader = loadCompileAndCreateShader("app_resources/raytrace.rmiss.hlsl");
    const auto missShadowShader = loadCompileAndCreateShader("app_resources/raytrace_shadow.rmiss.hlsl");
    const auto directionalLightCallShader = loadCompileAndCreateShader("app_resources/light_directional.rcall.hlsl");
    const auto pointLightCallShader = loadCompileAndCreateShader("app_resources/light_point.rcall.hlsl");
    const auto spotLightCallShader = loadCompileAndCreateShader("app_resources/light_spot.rcall.hlsl");
    const auto fragmentShader = loadCompileAndCreateShader("app_resources/present.frag.hlsl");

    core::smart_refctd_ptr<system::IFile> shaderWriteCacheFile;
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_system->deleteFile(shaderCachePath); // temp solution instead of trimming, to make sure we won't have corrupted json
        m_system->createFile(future, shaderCachePath.c_str(), system::IFile::ECF_WRITE);
        if (future.wait())
        {
            future.acquire().move_into(shaderWriteCacheFile);
            if (shaderWriteCacheFile)
            {
                auto serializedCache = shaderWriteCache->serialize();
                if (shaderWriteCacheFile)
                {
                    system::IFile::success_t succ;
                    shaderWriteCacheFile->write(succ, serializedCache->getPointer(), 0, serializedCache->getSize());
                    if (!succ)
                        m_logger->log("Failed Writing To Shader Cache File.", ILogger::ELL_ERROR);
                }
            }
            else
                m_logger->log("Failed Creating Shader Cache File.", ILogger::ELL_ERROR);
        }
        else
            m_logger->log("Failed Creating Shader Cache File.", ILogger::ELL_ERROR);
    }

    m_semaphore = m_device->createSemaphore(m_realFrameIx);
    if (!m_semaphore)
      return logFail("Failed to Create a Semaphore!");

    auto gQueue = getGraphicsQueue();

    // Create renderpass and init surface
    nbl::video::IGPURenderpass* renderpass;
    {
      ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
      if (!swapchainParams.deduceFormat(m_physicalDevice))
        return logFail("Could not choose a Surface Format for the Swapchain!");

      const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
      {
        {
          .srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
          .dstSubpass = 0,
          .memoryBarrier =
          {
            .srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
            .srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
            .dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
          }
        },
        {
          .srcSubpass = 0,
          .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
          .memoryBarrier =
          {
            .srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
          }
        },
        IGPURenderpass::SCreationParams::DependenciesEnd
      };

      auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
      renderpass = scResources->getRenderpass();

      if (!renderpass)
        return logFail("Failed to create Renderpass!");

      if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
        return logFail("Could not create Window & Surface or initialize the Surface!");
    }

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
          .samples = ICPUImage::ESCF_1_BIT,
          .format = EF_R16G16B16A16_SFLOAT,
          .extent = {WIN_W, WIN_H, 1},
          .mipLevels = 1,
          .arrayLayers = 1,
          .flags = IImage::ECF_NONE,
          .usage = bitflag(IImage::EUF_STORAGE_BIT) | IImage::EUF_TRANSFER_SRC_BIT | IImage::EUF_SAMPLED_BIT
        }
      });

    if (!m_hdrImage || !m_device->allocate(m_hdrImage->getMemoryReqs(), m_hdrImage.get()).isValid())
      return logFail("Could not create HDR Image");

    m_hdrImageView = m_device->createImageView({
      .flags = IGPUImageView::ECF_NONE,
      .subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT | IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
      .image = m_hdrImage,
      .viewType = IGPUImageView::E_TYPE::ET_2D,
      .format = asset::EF_R16G16B16A16_SFLOAT
    });



    // ray trace pipeline and descriptor set layout setup
    {
      const IGPUDescriptorSetLayout::SBinding bindings[] = {
        {
          .binding = 0,
          .type = asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
          .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_RAYGEN,
          .count = 1,
        },
        {
          .binding = 1,
          .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
          .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_RAYGEN,
          .count = 1,
        }
      };
      const auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

      const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayoutPtrs = { descriptorSetLayout.get() };
      m_rayTracingDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs.begin(), dsLayoutPtrs.end()));
      m_rayTracingDs = m_rayTracingDsPool->createDescriptorSet(descriptorSetLayout);

      const SPushConstantRange pcRange = {
        .stageFlags = IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
        .offset = 0u,
        .size = sizeof(SPushConstants),
      };
      const auto pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

      IGPURayTracingPipeline::SCreationParams params = {};

      enum RtDemoShader
      {
        RTDS_RAYGEN,
        RTDS_MISS,
        RTDS_MISS_SHADOW,
        RTDS_CLOSEST_HIT,
        RTDS_SPHERE_CLOSEST_HIT,
        RTDS_ANYHIT_PRIMARY,
        RTDS_ANYHIT_SHADOW,
        RTDS_INTERSECTION,
        RTDS_DIRECTIONAL_CALL,
        RTDS_POINT_CALL,
        RTDS_SPOT_CALL,
        RTDS_COUNT
      };

      IGPUShader::SSpecInfo shaders[RTDS_COUNT];
      shaders[RTDS_RAYGEN] = {.shader = raygenShader.get()};
      shaders[RTDS_MISS] = {.shader = missShader.get()};
      shaders[RTDS_MISS_SHADOW] = { .shader = missShadowShader.get() };
      shaders[RTDS_CLOSEST_HIT] = {.shader = closestHitShader.get()};
      shaders[RTDS_SPHERE_CLOSEST_HIT] = {.shader = proceduralClosestHitShader.get()};
      shaders[RTDS_ANYHIT_PRIMARY] = {.shader = anyHitShaderColorPayload.get()};
      shaders[RTDS_ANYHIT_SHADOW] = {.shader = anyHitShaderShadowPayload.get()};
      shaders[RTDS_INTERSECTION] = {.shader = intersectionHitShader.get() };
      shaders[RTDS_DIRECTIONAL_CALL] = {.shader = directionalLightCallShader.get()};
      shaders[RTDS_POINT_CALL] = {.shader = pointLightCallShader.get()};
      shaders[RTDS_SPOT_CALL] = {.shader = spotLightCallShader.get()};

      params.layout = pipelineLayout.get();
      params.shaders = std::span(shaders);
      using RayTracingFlags = IGPURayTracingPipeline::SCreationParams::FLAGS;
      params.flags = core::bitflag(RayTracingFlags::NO_NULL_MISS_SHADERS) |
        RayTracingFlags::NO_NULL_INTERSECTION_SHADERS | 
        RayTracingFlags::NO_NULL_ANY_HIT_SHADERS;

      auto& shaderGroups = params.shaderGroups;

      shaderGroups.raygen = { .index = RTDS_RAYGEN };

      IRayTracingPipelineBase::SGeneralShaderGroup missGroups[EMT_COUNT];
      missGroups[EMT_PRIMARY] = { .index = RTDS_MISS };
      missGroups[EMT_OCCLUSION] = { .index = RTDS_MISS_SHADOW };
      shaderGroups.misses = missGroups;

      auto getHitGroupIndex = [](E_GEOM_TYPE geomType, E_RAY_TYPE rayType)
        {
          return geomType * ERT_COUNT + rayType;
        };
      IRayTracingPipelineBase::SHitShaderGroup hitGroups[E_RAY_TYPE::ERT_COUNT * E_GEOM_TYPE::EGT_COUNT];
      hitGroups[getHitGroupIndex(EGT_TRIANGLES, ERT_PRIMARY)] = {
        .closestHit = RTDS_CLOSEST_HIT,
        .anyHit = RTDS_ANYHIT_PRIMARY,
      };
      hitGroups[getHitGroupIndex(EGT_TRIANGLES, ERT_OCCLUSION)] = {
        .closestHit = IGPURayTracingPipeline::SGeneralShaderGroup::Unused,
        .anyHit = RTDS_ANYHIT_SHADOW,
      };
      hitGroups[getHitGroupIndex(EGT_PROCEDURAL, ERT_PRIMARY)] = {
        .closestHit = RTDS_SPHERE_CLOSEST_HIT,
        .anyHit = RTDS_ANYHIT_PRIMARY,
        .intersection = RTDS_INTERSECTION,
      };
      hitGroups[getHitGroupIndex(EGT_PROCEDURAL, ERT_OCCLUSION)] = {
        .closestHit = IGPURayTracingPipeline::SGeneralShaderGroup::Unused,
        .anyHit = RTDS_ANYHIT_SHADOW,
        .intersection = RTDS_INTERSECTION,
      };
      shaderGroups.hits = hitGroups;

      IRayTracingPipelineBase::SGeneralShaderGroup callableGroups[ELT_COUNT];
      callableGroups[ELT_DIRECTIONAL] = { .index = RTDS_DIRECTIONAL_CALL };
      callableGroups[ELT_POINT] = { .index = RTDS_POINT_CALL };
      callableGroups[ELT_SPOT] = { .index = RTDS_SPOT_CALL };
      shaderGroups.callables = callableGroups;

      params.cached.maxRecursionDepth = 1;
      params.cached.dynamicStackSize = true;

      if (!m_device->createRayTracingPipelines(nullptr, { &params, 1 }, &m_rayTracingPipeline))
        return logFail("Failed to create ray tracing pipeline");

      calculateRayTracingStackSize(m_rayTracingPipeline);
      
      if (!createShaderBindingTable(gQueue, m_rayTracingPipeline))
        return logFail("Could not create shader binding table");

    }

    auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
    auto* geometryCreator = assetManager->getGeometryCreator();

    if (!createIndirectBuffer(gQueue))
      return logFail("Could not create indirect buffer");

    // create geometry objects
    if (!createGeometries(gQueue, geometryCreator))
      return logFail("Could not create geometries from geometry creator");

    if (!createAccelerationStructures(getComputeQueue()))
      return logFail("Could not create acceleration structures");

    ISampler::SParams samplerParams = {
      .AnisotropicFilter = 0
    };
    auto defaultSampler = m_device->createSampler(samplerParams);

    {
      const IGPUDescriptorSetLayout::SBinding bindings[] = {
        {
          .binding = 0u,
          .type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
          .createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
          .count = 1u,
          .immutableSamplers = &defaultSampler
        }
      };
      auto gpuPresentDescriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
      const video::IGPUDescriptorSetLayout* const layouts[] = { gpuPresentDescriptorSetLayout.get() };
      const uint32_t setCounts[] = { 1u };
      m_presentDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
      m_presentDs = m_presentDsPool->createDescriptorSet(gpuPresentDescriptorSetLayout);

      auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
      ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
      if (!fsTriProtoPPln)
        return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

      const IGPUShader::SSpecInfo fragSpec = {
        .entryPoint = "main",
        .shader = fragmentShader.get()
      };

      auto presentLayout = m_device->createPipelineLayout(
        {},
        core::smart_refctd_ptr(gpuPresentDescriptorSetLayout),
        nullptr,
        nullptr,
        nullptr
      );
      m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass());
      if (!m_presentPipeline)
        return logFail("Could not create Graphics Pipeline!");
    }

    // write descriptors
    IGPUDescriptorSet::SDescriptorInfo infos[3];
    infos[0].desc = m_gpuTlas;

    infos[1].desc = m_hdrImageView;
    if (!infos[1].desc)
      return logFail("Failed to create image view");
    infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;

    infos[2].desc = m_hdrImageView;
    infos[2].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

    IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
        {.dstSet = m_rayTracingDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
        {.dstSet = m_rayTracingDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
        {.dstSet = m_presentDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[2] },
    };
    m_device->updateDescriptorSets(std::span(writes), {});

    // gui descriptor setup
    {
      using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
      {
        IGPUSampler::SParams params;
        params.AnisotropicFilter = 1u;
        params.TextureWrapU = ETC_REPEAT;
        params.TextureWrapV = ETC_REPEAT;
        params.TextureWrapW = ETC_REPEAT;

        m_ui.samplers.gui = m_device->createSampler(params);
        m_ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
      }

      std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
      for (auto& it : immutableSamplers)
        it = smart_refctd_ptr(m_ui.samplers.scene);

      immutableSamplers[nbl::ext::imgui::UI::FontAtlasTexId] = smart_refctd_ptr(m_ui.samplers.gui);

      nbl::ext::imgui::UI::SCreationParameters params;

      params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
      params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
      params.assetManager = m_assetMgr;
      params.pipelineCache = nullptr;
      params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
      params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
      params.streamingBuffer = nullptr;
      params.subpassIx = 0u;
      params.transfer = getTransferUpQueue();
      params.utilities = m_utils;
      {
        m_ui.manager = ext::imgui::UI::create(std::move(params));

        // note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
        const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
        const auto& params = m_ui.manager->getCreationParameters();

        IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
        descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
        descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = MaxUITextureCount;
        descriptorPoolInfo.maxSets = 1u;
        descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

        m_guiDescriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
        assert(m_guiDescriptorSetPool);

        m_guiDescriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
        assert(m_ui.descriptorSet);
      }
    }

    m_ui.manager->registerListener(
      [this]() -> void {
        ImGuiIO& io = ImGui::GetIO();

        m_camera.setProjectionMatrix([&]()
        {
          static matrix4SIMD projection;

          projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
            core::radians(m_cameraSetting.fov), 
            io.DisplaySize.x / io.DisplaySize.y, 
            m_cameraSetting.zNear, 
            m_cameraSetting.zFar);

          return projection;
        }());

        ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

        // create a window and insert the inspector
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
        ImGui::Begin("Controls");

        ImGui::SameLine();

        ImGui::Text("Camera");

        ImGui::SliderFloat("Move speed", &m_cameraSetting.moveSpeed, 0.1f, 10.f);
        ImGui::SliderFloat("Rotate speed", &m_cameraSetting.rotateSpeed, 0.1f, 10.f);
        ImGui::SliderFloat("Fov", &m_cameraSetting.fov, 20.f, 150.f);
        ImGui::SliderFloat("zNear", &m_cameraSetting.zNear, 0.1f, 100.f);
        ImGui::SliderFloat("zFar", &m_cameraSetting.zFar, 110.f, 10000.f);
        Light m_oldLight = m_light;
        int light_type = m_light.type;
        ImGui::ListBox("LightType", &light_type, s_lightTypeNames, ELT_COUNT);
        m_light.type = static_cast<E_LIGHT_TYPE>(light_type);
        if (m_light.type == ELT_DIRECTIONAL)
        {
          ImGui::SliderFloat3("Light Direction", &m_light.direction.x, -1.f, 1.f);
        } else if (m_light.type == ELT_POINT)
        {
          ImGui::SliderFloat3("Light Position", &m_light.position.x, -20.f, 20.f);
        } else if (m_light.type == ELT_SPOT)
        {
          ImGui::SliderFloat3("Light Direction", &m_light.direction.x, -1.f, 1.f);
          ImGui::SliderFloat3("Light Position", &m_light.position.x, -20.f, 20.f);

          float32_t dOuterCutoff = hlsl::degrees(acos(m_light.outerCutoff));
          if (ImGui::SliderFloat("Light Outer Cutoff", &dOuterCutoff, 0.0f, 45.0f))
          {
            m_light.outerCutoff = cos(hlsl::radians(dOuterCutoff));
          }
        }
        ImGui::Checkbox("Use Indirect Command", &m_useIndirectCommand);
        if (m_light != m_oldLight)
        {
          m_frameAccumulationCounter = 0;
        }

        ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);

        ImGui::End();
      }
    );

    // Set Camera
    {
      core::vectorSIMDf cameraPosition(0, 5, -10);
      matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
        core::radians(60.0f),
        WIN_W / WIN_H,
        0.01f,
        500.0f
      );
      m_camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);
    }

    m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
    m_surface->recreateSwapchain();
    m_winMgr->show(m_window.get());
    m_oracle.reportBeginFrameRecord();
    m_camera.mapKeysToWASD();

    return true;
  }

  bool updateGUIDescriptorSet()
  {
    // texture atlas, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
    static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
    static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

    descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
    descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

    for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
    {
      writes[i].dstSet = m_ui.descriptorSet.get();
      writes[i].binding = 0u;
      writes[i].arrayElement = i;
      writes[i].count = 1u;
    }
    writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;

    return m_device->updateDescriptorSets(writes, {});
  }

  inline void workLoopBody() override
  {
    // framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
    const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
    // We block for semaphores for 2 reasons here:
      // A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
      // B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
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
    const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

    m_api->startCapture();

    update();

    auto queue = getGraphicsQueue();
    auto cmdbuf = m_cmdBufs[resourceIx].get();

    if (!keepRunning())
      return;

    cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
    cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
    cmdbuf->beginDebugMarker("RaytracingPipelineApp Frame");

    const auto viewMatrix = m_camera.getViewMatrix();
    const auto projectionMatrix = m_camera.getProjectionMatrix();
    const auto viewProjectionMatrix = m_camera.getConcatenatedMatrix();

    core::matrix3x4SIMD modelMatrix;
    modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
    modelMatrix.setRotation(quaternion(0, 0, 0));

    core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
    if (m_cachedModelViewProjectionMatrix != modelViewProjectionMatrix)
    {
      m_frameAccumulationCounter = 0;
      m_cachedModelViewProjectionMatrix = modelViewProjectionMatrix;
    }
    core::matrix4SIMD invModelViewProjectionMatrix;
    modelViewProjectionMatrix.getInverseTransform(invModelViewProjectionMatrix);

    {
      IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
      imageBarriers[0].barrier = {
         .dep = {
           .srcStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, // previous frame read from framgent shader
           .srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
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
      imageBarriers[0].oldLayout = m_frameAccumulationCounter == 0 ? IImage::LAYOUT::UNDEFINED : IImage::LAYOUT::READ_ONLY_OPTIMAL;
      imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
      cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }

    // Trace Rays Pass
    {
      SPushConstants pc;
      pc.light = m_light;
      pc.proceduralGeomInfoBuffer = m_proceduralGeomInfoBuffer->getDeviceAddress();
      pc.triangleGeomInfoBuffer = m_triangleGeomInfoBuffer->getDeviceAddress();
      pc.frameCounter = m_frameAccumulationCounter;
      const core::vector3df camPos = m_camera.getPosition().getAsVector3df();
      pc.camPos = { camPos.X, camPos.Y, camPos.Z };
      memcpy(&pc.invMVP, invModelViewProjectionMatrix.pointer(), sizeof(pc.invMVP));

      cmdbuf->bindRayTracingPipeline(m_rayTracingPipeline.get());
      cmdbuf->setRayTracingPipelineStackSize(m_rayTracingStackSize);
      cmdbuf->pushConstants(m_rayTracingPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING, 0, sizeof(SPushConstants), &pc);
      cmdbuf->bindDescriptorSets(EPBP_RAY_TRACING, m_rayTracingPipeline->getLayout(), 0, 1, &m_rayTracingDs.get());
      if (m_useIndirectCommand)
      {
        cmdbuf->traceRaysIndirect(
          SBufferBinding<const IGPUBuffer>{
            .offset = 0,
            .buffer = m_indirectBuffer,
          });
      }else
      {
        cmdbuf->traceRays(
          m_shaderBindingTable.raygenGroupRange,
          m_shaderBindingTable.missGroupsRange, m_shaderBindingTable.missGroupsStride,
          m_shaderBindingTable.hitGroupsRange, m_shaderBindingTable.hitGroupsStride,
          m_shaderBindingTable.callableGroupsRange, m_shaderBindingTable.callableGroupsStride,
          WIN_W, WIN_H, 1);
      }
    }

    // pipeline barrier
    {
      IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
      imageBarriers[0].barrier = {
        .dep = {
          .srcStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
          .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
          .dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
          .dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
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
      imageBarriers[0].oldLayout = IImage::LAYOUT::GENERAL;
      imageBarriers[0].newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

      cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }

    {
			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WIN_W;
				viewport.height = WIN_H;
			}
			cmdbuf->setViewport(0u, 1u, &viewport);


			VkRect2D defaultScisors[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
			cmdbuf->setScissor(defaultScisors);

      auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
      const VkRect2D currentRenderArea =
      {
        .offset = {0,0},
        .extent = {m_window->getWidth(),m_window->getHeight()}
      };
      const IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
      const IGPUCommandBuffer::SRenderpassBeginInfo info =
      {
        .framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
        .colorClearValues = &clearColor,
        .depthStencilClearValues = nullptr,
        .renderArea = currentRenderArea
      };
      nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

      cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

      cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
      cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_presentPipeline->getLayout(), 0, 1u, &m_presentDs.get());
      ext::FullScreenTriangle::recordDrawCall(cmdbuf);

      const auto uiParams = m_ui.manager->getCreationParameters();
      auto* uiPipeline = m_ui.manager->getPipeline();
      cmdbuf->bindGraphicsPipeline(uiPipeline);
      cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
      m_ui.manager->render(cmdbuf, waitInfo);

      cmdbuf->endRenderPass();

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

          updateGUIDescriptorSet();

          if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
            m_realFrameIx--;
        }
      }

      m_window->setCaption("[Nabla Engine] Ray Tracing Pipeline");
      m_surface->present(m_currentImageAcquire.imageIndex, rendered);
    }
    m_api->endCapture();
    m_frameAccumulationCounter++;
  }

  inline void update()
  {
    m_camera.setMoveSpeed(m_cameraSetting.moveSpeed);
    m_camera.setRotateSpeed(m_cameraSetting.rotateSpeed);

    static std::chrono::microseconds previousEventTimestamp{};

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

    struct
    {
      std::vector<SMouseEvent> mouse{};
      std::vector<SKeyboardEvent> keyboard{};
    } capturedEvents;

    m_camera.beginInputProcessing(nextPresentationTimestamp);
    {
      m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
        {
          m_camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

          for (const auto& e : events) // here capture
          {
            if (e.timeStamp < previousEventTimestamp)
              continue;

            previousEventTimestamp = e.timeStamp;
            capturedEvents.mouse.emplace_back(e);

          }
        }, m_logger.get());

      m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
        {
          m_camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

          for (const auto& e : events) // here capture
          {
            if (e.timeStamp < previousEventTimestamp)
              continue;

            previousEventTimestamp = e.timeStamp;
            capturedEvents.keyboard.emplace_back(e);
          }
        }, m_logger.get());

    }
    m_camera.endInputProcessing(nextPresentationTimestamp);

    const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
    const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
    const auto cursorPosition = m_window->getCursorControl()->getPosition();
    const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY());

    const ext::imgui::UI::SUpdateParameters params =
    {
      .mousePosition = mousePosition,
      .displaySize = { m_window->getWidth(), m_window->getHeight() },
      .mouseEvents = mouseEvents,
      .keyboardEvents = keyboardEvents
    };

    m_ui.manager->update(params);
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

  bool createIndirectBuffer(video::CThreadSafeQueueAdapter* queue)
  {
    const auto getBufferRangeAddress = [](const SBufferRange<IGPUBuffer>& range)
      {
        return range.buffer->getDeviceAddress() + range.offset;
      };
    const auto command = TraceRaysIndirectCommand_t{
      .raygenShaderRecordAddress = getBufferRangeAddress(m_shaderBindingTable.raygenGroupRange),
      .raygenShaderRecordSize = m_shaderBindingTable.raygenGroupRange.size,
      .missShaderBindingTableAddress = getBufferRangeAddress(m_shaderBindingTable.missGroupsRange),
      .missShaderBindingTableSize = m_shaderBindingTable.missGroupsRange.size,
      .missShaderBindingTableStride = m_shaderBindingTable.missGroupsStride,
      .hitShaderBindingTableAddress = getBufferRangeAddress(m_shaderBindingTable.hitGroupsRange),
      .hitShaderBindingTableSize = m_shaderBindingTable.hitGroupsRange.size,
      .hitShaderBindingTableStride = m_shaderBindingTable.hitGroupsStride,
      .callableShaderBindingTableAddress = getBufferRangeAddress(m_shaderBindingTable.callableGroupsRange),
      .callableShaderBindingTableSize = m_shaderBindingTable.callableGroupsRange.size,
      .callableShaderBindingTableStride = m_shaderBindingTable.callableGroupsStride,
      .width = WIN_W,
      .height = WIN_H,
      .depth = 1,
    };
    IGPUBuffer::SCreationParams params;
    params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INDIRECT_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
    params.size = sizeof(TraceRaysIndirectCommand_t);
    m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), &command).move_into(m_indirectBuffer);
    return true;
  }

  bool createGeometries(video::CThreadSafeQueueAdapter* queue, const IGeometryCreator* gc)
  {
    auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
    if (!pool)
      return logFail("Couldn't create Command Pool for geometry creation!");

    const auto defaultMaterial = Material{
      .ambient = {0.2, 0.1, 0.1},
      .diffuse = {0.8, 0.3, 0.3},
      .specular = {0.8, 0.8, 0.8},
      .shininess = 1.0f,
      .alpha = 1.0f,
    };

    auto getTranslationMatrix = [](float32_t x, float32_t y, float32_t z)
      {
        core::matrix3x4SIMD transform;
        transform.setTranslation(nbl::core::vectorSIMDf(x, y, z, 0));
        return transform;
      };

    core::matrix3x4SIMD planeTransform;
    planeTransform.setRotation(quaternion::fromAngleAxis(core::radians(-90.0f), vector3df_SIMD{ 1, 0, 0 }));

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
        .meta = {.type = OT_CUBE, .name = "Cube Mesh 2"},
        .data = gc->createCubeMesh(nbl::core::vector3df(1.5, 1.5, 1.5)),
        .material = Material{
          .ambient = {0.1, 0.1, 0.2},
          .diffuse = {0.2, 0.2, 0.8},
          .specular = {0.8, 0.8, 0.8},
          .shininess = 1.0f,
        },
        .transform = getTranslationMatrix(-5.0f, 1.0f, 0),
      },
      ReferenceObjectCpu {
        .meta = {.type = OT_CUBE, .name = "Transparent Cube Mesh"},
        .data = gc->createCubeMesh(nbl::core::vector3df(1.5, 1.5, 1.5)),
        .material = Material{
          .ambient = {0.1, 0.2, 0.1},
          .diffuse = {0.2, 0.8, 0.2},
          .specular = {0.8, 0.8, 0.8},
          .shininess = 1.0f,
          .alpha = 0.2,
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

    auto geomInfoBuffer = ICPUBuffer::create({ std::size(cpuObjects) * sizeof(STriangleGeomInfo) });
    STriangleGeomInfo* geomInfos = reinterpret_cast<STriangleGeomInfo*>(geomInfoBuffer->getPointer());

    m_gpuTriangleGeometries.reserve(std::size(cpuObjects));
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

        m_gpuTriangleGeometries.push_back(ReferenceObjectGpu{
          .meta = cpuObject.meta,
          .bindings = {
            .vertex = {.offset = 0, .buffer = buffers[2 * i + 0].value },
            .index = {.offset = 0, .buffer = buffers[2 * i + 1].value },
          },
          .vertexStride = cpuObject.data.inputParams.bindings[0].stride,
          .indexType = cpuObject.data.indexType,
          .indexCount = cpuObject.data.indexCount,
          .material = hlsl::_static_cast<MaterialPacked>(cpuObject.material),
          .transform = cpuObject.transform,
          });
      }

      for (uint32_t i = 0; i < m_gpuTriangleGeometries.size(); i++)
      {
        const auto& gpuObject = m_gpuTriangleGeometries[i];
        const uint64_t vertexBufferAddress = gpuObject.bindings.vertex.buffer->getDeviceAddress();
        geomInfos[i] = {
          .material = gpuObject.material,
          .vertexBufferAddress = vertexBufferAddress,
          .indexBufferAddress = gpuObject.useIndex() ? gpuObject.bindings.index.buffer->getDeviceAddress() : vertexBufferAddress,
          .vertexStride = gpuObject.vertexStride,
          .objType = gpuObject.meta.type,
          .indexType = gpuObject.indexType,
          .smoothNormals = s_smoothNormals[gpuObject.meta.type],
        };
      }
    }

    {
      IGPUBuffer::SCreationParams params;
      params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
      params.size = geomInfoBuffer->getSize();
      m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), geomInfos).move_into(m_triangleGeomInfoBuffer);
    }

    // intersection geometries setup
    {
      core::vector<SProceduralGeomInfo> proceduralGeoms;
      proceduralGeoms.reserve(NumberOfProceduralGeometries);
      using Aabb = IGPUBottomLevelAccelerationStructure::AABB_t;
      core::vector<Aabb> aabbs;
      aabbs.reserve(NumberOfProceduralGeometries);
      for (int32_t i = 0; i < NumberOfProceduralGeometries; i++)
      {
        const auto middle_i = NumberOfProceduralGeometries / 2.0;
        SProceduralGeomInfo sphere = {
          .material = hlsl::_static_cast<MaterialPacked>(Material{
            .ambient = {0.1, 0.05 * i, 0.1},
            .diffuse = {0.3, 0.2 * i, 0.3},
            .specular = {0.8, 0.8, 0.8},
            .shininess = 1.0f,
          }),
          .center = float32_t3((i - middle_i) * 4.0, 2, 5.0),
          .radius = 1,
        };

        proceduralGeoms.push_back(sphere);
        const auto sphereMin = sphere.center - sphere.radius;
        const auto sphereMax = sphere.center + sphere.radius;
        aabbs.emplace_back(
          vector3d(sphereMin.x, sphereMin.y, sphereMin.z), 
          vector3d(sphereMax.x, sphereMax.y, sphereMax.z));
      }

      {
        IGPUBuffer::SCreationParams params;
        params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
        params.size = proceduralGeoms.size() * sizeof(SProceduralGeomInfo);
        m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), proceduralGeoms.data()).move_into(m_proceduralGeomInfoBuffer);
      }

      {
        IGPUBuffer::SCreationParams params;
        params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT;
        params.size = aabbs.size() * sizeof(Aabb);
        m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), aabbs.data()).move_into(m_proceduralAabbBuffer);
      }
    }

    return true;
  }

  void calculateRayTracingStackSize(const smart_refctd_ptr<video::IGPURayTracingPipeline>& pipeline)
  {
    const auto raygenStackSize = pipeline->getRaygenStackSize();
    auto getMaxSize = [&](auto ranges, auto valProj) -> uint16_t
      {
        auto maxValue = 0;
        for (const auto& val : ranges)
        {
          maxValue = std::max<uint16_t>(maxValue, std::invoke(valProj, val));
        }
        return maxValue;
      };

    const auto closestHitStackMax = getMaxSize(pipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::closestHit);
    const auto anyHitStackMax = getMaxSize(pipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::anyHit);
    const auto intersectionStackMax = getMaxSize(pipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::intersection);
    const auto missStackMax = getMaxSize(pipeline->getMissStackSizes(), std::identity{});
    const auto callableStackMax = getMaxSize(pipeline->getCallableStackSizes(), std::identity{});
    auto firstDepthStackSizeMax = std::max(closestHitStackMax, missStackMax);
    firstDepthStackSizeMax = std::max<uint16_t>(firstDepthStackSizeMax, intersectionStackMax + anyHitStackMax);
    m_rayTracingStackSize = raygenStackSize + std::max(firstDepthStackSizeMax, callableStackMax);
  }

  bool createShaderBindingTable(video::CThreadSafeQueueAdapter* queue, const smart_refctd_ptr<video::IGPURayTracingPipeline>& pipeline)
  {
    const auto& limits = m_device->getPhysicalDevice()->getLimits();
    const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
    const auto handleSizeAligned = nbl::core::alignUp(handleSize, limits.shaderGroupHandleAlignment);

    auto& raygenRange = m_shaderBindingTable.raygenGroupRange;

    auto& hitRange = m_shaderBindingTable.hitGroupsRange;
    const auto hitHandles = pipeline->getHitHandles();

    auto& missRange = m_shaderBindingTable.missGroupsRange;
    const auto missHandles = pipeline->getMissHandles();

    auto& callableRange = m_shaderBindingTable.callableGroupsRange;
    const auto callableHandles = pipeline->getCallableHandles();

    raygenRange = {
      .offset = 0,
      .size = core::alignUp(handleSizeAligned, limits.shaderGroupBaseAlignment)
    };

    missRange = {
      .offset = raygenRange.size,
      .size = core::alignUp(missHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment),
    };
    m_shaderBindingTable.missGroupsStride = handleSizeAligned;

    hitRange = {
      .offset = missRange.offset + missRange.size,
      .size = core::alignUp(hitHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment),
    };
    m_shaderBindingTable.hitGroupsStride = handleSizeAligned;

    callableRange = {
      .offset = hitRange.offset + hitRange.size,
      .size = core::alignUp(callableHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment),
    };
    m_shaderBindingTable.callableGroupsStride = handleSizeAligned;

    const auto bufferSize = raygenRange.size + missRange.size + hitRange.size + callableRange.size;

    ICPUBuffer::SCreationParams cpuBufferParams;
    cpuBufferParams.size = bufferSize;
    auto cpuBuffer = ICPUBuffer::create(std::move(cpuBufferParams));
    uint8_t* pData = reinterpret_cast<uint8_t*>(cpuBuffer->getPointer());

    // copy raygen region
    memcpy(pData, &pipeline->getRaygen(), handleSize);

    // copy miss region
    uint8_t* pMissData = pData + missRange.offset;
    for (const auto& handle : missHandles)
    {
      memcpy(pMissData, &handle, handleSize);
      pMissData += m_shaderBindingTable.missGroupsStride;
    }

    // copy hit region
    uint8_t* pHitData = pData + hitRange.offset;
    for (const auto& handle : hitHandles)
    {
      memcpy(pHitData, &handle, handleSize);
      pHitData += m_shaderBindingTable.hitGroupsStride;
    }

    // copy callable region
    uint8_t* pCallableData = pData + callableRange.offset;
    for (const auto& handle : callableHandles)
    {
      memcpy(pCallableData, &handle, handleSize);
      pCallableData += m_shaderBindingTable.callableGroupsStride;
    }

    {
      IGPUBuffer::SCreationParams params;
      params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT;
      params.size = bufferSize;
      m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), pData).move_into(raygenRange.buffer);
      missRange.buffer = core::smart_refctd_ptr(raygenRange.buffer);
      hitRange.buffer = core::smart_refctd_ptr(raygenRange.buffer);
      callableRange.buffer = core::smart_refctd_ptr(raygenRange.buffer);
    }

    return true;
  }

  bool createAccelerationStructures(video::CThreadSafeQueueAdapter* queue)
  {
    // plus 1 blas for procedural geometry contains {{var::NumberOfProcedural}}
    // spheres. Each sphere is a primitive instead one instance or geometry
    const auto blasCount = m_gpuTriangleGeometries.size() + 1;
    const auto proceduralBlasIdx = m_gpuTriangleGeometries.size();

    IQueryPool::SCreationParams qParams{ .queryCount = static_cast<uint32_t>(blasCount), .queryType = IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE };
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
    const auto scratchOffsetAlignment = m_device->getPhysicalDevice()->getLimits().minAccelerationStructureScratchOffsetAlignment;

    // build bottom level ASes
    {
      core::vector<uint32_t> primitiveCounts(blasCount);
      core::vector<IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer>> triangles(m_gpuTriangleGeometries.size());
      core::vector<uint32_t> scratchSizes(blasCount);
      IGPUBottomLevelAccelerationStructure::AABBs<const IGPUBuffer> aabbs;

      auto blasFlags = bitflag(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT) | IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_COMPACTION_BIT;
      if (m_physicalDevice->getProperties().limits.rayTracingPositionFetch)
        blasFlags |= IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_DATA_ACCESS;

      IGPUBottomLevelAccelerationStructure::DeviceBuildInfo initBuildInfo;
      initBuildInfo.buildFlags = blasFlags;
      initBuildInfo.geometryCount = 1;	// only 1 geometry object per blas
      initBuildInfo.srcAS = nullptr;
      initBuildInfo.dstAS = nullptr;
      initBuildInfo.scratch = {};

      auto blasBuildInfos = core::vector(blasCount, initBuildInfo);

      m_gpuBlasList.resize(blasCount);
      // setup blas info for triangle geometries
      for (uint32_t i = 0; i < blasCount; i++)
      {
        const auto isProcedural = i == proceduralBlasIdx;
        if (isProcedural)
        {
          aabbs.data.buffer = smart_refctd_ptr(m_proceduralAabbBuffer);
          aabbs.data.offset = 0;
          aabbs.stride = sizeof(IGPUBottomLevelAccelerationStructure::AABB_t);
          aabbs.geometryFlags = IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::OPAQUE_BIT; // only allow opaque for now

          primitiveCounts[proceduralBlasIdx] = NumberOfProceduralGeometries;
          blasBuildInfos[proceduralBlasIdx].aabbs = &aabbs;
          blasBuildInfos[proceduralBlasIdx].buildFlags |= IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
        } else
        {
          const auto& gpuObject = m_gpuTriangleGeometries[i];

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
          triangles[i].geometryFlags = gpuObject.material.isTransparent() ?
            IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::NO_DUPLICATE_ANY_HIT_INVOCATION_BIT :
            IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::OPAQUE_BIT;

          blasBuildInfos[i].triangles = &triangles[i];
        }
        ILogicalDevice::AccelerationStructureBuildSizes buildSizes;
        {
          const uint32_t maxPrimCount[1] = { primitiveCounts[i] };
          if (isProcedural)
          {
            const auto* aabbData = &aabbs;
            buildSizes = m_device->getAccelerationStructureBuildSizes(blasBuildInfos[i].buildFlags, false, std::span{ aabbData, 1}, maxPrimCount);
          }
          else
          {
            const auto* trianglesData = triangles.data();
            buildSizes = m_device->getAccelerationStructureBuildSizes(blasBuildInfos[i].buildFlags, false, std::span{trianglesData,1}, maxPrimCount);
          }
          if (!buildSizes)
            return logFail("Failed to get BLAS build sizes");
        }

        scratchSizes[i] = buildSizes.buildScratchSize;
        totalScratchSize = core::alignUp(totalScratchSize, scratchOffsetAlignment);
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

      cmdbufBlas->resetQueryPool(queryPool.get(), 0, blasCount);

      smart_refctd_ptr<IGPUBuffer> scratchBuffer;
      {
        IGPUBuffer::SCreationParams params;
        params.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
        params.size = totalScratchSize;
        scratchBuffer = createBuffer(params);
      }

      core::vector<IGPUBottomLevelAccelerationStructure::BuildRangeInfo> buildRangeInfos(blasCount);
      core::vector<IGPUBottomLevelAccelerationStructure::BuildRangeInfo*> pRangeInfos(blasCount);
      for (uint32_t i = 0; i < blasCount; i++)
      {
        blasBuildInfos[i].dstAS = m_gpuBlasList[i].get();
        blasBuildInfos[i].scratch.buffer = scratchBuffer;
        if (i == 0)
        {
          blasBuildInfos[i].scratch.offset = 0u;
        } else
        {
          const auto unalignedOffset = blasBuildInfos[i - 1].scratch.offset + scratchSizes[i - 1];
          blasBuildInfos[i].scratch.offset = core::alignUp(unalignedOffset, scratchOffsetAlignment);
        }

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


      core::vector<const IGPUAccelerationStructure*> ases(blasCount);
      for (uint32_t i = 0; i < blasCount; i++)
        ases[i] = m_gpuBlasList[i].get();
      if (!cmdbufBlas->writeAccelerationStructureProperties(std::span(ases), IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE,
        queryPool.get(), 0))
        return logFail("Failed to write acceleration structure properties!");

      cmdbufBlas->endDebugMarker();
      cmdbufSubmitAndWait(cmdbufBlas, queue, 39);
    }

    auto cmdbufCompact = getSingleUseCommandBufferAndBegin(pool);
    cmdbufCompact->beginDebugMarker("Compact BLAS");

    // compact blas
    {
      core::vector<size_t> asSizes(blasCount);
      if (!m_device->getQueryPoolResults(queryPool.get(), 0, blasCount, asSizes.data(), sizeof(size_t), bitflag(IQueryPool::WAIT_BIT) | IQueryPool::_64_BIT))
        return logFail("Could not get query pool results for AS sizes");

      core::vector<smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>> cleanupBlas(blasCount);
      for (uint32_t i = 0; i < blasCount; i++)
      {
        if (asSizes[i] == 0) continue;
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
      const uint32_t instancesCount = blasCount;
      core::vector<IGPUTopLevelAccelerationStructure::DeviceStaticInstance> instances(instancesCount);
      for (uint32_t i = 0; i < instancesCount; i++)
      {
        const auto isProceduralInstance = i == proceduralBlasIdx;
        instances[i].base.blas.deviceAddress = m_gpuBlasList[i]->getReferenceForDeviceOperations().deviceAddress;
        instances[i].base.mask = 0xFF;
        instances[i].base.instanceCustomIndex = i;
        instances[i].base.instanceShaderBindingTableRecordOffset = isProceduralInstance ? 2 : 0;
        instances[i].base.flags = static_cast<uint32_t>(IGPUTopLevelAccelerationStructure::INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT);
        instances[i].transform = isProceduralInstance ? matrix3x4SIMD() : m_gpuTriangleGeometries[i].transform;
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
  uint32_t m_frameAccumulationCounter = 0;
  std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
  ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

  core::smart_refctd_ptr<InputSystem> m_inputSystem;
  InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
  InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

  struct CameraSetting
  {
    float fov = 60.f;
    float zNear = 0.1f;
    float zFar = 10000.f;
    float moveSpeed = 1.f;
    float rotateSpeed = 1.f;
    float viewWidth = 10.f;
    float camYAngle = 165.f / 180.f * 3.14159f;
    float camXAngle = 32.f / 180.f * 3.14159f;
    
  } m_cameraSetting;
  Camera m_camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

  Light m_light = {
    .direction = {-1.0f, -1.0f, -0.4f},
    .position = {10.0f, 15.0f, 8.0f},
    .outerCutoff = 0.866025404f, // {cos(radians(30.0f))}, 
    .type = ELT_DIRECTIONAL
  };

  video::CDumbPresentationOracle m_oracle;

  struct C_UI
  {
    nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

    struct
    {
      core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
    } samplers;

    core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
  } m_ui;
  core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

  core::vector<ReferenceObjectGpu> m_gpuTriangleGeometries;
  core::vector<SProceduralGeomInfo> m_gpuIntersectionSpheres;
  uint32_t m_intersectionHitGroupIdx;

  std::vector<smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>> m_gpuBlasList;
  smart_refctd_ptr<IGPUTopLevelAccelerationStructure> m_gpuTlas;
  smart_refctd_ptr<IGPUBuffer> m_instanceBuffer;

  smart_refctd_ptr<IGPUBuffer> m_triangleGeomInfoBuffer;
  smart_refctd_ptr<IGPUBuffer> m_proceduralGeomInfoBuffer;
  smart_refctd_ptr<IGPUBuffer> m_proceduralAabbBuffer;
  smart_refctd_ptr<IGPUBuffer> m_indirectBuffer;

  smart_refctd_ptr<IGPUImage> m_hdrImage;
  smart_refctd_ptr<IGPUImageView> m_hdrImageView;

  smart_refctd_ptr<IDescriptorPool> m_rayTracingDsPool;
  smart_refctd_ptr<IGPUDescriptorSet> m_rayTracingDs;
  smart_refctd_ptr<IGPURayTracingPipeline> m_rayTracingPipeline;
  uint64_t m_rayTracingStackSize;
  ShaderBindingTable m_shaderBindingTable;

  smart_refctd_ptr<IGPUDescriptorSet> m_presentDs;
  smart_refctd_ptr<IDescriptorPool> m_presentDsPool;
  smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;

  smart_refctd_ptr<CAssetConverter> m_converter;


  core::matrix4SIMD m_cachedModelViewProjectionMatrix;
  bool m_useIndirectCommand = false;

};
NBL_MAIN_FUNC(RaytracingPipelineApp)
