// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "app_resources/hlsl/benchmark/common.hlsl"
#include "app_resources/hlsl/common.hlsl"
#include "common.hpp"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include <nbl/builtin/hlsl/math/linalg/basic.hlsl>
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

//#include "app_resources/hlsl/silhouette.hlsl"
//#include "app_resources/hlsl/parallelogram_sampling.hlsl"
//#include "app_resources/hlsl/pyramid_sampling.hlsl"
//#include "app_resources/hlsl/triangle_sampling.hlsl"
//#include <nbl/builtin/hlsl/sampling/concepts.hlsl>

// ============================================================================
// Compile-time concept verification (mirrors example 37 main.cpp). Each
// example sampler must satisfy TractableSampler:
//   typedef domain_type, codomain_type, density_type, cache_type
//   codomain_type generate(domain_type, ref cache_type)
//   density_type  forwardPdf(domain_type, cache_type)
// SphericalPyramid is checked across all four (UseCaliper, InnerSampler)
// pairs that the frag shader / benchmark actually instantiate.
// ============================================================================

 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<Parallelogram>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<TriangleFanSampler<false>>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<TriangleFanSampler<true>>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<BilinearSampler>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<SphericalPyramid<false, nbl::hlsl::sampling::SphericalRectangle<float32_t>>>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<SphericalPyramid<true,  nbl::hlsl::sampling::SphericalRectangle<float32_t>>>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<SphericalPyramid<false, nbl::hlsl::sampling::ProjectedSphericalRectangle<float32_t>>>);
 //static_assert(nbl::hlsl::sampling::concepts::TractableSampler<SphericalPyramid<false, BilinearSampler>>);

// App execution mode -- pick at compile time via -DAPP_MODE=N
//   APP_MODE_VISUALIZER       (1) full visualization with debug + ImGui editor (default)
//   APP_MODE_NSIGHT_BENCHMARKS(2) submits one dispatch per SAMPLING_MODE_FLAGS in a single capture, then exits
#define APP_MODE_VISUALIZER 1
#define APP_MODE_NSIGHT_BENCHMARKS 2
#ifndef APP_MODE
#define APP_MODE APP_MODE_VISUALIZER
#endif

/*
Renders scene texture to an offscreen framebuffer whose color attachment is then sampled into a imgui window.

Written with Nabla's UI extension and got integrated with ImGuizmo to handle scene's object translations.
*/
class SolidAngleVisualizer final : public MonoWindowApplication, public BuiltinResourcesApplication
{
   using device_base_t = MonoWindowApplication;
   using asset_base_t  = BuiltinResourcesApplication;

   public:
   inline SolidAngleVisualizer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
      : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
        device_base_t({2048, 1024}, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
   {
   }

   virtual SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
   {
      auto retval                   = device_base_t::getPreferredDeviceFeatures();
      retval.pipelineExecutableInfo = true;
      return retval;
   }

   inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
   {
      if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
         return false;
      if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
         return false;

      interface.m_visualizer = this;

      m_semaphore = m_device->createSemaphore(m_realFrameIx);
      if (!m_semaphore)
         return logFail("Failed to Create a Semaphore!");

      auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      for (auto i = 0u; i < MaxFramesInFlight; i++)
      {
         if (!pool)
            return logFail("Couldn't create Command Pool!");
         if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, {m_cmdBufs.data() + i, 1}))
            return logFail("Couldn't create Command Buffer!");
      }

#if APP_MODE == APP_MODE_VISUALIZER
      const uint32_t addtionalBufferOwnershipFamilies[] = {getGraphicsQueue()->getFamilyIndex()};
      m_scene                                           = CGeometryCreatorScene::create(
         {.transferQueue                      = getTransferUpQueue(),
                                                      .utilities                        = m_utils.get(),
                                                      .logger                           = m_logger.get(),
                                                      .addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies},
         CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
#endif

      // for the scene drawing pass
      {
         IGPURenderpass::SCreationParams                                           params             = {};
         const IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
            {{{.format     = sceneRenderDepthFormat,
                 .samples  = IGPUImage::ESCF_1_BIT,
                 .mayAlias = false},
               /*.loadOp =*/ {IGPURenderpass::LOAD_OP::CLEAR},
               /*.storeOp =*/ {IGPURenderpass::STORE_OP::STORE},
               /*.initialLayout =*/ {IGPUImage::LAYOUT::UNDEFINED},
               /*.finalLayout =*/ {IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}},
            IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd};
         params.depthStencilAttachments                                                        = depthAttachments;
         const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
            {{
               {.format     = finalSceneRenderFormat,
                  .samples  = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
                  .mayAlias = false},
               /*.loadOp =*/IGPURenderpass::LOAD_OP::CLEAR,
               /*.storeOp =*/IGPURenderpass::STORE_OP::STORE,
               /*.initialLayout =*/IGPUImage::LAYOUT::UNDEFINED,
               /*.finalLayout =*/IGPUImage::LAYOUT::READ_ONLY_OPTIMAL // ImGUI shall read
            }},
            IGPURenderpass::SCreationParams::ColorAttachmentsEnd};
         params.colorAttachments                                          = colorAttachments;
         IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
            {},
            IGPURenderpass::SCreationParams::SubpassesEnd};
         subpasses[0].depthStencilAttachment = {{.render = {.attachmentIndex = 0, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}};
         subpasses[0].colorAttachments[0]    = {.render = {.attachmentIndex = 0, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
         params.subpasses                    = subpasses;

         const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
            // wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
            {
               .srcSubpass    = IGPURenderpass::SCreationParams::SSubpassDependency::External,
               .dstSubpass    = 0,
               .memoryBarrier = {
                  // last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
                  // while color is sampled by ImGUI
                  .srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
                  // don't want any writes to be available, as we are clearing both attachments
                  .srcAccessMask = ACCESS_FLAGS::NONE,
                  // destination needs to wait as early as possible
                  // TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
                  .dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                  // because depth and color get cleared first no read mask
                  .dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT}
               // leave view offsets and flags default
            },
            {
               .srcSubpass = 0, .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External, .memoryBarrier = {// last place where the color can get modified, depth is implicitly earlier
                                                                                                                .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                                                                                                                // only write ops, reads can't be made available, also won't be using depth so don't care about it being visible to anyone else
                                                                                                                .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
                                                                                                                // the ImGUI will sample the color, then next frame we overwrite both attachments
                                                                                                                .dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT | PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
                                                                                                                // but we only care about the availability-visibility chain between renderpass and imgui
                                                                                                                .dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT}
               // leave view offsets and flags default
            },
            IGPURenderpass::SCreationParams::DependenciesEnd};
         params.dependencies             = dependencies;
         auto solidAngleRenderpassParams = params;
         m_mainRenderpass                = m_device->createRenderpass(std::move(params));
         if (!m_mainRenderpass)
            return logFail("Failed to create Main Renderpass!");

         m_solidAngleRenderpass = m_device->createRenderpass(std::move(solidAngleRenderpassParams));
         if (!m_solidAngleRenderpass)
            return logFail("Failed to create Solid Angle Renderpass!");
      }

#if APP_MODE == APP_MODE_VISUALIZER
      const auto& geometries = m_scene->getInitParams().geometries;
      m_renderer             = CSimpleDebugRenderer::create(m_assetMgr.get(), m_solidAngleRenderpass.get(), 0, {&geometries.front().get(), geometries.size()});
      // special case
      {
         const auto& pipelines = m_renderer->getInitParams().pipelines;
         auto        ix        = 0u;
         for (const auto& name : m_scene->getInitParams().geometryNames)
         {
            if (name == "Cone")
               m_renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
            ix++;
         }
      }
      // we'll only display one thing at a time
      m_renderer->m_instances.resize(1);
#endif

      // Create graphics pipeline
      {
         auto loadPrecompiledShader = [&](auto key) -> smart_refctd_ptr<IShader>
         {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger                         = m_logger.get();
            lp.workingDirectory               = "app_resources";
            auto       assetBundle            = m_assetMgr->getAsset(key.data(), lp);
            const auto assets                 = assetBundle.getContents();
            if (assets.empty())
            {
               m_logger->log("Could not load precompiled shader!", ILogger::ELL_ERROR);
               std::exit(-1);
            }
            assert(assets.size() == 1);
            auto shader = IAsset::castDown<IShader>(assets[0]);
            if (!shader)
            {
               m_logger->log("Failed to load precompiled shader!", ILogger::ELL_ERROR);
               std::exit(-1);
            }
            return shader;
         };

         ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
         if (!fsTriProtoPPln)
            return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

         smart_refctd_ptr<IShader> saVisShaders[SAMPLING_MODE_FLAGS::Count * DebugPermutations];

         auto addSaVis = [&]<nbl::core::StringLiteral ReleaseKey, nbl::core::StringLiteral DebugKey>(SAMPLING_MODE_FLAGS mode)
         {
            saVisShaders[denseIdOf(mode) * DebugPermutations + 0] = loadPrecompiledShader(nbl::this_example::builtin::build::get_spirv_key<ReleaseKey>(m_device.get()));
            saVisShaders[denseIdOf(mode) * DebugPermutations + 1] = loadPrecompiledShader(nbl::this_example::builtin::build::get_spirv_key<DebugKey>(m_device.get()));
         };

         addSaVis.template operator()<"sa_vis_tri_sa", "sa_vis_tri_sa_dbg">(SAMPLING_MODE_FLAGS::TRIANGLE_SOLID_ANGLE);
         addSaVis.template operator()<"sa_vis_tri_psa", "sa_vis_tri_psa_dbg">(SAMPLING_MODE_FLAGS::TRIANGLE_PROJECTED_SOLID_ANGLE);
         addSaVis.template operator()<"sa_vis_para", "sa_vis_para_dbg">(SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE);
         addSaVis.template operator()<"sa_vis_rectangle", "sa_vis_rectangle_dbg">(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID);
         addSaVis.template operator()<"sa_vis_bilinear", "sa_vis_bilinear_dbg">(SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID);
         addSaVis.template operator()<"sa_vis_proj_rectangle", "sa_vis_proj_rectangle_dbg">(SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID);
         addSaVis.template operator()<"sa_vis_silhouette", "sa_vis_silhouette_dbg">(SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY);
         addSaVis.template operator()<"sa_vis_pyramid", "sa_vis_pyramid_dbg">(SAMPLING_MODE_FLAGS::PYRAMID_CREATION_ONLY);
         addSaVis.template operator()<"sa_vis_caliper_pyramid", "sa_vis_caliper_pyramid_dbg">(SAMPLING_MODE_FLAGS::CALIPER_PYRAMID_CREATION_ONLY);
         addSaVis.template operator()<"sa_vis_caliper_rectangle", "sa_vis_caliper_rectangle_dbg">(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID);

         smart_refctd_ptr<IShader> rayVisShaders[DebugPermutations];
         rayVisShaders[0] = loadPrecompiledShader(nbl::this_example::builtin::build::get_spirv_key<"ray_vis">(m_device.get()));
         rayVisShaders[1] = loadPrecompiledShader(nbl::this_example::builtin::build::get_spirv_key<"ray_vis_dbg">(m_device.get()));

         smart_refctd_ptr<IGPUPipelineLayout>          solidAngleVisLayout, rayVisLayout;
         nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] =
            {
               {.binding       = 0,
                  .type        = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                  .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                  .stageFlags  = ShaderStage::ESS_FRAGMENT,
                  .count       = 1}};
         smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout(bindings);

         const asset::SPushConstantRange saRanges[]  = {{.stageFlags = hlsl::ShaderStage::ESS_FRAGMENT, .offset = 0, .size = sizeof(PushConstants)}};
         const asset::SPushConstantRange rayRanges[] = {{.stageFlags = hlsl::ShaderStage::ESS_FRAGMENT, .offset = 0, .size = sizeof(PushConstantRayVis)}};

         if (!dsLayout)
            logFail("Failed to create a Descriptor Layout!\n");

         solidAngleVisLayout = m_device->createPipelineLayout(saRanges, dsLayout);

         rayVisLayout = m_device->createPipelineLayout(rayRanges, dsLayout);

         {
            // Create all SolidAngleVis pipeline variants
            for (uint32_t i = 0; i < SAMPLING_MODE_FLAGS::Count * DebugPermutations; i++)
            {
               const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
                  .shader     = saVisShaders[i].get(),
                  .entryPoint = "main"};
               m_solidAngleVisPipelines[i] = fsTriProtoPPln.createPipeline(fragSpec, solidAngleVisLayout.get(), m_solidAngleRenderpass.get());
               if (!m_solidAngleVisPipelines[i])
                  return logFail("Could not create SolidAngleVis Graphics Pipeline variant %d!", i);
            }

            asset::SRasterizationParams rasterParams = ext::FullScreenTriangle::ProtoPipeline::DefaultRasterParams;
            rasterParams.depthWriteEnable            = true;
            rasterParams.depthCompareOp              = asset::E_COMPARE_OP::ECO_GREATER;

            // Create all RayVis pipeline variants
            for (uint32_t i = 0; i < DebugPermutations; i++)
            {
               const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
                  .shader     = rayVisShaders[i].get(),
                  .entryPoint = "main"};
               m_rayVisPipelines[i] = fsTriProtoPPln.createPipeline(fragSpec, rayVisLayout.get(), m_mainRenderpass.get(), 0, {}, rasterParams);
               if (!m_rayVisPipelines[i])
                  return logFail("Could not create RayVis Graphics Pipeline variant %d!", i);
            }
         }
         // Allocate the memory
         {
            constexpr size_t BufferSize = sizeof(ResultData);

            nbl::video::IGPUBuffer::SCreationParams params = {};
            params.size                                    = BufferSize;
            params.usage                                   = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
            m_outputStorageBuffer                          = m_device->createBuffer(std::move(params));
            if (!m_outputStorageBuffer)
               logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

            m_outputStorageBuffer->setObjectDebugName("ResultData output buffer");

            nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_outputStorageBuffer->getMemoryReqs();
            reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

            m_allocation = m_device->allocate(reqs, m_outputStorageBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
            if (!m_allocation.isValid())
               logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

            assert(m_outputStorageBuffer->getBoundMemory().memory == m_allocation.memory.get());
            smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, {&dsLayout.get(), 1});

            m_ds = pool->createDescriptorSet(std::move(dsLayout));
            {
               IGPUDescriptorSet::SDescriptorInfo info[1];
               info[0].desc                                     = smart_refctd_ptr(m_outputStorageBuffer);
               info[0].info.buffer                              = {.offset = 0, .size = BufferSize};
               IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                  {.dstSet = m_ds.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = info}};
               m_device->updateDescriptorSets(writes, {});
            }
         }

         if (!m_allocation.memory->map({0ull, m_allocation.memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ))
            logFail("Failed to map the Device Memory!\n");

         // if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
         const ILogicalDevice::MappedMemoryRange memoryRange(m_allocation.memory.get(), 0ull, m_allocation.memory->getAllocationSize());
         if (!m_allocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);
      }

#if APP_MODE == APP_MODE_VISUALIZER
      // Create ImGUI
      {
         auto                                scRes  = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
         ext::imgui::UI::SCreationParameters params = {};
         params.resources.texturesInfo              = {.setIx = 0u, .bindingIx = TexturesImGUIBindingIndex};
         params.resources.samplersInfo              = {.setIx = 0u, .bindingIx = 1u};
         params.utilities                           = m_utils;
         params.transfer                            = getTransferUpQueue();
         params.pipelineLayout                      = ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxImGUITextures);
         params.assetManager                        = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(m_system));
         params.renderpass                          = smart_refctd_ptr<IGPURenderpass>(scRes->getRenderpass());
         params.subpassIx                           = 0u;
         params.pipelineCache                       = nullptr;
         interface.imGUI                            = ext::imgui::UI::create(std::move(params));
         if (!interface.imGUI)
            return logFail("Failed to create `nbl::ext::imgui::UI` class");
      }

      // create rest of User Interface
      {
         auto* imgui = interface.imGUI.get();
         // create the suballocated descriptor set
         {
            // note that we use default layout provided by our extension, but you are free to create your own by filling ext::imgui::UI::S_CREATION_PARAMETERS::resources
            const auto* layout   = interface.imGUI->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
            auto        pool     = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, {&layout, 1});
            auto        ds       = pool->createDescriptorSet(smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout));
            interface.subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
            if (!interface.subAllocDS)
               return logFail("Failed to create the descriptor set");
            // make sure Texture Atlas slot is taken for eternity
            {
               auto dummy = SubAllocatedDescriptorSet::invalid_value;
               interface.subAllocDS->multi_allocate(0, 1, &dummy);
               assert(dummy == ext::imgui::UI::FontAtlasTexId);
            }
            // write constant descriptors, note we don't create info & write pair for the samplers because UI extension's are immutable and baked into DS layout
            IGPUDescriptorSet::SDescriptorInfo info            = {};
            info.desc                                          = smart_refctd_ptr<nbl::video::IGPUImageView>(interface.imGUI->getFontAtlasView());
            info.info.image.imageLayout                        = IImage::LAYOUT::READ_ONLY_OPTIMAL;
            const IGPUDescriptorSet::SWriteDescriptorSet write = {
               .dstSet       = interface.subAllocDS->getDescriptorSet(),
               .binding      = TexturesImGUIBindingIndex,
               .arrayElement = ext::imgui::UI::FontAtlasTexId,
               .count        = 1,
               .info         = &info};
            if (!m_device->updateDescriptorSets({&write, 1}, {}))
               return logFail("Failed to write the descriptor set");
         }
         imgui->registerListener([this]()
            { interface(); });
      }

      interface.camera.mapKeysToWASD();
#endif

#if APP_MODE == APP_MODE_NSIGHT_BENCHMARKS
      // The actual one-shot runs from inside the first renderFrame() so NSight's Shader Profiler has
      // the same render-loop context as the working UI-button-triggered benchmark. Just seed the OBB
      // matrix here from the default TRS so the bench shaders see sane inputs.
      ImGuizmo::RecomposeMatrixFromComponents(&interface.m_TRS.translation.x, &interface.m_TRS.rotation.x, &interface.m_TRS.scale.x, &interface.m_OBBModelMatrix[0][0]);
#endif
      onAppInitializedFinish();
      return true;
   }

   virtual inline bool keepRunning() override
   {
      if (!m_keepRunning)
         return false;
      return device_base_t::keepRunning();
   }

   //
   virtual inline bool onAppTerminated()
   {
#if APP_MODE == APP_MODE_VISUALIZER
      SubAllocatedDescriptorSet::value_type fontAtlasDescIx = ext::imgui::UI::FontAtlasTexId;
      IGPUDescriptorSet::SDropDescriptorSet dummy[1];
      interface.subAllocDS->multi_deallocate(dummy, TexturesImGUIBindingIndex, 1, &fontAtlasDescIx);
#endif
      return device_base_t::onAppTerminated();
   }

   inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
   {
#if APP_MODE == APP_MODE_NSIGHT_BENCHMARKS
      // Minimal frame: run the one-shot once (inside the render loop so NSight's Shader Profiler
      // has the same context as the UI-triggered benchmark), then submit a bare swapchain clear
      // to satisfy the framework's frame contract, and signal exit on the next loop iteration.
      if (!m_nsightBenchDone)
      {
         SamplingBenchmark(*this).runNSightOneShot();
         m_nsightBenchDone = true;
         m_keepRunning     = false;
      }

      const auto resourceIx = m_realFrameIx % MaxFramesInFlight;
      auto* const cb        = m_cmdBufs.data()[resourceIx].get();
      cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
      cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
      {
         auto*                                         scRes      = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
         const IGPUCommandBuffer::SClearColorValue     clearValue = {.float32 = {0.f, 0.f, 0.f, 1.f}};
         const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
            {.framebuffer             = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
               .colorClearValues      = &clearValue,
               .depthStencilClearValues = nullptr,
               .renderArea              = {.offset = {0, 0}, .extent = {m_window->getWidth(), m_window->getHeight()}}};
         beginRenderpass(cb, renderpassInfo);
         cb->endRenderPass();
      }
      cb->end();

      IQueue::SSubmitInfo::SSemaphoreInfo retval =
         {.semaphore = m_semaphore.get(),
            .value   = ++m_realFrameIx,
            .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS};
      const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = {{.cmdbuf = cb}};
      const IQueue::SSubmitInfo::SSemaphoreInfo     acquired[]       = {
         {.semaphore   = device_base_t::getCurrentAcquire().semaphore,
            .value     = device_base_t::getCurrentAcquire().acquireCount,
            .stageMask = PIPELINE_STAGE_FLAGS::NONE}};
      const IQueue::SSubmitInfo infos[] = {
         {.waitSemaphores = acquired, .commandBuffers = commandBuffers, .signalSemaphores = {&retval, 1}}};
      if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
      {
         retval.semaphore = nullptr;
         m_realFrameIx--;
      }
      return retval;
#else
      // CPU events
      update(nextPresentationTimestamp);

      {
         const auto& virtualSolidAngleWindowRes = interface.solidAngleViewTransformReturnInfo.sceneResolution;
         const auto& virtualMainWindowRes       = interface.mainViewTransformReturnInfo.sceneResolution;
         if (!m_solidAngleViewFramebuffer || m_solidAngleViewFramebuffer->getCreationParameters().width != virtualSolidAngleWindowRes[0] || m_solidAngleViewFramebuffer->getCreationParameters().height != virtualSolidAngleWindowRes[1] ||
            !m_mainViewFramebuffer || m_mainViewFramebuffer->getCreationParameters().width != virtualMainWindowRes[0] || m_mainViewFramebuffer->getCreationParameters().height != virtualMainWindowRes[1])
            recreateFramebuffers();
      }

      //
      const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

      auto* const cb = m_cmdBufs.data()[resourceIx].get();
      cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
      cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

      if (m_solidAngleViewFramebuffer)
      {
         asset::SBufferRange<IGPUBuffer> range {
            .offset = 0,
            .size   = m_outputStorageBuffer->getSize(),
            .buffer = m_outputStorageBuffer};
         cb->fillBuffer(range, 0u);
         {
            const auto& creationParams = m_solidAngleViewFramebuffer->getCreationParameters();
            cb->beginDebugMarker("Draw Circle View Frame");
            {
               const IGPUCommandBuffer::SClearDepthStencilValue farValue   = {.depth = 0.f};
               const IGPUCommandBuffer::SClearColorValue        clearValue = {.float32 = {0.f, 0.f, 0.f, 1.f}};
               const IGPUCommandBuffer::SRenderpassBeginInfo    renderpassInfo =
                  {
                     .framebuffer             = m_solidAngleViewFramebuffer.get(),
                     .colorClearValues        = &clearValue,
                     .depthStencilClearValues = &farValue,
                     .renderArea              = {
                                     .offset = {0, 0},
                                     .extent = {creationParams.width, creationParams.height}}};
               beginRenderpass(cb, renderpassInfo);
            }
            // draw scene
            {
               static uint32_t lastFrameSeed = 0u;
               lastFrameSeed                 = m_frameSeeding ? static_cast<uint32_t>(m_realFrameIx) : lastFrameSeed;
               PushConstants pc {
                  .modelMatrix = hlsl::float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix)),
                  .viewport    = {0.f, 0.f, static_cast<float>(creationParams.width), static_cast<float>(creationParams.height)},
                  .sampleCount = static_cast<uint32_t>(m_SampleCount),
                  .frameIndex  = lastFrameSeed};
               const uint32_t debugIdx = m_debugVisualization ? 1u : 0u;
               auto           pipeline = m_solidAngleVisPipelines[denseIdOf(m_samplingMode) * DebugPermutations + debugIdx];
               cb->bindGraphicsPipeline(pipeline.get());
               cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
               cb->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, pipeline->getLayout(), 0, 1, &m_ds.get());
               ext::FullScreenTriangle::recordDrawCall(cb);
            }
            cb->endRenderPass();
            cb->endDebugMarker();
         }

         if (m_debugVisualization)
         {
            m_device->waitIdle();
            std::memcpy(&m_GPUOutResulData, static_cast<ResultData*>(m_allocation.memory->getMappedPointer()), sizeof(ResultData));
            m_device->waitIdle();
         }
      }
      // draw main view
      if (m_mainViewFramebuffer)
      {
         {
            auto                                             creationParams = m_mainViewFramebuffer->getCreationParameters();
            const IGPUCommandBuffer::SClearDepthStencilValue farValue       = {.depth = 0.f};
            const IGPUCommandBuffer::SClearColorValue        clearValue     = {.float32 = {0.1f, 0.1f, 0.1f, 1.f}};
            const IGPUCommandBuffer::SRenderpassBeginInfo    renderpassInfo =
               {
                  .framebuffer             = m_mainViewFramebuffer.get(),
                  .colorClearValues        = &clearValue,
                  .depthStencilClearValues = &farValue,
                  .renderArea              = {
                                  .offset = {0, 0},
                                  .extent = {creationParams.width, creationParams.height}}};
            beginRenderpass(cb, renderpassInfo);
         }
         { // draw rays visualization
            auto creationParams = m_mainViewFramebuffer->getCreationParameters();

            cb->beginDebugMarker("Draw Rays visualization");
            // draw scene
            {
               float32_t4x4       viewProj = *reinterpret_cast<const float32_t4x4*>(&interface.camera.getConcatenatedMatrix());
               float32_t3x4       view     = *reinterpret_cast<const float32_t3x4*>(&interface.camera.getViewMatrix());
               PushConstantRayVis pc {
                  .viewProjMatrix = viewProj,
                  .viewMatrix     = view,
                  .modelMatrix    = hlsl::float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix)),
                  .invModelMatrix = hlsl::float32_t3x4(hlsl::transpose(hlsl::inverse(interface.m_OBBModelMatrix))),
                  .viewport       = {0.f, 0.f, static_cast<float>(creationParams.width), static_cast<float>(creationParams.height)},
                  .frameIndex     = m_frameSeeding ? static_cast<uint32_t>(m_realFrameIx) : 0u};
               auto pipeline = m_rayVisPipelines[m_debugVisualization ? 1u : 0u];
               cb->bindGraphicsPipeline(pipeline.get());
               cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
               cb->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, pipeline->getLayout(), 0, 1, &m_ds.get());
               ext::FullScreenTriangle::recordDrawCall(cb);
            }
            cb->endDebugMarker();
         }
         // draw scene
         {
            cb->beginDebugMarker("Main Scene Frame");

            float32_t3x4 viewMatrix;
            float32_t4x4 viewProjMatrix;
            // TODO: get rid of legacy matrices
            {
               const auto& camera = interface.camera;
               memcpy(&viewMatrix, &camera.getViewMatrix(), sizeof(viewMatrix));
               memcpy(&viewProjMatrix, &camera.getConcatenatedMatrix(), sizeof(viewProjMatrix));
            }
            const auto viewParams = CSimpleDebugRenderer::SViewParams(viewMatrix, viewProjMatrix);

            // tear down scene every frame
            auto& instance     = m_renderer->m_instances[0];
            instance.world     = float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix));
            instance.packedGeo = m_renderer->getGeometries().data(); // cube // +interface.gcIndex;
            m_renderer->render(cb, viewParams); // draw the cube/OBB

            instance.world     = float32_t3x4(1.0f);
            instance.packedGeo = m_renderer->getGeometries().data() + 2; // disk
            m_renderer->render(cb, viewParams);
         }

         cb->endDebugMarker();
         cb->endRenderPass();
      }

      {
         cb->beginDebugMarker("SolidAngleVisualizer IMGUI Frame");
         {
            auto                                          scRes      = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
            const IGPUCommandBuffer::SClearColorValue     clearValue = {.float32 = {0.f, 0.f, 0.f, 1.f}};
            const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
               {
                  .framebuffer             = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
                  .colorClearValues        = &clearValue,
                  .depthStencilClearValues = nullptr,
                  .renderArea              = {
                                  .offset = {0, 0},
                                  .extent = {m_window->getWidth(), m_window->getHeight()}}};
            beginRenderpass(cb, renderpassInfo);
         }
         // draw ImGUI
         {
            auto* imgui    = interface.imGUI.get();
            auto* pipeline = imgui->getPipeline();
            cb->bindGraphicsPipeline(pipeline);
            // note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx
            const auto* ds = interface.subAllocDS->getDescriptorSet();
            cb->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), imgui->getCreationParameters().resources.texturesInfo.setIx, 1u, &ds);
            // a timepoint in the future to release streaming resources for geometry
            const ISemaphore::SWaitInfo drawFinished = {.semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u};
            if (!imgui->render(cb, drawFinished))
            {
               m_logger->log("TODO: need to present acquired image before bailing because its already acquired.", ILogger::ELL_ERROR);
               return {};
            }
         }
         cb->endRenderPass();
         cb->endDebugMarker();
      }
      cb->end();

      IQueue::SSubmitInfo::SSemaphoreInfo retval =
         {
            .semaphore = m_semaphore.get(),
            .value     = ++m_realFrameIx,
            .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS};
      const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
         {
            {.cmdbuf = cb}};
      const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
         {.semaphore   = device_base_t::getCurrentAcquire().semaphore,
            .value     = device_base_t::getCurrentAcquire().acquireCount,
            .stageMask = PIPELINE_STAGE_FLAGS::NONE}};
      const IQueue::SSubmitInfo infos[] =
         {
            {.waitSemaphores     = acquired,
               .commandBuffers   = commandBuffers,
               .signalSemaphores = {&retval, 1}}};

      if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
      {
         retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
         m_realFrameIx--;
      }

      m_window->setCaption("[Nabla Engine] UI App Test Demo");
      return retval;
#endif
   }

   protected:
   const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
   {
      // Subsequent submits don't wait for each other, but they wait for acquire and get waited on by present
      const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
         // don't want any writes to be available, we'll clear, only thing to worry about is the layout transition
         {
            .srcSubpass    = IGPURenderpass::SCreationParams::SSubpassDependency::External,
            .dstSubpass    = 0,
            .memoryBarrier = {
               .srcStageMask  = PIPELINE_STAGE_FLAGS::NONE, // should sync against the semaphore wait anyway
               .srcAccessMask = ACCESS_FLAGS::NONE,
               // layout transition needs to finish before the color write
               .dstStageMask  = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
               .dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT}
            // leave view offsets and flags default
         },
         // want layout transition to begin after all color output is done
         {
            .srcSubpass = 0, .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External, .memoryBarrier = {
                                                                                                             // last place where the color can get modified, depth is implicitly earlier
                                                                                                             .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                                                                                                             // only write ops, reads can't be made available
                                                                                                             .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
                                                                                                             // spec says nothing is needed when presentation is the destination
                                                                                                          }
            // leave view offsets and flags default
         },
         IGPURenderpass::SCreationParams::DependenciesEnd};
      return dependencies;
   }

   private:
   inline void update(const std::chrono::microseconds nextPresentationTimestamp)
   {
      auto& camera = interface.camera;
      camera.setMoveSpeed(interface.moveSpeed);
      camera.setRotateSpeed(interface.rotateSpeed);

      m_inputSystem->getDefaultMouse(&mouse);
      m_inputSystem->getDefaultKeyboard(&keyboard);

      struct
      {
         std::vector<SMouseEvent>    mouse {};
         std::vector<SKeyboardEvent> keyboard {};
      } uiEvents;

      // TODO: should be a member really
      static std::chrono::microseconds previousEventTimestamp {};

      // I think begin/end should always be called on camera, just events shouldn't be fed, why?
      // If you stop begin/end, whatever keys were up/down get their up/down values frozen leading to
      // `perActionDt` becoming obnoxiously large the first time the even processing resumes due to
      // `timeDiff` being computed since `lastVirtualUpTimeStamp`
      camera.beginInputProcessing(nextPresentationTimestamp);
      {
         mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
            {
					if (interface.move)
						camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl
					else
						camera.mouseKeysUp();

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						uiEvents.mouse.emplace_back(e);

						//if (e.type == nbl::ui::SMouseEvent::EET_SCROLL && m_renderer)
						//{
						//	interface.gcIndex += int16_t(core::sign(e.scrollEvent.verticalScroll));
						//	interface.gcIndex = core::clamp(interface.gcIndex, 0ull, m_renderer->getGeometries().size() - 1);
						//}
					} },
            m_logger.get());
         keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
            {
					if (interface.move)
						camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						uiEvents.keyboard.emplace_back(e);
					} },
            m_logger.get());
      }
      camera.endInputProcessing(nextPresentationTimestamp);

      const auto cursorPosition = m_window->getCursorControl()->getPosition();

      ext::imgui::UI::SUpdateParameters params =
         {
            .mousePosition  = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY()),
            .displaySize    = {m_window->getWidth(), m_window->getHeight()},
            .mouseEvents    = uiEvents.mouse,
            .keyboardEvents = uiEvents.keyboard};

      // interface.objectName = m_scene->getInitParams().geometryNames[interface.gcIndex];
      interface.imGUI->update(params);
   }

   void recreateFramebuffers()
   {
      auto createImageAndView = [&](const uint16_t2 resolution, E_FORMAT format) -> smart_refctd_ptr<IGPUImageView>
      {
         auto image = m_device->createImage({{.type = IGPUImage::ET_2D,
            .samples                                = IGPUImage::ESCF_1_BIT,
            .format                                 = format,
            .extent                                 = {resolution.x, resolution.y, 1},
            .mipLevels                              = 1,
            .arrayLayers                            = 1,
            .usage                                  = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT}});
         if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
            return nullptr;
         IGPUImageView::SCreationParams params = {
            .image    = std::move(image),
            .viewType = IGPUImageView::ET_2D,
            .format   = format};
         params.subresourceRange.aspectMask = isDepthOrStencilFormat(format) ? IGPUImage::EAF_DEPTH_BIT : IGPUImage::EAF_COLOR_BIT;
         return m_device->createImageView(std::move(params));
      };

      smart_refctd_ptr<IGPUImageView> solidAngleView;
      smart_refctd_ptr<IGPUImageView> mainView;
      const uint16_t2                 solidAngleViewRes = interface.solidAngleViewTransformReturnInfo.sceneResolution;
      const uint16_t2                 mainViewRes       = interface.mainViewTransformReturnInfo.sceneResolution;

      // detect window minimization
      if (solidAngleViewRes.x < 0x4000 && solidAngleViewRes.y < 0x4000 || mainViewRes.x < 0x4000 && mainViewRes.y < 0x4000)
      {
         solidAngleView              = createImageAndView(solidAngleViewRes, finalSceneRenderFormat);
         auto solidAngleDepthView    = createImageAndView(solidAngleViewRes, sceneRenderDepthFormat);
         m_solidAngleViewFramebuffer = m_device->createFramebuffer({{.renderpass = m_solidAngleRenderpass,
            .depthStencilAttachments                                             = &solidAngleDepthView.get(),
            .colorAttachments                                                    = &solidAngleView.get(),
            .width                                                               = solidAngleViewRes.x,
            .height                                                              = solidAngleViewRes.y}});

         mainView              = createImageAndView(mainViewRes, finalSceneRenderFormat);
         auto mainDepthView    = createImageAndView(mainViewRes, sceneRenderDepthFormat);
         m_mainViewFramebuffer = m_device->createFramebuffer({{.renderpass = m_mainRenderpass,
            .depthStencilAttachments                                       = &mainDepthView.get(),
            .colorAttachments                                              = &mainView.get(),
            .width                                                         = mainViewRes.x,
            .height                                                        = mainViewRes.y}});
      }
      else
      {
         m_solidAngleViewFramebuffer = nullptr;
         m_mainViewFramebuffer       = nullptr;
      }

      // release previous slot and its image
      interface.subAllocDS->multi_deallocate(0, static_cast<int>(CInterface::Count), interface.renderColorViewDescIndices, {.semaphore = m_semaphore.get(), .value = m_realFrameIx + 1});
      //
      if (solidAngleView && mainView)
      {
         interface.subAllocDS->multi_allocate(0, static_cast<int>(CInterface::Count), interface.renderColorViewDescIndices);
         // update descriptor set
         IGPUDescriptorSet::SDescriptorInfo infos[static_cast<int>(CInterface::Count)]           = {};
         infos[0].desc                                                                           = mainView;
         infos[0].info.image.imageLayout                                                         = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
         infos[1].desc                                                                           = solidAngleView;
         infos[1].info.image.imageLayout                                                         = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
         const IGPUDescriptorSet::SWriteDescriptorSet write[static_cast<int>(CInterface::Count)] = {
            {.dstSet         = interface.subAllocDS->getDescriptorSet(),
               .binding      = TexturesImGUIBindingIndex,
               .arrayElement = interface.renderColorViewDescIndices[static_cast<int>(CInterface::ERV_MAIN_VIEW)],
               .count        = 1,
               .info         = &infos[static_cast<int>(CInterface::ERV_MAIN_VIEW)]},
            {.dstSet         = interface.subAllocDS->getDescriptorSet(),
               .binding      = TexturesImGUIBindingIndex,
               .arrayElement = interface.renderColorViewDescIndices[static_cast<int>(CInterface::ERV_SOLID_ANGLE_VIEW)],
               .count        = 1,
               .info         = &infos[static_cast<int>(CInterface::ERV_SOLID_ANGLE_VIEW)]}};
         m_device->updateDescriptorSets({write, static_cast<int>(CInterface::Count)}, {});
      }
      interface.transformParams.sceneTexDescIx = interface.renderColorViewDescIndices[CInterface::ERV_MAIN_VIEW];
   }

   inline void beginRenderpass(IGPUCommandBuffer* cb, const IGPUCommandBuffer::SRenderpassBeginInfo& info)
   {
      cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
      cb->setScissor(0, 1, &info.renderArea);
      const SViewport viewport = {
         .x      = 0,
         .y      = 0,
         .width  = static_cast<float>(info.renderArea.extent.width),
         .height = static_cast<float>(info.renderArea.extent.height)};
      cb->setViewport(0u, 1u, &viewport);
   }

   ~SolidAngleVisualizer() override
   {
      m_allocation.memory->unmap();
   }

   // Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
   constexpr static inline uint32_t MaxFramesInFlight         = 3u;
   constexpr static inline auto     sceneRenderDepthFormat    = EF_D32_SFLOAT;
   constexpr static inline auto     finalSceneRenderFormat    = EF_R8G8B8A8_SRGB;
   constexpr static inline auto     TexturesImGUIBindingIndex = 0u;
   // we create the Descriptor Set with a few slots extra to spare, so we don't have to `waitIdle` the device whenever ImGUI virtual window resizes
   constexpr static inline auto MaxImGUITextures = 2u + MaxFramesInFlight;

   static inline SAMPLING_MODE_FLAGS m_samplingMode         = SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID;
   static inline bool          m_debugVisualization   = true;
   static inline int           m_SampleCount          = 64;
   static inline int           m_BenchmarkSampleCount = 128;
   static inline bool          m_frameSeeding         = true;
   static inline ResultData    m_GPUOutResulData;
   bool                        m_keepRunning          = true;
   bool                        m_nsightBenchDone      = false;
   //
   smart_refctd_ptr<CGeometryCreatorScene> m_scene;
   smart_refctd_ptr<IGPURenderpass>        m_solidAngleRenderpass;
   smart_refctd_ptr<IGPURenderpass>        m_mainRenderpass;
   smart_refctd_ptr<CSimpleDebugRenderer>  m_renderer;
   smart_refctd_ptr<IGPUFramebuffer>       m_solidAngleViewFramebuffer;
   smart_refctd_ptr<IGPUFramebuffer>       m_mainViewFramebuffer;
   // Pipeline variants: SolidAngleVis indexed by [mode * 2 + debugFlag], RayVis by [debugFlag]
   static constexpr uint32_t              DebugPermutations = 2;
   smart_refctd_ptr<IGPUGraphicsPipeline> m_solidAngleVisPipelines[SAMPLING_MODE_FLAGS::Count * DebugPermutations];
   smart_refctd_ptr<IGPUGraphicsPipeline> m_rayVisPipelines[DebugPermutations];
   //
   nbl::video::IDeviceMemoryAllocator::SAllocation                    m_allocation = {};
   smart_refctd_ptr<IGPUBuffer>                                       m_outputStorageBuffer;
   smart_refctd_ptr<nbl::video::IGPUDescriptorSet>                    m_ds = nullptr;
   smart_refctd_ptr<ISemaphore>                                       m_semaphore;
   uint64_t                                                           m_realFrameIx = 0;
   std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
   //
   InputSystem::ChannelReader<IMouseEventChannel>    mouse;
   InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
   // UI stuff
   struct CInterface
   {
      void operator()()
      {
         ImGuiIO& io = ImGui::GetIO();

         // TODO: why is this a lambda and not just an assignment in a scope ?
         camera.setProjectionMatrix([&]()
            {
               hlsl::float32_t4x4 projection;

               if (isPerspective)
                  if (isLH)
                     projection = hlsl::math::thin_lens::lhPerspectiveFovMatrix<float>(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y * 0.5f, zNear, zFar); // TODO: why do I need to divide aspect ratio by 2?
                  else
                     projection = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y * 0.5f, zNear, zFar);
               else
               {
                  float viewHeight = viewWidth * io.DisplaySize.y / io.DisplaySize.x;

                  if (isLH)
                     projection = hlsl::math::thin_lens::lhPerspectiveFovMatrix<float>(viewWidth, viewHeight, zNear, zFar);
                  else
                     projection = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(viewWidth, viewHeight, zNear, zFar);
               }

               return projection;
            }());

         ImGuizmo::SetOrthographic(!isPerspective);
         ImGuizmo::BeginFrame();

         ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
         ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

         // create a window and insert the inspector
         ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
         ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
         ImGui::Begin("Editor");

         ImGui::Text("Benchmarking Solid Angle Visualizer");

         if (ImGui::Button("Run Benchmark"))
         {
            SolidAngleVisualizer::SamplingBenchmark benchmark(*m_visualizer);
            benchmark.run();
         }
         ImGui::Separator();

         ImGui::Text("Sampling Mode:");
         ImGui::SameLine();

         const char* samplingModes[SAMPLING_MODE_FLAGS::Count - 3]                       = {};
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID)]            = "Spherical Rectangle From Pyramid";
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID)]    = "Caliper Rectangle From Pyramid";
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID)]       = "Projected Spherical Rectangle From Pyramid";
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::TRIANGLE_SOLID_ANGLE)]             = "Spherical Triangle";
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::TRIANGLE_PROJECTED_SOLID_ANGLE)]   = "Projected Spherical Triangle";
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)] = "Projected Parallelogram";
         samplingModes[denseIdOf(SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID)]            = "Bilinear Pyramid";

         int currentMode = static_cast<int>(denseIdOf(m_samplingMode));

         if (ImGui::Combo("##SamplingMode", &currentMode, samplingModes, SAMPLING_MODE_FLAGS::Count - 3))
         {
            m_samplingMode = kAllModes[currentMode];
         }

         ImGui::Checkbox("Debug Visualization", &m_debugVisualization);
         ImGui::Text("Pipeline idx: SA=%d, Ray=%d",
            static_cast<int>(denseIdOf(m_samplingMode)) * DebugPermutations + (m_debugVisualization ? 1 : 0),
            m_debugVisualization ? 1 : 0);
         ImGui::Checkbox("Frame seeding", &m_frameSeeding);

         ImGui::SliderInt("Sample Count", &m_SampleCount, 0, 512);
         ImGui::SliderInt("Benchmark Sample Count", &m_BenchmarkSampleCount, 0, 8096);

         ImGui::Separator();

         ImGui::Text("Camera");

         if (ImGui::RadioButton("LH", isLH))
            isLH = true;

         ImGui::SameLine();

         if (ImGui::RadioButton("RH", !isLH))
            isLH = false;

         if (ImGui::RadioButton("Perspective", isPerspective))
            isPerspective = true;

         ImGui::SameLine();

         if (ImGui::RadioButton("Orthographic", !isPerspective))
            isPerspective = false;

         ImGui::Checkbox("Enable \"view manipulate\"", &transformParams.enableViewManipulate);
         // ImGui::Checkbox("Enable camera movement", &move);
         ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
         ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);

         // ImGui::Checkbox("Flip Gizmo's Y axis", &flipGizmoY); // let's not expose it to be changed in UI but keep the logic in case

         if (isPerspective)
            ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
         else
            ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);

         ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
         ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

         if (firstFrame)
         {
            camera.setPosition(cameraIntialPosition);
            camera.setTarget(cameraInitialTarget);
            camera.setUpVector(cameraInitialUp);

            camera.recomputeViewMatrix();
         }
         firstFrame = false;

         ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
         if (ImGuizmo::IsUsing())
         {
            ImGui::Text("Using gizmo");
         }
         else
         {
            ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
            ImGui::SameLine();
            ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
            ImGui::SameLine();
            ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
            ImGui::SameLine();
            ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
         }
         ImGui::Separator();

         /*
			* ImGuizmo expects view & perspective matrix to be column major both with 4x4 layout
			* and Nabla uses row major matricies - 3x4 matrix for view & 4x4 for projection

			- VIEW:

				ImGuizmo

				|     X[0]          Y[0]          Z[0]         0.0f |
				|     X[1]          Y[1]          Z[1]         0.0f |
				|     X[2]          Y[2]          Z[2]         0.0f |
				| -Dot(X, eye)  -Dot(Y, eye)  -Dot(Z, eye)     1.0f |

				Nabla

				|     X[0]         X[1]           X[2]     -Dot(X, eye)  |
				|     Y[0]         Y[1]           Y[2]     -Dot(Y, eye)  |
				|     Z[0]         Z[1]           Z[2]     -Dot(Z, eye)  |

				<ImGuizmo View Matrix> = transpose(nbl::core::matrix4SIMD(<Nabla View Matrix>))

			- PERSPECTIVE [PROJECTION CASE]:

				ImGuizmo

				|      (temp / temp2)                 (0.0)                       (0.0)                   (0.0)  |
				|          (0.0)                  (temp / temp3)                  (0.0)                   (0.0)  |
				| ((right + left) / temp2)   ((top + bottom) / temp3)    ((-zfar - znear) / temp4)       (-1.0f) |
				|          (0.0)                      (0.0)               ((-temp * zfar) / temp4)        (0.0)  |

				Nabla

				|            w                        (0.0)                       (0.0)                   (0.0)               |
				|          (0.0)                       -h                         (0.0)                   (0.0)               |
				|          (0.0)                      (0.0)               (-zFar/(zFar-zNear))     (-zNear*zFar/(zFar-zNear)) |
				|          (0.0)                      (0.0)                      (-1.0)                   (0.0)               |

				<ImGuizmo Projection Matrix> = transpose(<Nabla Projection Matrix>)

			*
			* the ViewManipulate final call (inside EditTransform) returns world space column major matrix for an object,
			* note it also modifies input view matrix but projection matrix is immutable
			*/

         if (ImGui::IsKeyPressed(ImGuiKey_End))
         {
            m_TRS = TRS {};
         }

         {
            static struct
            {
               float32_t4x4 view, projection, model;
            } imguizmoM16InOut;

            ImGuizmo::SetID(0u);

            // TODO: camera will return hlsl::float32_tMxN
            auto view             = camera.getViewMatrix();
            imguizmoM16InOut.view = hlsl::transpose(hlsl::math::linalg::promote_affine<4, 4>(view));

            // TODO: camera will return hlsl::float32_tMxN
            imguizmoM16InOut.projection = hlsl::transpose(camera.getProjectionMatrix());
            ImGuizmo::RecomposeMatrixFromComponents(&m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x, &imguizmoM16InOut.model[0][0]);

            if (flipGizmoY) // note we allow to flip gizmo just to match our coordinates
               imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/

            transformParams.editTransformDecomposition = true;
            mainViewTransformReturnInfo                = EditTransform(&imguizmoM16InOut.view[0][0], &imguizmoM16InOut.projection[0][0], &imguizmoM16InOut.model[0][0], transformParams);
            move                                       = mainViewTransformReturnInfo.allowCameraMovement;

            ImGuizmo::DecomposeMatrixToComponents(&imguizmoM16InOut.model[0][0], &m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x);
            ImGuizmo::RecomposeMatrixFromComponents(&m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x, &imguizmoM16InOut.model[0][0]);
         }
         // object meta display
         //{
         //	ImGui::Begin("Object");
         //	ImGui::Text("type: \"%s\"", objectName.data());
         //	ImGui::End();
         //}

         // solid angle view window
         {
            ImGui::SetNextWindowSize(ImVec2(800, 800), ImGuiCond_Appearing);
            ImGui::SetNextWindowPos(ImVec2(1240, 20), ImGuiCond_Appearing);
            static bool isOpen = true;
            ImGui::Begin("Projected Solid Angle View", &isOpen, 0);

            ImVec2 contentRegionSize                              = ImGui::GetContentRegionAvail();
            solidAngleViewTransformReturnInfo.sceneResolution     = uint16_t2(static_cast<uint16_t>(contentRegionSize.x), static_cast<uint16_t>(contentRegionSize.y));
            solidAngleViewTransformReturnInfo.allowCameraMovement = false; // not used in this view
            ImGui::Image({renderColorViewDescIndices[ERV_SOLID_ANGLE_VIEW]}, contentRegionSize);
            ImGui::End();
         }

         // Show data coming from GPU
         if (m_debugVisualization)
         {
            if (ImGui::Begin("Result Data"))
            {
               auto drawColorField = [&](const char* fieldName, uint32_t index)
               {
                  ImGui::Text("%s: %u", fieldName, index);

                  if (index >= 27)
                  {
                     ImGui::SameLine();
                     ImGui::Text("<invalid>");
                     return;
                  }

                  const auto& c = colorLUT[index]; // uses the combined LUT we made earlier

                  ImGui::SameLine();

                  // Color preview button
                  ImGui::ColorButton(
                     fieldName,
                     ImVec4(c.r, c.g, c.b, 1.0f),
                     0,
                     ImVec2(20, 20));

                  ImGui::SameLine();
                  ImGui::Text("%s", colorNames[index]);
               };

               // Vertices
               if (ImGui::CollapsingHeader("Vertices", ImGuiTreeNodeFlags_DefaultOpen))
               {
                  for (uint32_t i = 0; i < 6; ++i)
                  {
                     if (i < m_GPUOutResulData.silhouette.silhouetteVertexCount)
                     {
                        ImGui::Text("corners[%u]", i);
                        ImGui::SameLine();
                        drawColorField(":", m_GPUOutResulData.silhouette.vertices[i]);
                        ImGui::SameLine();
                        static const float32_t3 constCorners[8] = {
                           float32_t3(0, 0, 0), float32_t3(1, 0, 0), float32_t3(0, 1, 0), float32_t3(1, 1, 0),
                           float32_t3(0, 0, 1), float32_t3(1, 0, 1), float32_t3(0, 1, 1), float32_t3(1, 1, 1)};
                        float32_t3 vertexLocation = constCorners[m_GPUOutResulData.silhouette.vertices[i]];
                        ImGui::Text(" : (%.3f, %.3f, %.3f", vertexLocation.x, vertexLocation.y, vertexLocation.z);
                     }
                     else
                     {
                        ImGui::Text("corners[%u] ::  ", i);
                        ImGui::SameLine();
                        ImGui::ColorButton(
                           "<unused>",
                           ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
                           0,
                           ImVec2(20, 20));
                        ImGui::SameLine();
                        ImGui::Text("<unused>");
                     }
                  }
               }

               if (ImGui::CollapsingHeader("Color LUT Map"))
               {
                  for (int i = 0; i < 27; i++)
                     drawColorField(" ", i);
               }

               ImGui::Separator();
               ImGui::Text("Valid Samples: %u / %u", m_GPUOutResulData.sampling.validSampleCount / hlsl::max(m_GPUOutResulData.sampling.threadCount, 1u), m_GPUOutResulData.sampling.sampleCount);
               ImGui::ProgressBar(static_cast<float>(m_GPUOutResulData.sampling.validSampleCount / hlsl::max(m_GPUOutResulData.sampling.threadCount, 1u)) / static_cast<float>(m_GPUOutResulData.sampling.sampleCount));
               ImGui::Separator();

               // Silhouette
               if (ImGui::CollapsingHeader("Silhouette"))
               {
                  drawColorField("silhouetteIndex", m_GPUOutResulData.silhouette.silhouetteIndex);
                  ImGui::Text("Region: (%u, %u, %u)", m_GPUOutResulData.silhouette.region.x, m_GPUOutResulData.silhouette.region.y, m_GPUOutResulData.silhouette.region.z);
                  ImGui::Text("Silhouette Vertex Count: %u", m_GPUOutResulData.silhouette.silhouetteVertexCount);
                  ImGui::Text("Positive Vertex Count: %u", m_GPUOutResulData.silhouette.positiveVertCount);
                  ImGui::Text("Edge Visibility Mismatch: %s", m_GPUOutResulData.silhouette.edgeVisibilityMismatch ? "true" : "false");
                  ImGui::Text("Max Triangles Exceeded: %s", m_GPUOutResulData.triangleFan.maxTrianglesExceeded ? "true" : "false");
                  for (uint32_t i = 0; i < 6; i++)
                     ImGui::Text("Vertex[%u]: %u", i, m_GPUOutResulData.silhouette.vertices[i]);
                  ImGui::Text("Clipped Silhouette Vertex Count: %u", m_GPUOutResulData.silhouette.clippedVertexCount);
                  for (uint32_t i = 0; i < 7; i++)
                     ImGui::Text("Clipped Vertex[%u]: (%.3f, %.3f, %.3f) Index: %u", i,
                        m_GPUOutResulData.silhouette.clippedVertices[i].x,
                        m_GPUOutResulData.silhouette.clippedVertices[i].y,
                        m_GPUOutResulData.silhouette.clippedVertices[i].z,
                        m_GPUOutResulData.silhouette.clippedVertexIndices[i]);

                  // Silhouette mask printed in binary
                  auto printBin = [](uint32_t bin, const char* name)
                  {
                     char buf[33];
                     for (int i = 0; i < 32; i++)
                        buf[i] = (bin & (1u << (31 - i))) ? '1' : '0';
                     buf[32] = '\0';
                     ImGui::Text("%s: 0x%08X", name, bin);
                     ImGui::Text("binary: 0b%s", buf);
                     ImGui::Separator();
                  };
                  printBin(m_GPUOutResulData.silhouette.silhouette, "Silhouette");
                  printBin(m_GPUOutResulData.silhouette.rotatedSil, "rotatedSilhouette");

                  printBin(m_GPUOutResulData.silhouette.clipCount, "clipCount");
                  printBin(m_GPUOutResulData.silhouette.clipMask, "clipMask");
                  printBin(m_GPUOutResulData.silhouette.rotatedClipMask, "rotatedClipMask");
                  printBin(m_GPUOutResulData.silhouette.rotateAmount, "rotateAmount");
                  printBin(m_GPUOutResulData.silhouette.wrapAround, "wrapAround");
               }

               // Parallelogram
               if (m_samplingMode == PROJECTED_PARALLELOGRAM_SOLID_ANGLE && ImGui::CollapsingHeader("Projected Parallelogram", ImGuiTreeNodeFlags_DefaultOpen))
               {
                  ImGui::Text("Area: %.3f", m_GPUOutResulData.parallelogram.area);
                  ImGui::Text("N3 Mask: 0x%02X", m_GPUOutResulData.parallelogram.n3Mask);
                  for (uint32_t i = 0; i < 4; i++)
                  {
                     bool convex = m_GPUOutResulData.parallelogram.edgeIsConvex[i] != 0;
                     bool n3     = (m_GPUOutResulData.parallelogram.n3Mask >> i) & 1u;
                     ImGui::Text("Edge[%u]: %s%s", i,
                        convex ? "convex" : "concave",
                        n3 ? " (N3 split)" : "");
                  }
                  for (uint32_t i = 0; i < 4; i++)
                     ImGui::Text("Corner[%u]: (%.3f, %.3f)", i, m_GPUOutResulData.parallelogram.corners[i].x, m_GPUOutResulData.parallelogram.corners[i].y);
               }
               else if ((m_samplingMode == SPH_RECT_FROM_PYRAMID || m_samplingMode == PROJ_SPH_RECT_FROM_PYRAMID || m_samplingMode == BILINEAR_FROM_PYRAMID || m_samplingMode == SPH_RECT_FROM_CALIPER_PYRAMID) && ImGui::CollapsingHeader("Spherical Pyramid", ImGuiTreeNodeFlags_DefaultOpen))
               {
                  ImGui::Text("Best Caliper Edge: %u", m_GPUOutResulData.pyramid.bestEdge);
                  ImGui::Separator();

                  ImGui::Text("Axis 1: (%.4f, %.4f, %.4f)",
                     m_GPUOutResulData.pyramid.axis1.x, m_GPUOutResulData.pyramid.axis1.y, m_GPUOutResulData.pyramid.axis1.z);
                  ImGui::Text("  Half-Width: %.4f  Offset: %.4f",
                     m_GPUOutResulData.pyramid.halfWidth1, m_GPUOutResulData.pyramid.offset1);
                  ImGui::Text("  Bounds: [%.4f, %.4f]",
                     m_GPUOutResulData.pyramid.min1, m_GPUOutResulData.pyramid.max1);

                  ImGui::Text("Axis 2: (%.4f, %.4f, %.4f)",
                     m_GPUOutResulData.pyramid.axis2.x, m_GPUOutResulData.pyramid.axis2.y, m_GPUOutResulData.pyramid.axis2.z);
                  ImGui::Text("  Half-Width: %.4f  Offset: %.4f",
                     m_GPUOutResulData.pyramid.halfWidth2, m_GPUOutResulData.pyramid.offset2);
                  ImGui::Text("  Bounds: [%.4f, %.4f]",
                     m_GPUOutResulData.pyramid.min2, m_GPUOutResulData.pyramid.max2);

                  ImGui::Separator();
                  ImGui::Text("Center: (%.4f, %.4f, %.4f)",
                     m_GPUOutResulData.pyramid.center.x, m_GPUOutResulData.pyramid.center.y, m_GPUOutResulData.pyramid.center.z);
                  ImGui::Text("Solid Angle (bound): %.6f sr", m_GPUOutResulData.pyramid.solidAngle);
               }
               else if (m_samplingMode == TRIANGLE_SOLID_ANGLE || m_samplingMode == TRIANGLE_PROJECTED_SOLID_ANGLE && ImGui::CollapsingHeader("Spherical Triangle", ImGuiTreeNodeFlags_DefaultOpen))
               {
                  ImGui::Text("Spherical Lune Detected: %s", m_GPUOutResulData.triangleFan.sphericalLuneDetected ? "true" : "false");
                  ImGui::Text("Triangle Count: %u", m_GPUOutResulData.triangleFan.triangleCount);
                  // print solidAngles for each triangle
                  {
                     ImGui::Text("Solid Angles per Triangle:");
                     ImGui::BeginTable("SolidAnglesTable", 2);
                     ImGui::TableSetupColumn("Triangle Index");
                     ImGui::TableSetupColumn("Solid Angle");
                     ImGui::TableHeadersRow();
                     for (uint32_t i = 0; i < m_GPUOutResulData.triangleFan.triangleCount; ++i)
                     {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%u", i);
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%.6f", m_GPUOutResulData.triangleFan.solidAngles[i]);
                     }
                     ImGui::Text("Total: %.6f", m_GPUOutResulData.triangleFan.totalSolidAngles);
                     ImGui::EndTable();
                  }
               }

               {
                  float32_t3 xAxis = m_OBBModelMatrix[0].xyz;
                  float32_t3 yAxis = m_OBBModelMatrix[1].xyz;
                  float32_t3 zAxis = m_OBBModelMatrix[2].xyz;

                  float32_t3 nx = normalize(xAxis);
                  float32_t3 ny = normalize(yAxis);
                  float32_t3 nz = normalize(zAxis);

                  const float epsilon = 1e-4;
                  bool        hasSkew = false;
                  if (abs(dot(nx, ny)) > epsilon || abs(dot(nx, nz)) > epsilon || abs(dot(ny, nz)) > epsilon)
                     hasSkew = true;
                  ImGui::Separator();
                  ImGui::Text("Matrix Has Skew: %s", hasSkew ? "true" : "false");
               }

               static bool     modalShown          = false;
               static bool     modalDismissed      = false;
               static uint32_t lastSilhouetteIndex = ~0u;

               // Reset modal flags if silhouette configuration changed
               if (m_GPUOutResulData.silhouette.silhouetteIndex != lastSilhouetteIndex)
               {
                  modalShown          = false;
                  modalDismissed      = false; // Allow modal to show again for new configuration
                  lastSilhouetteIndex = m_GPUOutResulData.silhouette.silhouetteIndex;
               }

               // Reset flags when mismatch is cleared
               if (!m_GPUOutResulData.silhouette.edgeVisibilityMismatch && !m_GPUOutResulData.triangleFan.maxTrianglesExceeded && !m_GPUOutResulData.triangleFan.sphericalLuneDetected)
               {
                  modalShown     = false;
                  modalDismissed = false;
               }

               // Open modal only if not already shown/dismissed
               if ((m_GPUOutResulData.silhouette.edgeVisibilityMismatch || m_GPUOutResulData.triangleFan.maxTrianglesExceeded || m_GPUOutResulData.triangleFan.sphericalLuneDetected) && m_GPUOutResulData.silhouette.silhouetteIndex != 13 && !modalShown && !modalDismissed) // Don't reopen if user dismissed it
               {
                  ImGui::OpenPopup("Edge Visibility Mismatch Warning");
                  modalShown = true;
               }

               // Modal popup
               if (ImGui::BeginPopupModal("Edge Visibility Mismatch Warning", NULL, ImGuiWindowFlags_AlwaysAutoResize))
               {
                  ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Warning: Edge Visibility Mismatch Detected!");
                  ImGui::Separator();
                  ImGui::Text("The silhouette lookup table (LUT) does not match the computed edge visibility.");
                  ImGui::Text("This indicates the pre-computed silhouette data may be incorrect.");
                  ImGui::Spacing();
                  ImGui::TextWrapped("Configuration Index: %u", m_GPUOutResulData.silhouette.silhouetteIndex);
                  ImGui::TextWrapped("Region: (%u, %u, %u)", m_GPUOutResulData.silhouette.region.x, m_GPUOutResulData.silhouette.region.y, m_GPUOutResulData.silhouette.region.z);
                  ImGui::Spacing();
                  ImGui::Text("Mismatched Vertices (bitmask): 0x%08X", m_GPUOutResulData.silhouette.edgeVisibilityMismatch);
                  ImGui::Text("Vertices involved in mismatched edges:");
                  ImGui::Indent();
                  for (int i = 0; i < 8; i++)
                  {
                     if (m_GPUOutResulData.silhouette.edgeVisibilityMismatch & (1u << i))
                     {
                        ImGui::BulletText("Vertex %d", i);
                     }
                  }
                  ImGui::Unindent();
                  ImGui::Spacing();
                  if (ImGui::Button("OK", ImVec2(120, 0)))
                  {
                     ImGui::CloseCurrentPopup();
                     modalShown     = false;
                     modalDismissed = true; // Mark as dismissed to prevent reopening
                  }
                  ImGui::EndPopup();
               }
            }
            ImGui::End();
         }

         // view matrices editor
         {
            ImGui::Begin("Matrices");

            auto addMatrixTable = [&](const char* topText, const char* tableName, const int rows, const int columns, const float* pointer, const bool withSeparator = true)
            {
               ImGui::Text(topText);
               if (ImGui::BeginTable(tableName, columns))
               {
                  for (int y = 0; y < rows; ++y)
                  {
                     ImGui::TableNextRow();
                     for (int x = 0; x < columns; ++x)
                     {
                        ImGui::TableSetColumnIndex(x);
                        ImGui::Text("%.3f", *(pointer + (y * columns) + x));
                     }
                  }
                  ImGui::EndTable();
               }

               if (withSeparator)
                  ImGui::Separator();
            };

            static RandomSampler rng(0x45); // Initialize RNG with seed

            // Helper function to check if cube intersects unit sphere at origin
            auto isCubeOutsideUnitSphere = [](const float32_t3& translation, const float32_t3& scale) -> bool
            {
               float cubeRadius       = glm::length(scale) * 0.5f;
               float distanceToCenter = glm::length(translation);
               return (distanceToCenter - cubeRadius) > 1.0f;
            };

            static TRS lastTRS = {};
            if (ImGui::Button("Randomize Translation"))
            {
               lastTRS      = m_TRS; // Backup before randomizing
               int attempts = 0;
               do
               {
                  m_TRS.translation = float32_t3(rng.nextFloat(-3.f, 3.f), rng.nextFloat(-3.f, 3.f), rng.nextFloat(-1.f, 3.f));
                  attempts++;
               } while (!isCubeOutsideUnitSphere(m_TRS.translation, m_TRS.scale) && attempts < 100);
            }
            ImGui::SameLine();
            if (ImGui::Button("Randomize Rotation"))
            {
               lastTRS        = m_TRS; // Backup before randomizing
               m_TRS.rotation = float32_t3(rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f));
            }
            ImGui::SameLine();
            if (ImGui::Button("Randomize Scale"))
            {
               lastTRS      = m_TRS; // Backup before randomizing
               int attempts = 0;
               do
               {
                  m_TRS.scale = float32_t3(rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f));
                  attempts++;
               } while (!isCubeOutsideUnitSphere(m_TRS.translation, m_TRS.scale) && attempts < 100);
            }
            // ImGui::SameLine();
            if (ImGui::Button("Randomize All"))
            {
               lastTRS      = m_TRS; // Backup before randomizing
               int attempts = 0;
               do
               {
                  m_TRS.translation = float32_t3(rng.nextFloat(-3.f, 3.f), rng.nextFloat(-3.f, 3.f), rng.nextFloat(-1.f, 3.f));
                  m_TRS.rotation    = float32_t3(rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f));
                  m_TRS.scale       = float32_t3(rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f));
                  attempts++;
               } while (!isCubeOutsideUnitSphere(m_TRS.translation, m_TRS.scale) && attempts < 100);
            }
            ImGui::SameLine();
            if (ImGui::Button("Revert to Last"))
            {
               m_TRS = lastTRS; // Restore backed-up TRS
            }

            addMatrixTable("Model Matrix", "ModelMatrixTable", 4, 4, &m_OBBModelMatrix[0][0]);
            addMatrixTable("Camera View Matrix", "ViewMatrixTable", 3, 4, &camera.getViewMatrix()[0].x);
            addMatrixTable("Camera View Projection Matrix", "ViewProjectionMatrixTable", 4, 4, &camera.getProjectionMatrix()[0].x, false);

            ImGui::End();
         }

         // Nabla Imgui backend MDI buffer info
         // To be 100% accurate and not overly conservative we'd have to explicitly `cull_frees` and defragment each time,
         // so unless you do that, don't use this basic info to optimize the size of your IMGUI buffer.
         {
            auto* streaminingBuffer = imGUI->getStreamingBuffer();

            const size_t total          = streaminingBuffer->get_total_size(); // total memory range size for which allocation can be requested
            const size_t freeSize       = streaminingBuffer->getAddressAllocator().get_free_size(); // max total free bloock memory size we can still allocate from total memory available
            const size_t consumedMemory = total - freeSize; // memory currently consumed by streaming buffer

            float freePercentage      = 100.0f * (float)(freeSize) / (float)total;
            float allocatedPercentage = (float)(consumedMemory) / (float)total;

            ImVec2 barSize         = ImVec2(400, 30);
            float  windowPadding   = 10.0f;
            float  verticalPadding = ImGui::GetStyle().FramePadding.y;

            ImGui::SetNextWindowSize(ImVec2(barSize.x + 2 * windowPadding, 110 + verticalPadding), ImGuiCond_Always);
            ImGui::Begin("Nabla Imgui MDI Buffer Info", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);

            ImGui::Text("Total Allocated Size: %zu bytes", total);
            ImGui::Text("In use: %zu bytes", consumedMemory);
            ImGui::Text("Buffer Usage:");

            ImGui::SetCursorPosX(windowPadding);

            if (freePercentage > 70.0f)
               ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 1.0f, 0.0f, 0.4f)); // Green
            else if (freePercentage > 30.0f)
               ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 1.0f, 0.0f, 0.4f)); // Yellow
            else
               ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 0.4f)); // Red

            ImGui::ProgressBar(allocatedPercentage, barSize, "");

            ImGui::PopStyleColor();

            ImDrawList* drawList = ImGui::GetWindowDrawList();

            ImVec2 progressBarPos  = ImGui::GetItemRectMin();
            ImVec2 progressBarSize = ImGui::GetItemRectSize();

            const char* text = "%.2f%% free";
            char        textBuffer[64];
            snprintf(textBuffer, sizeof(textBuffer), text, freePercentage);

            ImVec2 textSize = ImGui::CalcTextSize(textBuffer);
            ImVec2 textPos  = ImVec2(
               progressBarPos.x + (progressBarSize.x - textSize.x) * 0.5f,
               progressBarPos.y + (progressBarSize.y - textSize.y) * 0.5f);

            ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
            drawList->AddRectFilled(
               ImVec2(textPos.x - 5, textPos.y - 2),
               ImVec2(textPos.x + textSize.x + 5, textPos.y + textSize.y + 2),
               ImGui::GetColorU32(bgColor));

            ImGui::SetCursorScreenPos(textPos);
            ImGui::Text("%s", textBuffer);

            ImGui::Dummy(ImVec2(0.0f, verticalPadding));

            ImGui::End();
         }
         ImGui::End();

         ImGuizmo::RecomposeMatrixFromComponents(&m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x, &m_OBBModelMatrix[0][0]);
      }

      smart_refctd_ptr<ext::imgui::UI> imGUI;

      // descriptor set
      smart_refctd_ptr<SubAllocatedDescriptorSet> subAllocDS;
      enum E_RENDER_VIEWS : uint8_t
      {
         ERV_MAIN_VIEW,
         ERV_SOLID_ANGLE_VIEW,
         Count
      };
      SubAllocatedDescriptorSet::value_type renderColorViewDescIndices[E_RENDER_VIEWS::Count] = {SubAllocatedDescriptorSet::invalid_value, SubAllocatedDescriptorSet::invalid_value};
      //
      Camera camera = Camera(cameraIntialPosition, cameraInitialTarget, {}, 1, 1, nbl::core::vectorSIMDf(0.0f, 0.0f, 1.0f));
      // mutables
      struct TRS // Source of truth
      {
         float32_t3 translation {0.0f, 0.0f, 1.5f};
         float32_t3 rotation {0.0f}; // MUST stay orthonormal
         float32_t3 scale {1.0f};
      } m_TRS;
      float32_t4x4 m_OBBModelMatrix; // always overwritten from TRS

      // std::string_view objectName;
      TransformRequestParams transformParams;
      TransformReturnInfo    mainViewTransformReturnInfo;
      TransformReturnInfo    solidAngleViewTransformReturnInfo;

      const static inline core::vectorSIMDf cameraIntialPosition {-3.0f, 6.0f, 3.0f};
      const static inline core::vectorSIMDf cameraInitialTarget {0.f, 0.0f, 3.f};
      const static inline core::vectorSIMDf cameraInitialUp {0.f, 0.f, 1.f};

      float fov = 90.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
      float viewWidth = 10.f;
      // uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed
      bool isPerspective = true, isLH = true, flipGizmoY = true, move = true;
      bool firstFrame = true;

      SolidAngleVisualizer* m_visualizer;
   } interface;

   class SamplingBenchmark final
   {
  public:
      SamplingBenchmark(SolidAngleVisualizer& base)
         : m_api(base.m_api), m_device(base.m_device), m_logger(base.m_logger), m_visualizer(&base)
      {
         // setting up pipeline in the constructor
         m_queueFamily = base.getComputeQueue()->getFamilyIndex();
         m_cmdpool     = base.m_device->createCommandPool(m_queueFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
         if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_cmdbuf))
            base.logFail("Failed to create Command Buffers!\n");

         // Load shaders, set up pipelines (one per sampling mode)
         {
            auto loadShader = [&](auto key) -> smart_refctd_ptr<IShader>
            {
               IAssetLoader::SAssetLoadParams lp = {};
               lp.logger                         = base.m_logger.get();
               lp.workingDirectory               = "app_resources";
               auto       assetBundle            = base.m_assetMgr->getAsset(key.data(), lp);
               const auto assets                 = assetBundle.getContents();
               if (assets.empty())
               {
                  base.logFail("Could not load shader!");
                  assert(0);
               }
               assert(assets.size() == 1);
               auto shader = IAsset::castDown<IShader>(assets[0]);
               if (!shader)
                  base.logFail("Failed to load precompiled benchmark shader!\n");
               return shader;
            };

            const char*               shaderNames[SAMPLING_MODE_FLAGS::Count] = {};
            smart_refctd_ptr<IShader> shaders[SAMPLING_MODE_FLAGS::Count];

            auto addBench = [&]<nbl::core::StringLiteral Key>(SAMPLING_MODE_FLAGS mode)
            {
               shaderNames[denseIdOf(mode)] = Key.value;
               shaders[denseIdOf(mode)]     = loadShader(nbl::this_example::builtin::build::get_spirv_key<Key>(m_device.get()));
            };

            addBench.template operator()<"benchmark_tri_sa">(SAMPLING_MODE_FLAGS::TRIANGLE_SOLID_ANGLE);
            addBench.template operator()<"benchmark_tri_psa">(SAMPLING_MODE_FLAGS::TRIANGLE_PROJECTED_SOLID_ANGLE);
            addBench.template operator()<"benchmark_para">(SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE);
            addBench.template operator()<"benchmark_rectangle">(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID);
            addBench.template operator()<"benchmark_bilinear">(SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID);
            addBench.template operator()<"benchmark_proj_rectangle">(SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID);
            addBench.template operator()<"benchmark_silhouette">(SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY);
            addBench.template operator()<"benchmark_pyramid_creation">(SAMPLING_MODE_FLAGS::PYRAMID_CREATION_ONLY);
            addBench.template operator()<"benchmark_caliper_pyramid_creation">(SAMPLING_MODE_FLAGS::CALIPER_PYRAMID_CREATION_ONLY);
            addBench.template operator()<"benchmark_caliper_rectangle">(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID);

            nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
               {.binding       = 0,
                  .type        = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                  .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                  .stageFlags  = ShaderStage::ESS_COMPUTE,
                  .count       = 1}};
            smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = base.m_device->createDescriptorSetLayout(bindings);
            if (!dsLayout)
               base.logFail("Failed to create a Descriptor Layout!\n");

            SPushConstantRange pushConstantRanges[] = {
               {.stageFlags = ShaderStage::ESS_COMPUTE,
                  .offset   = 0,
                  .size     = sizeof(BenchmarkPushConstants)}};
            m_pplnLayout = base.m_device->createPipelineLayout(pushConstantRanges, smart_refctd_ptr(dsLayout));
            if (!m_pplnLayout)
               base.logFail("Failed to create a Pipeline Layout!\n");

            for (uint32_t i = 0; i < SAMPLING_MODE_FLAGS::Count; i++)
            {
               IGPUComputePipeline::SCreationParams params = {};
               params.layout                               = m_pplnLayout.get();
               params.shader.entryPoint                    = "main";
               params.shader.shader                        = shaders[i].get();
               if (base.m_device->getEnabledFeatures().pipelineExecutableInfo)
               {
                  params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS;
                  params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
               }
               if (!base.m_device->createComputePipelines(nullptr, {&params, 1}, &m_pipelines[i]))
                  base.logFail("Failed to create pipelines (compile & link shaders)!\n");
               if (base.m_device->getEnabledFeatures().pipelineExecutableInfo)
               {
                  m_pipelineReports[i]     = system::to_string(m_pipelines[i]->getExecutableInfo());
                  m_pipelineReportNames[i] = shaderNames[i];
               }
            }

            // Allocate the memory
            {
               constexpr size_t BufferSize = BENCHMARK_WORKGROUP_COUNT * BENCHMARK_WORKGROUP_DIMENSION_SIZE_X * BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y * BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z * sizeof(uint32_t);

               nbl::video::IGPUBuffer::SCreationParams params = {};
               params.size                                    = BufferSize;
               params.usage                                   = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
               smart_refctd_ptr<IGPUBuffer> dummyBuff         = base.m_device->createBuffer(std::move(params));
               if (!dummyBuff)
                  base.logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

               dummyBuff->setObjectDebugName("benchmark buffer");

               nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = dummyBuff->getMemoryReqs();

               m_allocation = base.m_device->allocate(reqs, dummyBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
               if (!m_allocation.isValid())
                  base.logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

               assert(dummyBuff->getBoundMemory().memory == m_allocation.memory.get());
               smart_refctd_ptr<nbl::video::IDescriptorPool> pool = base.m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, {&dsLayout.get(), 1});

               m_ds = pool->createDescriptorSet(std::move(dsLayout));
               {
                  IGPUDescriptorSet::SDescriptorInfo info[1];
                  info[0].desc                                     = smart_refctd_ptr(dummyBuff);
                  info[0].info.buffer                              = {.offset = 0, .size = BufferSize};
                  IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                     {.dstSet = m_ds.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = info}};
                  base.m_device->updateDescriptorSets(writes, {});
               }
            }
         }

         IQueryPool::SCreationParams queryPoolCreationParams {};
         queryPoolCreationParams.queryType               = IQueryPool::TYPE::TIMESTAMP;
         queryPoolCreationParams.queryCount              = 2;
         queryPoolCreationParams.pipelineStatisticsFlags = IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
         m_queryPool                                     = m_device->createQueryPool(queryPoolCreationParams);

         m_computeQueue      = m_device->getQueue(m_queueFamily, 0);
         m_physicalDevice    = base.m_device->getPhysicalDevice();
         m_timestampPeriodNs = float64_t(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);
      }

      void run()
      {
         // Pipeline executable reports first so the timings cluster at the bottom of the log.
         for (uint32_t i = 0; i < SAMPLING_MODE_FLAGS::Count; i++)
         {
            if (!m_pipelineReports[i].empty())
               m_logger->log("%s Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, m_pipelineReportNames[i], m_pipelineReports[i].c_str());
         }

         const uint64_t totalThreads = (uint64_t)BENCHMARK_WORKGROUP_COUNT * BENCHMARK_WORKGROUP_DIMENSION_SIZE_X;
         m_logger->log("\n\n=== GPU Sampler Benchmarks (%d dispatches, %llu threads/dispatch, %d samples/thread, ps/sample is per all GPU threads) ===",
            ILogger::ELL_PERFORMANCE, Dispatches, totalThreads, m_BenchmarkSampleCount);
         m_logger->log("  timestampPeriod = %.1f ps/tick", ILogger::ELL_PERFORMANCE, m_timestampPeriodNs * 1000.0);
         m_logger->log("%-29s | %-12s | %9s | %10s | %10s",
            ILogger::ELL_PERFORMANCE, "Sampler", "Mode", "ps/sample", "GSamples/s", "ms total");

         struct SamplerEntry
         {
            const char*   name;
            SAMPLING_MODE_FLAGS mode;
         };
         const SamplerEntry samplers[] = {
            {.name = "PYRAMID_RECTANGLE", .mode = SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID},
            {.name = "CALIPER_PYRAMID_RECTANGLE", .mode = SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID},
            {.name = "PYRAMID_PROJ_RECTANGLE", .mode = SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID},
            {.name = "PYRAMID_BILINEAR", .mode = SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID},
            {.name = "PARALLELOGRAM", .mode = SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE},
            {.name = "TRIANGLE_SA", .mode = SAMPLING_MODE_FLAGS::TRIANGLE_SOLID_ANGLE},
            {.name = "TRIANGLE_PSA", .mode = SAMPLING_MODE_FLAGS::TRIANGLE_PROJECTED_SOLID_ANGLE},
         };

         // Creation-only modes: report per-creation, not per-sample.
         performBenchmark("SILHOUETTE_CREATION_ONLY", SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY, totalThreads, 0);
         performBenchmark("PYRAMID_CREATION_ONLY", SAMPLING_MODE_FLAGS::PYRAMID_CREATION_ONLY, totalThreads, 0);
         performBenchmark("CALIPER_PYRAMID_CREATION_ONLY", SAMPLING_MODE_FLAGS::CALIPER_PYRAMID_CREATION_ONLY, totalThreads, 0);

         // Modes per sampler: 1 creation per N samples. 1 = no amortization, sampleCount = full amortization.
         const uint32_t modeRatios[] = {1u, 16u, static_cast<uint32_t>(m_BenchmarkSampleCount)};
         for (uint32_t spc : modeRatios)
            for (const auto& s : samplers)
               performBenchmark(s.name, s.mode, totalThreads, spc);
      }

      // Many dispatches per SAMPLING_MODE_FLAGS, all in a single capture. Intended for NSight submit-mode
      // captures with the Shader Profiler -- each mode's range needs sustained execution so PC sampling
      // can gather enough source-line hits.
      void runNSightOneShot()
      {
         const char* modeNames[SAMPLING_MODE_FLAGS::Count]                       = {};
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID)]       = "CALIPER_PYRAMID_RECTANGLE";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID)]               = "PYRAMID_RECTANGLE";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID)]          = "PYRAMID_PROJ_RECTANGLE";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::TRIANGLE_SOLID_ANGLE)]                = "TRIANGLE_SA";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::TRIANGLE_PROJECTED_SOLID_ANGLE)]      = "TRIANGLE_PSA";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)] = "PARALLELOGRAM";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID)]               = "PYRAMID_BILINEAR";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY)]            = "SILHOUETTE_CREATION_ONLY";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::PYRAMID_CREATION_ONLY)]               = "PYRAMID_CREATION_ONLY";
         modeNames[denseIdOf(SAMPLING_MODE_FLAGS::CALIPER_PYRAMID_CREATION_ONLY)]       = "CALIPER_PYRAMID_CREATION_ONLY";

         m_pushConstants.modelMatrix        = float32_t3x4(transpose(m_visualizer->interface.m_OBBModelMatrix));
         m_pushConstants.sampleCount        = static_cast<uint32_t>(m_BenchmarkSampleCount);
         m_pushConstants.samplesPerCreation = m_pushConstants.sampleCount; // full amortization: 1 creation per dispatch

         m_cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
         m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
         m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
         m_cmdbuf->pushConstants(m_pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(BenchmarkPushConstants), &m_pushConstants);

         const asset::SMemoryBarrier serializeDispatch = {
            .srcStageMask  = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
            .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
            .dstStageMask  = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
            .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
         };
         const IGPUCommandBuffer::SPipelineBarrierDependencyInfo barrierInfo = {.memBarriers = {&serializeDispatch, 1}};

         for (uint32_t mode = 0; mode < SAMPLING_MODE_FLAGS::Count; ++mode)
         {
            m_cmdbuf->beginDebugMarker(modeNames[mode], vectorSIMDf(0, 1, 0, 1));
            m_cmdbuf->bindComputePipeline(m_pipelines[mode].get());
            for (int i = 0; i < NSightDispatchesPerMode; ++i)
            {
               m_cmdbuf->dispatch(BENCHMARK_WORKGROUP_COUNT, 1, 1);
               if (i + 1 < NSightDispatchesPerMode)
                  m_cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, barrierInfo);
            }
            m_cmdbuf->endDebugMarker();
            if (mode + 1u < SAMPLING_MODE_FLAGS::Count)
               m_cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, barrierInfo);
         }
         m_cmdbuf->end();

         smart_refctd_ptr<ISemaphore>              done       = m_device->createSemaphore(0);
         const IQueue::SSubmitInfo::SSemaphoreInfo signals[]  = {{.semaphore = done.get(), .value = 1, .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS}};
         IQueue::SSubmitInfo                           submitInfos[1] = {};
         const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[]      = {{.cmdbuf = m_cmdbuf.get()}};
         submitInfos[0].commandBuffers                                = cmdbufs;
         submitInfos[0].signalSemaphores                              = signals;

         m_api->startCapture();
         m_computeQueue->submit(submitInfos);
         const ISemaphore::SWaitInfo waitInfo[] = {{.semaphore = done.get(), .value = 1}};
         m_device->blockForSemaphores(waitInfo);
         m_api->endCapture();

         m_logger->log("NSight benchmarks: dispatched %u sampling modes in one submit.", ILogger::ELL_INFO, static_cast<uint32_t>(SAMPLING_MODE_FLAGS::Count));
      }

  private:
      // samplesPerCreation: > 0 selects sampling mode with that 1:N ratio; 0 means create-only mode (label "create-only").
      void performBenchmark(const char* name, SAMPLING_MODE_FLAGS mode, uint64_t totalThreads, uint32_t samplesPerCreation)
      {
         m_device->waitIdle();

         const bool isCreationBenchmark = (mode == SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY || mode == SAMPLING_MODE_FLAGS::PYRAMID_CREATION_ONLY || mode == SAMPLING_MODE_FLAGS::CALIPER_PYRAMID_CREATION_ONLY);

         m_pushConstants.modelMatrix = float32_t3x4(transpose(m_visualizer->interface.m_OBBModelMatrix));
         m_pushConstants.sampleCount = m_BenchmarkSampleCount;
         // For create-only modes the inner loop is unused; pick any divisor of sampleCount to keep the shader's `creations = sampleCount / samplesPerCreation` well-defined.
         m_pushConstants.samplesPerCreation = isCreationBenchmark ? uint32_t(m_BenchmarkSampleCount) : samplesPerCreation;
         recordCmdBuff(mode);

         // Nabla's IQueue::submit rejects submissions without a signal semaphore
         // (SSubmitInfo::valid() requires signalSemaphores non-empty so the
         // submission's resources can be tracked on a timeline).
         smart_refctd_ptr<ISemaphore>              done      = m_device->createSemaphore(0);
         const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {{.semaphore = done.get(), .value = 1, .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS}};

         IQueue::SSubmitInfo                           submitInfos[1] = {};
         const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[]      = {{.cmdbuf = m_cmdbuf.get()}};
         submitInfos[0].commandBuffers                                = cmdbufs;
         submitInfos[0].signalSemaphores                              = signals;

         m_api->startCapture();
         m_computeQueue->submit(submitInfos);
         const ISemaphore::SWaitInfo waitInfo[] = {{.semaphore = done.get(), .value = 1}};
         m_device->blockForSemaphores(waitInfo);
         m_api->endCapture();

         const float64_t elapsed_ps = float64_t(calcTimeElapsed()) * m_timestampPeriodNs * 1000.0;

         const uint64_t  totalOps   = uint64_t(Dispatches) * totalThreads * uint64_t(m_BenchmarkSampleCount);
         const float64_t ps_per_op  = elapsed_ps / float64_t(totalOps);
         const float64_t gops_per_s = float64_t(totalOps) / elapsed_ps * 1e3; // ops / (ps × 1e-12) / 1e9
         const float64_t elapsed_ms = elapsed_ps * 1e-9;

         char modeBuf[16];
         if (isCreationBenchmark)
            snprintf(modeBuf, sizeof(modeBuf), "create-only");
         else
            snprintf(modeBuf, sizeof(modeBuf), "1:%u", samplesPerCreation);

         m_logger->log("%-29s | %-12s | %9.2f | %10.2f | %10.3f", ILogger::ELL_PERFORMANCE, name, modeBuf, ps_per_op, gops_per_s, elapsed_ms);
      }

      void recordCmdBuff(SAMPLING_MODE_FLAGS mode) const
      {
         m_cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
         m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
         m_cmdbuf->resetQueryPool(m_queryPool.get(), 0, 2);
         m_cmdbuf->beginDebugMarker("sampling compute dispatch", vectorSIMDf(0, 1, 0, 1));
         m_cmdbuf->bindComputePipeline(m_pipelines[denseIdOf(mode)].get());
         m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
         m_cmdbuf->pushConstants(m_pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(BenchmarkPushConstants), &m_pushConstants);

         // Serialize back-to-back dispatches so each completes before the next begins
         // (matches the original semaphore-chain methodology — measurement is per-dispatch
         // time, not pipelined throughput).
         const asset::SMemoryBarrier serializeDispatch = {
            .srcStageMask  = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
            .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
            .dstStageMask  = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
            .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
         };
         const IGPUCommandBuffer::SPipelineBarrierDependencyInfo barrierInfo = {.memBarriers = {&serializeDispatch, 1}};

         for (int i = 0; i < WarmupDispatches; ++i)
         {
            m_cmdbuf->dispatch(BENCHMARK_WORKGROUP_COUNT, 1, 1);
            m_cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, barrierInfo);
         }

         m_cmdbuf->writeTimestamp(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 0);

         for (int i = 0; i < Dispatches; ++i)
         {
            m_cmdbuf->dispatch(BENCHMARK_WORKGROUP_COUNT, 1, 1);
            if (i + 1 < Dispatches)
               m_cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, barrierInfo);
         }

         m_cmdbuf->writeTimestamp(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 1);
         m_cmdbuf->endDebugMarker();
         m_cmdbuf->end();
      }

      uint64_t calcTimeElapsed() const
      {
         uint64_t            timestamps[2];
         const core::bitflag flags = core::bitflag(IQueryPool::RESULTS_FLAGS::_64_BIT) | core::bitflag(IQueryPool::RESULTS_FLAGS::WAIT_BIT);
         m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, &timestamps, sizeof(uint64_t), flags);
         return timestamps[1] - timestamps[0];
      }

  private:
      core::smart_refctd_ptr<video::CVulkanConnection> m_api;
      smart_refctd_ptr<ILogicalDevice>                 m_device;
      smart_refctd_ptr<ILogger>                        m_logger;
      SolidAngleVisualizer*                            m_visualizer;

      nbl::video::IDeviceMemoryAllocator::SAllocation   m_allocation = {};
      smart_refctd_ptr<nbl::video::IGPUCommandPool>     m_cmdpool    = nullptr;
      smart_refctd_ptr<nbl::video::IGPUCommandBuffer>   m_cmdbuf     = nullptr;
      smart_refctd_ptr<nbl::video::IGPUDescriptorSet>   m_ds         = nullptr;
      smart_refctd_ptr<nbl::video::IGPUPipelineLayout>  m_pplnLayout = nullptr;
      BenchmarkPushConstants                            m_pushConstants;
      smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_pipelines[SAMPLING_MODE_FLAGS::Count];

      smart_refctd_ptr<nbl::video::IQueryPool> m_queryPool = nullptr;

      std::string m_pipelineReports[SAMPLING_MODE_FLAGS::Count];
      const char* m_pipelineReportNames[SAMPLING_MODE_FLAGS::Count] = {};

      uint32_t                           m_queueFamily;
      IQueue*                            m_computeQueue;
      const nbl::video::IPhysicalDevice* m_physicalDevice    = nullptr;
      float64_t                          m_timestampPeriodNs = 1.0;
      static constexpr int               WarmupDispatches    = 100;
      static constexpr int               Dispatches          = 1000;
      // PC sampling needs sustained execution per range; one dispatch is too short. Tune up if NSight still reports too few samples.
      static constexpr int               NSightDispatchesPerMode = 16;
   };

   template<typename... Args>
   inline bool logFail(const char* msg, Args&&... args)
   {
      m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
      return false;
   }

   std::ofstream m_logFile;
};

NBL_MAIN_FUNC(SolidAngleVisualizer)