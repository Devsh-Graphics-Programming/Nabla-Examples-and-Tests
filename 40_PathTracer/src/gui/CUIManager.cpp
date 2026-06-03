// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "gui/CUIManager.h"
#include "ImGuizmo.h"
#include "renderer/CScene.h"
#include "renderer/CSession.h"
#include "renderer/CRenderer.h"

namespace nbl::this_example::gui
{

using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

core::smart_refctd_ptr<CUIManager> CUIManager::create(SCreationParams&& params)
{
   if (!params.assetManager || !params.utilities || !params.transferQueue)
      return nullptr;

   return core::make_smart_refctd_ptr<CUIManager>(SCachedParams { std::move(params) });
}

bool CUIManager::init(const SInitParams& params)
{
   if (!params.renderpass)
      return false;

   auto* device = m_params.utilities->getLogicalDevice();

   // Set callbacks
   m_sceneWindow.setCallbacks({ .onSensorSelected = params.onSensorSelected,
      .onLoadRequested                            = params.onLoadSceneRequested,
      .onReloadRequested                          = params.onReloadSceneRequested,
      .onEmitterDensityChanged                    = params.onEmitterDensityChanged,
      .onUseAliasNEEChanged                       = params.onUseAliasNEEChanged,
      .onMisModeChanged                           = params.onMisModeChanged,
      .onCameraMoveSpeedChanged                   = params.onCameraMoveSpeedChanged,
      .onProbeChanged                             = params.onProbeChanged });
   m_sessionWindow.setCallbacks({ .onRenderModeChanged = params.onRenderModeChanged,
      .onResolutionChanged                             = params.onResolutionChanged,
      .onMutablesChanged                               = params.onMutablesChanged,
      .onDynamicsChanged                               = params.onDynamicsChanged,
      .onMaxPathDepthChanged                           = params.onMaxPathDepthChanged,
      .onBufferSelected                                = params.onBufferSelected,
      .onBenchmarkRequested                            = params.onBenchmarkRequested,
      .onDumpImageRequested                            = params.onDumpImageRequested });

   // Initialize session texture indices to invalid
   for (auto& idx : m_sessionTextureIndices)
      idx = SubAllocatedDescriptorSet::invalid_value;

   // Create samplers
   {
      IGPUSampler::SParams samplerParams;
      samplerParams.AnisotropicFilter = 1u;
      samplerParams.TextureWrapU      = hlsl::ETC_REPEAT;
      samplerParams.TextureWrapV      = hlsl::ETC_REPEAT;
      samplerParams.TextureWrapW      = hlsl::ETC_REPEAT;

      m_samplers.gui = device->createSampler(samplerParams);
      if (!m_samplers.gui)
         return false;
      m_samplers.gui->setObjectDebugName("UI GUI Sampler");

      // User sampler for session textures
      m_samplers.user = device->createSampler(samplerParams);
      if (!m_samplers.user)
         return false;
      m_samplers.user->setObjectDebugName("UI User Sampler");
   }

   // Create ImGui manager
   {
      nbl::ext::imgui::UI::SCreationParameters imguiParams;

      imguiParams.resources.texturesInfo = { .setIx = 0u, .bindingIx = TexturesBindingIndex };
      imguiParams.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
      imguiParams.assetManager           = m_params.assetManager;
      imguiParams.pipelineCache          = nullptr;
      imguiParams.pipelineLayout         = nbl::ext::imgui::UI::createDefaultPipelineLayout(device, imguiParams.resources.texturesInfo, imguiParams.resources.samplersInfo, MaxUITextureCount);
      imguiParams.renderpass             = core::smart_refctd_ptr<IGPURenderpass>(params.renderpass);
      imguiParams.streamingBuffer        = nullptr;
      imguiParams.subpassIx              = 0u;
      imguiParams.transfer               = m_params.transferQueue;
      imguiParams.utilities              = m_params.utilities;

      m_imguiManager = nbl::ext::imgui::UI::create(std::move(imguiParams));
      if (!m_imguiManager)
      {
         if (m_params.logger.get())
            m_params.logger.log("Failed to create ImGui manager", system::ILogger::ELL_ERROR);
         return false;
      }
      // Increase double-click time threshold (default 0.30s is too fast)
      ImGui::GetIO().MouseDoubleClickTime = 0.5f;
   }

   // Create SubAllocated descriptor set
   {
      const auto* layout = m_imguiManager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
      auto        pool   = device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, { &layout, 1 });
      if (!pool)
      {
         if (m_params.logger.get())
            m_params.logger.log("Failed to create UI descriptor pool", system::ILogger::ELL_ERROR);
         return false;
      }

      auto ds = pool->createDescriptorSet(smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout));
      if (!ds)
      {
         if (m_params.logger.get())
            m_params.logger.log("Failed to create UI descriptor set", system::ILogger::ELL_ERROR);
         return false;
      }

      m_subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
      if (!m_subAllocDS)
      {
         if (m_params.logger.get())
            m_params.logger.log("Failed to create SubAllocatedDescriptorSet", system::ILogger::ELL_ERROR);
         return false;
      }

      // make sure Texture Atlas slot is taken for eternity
      {
         auto dummy = SubAllocatedDescriptorSet::invalid_value;
         m_subAllocDS->multi_allocate(0, 1, &dummy);
         assert(dummy == ext::imgui::UI::FontAtlasTexId);
      }

      // Write font atlas descriptor
      IGPUDescriptorSet::SDescriptorInfo info            = {};
      info.desc                                          = smart_refctd_ptr<IGPUImageView>(m_imguiManager->getFontAtlasView());
      info.info.image.imageLayout                        = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
      const IGPUDescriptorSet::SWriteDescriptorSet write = { .dstSet = m_subAllocDS->getDescriptorSet(), .binding = TexturesBindingIndex, .arrayElement = nbl::ext::imgui::UI::FontAtlasTexId, .count = 1, .info = &info };
      if (!device->updateDescriptorSets({ &write, 1 }, {}))
      {
         if (m_params.logger.get())
            m_params.logger.log("Failed to write font atlas descriptor", system::ILogger::ELL_ERROR);
         return false;
      }
   }

   m_initialized = true;
   return true;
}

void CUIManager::deinit()
{
   if (!m_subAllocDS)
      return;

   auto* device = m_params.utilities->getLogicalDevice();

   // Deallocate session texture indices
   unbindSessionTextures(nullptr, 0);

   // Deallocate font atlas
   SubAllocatedDescriptorSet::value_type fontAtlasIdx = nbl::ext::imgui::UI::FontAtlasTexId;
   IGPUDescriptorSet::SDropDescriptorSet dummy[1];
   m_subAllocDS->multi_deallocate(dummy, TexturesBindingIndex, 1, &fontAtlasIdx);

   m_subAllocDS  = nullptr;
   m_initialized = false;
}

void CUIManager::setScene(const CScene* scene, const std::string& scenePath)
{
   m_sceneWindow.setScene(scene);
   m_sceneWindow.setScenePath(scenePath);
}

// TODO: actually test this
void CUIManager::setSession(CSession* session, ISemaphore* semaphore, uint64_t semaphoreValue)
{
   // Only rebind if session actually changed
   if (m_currentSession == session)
   {
      m_sessionWindow.setSession(session);
      return;
   }

   // Unbind old session textures
   if (m_currentSession)
      unbindSessionTextures(semaphore, semaphoreValue);

   m_currentSession = session;
   m_sessionWindow.setSession(session);

   // Bind new session textures and pass IDs to session window
   if (session)
   {
      bindSessionTextures(session);

      std::array<uint32_t, static_cast<size_t>(CSessionWindow::BufferType::Count)> textureIDs;
      for (size_t i = 0; i < textureIDs.size(); i++)
         textureIDs[i] = m_sessionTextureIndices[i];
      m_sessionWindow.setBufferTextureIDs(textureIDs);
   }
   else
   {
      std::array<uint32_t, static_cast<size_t>(CSessionWindow::BufferType::Count)> invalidIDs;
      invalidIDs.fill(SubAllocatedDescriptorSet::invalid_value);
      m_sessionWindow.setBufferTextureIDs(invalidIDs);
   }
}

void CUIManager::bindSessionTextures(CSession* session)
{
   if (!session || !m_subAllocDS)
      return;

   auto*       device     = m_params.utilities->getLogicalDevice();
   const auto& resources  = session->getActiveResources();
   const auto& immutables = resources.immutables;

   // Helper to bind a single texture. ImGui's fragment shader declares its
   // texture array as `Texture2D` (see ext/ImGui/builtin/hlsl/fragment.hlsl),
   // so we can't reuse the session's `ET_2D_ARRAY` views - we synthesize a
   // fresh `ET_2D` view of layer 0 from the underlying image.
   auto& thumbnailViews = m_sessionThumbnailViews;
   auto  bindTexture    = [&](SessionTextureIndex texIdx, const CSession::SImageWithViews& imageWithViews, asset::E_FORMAT format)
   {
      auto& allocIdx   = m_sessionTextureIndices[static_cast<size_t>(texIdx)];
      auto& cachedView = thumbnailViews[static_cast<size_t>(texIdx)];

      if (!imageWithViews || !imageWithViews.image)
      {
         allocIdx   = SubAllocatedDescriptorSet::invalid_value;
         cachedView = nullptr;
         return;
      }

      // Path tracer accumulators don't write the alpha channel, so for formats that
      // have alpha (A2B10G10R10, RGBA16F, ...) the sampled .a is 0 and ImGui blends
      // the thumbnail to fully transparent. Force alpha=1 via component swizzle.
      IGPUImageView::SComponentMapping swizzle = {};
      swizzle.a                                = IGPUImageView::SComponentMapping::ES_ONE;
      cachedView                               = device->createImageView({
                                       .subUsages        = imageWithViews.image->getCreationParameters().usage & IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
                                       .image            = imageWithViews.image,
                                       .viewType         = IGPUImageView::E_TYPE::ET_2D,
                                       .format           = format,
                                       .components       = swizzle,
                                       .subresourceRange = { .aspectMask = IGPUImage::EAF_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
      });
      if (!cachedView)
      {
         allocIdx = SubAllocatedDescriptorSet::invalid_value;
         return;
      }

      m_subAllocDS->multi_allocate(TexturesBindingIndex, 1, &allocIdx);
      if (allocIdx == SubAllocatedDescriptorSet::invalid_value)
         return;

      IGPUDescriptorSet::SDescriptorInfo info            = {};
      info.desc                                          = cachedView;
      info.info.image.imageLayout                        = IGPUImage::LAYOUT::GENERAL;
      const IGPUDescriptorSet::SWriteDescriptorSet write = { .dstSet = m_subAllocDS->getDescriptorSet(), .binding = TexturesBindingIndex, .arrayElement = allocIdx, .count = 1, .info = &info };
      device->updateDescriptorSets({ &write, 1 }, {});
   };

   bindTexture(SessionTextureIndex::Beauty, immutables.beauty, asset::EF_R32G32B32A32_SFLOAT);
   bindTexture(SessionTextureIndex::Albedo, immutables.albedo, asset::EF_A2B10G10R10_UNORM_PACK32);
   bindTexture(SessionTextureIndex::Normal, immutables.normal, asset::EF_A2B10G10R10_UNORM_PACK32);
   bindTexture(SessionTextureIndex::Motion, immutables.motion, asset::EF_A2B10G10R10_UNORM_PACK32);
   bindTexture(SessionTextureIndex::Mask, immutables.mask, asset::EF_R16_UNORM);
   bindTexture(SessionTextureIndex::RWMCCascades, immutables.rwmcCascades, asset::EF_R16G16B16A16_SFLOAT);
   // SampleCount intentionally NOT bound: it's EF_R16_UINT (integer format),
   // and ImGui's pixel shader samples its textures array as float. Binding it
   // anyway violates VUID-vkCmdDrawIndexedIndirect-format-07753 and produces a
   // GPU fault on most drivers -> device lost. Re-add only if you also expose
   // an EF_R16_UNORM aliased view (which needs MUTABLE_FORMAT_BIT + the unorm
   // format in the image's viewFormats bitset over in CSession.cpp).
}

void CUIManager::unbindSessionTextures(ISemaphore* semaphore, uint64_t semaphoreValue)
{
   if (!m_subAllocDS)
      return;

   ISemaphore::SWaitInfo waitInfo = {};
   if (semaphore)
   {
      waitInfo.semaphore = semaphore;
      waitInfo.value     = semaphoreValue;
   }

   for (auto& idx : m_sessionTextureIndices)
   {
      if (idx != SubAllocatedDescriptorSet::invalid_value)
      {
         m_subAllocDS->multi_deallocate(TexturesBindingIndex, 1, &idx, waitInfo);
         idx = SubAllocatedDescriptorSet::invalid_value;
      }
   }
   for (auto& view : m_sessionThumbnailViews)
      view = nullptr;
}

void CUIManager::update(const nbl::ext::imgui::UI::SUpdateParameters& params)
{
   if (m_imguiManager)
      m_imguiManager->update(params);
}

void CUIManager::drawWindows()
{
   const bool reposition    = m_needsRepositionWindows;
   m_needsRepositionWindows = false;

   // ImGuizmo lives in the same ImGui frame as our windows. Must call BeginFrame
   // exactly once between ImGui::NewFrame and ImGui::Render.
   ImGuizmo::BeginFrame();

   // Draw Scene Window
   m_sceneWindow.draw(reposition);

   // Draw Session Window
   m_sessionWindow.draw(reposition);
}

bool CUIManager::render(IGPUCommandBuffer* cmdbuf, ISemaphore::SWaitInfo waitInfo)
{
   if (!m_initialized || !m_imguiManager || !m_subAllocDS)
      return false;

   const auto& uiParams   = m_imguiManager->getCreationParameters();
   auto*       uiPipeline = m_imguiManager->getPipeline();

   cmdbuf->bindGraphicsPipeline(uiPipeline);
   const auto* ds = m_subAllocDS->getDescriptorSet();
   cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &ds);

   return m_imguiManager->render(cmdbuf, waitInfo);
}

} // namespace nbl::this_example::gui
