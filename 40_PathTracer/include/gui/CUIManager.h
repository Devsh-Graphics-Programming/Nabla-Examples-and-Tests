// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_UI_MANAGER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_UI_MANAGER_H_INCLUDED_

#include "nbl/ext/ImGui/ImGui.h"
#include "nbl/video/alloc/SubAllocatedDescriptorSet.h"
#include "gui/CSceneWindow.h"
#include "gui/CSessionWindow.h"
#include "renderer/shaders/pathtrace/push_constants.hlsl"

#include <functional>

namespace nbl::this_example
{
// Forward declarations
class CScene;
class CSession;
class CWindowPresenter;
}

namespace nbl::this_example::gui
{

class CUIManager final : public core::IReferenceCounted
{
public:
    static constexpr uint32_t MaxUITextureCount = 16u;
    static constexpr uint32_t TexturesBindingIndex = 0u;

    // Texture indices for session buffers (after font atlas at index 0)
    enum class SessionTextureIndex : uint32_t
    {
        Beauty = 0,
        Albedo,
        Normal,
        Motion,
        Mask,
        RWMCCascades,
        SampleCount,
        Count
    };

    struct SCreationParams
    {
        core::smart_refctd_ptr<asset::IAssetManager> assetManager;
        core::smart_refctd_ptr<video::IUtilities> utilities;
        video::IQueue* transferQueue = nullptr;
        system::logger_opt_smart_ptr logger = nullptr;
    };

    struct SCachedParams : SCreationParams
    {
    };

    CUIManager(SCachedParams&& params) : m_params(std::move(params)) {}

    struct SInitParams
    {
        video::IGPURenderpass* renderpass = nullptr;
        std::function<void(size_t sensorIndex)> onSensorSelected = nullptr;
        std::function<void(const std::string& path)> onLoadSceneRequested = nullptr;
        std::function<void()> onReloadSceneRequested = nullptr;

        // Session Callbacks
        std::function<void(CSession::RenderMode mode, CSession* session)> onRenderModeChanged = nullptr;
        std::function<void(uint16_t width, uint16_t height)> onResolutionChanged = nullptr;
        std::function<void(const SSensorDynamics& dynamics, CSession* session)> onMutablesChanged = nullptr;
        std::function<void(const SSensorDynamics& dynamics, CSession* session)> onDynamicsChanged = nullptr;
        std::function<void(int bufferIndex)> onBufferSelected = nullptr;

    };

    static core::smart_refctd_ptr<CUIManager> create(SCreationParams&& params);

    // Initialize GPU resources
    bool init(const SInitParams& params);

    // Cleanup (call before destruction)
    void deinit();

    // Set current scene for the scene window
    void setScene(const CScene* scene, const std::string& scenePath = "");

    // Set current active session for session window and bind its textures
    void setSession(CSession* session, video::ISemaphore* semaphore = nullptr, uint64_t semaphoreValue = 0);

    // Update ImGui state with input events
    void update(const nbl::ext::imgui::UI::SUpdateParameters& params);

    // Draw all UI windows - called between beginRenderpass() and endRenderpassAndPresent()
    void drawWindows();
    bool render(video::IGPUCommandBuffer* cmdbuf, video::ISemaphore::SWaitInfo waitInfo = {});

    nbl::ext::imgui::UI* getImGuiManager() { return m_imguiManager.get(); }
    video::IGPUDescriptorSet* getDescriptorSet() { return m_subAllocDS ? m_subAllocDS->getDescriptorSet() : nullptr; }

    CSceneWindow& getSceneWindow() { return m_sceneWindow; }
    CSessionWindow& getSessionWindow() { return m_sessionWindow; }

    // Reset window positions (call when viewport/resolution changes)
    void resetWindowPositions() { m_needsRepositionWindows = true; }

private:
    // Bind session textures to descriptor set
    void bindSessionTextures(CSession* session);
    // Unbind session textures (deallocate indices)
    void unbindSessionTextures(video::ISemaphore* semaphore, uint64_t semaphoreValue);

    SCachedParams m_params;

    // ImGui extension manager
    core::smart_refctd_ptr<nbl::ext::imgui::UI> m_imguiManager;

    // SubAllocated descriptor set for dynamic texture management
    core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> m_subAllocDS;

    // Allocated texture indices for session buffers
    std::array<video::SubAllocatedDescriptorSet::value_type, static_cast<size_t>(SessionTextureIndex::Count)> m_sessionTextureIndices;

    // Samplers
    struct
    {
        core::smart_refctd_ptr<video::IGPUSampler> gui;
        core::smart_refctd_ptr<video::IGPUSampler> user;
    } m_samplers;

    // Current session (for tracking when it changes)
    CSession* m_currentSession = nullptr;

    // UI Windows
    CSceneWindow m_sceneWindow;
    CSessionWindow m_sessionWindow;

    bool m_initialized = false;
    bool m_needsRepositionWindows = true; // Start true to position on first frame
};

} // namespace nbl::this_example::gui

#endif // _NBL_THIS_EXAMPLE_C_UI_MANAGER_H_INCLUDED_
