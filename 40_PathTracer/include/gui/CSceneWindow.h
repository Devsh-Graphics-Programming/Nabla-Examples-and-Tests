// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_WINDOW_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_WINDOW_H_INCLUDED_

#include "imgui.h"

#include <functional>
#include <string>
#include <vector>

namespace nbl::this_example
{
// Forward declarations
class CScene;
class CSession;
}

namespace nbl::this_example::gui
{

class CSceneWindow final
{
public:
    // Callbacks for user actions - main app handles the actual logic
    struct SCallbacks
    {
        std::function<void(size_t sensorIndex)> onSensorSelected = nullptr;
        std::function<void(const std::string& path)> onLoadRequested = nullptr;
        std::function<void()> onReloadRequested = nullptr;
        // Fired on slider release: should update renderer density then rebuild the tree (typically by re-firing onReloadRequested).
        std::function<void(float density)> onEmitterDensityChanged = nullptr;
        // push-constant uniform branch on the next frame, no rebuild needed.
        std::function<void(bool useAlias)> onUseAliasNEEChanged = nullptr;
        // NEE architecture combo (0=inline, 1=batched deferral, 2=per-bounce wavefront). All three
        // converge to the same image, no accumulation restart needed.
        std::function<void(int deferredMode)> onDeferredNEEChanged = nullptr;
        // Batched band count (request-buffer memory / bandCount); runtime-only, image unchanged.
        std::function<void(int bandCount)> onNeeBandCountChanged = nullptr;
        // handler must restart accumulation (unlike the alias toggle).
        std::function<void(int misMode)> onMisModeChanged = nullptr;
        // Light sampler combo (0=OBB,1=TriUniform,2=TriArvo,3=TriProjected). Handler sets the sampler and,
        // when OBB<->Triangle granularity changes, rebuilds the light tree; restarts accumulation.
        std::function<void(int sampler)> onLeafSamplerChanged = nullptr;
        // Fired while the move-speed slider is dragged: updates the live camera speed.
        std::function<void(float moveSpeed)> onCameraMoveSpeedChanged = nullptr;
        // Debug probe (consumed by debug.hlsl pdf viz). Fired whenever any field is edited.
        std::function<void(float px, float py, float pz, float nx, float ny, float nz)> onProbeChanged = nullptr;
    };
	void setCallbacks(const SCallbacks& callbacks) { m_callbacks = callbacks; }

    // Initial slider value; lets main sync the panel to the renderer's current density.
    void setEmitterDensity(float d) { m_emitterDensity = d; }
    void setUseAliasNEE(bool v) { m_useAliasNEE = v; }
    void setDeferredNEE(int v) { m_deferredNEE = v; }
    void setNeeBandCount(int v) { m_neeBandCount = v; }
    // Sync the MIS-mode combo to the renderer's current value (0=NEEOnly, 1=BxDFOnly, 2=Both).
    void setMisMode(int v) { m_misMode = v; }
    // Sync the light-sampler combo (0=OBB,1=TriUniform,2=TriArvo,3=TriProjected).
    void setLeafSampler(int v) { m_leafSampler = v; }
    // Sync the panel to the camera's current move speed (called when sensor changes).
    void setCameraMoveSpeed(float s) { m_cameraMoveSpeed = s; }
    // Sync the debug-probe panel (called when probe is reset / scene reloaded).
    void setProbe(float px, float py, float pz, float nx, float ny, float nz)
    {
        m_probe[0]=px; m_probe[1]=py; m_probe[2]=pz;
        m_probeN[0]=nx; m_probeN[1]=ny; m_probeN[2]=nz;
    }
    inline const float* getProbePoint()  const { return m_probe; }
    inline const float* getProbeNormal() const { return m_probeN; }
    // Sum of all emitters' backward NEE pdfs at the current probe (~1.0 when the
    // sampler is a valid distribution). Pushed by main.cpp from the renderer.
    void setProbePdfSum(float s) { m_probePdfSum = s; }
    // NEE stats window (renderer's getNeeStats order: calls, selFail, silhDegen, draws, dirDegen,
    // zeroTarget, traced, confirmed, zeroContrib).
    void setNeeStats(const uint32_t* c) { for (int i = 0; i < 12; ++i) m_neeStats[i] = c[i]; }

    // Caller (main.cpp) supplies the current view + perspective matrices each frame
    // so we can draw a translation gizmo over the rendered image. Matrices are in
    // OpenGL column-major float[16]. Set to zero pointer to skip the gizmo.
    void setGizmoCameraMatrices(const float* view, const float* proj)
    {
        m_haveCameraMatrices = (view && proj);
        if (m_haveCameraMatrices)
        {
            for (int i=0;i<16;++i) { m_viewMat[i]=view[i]; m_projMat[i]=proj[i]; }
        }
    }

    CSceneWindow() = default;
    ~CSceneWindow() = default;

    // Set the scene to display (can be null)
    void setScene(const CScene* scene) { m_scene = scene; }

    // Get current scene path for display
    void setScenePath(const std::string& path) { m_scenePath = path; }

    // Main draw call - renders the window
    // forceReposition: if true, window will reposition to default location
    void draw(bool forceReposition = false);

    // Current selection state
    int getSelectedSensorIndex() const { return m_selectedSensorIndex; }
    void setSelectedSensorIndex(int idx) { m_selectedSensorIndex = idx; }

    // Window visibility
    bool isOpen() const { return m_isOpen; }
    void setOpen(bool open) { m_isOpen = open; }

private:
    const CScene* m_scene = nullptr;
    std::string m_scenePath = "";
    int m_selectedSensorIndex = -1;
    bool m_isOpen = true;
    float m_emitterDensity = 0.1f;
    bool  m_useAliasNEE    = true;
    int   m_deferredNEE    = 1; // 0=inline, 1=batched, 2=wavefront (matches CRenderer::DeferredNEEMode)
    int   m_neeBandCount   = 4; // batched horizontal bands (matches CRenderer default)
    int   m_misMode        = 2; // 0=NEEOnly, 1=BxDFOnly, 2=Both (matches CSession::MisMode + renderer default)
    int   m_leafSampler    = 0; // 0=OBB,1=TriUniform,2=TriArvo,3=TriProjected (matches CSession::LightSampler)
    float m_cameraMoveSpeed = 1.0f;
    float m_probe[3]  = {0.f, 0.f, 0.f};
    float m_probeN[3] = {0.f, 1.f, 0.f};
    float m_probePdfSum = 0.f;
    uint32_t m_neeStats[12] = {};
    bool  m_haveCameraMatrices = false;
    float m_viewMat[16] = {};
    float m_projMat[16] = {};
	SCallbacks m_callbacks;

    // Section drawing helpers
    void drawLoadSection();
    void drawSensorsSection();
    void drawGlobalsSection();
    void drawEmittersSection();
    void drawEditorSection();
    void drawDebugProbeSection();
};

} // namespace nbl::this_example::gui

#endif // _NBL_THIS_EXAMPLE_C_SCENE_WINDOW_H_INCLUDED_
