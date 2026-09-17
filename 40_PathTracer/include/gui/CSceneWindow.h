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
    };
	void setCallbacks(const SCallbacks& callbacks) { m_callbacks = callbacks; }

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
	SCallbacks m_callbacks;

    // Section drawing helpers
    void drawLoadSection();
    void drawSensorsSection();
    void drawGlobalsSection();
    void drawEmittersSection();
    void drawEditorSection();
};

} // namespace nbl::this_example::gui

#endif // _NBL_THIS_EXAMPLE_C_SCENE_WINDOW_H_INCLUDED_
