// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SESSION_WINDOW_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SESSION_WINDOW_H_INCLUDED_

#include "imgui.h"
#include "renderer/shaders/pathtrace/push_constants.hlsl"
#include <functional>
#include <string>
#include <array>
#include <renderer/CSession.h>

namespace nbl::this_example
{
	class CSession;
}

namespace nbl::this_example::gui
{

class CSessionWindow final
{
public:
	// Buffer types that can be displayed
	enum class BufferType : uint32_t
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

	// Callbacks to main app
	struct SCallbacks
	{
		// Requires session recreation
		std::function<void(CSession::RenderMode mode, CSession* session)> onRenderModeChanged = nullptr;
		std::function<void(uint16_t width, uint16_t height)> onResolutionChanged = nullptr;

		// Requires reset()
		std::function<void(const SSensorDynamics& dynamics, CSession* session)> onMutablesChanged = nullptr;

		// Immediate update()
		std::function<void(const SSensorDynamics& dynamics, CSession* session)> onDynamicsChanged = nullptr;

		// Buffer view change
		std::function<void(int bufferIndex)> onBufferSelected = nullptr;
	};
	void setCallbacks(const SCallbacks& callbacks) { m_callbacks = callbacks; }

	CSessionWindow() = default;
	~CSessionWindow() = default;

	// Set active session to display/control
	void setSession(CSession* session);

	// Set texture IDs for buffer thumbnails (called by CUIManager)
	void setBufferTextureIDs(const std::array<uint32_t, static_cast<size_t>(BufferType::Count)>& textureIDs);

	// forceReposition: if true, window will reposition to default location
	void draw(bool forceReposition = false);

	bool isOpen() const { return m_isOpen; }
	void setOpen(bool open) { m_isOpen = open; }

	// Get currently selected buffer
	BufferType getSelectedBuffer() const { return static_cast<BufferType>(m_state.selectedBufferIndex); }

private:
	CSession* m_session = nullptr;
	bool m_isOpen = true;
	SCallbacks m_callbacks;

	// Texture IDs for buffer thumbnails
	std::array<uint32_t, static_cast<size_t>(BufferType::Count)> m_bufferTextureIDs = {};

	// Local state for UI controls
	struct SState
	{
		// Mode
		CSession::RenderMode renderMode = CSession::RenderMode::Beauty;

		// Dynamics
		float cropOffsetX = 0.0f;
		float cropOffsetY = 0.0f;
		float tMax = 10000.0f;

		// Mutables
		int cropWidth = 1920;
		int cropHeight = 1080;
		float nearClip = 0.1f;
		float farClip = 10000.0f;

		// View
		int selectedBufferIndex = 0;
	} m_state;

	// Copies to check for changes
	SSensorDynamics m_cachedDynamics;

	void drawRenderModeSection();
	void drawDynamicsSection();
	void drawMutablesSection();
	void drawOutputBufferSection();
};

} // namespace nbl::this_example::gui

#endif // _NBL_THIS_EXAMPLE_C_SESSION_WINDOW_H_INCLUDED_
