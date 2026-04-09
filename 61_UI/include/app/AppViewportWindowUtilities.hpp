#ifndef _NBL_THIS_EXAMPLE_APP_VIEWPORT_WINDOW_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_VIEWPORT_WINDOW_UTILITIES_HPP_

#include "app/AppTypes.hpp"

namespace nbl::ui
{

struct SViewportWindowRuntime final
{
	SViewportWindowFrame frame = {};
	SBoundViewportCameraState viewportState = {};
};

inline SViewportWindowFrame buildViewportWindowFrame()
{
	SViewportWindowFrame frame = {};
	frame.contentRegionSize = ImGui::GetContentRegionAvail();
	frame.cursorPos = ImGui::GetCursorScreenPos();
	frame.overlayRect = { frame.cursorPos, frame.contentRegionSize };
	frame.hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
	frame.focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

	const auto mousePos = ImGui::GetMousePos();
	frame.mouseInside =
		mousePos.x >= frame.cursorPos.x &&
		mousePos.y >= frame.cursorPos.y &&
		mousePos.x <= frame.cursorPos.x + frame.contentRegionSize.x &&
		mousePos.y <= frame.cursorPos.y + frame.contentRegionSize.y;
	return frame;
}

inline bool tryBuildViewportWindowRuntime(
	std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
	SWindowControlBinding& binding,
	const bool flipGizmoY,
	SViewportWindowRuntime& outRuntime)
{
	outRuntime = {};
	outRuntime.frame = buildViewportWindowFrame();
	return tryBuildViewportBoundCameraState(
		planarProjections,
		binding,
		outRuntime.frame.contentRegionSize,
		flipGizmoY,
		outRuntime.viewportState);
}

inline void updateViewportWindowMoveFlag(ImGuiWindow* const window, const SViewportWindowFrame& frame)
{
	if (!window)
		return;

	if (frame.mouseInside)
		window->Flags |= ImGuiWindowFlags_NoMove;
	else
		window->Flags &= ~ImGuiWindowFlags_NoMove;
}

inline void drawFollowTargetOverlayIfActive(
	ImDrawList* const drawList,
	const SBoundViewportCameraState& viewportState,
	const nbl::core::CTrackedTarget& followTarget,
	const SViewportOverlayRect& viewportRect,
	const SScriptedInputRuntimeState& scriptedInput)
{
	if (!drawList)
		return;
	if (!(scriptedInput.enabled && scriptedInput.visualDebug && scriptedInput.visualFollow.active))
		return;

	drawFollowTargetViewportOverlay(
		*drawList,
		{
			.viewMatrix = viewportState.viewMatrix,
			.projectionMatrix = viewportState.projectionMatrix
		},
		followTarget,
		viewportRect);
}

template<typename OverlayDrawFn>
inline void drawViewportTextureAndOverlay(
	SImResourceInfo& info,
	const SViewportWindowRuntime& viewportRuntime,
	const nbl::core::CTrackedTarget& followTarget,
	const SScriptedInputRuntimeState& scriptedInput,
	OverlayDrawFn&& drawOverlay)
{
	const auto& frame = viewportRuntime.frame;
	ImGui::Image(info, frame.contentRegionSize);
	ImGuizmo::SetRect(frame.cursorPos.x, frame.cursorPos.y, frame.contentRegionSize.x, frame.contentRegionSize.y);
	if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
	{
		drawOverlay(*drawList, frame.overlayRect, viewportRuntime.viewportState);
		drawFollowTargetOverlayIfActive(drawList, viewportRuntime.viewportState, followTarget, frame.overlayRect, scriptedInput);
	}
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_VIEWPORT_WINDOW_UTILITIES_HPP_
