#ifndef __NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED__

#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"

static constexpr inline auto OfflineSceneTextureIx = 1u;

struct TransformRequestParams
{
	bool useWindow = true, editTransformDecomposition = false, enableViewManipulate = false; 
	float camDistance = 8.f, aspectRatio = {}, invAspectRatio = {};
};

void EditTransform(float* cameraView, const float* cameraProjection, float* matrix, TransformRequestParams& params)
{
	static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);
	static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
	static bool useSnap = false;
	static float snap[3] = { 1.f, 1.f, 1.f };
	static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
	static bool boundSizing = false;
	static bool boundSizingSnap = false;

	if (params.editTransformDecomposition)
	{
		if (ImGui::IsKeyPressed(ImGuiKey_T))
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		if (ImGui::IsKeyPressed(ImGuiKey_R))
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		if (ImGui::IsKeyPressed(ImGuiKey_S))
			mCurrentGizmoOperation = ImGuizmo::SCALE;
		if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
			mCurrentGizmoOperation = ImGuizmo::SCALE;
		if (ImGui::RadioButton("Universal", mCurrentGizmoOperation == ImGuizmo::UNIVERSAL))
			mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;
		float matrixTranslation[3], matrixRotation[3], matrixScale[3];
		ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
		ImGui::InputFloat3("Tr", matrixTranslation);
		ImGui::InputFloat3("Rt", matrixRotation);
		ImGui::InputFloat3("Sc", matrixScale);
		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

		if (mCurrentGizmoOperation != ImGuizmo::SCALE)
		{
			if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
				mCurrentGizmoMode = ImGuizmo::LOCAL;
			ImGui::SameLine();
			if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
				mCurrentGizmoMode = ImGuizmo::WORLD;
		}
		if (ImGui::IsKeyPressed(ImGuiKey_S) && ImGui::IsKeyPressed(ImGuiKey_LeftShift))
			useSnap = !useSnap;
		ImGui::Checkbox("##UseSnap", &useSnap);
		ImGui::SameLine();

		switch (mCurrentGizmoOperation)
		{
		case ImGuizmo::TRANSLATE:
			ImGui::InputFloat3("Snap", &snap[0]);
			break;
		case ImGuizmo::ROTATE:
			ImGui::InputFloat("Angle Snap", &snap[0]);
			break;
		case ImGuizmo::SCALE:
			ImGui::InputFloat("Scale Snap", &snap[0]);
			break;
		}
		ImGui::Checkbox("Bound Sizing", &boundSizing);
		if (boundSizing)
		{
			ImGui::PushID(3);
			ImGui::Checkbox("##BoundSizing", &boundSizingSnap);
			ImGui::SameLine();
			ImGui::InputFloat3("Snap", boundsSnap);
			ImGui::PopID();
		}
	}

	ImGuiIO& io = ImGui::GetIO();
	float viewManipulateRight = io.DisplaySize.x;
	float viewManipulateTop = 0;
	static ImGuiWindowFlags gizmoWindowFlags = 0;

	/*
		for the "useWindow" case we just render to a gui area, 
		otherwise to fake full screen transparent window

		note that for both cases we make sure gizmo being 
		rendered is aligned to our texture scene using 
        imgui  "cursor" screen positions
	*/

	SImResourceInfo info;
	info.textureID = OfflineSceneTextureIx;
	info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

	if (params.useWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_Appearing);
		ImGui::SetNextWindowPos(ImVec2(400, 20), ImGuiCond_Appearing);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4)ImColor(0.35f, 0.3f, 0.3f));
		ImGui::Begin("Gizmo", 0, gizmoWindowFlags);
		ImGuizmo::SetDrawlist();

		ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
		ImVec2 windowPos = ImGui::GetWindowPos();
		ImVec2 cursorPos = ImGui::GetCursorScreenPos();

		params.aspectRatio = contentRegionSize.x / contentRegionSize.y;
		params.invAspectRatio = contentRegionSize.y / contentRegionSize.x;

		ImGui::Image(info, contentRegionSize);
		ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);

		viewManipulateRight = cursorPos.x + contentRegionSize.x;
		viewManipulateTop = cursorPos.y;

		ImGuiWindow* window = ImGui::GetCurrentWindow();
		gizmoWindowFlags = (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max) ? ImGuiWindowFlags_NoMove : 0);
	}
	else
	{
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::SetNextWindowSize(io.DisplaySize);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // fully transparent fake window
		ImGui::Begin("FullScreenWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);

		ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
		ImVec2 cursorPos = ImGui::GetCursorScreenPos();

		ImGui::Image(info, contentRegionSize);
		ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);

		viewManipulateRight = cursorPos.x + contentRegionSize.x;
		viewManipulateTop = cursorPos.y;

		params.aspectRatio = io.DisplaySize.x / io.DisplaySize.y;
		params.invAspectRatio = io.DisplaySize.y / io.DisplaySize.x;
	}

	ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix, NULL, useSnap ? &snap[0] : NULL, boundSizing ? bounds : NULL, boundSizingSnap ? boundsSnap : NULL);

	if(params.enableViewManipulate)
		ImGuizmo::ViewManipulate(cameraView, params.camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

	ImGui::End();
	ImGui::PopStyleColor();
}

#endif // __NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED__