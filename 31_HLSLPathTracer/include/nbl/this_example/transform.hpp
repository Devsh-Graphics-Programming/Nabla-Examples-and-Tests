#ifndef _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_

#include "nbl/ui/ICursorControl.h"

#include "nbl/ext/ImGui/ImGui.h"

#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"

struct TransformRequestParams
{
	float camDistance = 8.f;
	bool isSphere = false;
	ImGuizmo::OPERATION allowedOp;
	uint8_t sceneTexDescIx = ~0;
	bool useWindow = false, editTransformDecomposition = false, enableViewManipulate = false;
};

nbl::hlsl::uint16_t2 EditTransform(float* cameraView, const float* cameraProjection, float* matrix, const TransformRequestParams& params)
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
		if (ImGui::IsKeyPressed(ImGuiKey_T)) // Always translate
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		if (ImGui::IsKeyPressed(ImGuiKey_R) && params.allowedOp & ImGuizmo::OPERATION::ROTATE)
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		if (ImGui::IsKeyPressed(ImGuiKey_S) && params.allowedOp & ImGuizmo::OPERATION::SCALEU) // for sphere
			mCurrentGizmoOperation = ImGuizmo::SCALEU;
		if (ImGui::IsKeyPressed(ImGuiKey_S) && params.allowedOp & ImGuizmo::OPERATION::SCALE) // for triangle/rectangle
			mCurrentGizmoOperation = ImGuizmo::SCALE_X | ImGuizmo::SCALE_Y;

#if 0
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
#endif
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
		// TODO: this shouldn't be handled here I think
	SImResourceInfo info;
	info.textureID = params.sceneTexDescIx;
	info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

	nbl::hlsl::uint16_t2 retval;
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

		ImGui::Image(info, contentRegionSize);
		ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);
		retval = { contentRegionSize.x, contentRegionSize.y };

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
		retval = { contentRegionSize.x, contentRegionSize.y };

		viewManipulateRight = cursorPos.x + contentRegionSize.x;
		viewManipulateTop = cursorPos.y;
	}

	ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix, NULL, useSnap ? &snap[0] : NULL, boundSizing ? bounds : NULL, boundSizingSnap ? boundsSnap : NULL);

	//if (params.enableViewManipulate)
		//ImGuizmo::ViewManipulate(cameraView, params.camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

	ImGui::End();
	ImGui::PopStyleColor();

	return retval;
}

#endif // __NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED__
