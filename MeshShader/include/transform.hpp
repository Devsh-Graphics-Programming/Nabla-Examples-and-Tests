#ifndef _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_

#include "nbl/ext/ImGui/ImGui.h"

#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"


struct TransformRequestParams
{
	float camDistance = 8.f;
	uint8_t sceneTexDescIx = ~0;
	bool useWindow = true;
	bool editTransformDecomposition = false;
	bool enableViewManipulate = false;
};

struct TransformWidget {
	ImGuizmo::OPERATION mCurrentGizmoOperation{ ImGuizmo::TRANSLATE };
	ImGuizmo::MODE mCurrentGizmoMode{ImGuizmo::LOCAL};
	bool useSnap = false;
	float snap[3] = { 1.f, 1.f, 1.f };
	float bounds[6] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	float boundsSnap[3] = { 0.1f, 0.1f, 0.1f };
	bool boundSizing = false;
	bool boundSizingSnap = false;


	void EditTransform(float* matrix, const TransformRequestParams& params) {


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
	
	}


	ImVec2 ViewingGizmo(float* cameraView, const float* cameraProjection, float* matrix, const TransformRequestParams& params) {
		ImGuiIO& io = ImGui::GetIO();
		float viewManipulateRight = io.DisplaySize.x;
		float viewManipulateTop = 0;
		static ImGuiWindowFlags gizmoWindowFlags = 0;
		SImResourceInfo info;
		info.textureID = params.sceneTexDescIx;
		info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;


		ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_Appearing);
		ImGui::SetNextWindowPos(ImVec2(400, 20), ImGuiCond_Appearing);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4)ImColor(0.35f, 0.3f, 0.3f));
		ImGui::Begin("Gizmo", 0, gizmoWindowFlags);
		ImGuizmo::SetDrawlist();

		ImVec2 windowPos = ImGui::GetWindowPos();
		ImVec2 cursorPos = ImGui::GetCursorScreenPos();

		ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
		ImGui::Image(info, contentRegionSize);
		ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);

		viewManipulateRight = cursorPos.x + contentRegionSize.x;
		viewManipulateTop = cursorPos.y;

		ImGuiWindow* window = ImGui::GetCurrentWindow();
		gizmoWindowFlags = (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max) ? ImGuiWindowFlags_NoMove : 0);

		ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix, NULL, useSnap ? &snap[0] : NULL, boundSizing ? bounds : NULL, boundSizingSnap ? boundsSnap : NULL);

		if (params.enableViewManipulate)
			ImGuizmo::ViewManipulate(cameraView, params.camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

		ImGui::End();

		return contentRegionSize;
	}

	ImVec2 Update(float* cameraView, const float* cameraProjection, float* matrix, const TransformRequestParams& params) {
		EditTransform(matrix, params);
		return ViewingGizmo(cameraView, cameraProjection, matrix, params);
	}

};


#endif // __NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED__