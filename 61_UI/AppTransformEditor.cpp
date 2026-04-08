#include "app/App.hpp"

void App::TransformEditorContents()
{
			static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
			static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
			static bool boundSizing = false;
			static bool boundSizingSnap = false;

			const size_t objectsCount = getManipulableObjectCount();
			assert(objectsCount);

			std::vector<std::string> sbels(objectsCount);
			for (size_t i = 0; i < objectsCount; ++i)
				sbels[i] = getManipulableObjectLabel(static_cast<uint32_t>(i));

			std::vector<const char*> labels(objectsCount);
			for (size_t i = 0; i < objectsCount; ++i)
				labels[i] = sbels[i].c_str();

			int activeObject = static_cast<int>(getManipulatedObjectIx());
			if (ImGui::Combo("Active Object", &activeObject, labels.data(), static_cast<int>(labels.size())))
				bindManipulatedObjectByIx(static_cast<uint32_t>(activeObject));

			ImGuizmoModelM16InOut imguizmoModel;

			imguizmoModel.inTRS = getManipulableObjectTransform(static_cast<uint32_t>(activeObject));

			imguizmoModel.outTRS = imguizmoModel.inTRS;
			float* m16TRSmatrix = &imguizmoModel.outTRS[0][0];

			std::string indent; 
			if (m_manipulatedObjectKind == SceneManipulatedObjectKind::Camera && boundCameraToManipulate)
				indent = boundCameraToManipulate->getIdentifier();
			else if (m_manipulatedObjectKind == SceneManipulatedObjectKind::FollowTarget)
				indent = m_followTarget.getIdentifier();
			else
				indent = "Geometry Creator Object";

			ImGui::Text("Identifier: \"%s\"", indent.c_str());
			{
				if (ImGuizmo::IsUsingAny())
					ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Gizmo: In Use");
				else
					ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Gizmo: Idle");

				if (ImGui::IsItemHovered())
				{
					nbl::ui::beginHoverInfoOverlay("HoverOverlay", ImGui::GetMousePos());
					ImGui::Text("Right-click and drag on the gizmo to manipulate the object.");
					nbl::ui::endHoverInfoOverlay();
				}
			}

			ImGui::Separator();

			if (m_manipulatedObjectKind == SceneManipulatedObjectKind::Model)
			{
				const auto& names = m_scene->getInitParams().geometryNames;
				if (!names.empty())
				{
					if (gcIndex >= names.size())
						gcIndex = 0;

					if (ImGui::BeginCombo("Object Type", names[gcIndex].c_str()))
					{
						for (uint32_t i = 0u; i < names.size(); ++i)
						{
							const bool isSelected = (gcIndex == i);
							if (ImGui::Selectable(names[i].c_str(), isSelected))
								gcIndex = static_cast<uint16_t>(i);

							if (isSelected)
								ImGui::SetItemDefaultFocus();
						}
						ImGui::EndCombo();
					}
				}

			}

			addMatrixTable("Model (TRS) Matrix", "ModelMatrixTable", 4, 4, m16TRSmatrix);

			if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;

			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
				mCurrentGizmoOperation = ImGuizmo::SCALE;

			hlsl::SRigidTransformComponents<hlsl::float32_t> transformState = {};
			float32_t3 matrixRotation = float32_t3(0.0f);
			imguizmoModel.outDeltaTRS = hlsl::float32_t4x4(1.0f);

			if (!hlsl::tryExtractRigidTransformComponents(imguizmoModel.outTRS, transformState))
			{
				transformState.translation = float32_t3(imguizmoModel.outTRS[3].x, imguizmoModel.outTRS[3].y, imguizmoModel.outTRS[3].z);
				transformState.orientation = hlsl::makeIdentityQuaternion<hlsl::float32_t>();
				transformState.scale = float32_t3(1.0f);
			}
			matrixRotation = hlsl::getQuaternionEulerDegrees(transformState.orientation);
			{
				ImGuiInputTextFlags flags = 0;

				ImGui::InputFloat3("Tr", &transformState.translation[0], "%.3f", flags);
				ImGui::InputFloat3("Rt", &matrixRotation[0], "%.3f", flags);
				ImGui::InputFloat3("Sc", &transformState.scale[0], "%.3f", flags);
			}
			imguizmoModel.outTRS = hlsl::composeTransformMatrix(
				transformState.translation,
				hlsl::makeQuaternionFromEulerDegrees(matrixRotation),
				transformState.scale);
			m16TRSmatrix = &imguizmoModel.outTRS[0][0];

			if (mCurrentGizmoOperation != ImGuizmo::SCALE)
			{
				if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
					mCurrentGizmoMode = ImGuizmo::LOCAL;
				ImGui::SameLine();
				if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
					mCurrentGizmoMode = ImGuizmo::WORLD;
			}

			ImGui::Checkbox(" ", &useSnap);
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

			// generate virtual events given delta TRS matrix
			if (m_manipulatedObjectKind == SceneManipulatedObjectKind::Camera && boundCameraToManipulate)
			{
				auto referenceFrame = getCastedMatrix<float64_t>(imguizmoModel.outTRS);
				nbl::core::applyReferenceFrameToCamera(boundCameraToManipulate.get(), referenceFrame);
				if (boundPlanarCameraIxToManipulate.has_value())
					refreshFollowOffsetConfigForPlanar(boundPlanarCameraIxToManipulate.value());
			}
			else if (m_manipulatedObjectKind == SceneManipulatedObjectKind::FollowTarget)
			{
				setFollowTargetTransform(getCastedMatrix<float64_t>(imguizmoModel.outTRS));
				applyFollowToConfiguredCameras();
			}
			else
			{
				m_model = float32_t3x4(hlsl::transpose(imguizmoModel.outTRS));
			}

}


