#include "app/App.hpp"
#include "app/AppGizmoUtilities.hpp"

void App::TransformEditorContents()
{
	const size_t objectsCount = getManipulableObjectCount();
	assert(objectsCount);

	int activeObject = static_cast<int>(getManipulatedObjectIx());
	std::string activeObjectLabel = getManipulableObjectLabel(static_cast<uint32_t>(activeObject));
	if (ImGui::BeginCombo("Active Object", activeObjectLabel.c_str()))
	{
		for (size_t i = 0u; i < objectsCount; ++i)
		{
			const bool isSelected = activeObject == static_cast<int>(i);
			const auto label = getManipulableObjectLabel(static_cast<uint32_t>(i));
			if (ImGui::Selectable(label.c_str(), isSelected))
			{
				activeObject = static_cast<int>(i);
				bindManipulatedObjectByIx(static_cast<uint32_t>(activeObject));
			}

			if (isSelected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	SManipulableObjectContext objectContext = {};
	if (!tryBuildManipulableObjectContext(static_cast<uint32_t>(activeObject), objectContext))
		return;

	auto imguizmoModel = nbl::ui::makeImGuizmoModel(objectContext.transform);
	float* m16TRSmatrix = &imguizmoModel.outTRS[0][0];

	ImGui::Text("Identifier: \"%s\"", objectContext.label.c_str());
	if (ImGuizmo::IsUsingAny())
		ImGui::TextColored(SCameraAppTransformEditorUiDefaults::GizmoActiveStatusColor, "Gizmo: In Use");
	else
		ImGui::TextColored(SCameraAppTransformEditorUiDefaults::GizmoIdleStatusColor, "Gizmo: Idle");

	if (ImGui::IsItemHovered())
	{
		nbl::ui::CCameraViewportOverlayUtilities::beginHoverInfoOverlay("HoverOverlay", ImGui::GetMousePos());
		ImGui::Text("Right-click and drag on the gizmo to manipulate the object.");
		nbl::ui::CCameraViewportOverlayUtilities::endHoverInfoOverlay();
	}

	ImGui::Separator();

	if (objectContext.kind == SceneManipulatedObjectKind::Model)
	{
		const auto& names = m_debugScene.scene->getInitParams().geometryNames;
		if (!names.empty())
		{
			if (m_debugScene.geometrySelectionIx >= names.size())
				m_debugScene.geometrySelectionIx = 0;

			if (ImGui::BeginCombo("Object Type", names[m_debugScene.geometrySelectionIx].c_str()))
			{
				for (uint32_t i = 0u; i < names.size(); ++i)
				{
					const bool isSelected = (m_debugScene.geometrySelectionIx == i);
					if (ImGui::Selectable(names[i].c_str(), isSelected))
						m_debugScene.geometrySelectionIx = static_cast<uint16_t>(i);

					if (isSelected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
	}

	addMatrixTable("Model (TRS) Matrix", "ModelMatrixTable", 4, 4, m16TRSmatrix);

	if (ImGui::RadioButton("Translate", m_gizmoState.operation == ImGuizmo::TRANSLATE))
		m_gizmoState.operation = ImGuizmo::TRANSLATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Rotate", m_gizmoState.operation == ImGuizmo::ROTATE))
		m_gizmoState.operation = ImGuizmo::ROTATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Scale", m_gizmoState.operation == ImGuizmo::SCALE))
		m_gizmoState.operation = ImGuizmo::SCALE;

	auto transformState = nbl::ui::extractRigidTransformComponentsOrDefault(imguizmoModel.outTRS);

	float32_t3 matrixRotation = hlsl::CCameraMathUtilities::getQuaternionEulerDegrees(transformState.orientation);
	ImGui::InputFloat3("Tr", &transformState.translation[0], "%.3f");
	ImGui::InputFloat3("Rt", &matrixRotation[0], "%.3f");
	ImGui::InputFloat3("Sc", &transformState.scale[0], "%.3f");

	imguizmoModel.outTRS = nbl::ui::composeRigidTransform(
		transformState.translation,
		matrixRotation,
		transformState.scale);
	m16TRSmatrix = &imguizmoModel.outTRS[0][0];

	if (m_gizmoState.operation != ImGuizmo::SCALE)
	{
		if (ImGui::RadioButton("Local", m_gizmoState.mode == ImGuizmo::LOCAL))
			m_gizmoState.mode = ImGuizmo::LOCAL;
		ImGui::SameLine();
		if (ImGui::RadioButton("World", m_gizmoState.mode == ImGuizmo::WORLD))
			m_gizmoState.mode = ImGuizmo::WORLD;
	}

	ImGui::Checkbox(" ", &m_gizmoState.useSnap);
	ImGui::SameLine();
	switch (m_gizmoState.operation)
	{
		case ImGuizmo::TRANSLATE:
			ImGui::InputFloat3("Snap", &m_gizmoState.snap[0]);
			break;
		case ImGuizmo::ROTATE:
			ImGui::InputFloat("Angle Snap", &m_gizmoState.snap[0]);
			break;
		case ImGuizmo::SCALE:
			ImGui::InputFloat("Scale Snap", &m_gizmoState.snap[0]);
			break;
	}

	applyManipulableObjectTransform(objectContext, getCastedMatrix<float64_t>(imguizmoModel.outTRS));
}
