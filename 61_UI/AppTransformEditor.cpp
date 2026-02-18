#include "app/App.hpp"

void App::TransformEditorContents()
{
			static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
			static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
			static bool boundSizing = false;
			static bool boundSizingSnap = false;

			const size_t objectsCount = m_planarProjections.size() + 1u;
			assert(objectsCount);

			std::vector<std::string> sbels(objectsCount);
			for (size_t i = 0; i < objectsCount; ++i)
				sbels[i] = "Object " + std::to_string(i);

			std::vector<const char*> labels(objectsCount);
			for (size_t i = 0; i < objectsCount; ++i)
				labels[i] = sbels[i].c_str();

			int activeObject = boundCameraToManipulate ? static_cast<int>(boundPlanarCameraIxToManipulate.value() + 1u) : 0;
			if (ImGui::Combo("Active Object", &activeObject, labels.data(), static_cast<int>(labels.size())))
			{
				const auto newActiveObject = static_cast<uint32_t>(activeObject);

				if (newActiveObject) // camera
				{
					boundPlanarCameraIxToManipulate = newActiveObject - 1u;
					ICamera* const targetGimbalManipulationCamera = m_planarProjections[boundPlanarCameraIxToManipulate.value()]->getCamera();
					boundCameraToManipulate = smart_refctd_ptr<ICamera>(targetGimbalManipulationCamera);
				}
				else // gc model
				{
					boundPlanarCameraIxToManipulate = std::nullopt;
					boundCameraToManipulate = nullptr;
				}
			}

			ImGuizmoModelM16InOut imguizmoModel;

			if (boundCameraToManipulate)
				imguizmoModel.inTRS = getCastedMatrix<float32_t>(boundCameraToManipulate->getGimbal().template operator() < float64_t4x4 > ());
			else
				imguizmoModel.inTRS = hlsl::transpose(getMatrix3x4As4x4(m_model));

			imguizmoModel.outTRS = imguizmoModel.inTRS;
			float* m16TRSmatrix = &imguizmoModel.outTRS[0][0];

			std::string indent; 
			if (boundCameraToManipulate)
				indent = boundCameraToManipulate->getIdentifier();
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
					ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.2f, 0.2f, 0.2f, 0.8f));
					ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
					ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

					ImVec2 mousePos = ImGui::GetMousePos();
					ImGui::SetNextWindowPos(ImVec2(mousePos.x + 10, mousePos.y + 10), ImGuiCond_Always);

					ImGui::Begin("HoverOverlay", nullptr,
						ImGuiWindowFlags_NoDecoration |
						ImGuiWindowFlags_AlwaysAutoResize |
						ImGuiWindowFlags_NoSavedSettings);

					ImGui::Text("Right-click and drag on the gizmo to manipulate the object.");

					ImGui::End();

					ImGui::PopStyleVar();
					ImGui::PopStyleColor(2);
				}
			}

			ImGui::Separator();

			if (!boundCameraToManipulate)
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

			float32_t3 matrixTranslation, matrixRotation, matrixScale;
			IGimbalController::input_imguizmo_event_t decomposed, recomposed;
			imguizmoModel.outDeltaTRS = IGimbalController::input_imguizmo_event_t(1);

			ImGuizmo::DecomposeMatrixToComponents(m16TRSmatrix, &matrixTranslation[0], &matrixRotation[0], &matrixScale[0]);
			decomposed = *reinterpret_cast<float32_t4x4*>(m16TRSmatrix);
			{
				ImGuiInputTextFlags flags = 0;

				ImGui::InputFloat3("Tr", &matrixTranslation[0], "%.3f", flags);
				ImGui::InputFloat3("Rt", &matrixRotation[0], "%.3f", flags);
				ImGui::InputFloat3("Sc", &matrixScale[0], "%.3f", flags);
			}
			ImGuizmo::RecomposeMatrixFromComponents(&matrixTranslation[0], &matrixRotation[0], &matrixScale[0], m16TRSmatrix);
			recomposed = *reinterpret_cast<float32_t4x4*>(m16TRSmatrix);

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
			if (boundCameraToManipulate)
			{
				const float pmSpeed = boundCameraToManipulate->getMoveSpeedScale();
				const float prSpeed = boundCameraToManipulate->getRotationSpeedScale();

				boundCameraToManipulate->setMoveSpeedScale(1);
				boundCameraToManipulate->setRotationSpeedScale(1);

				auto referenceFrame = getCastedMatrix<float64_t>(imguizmoModel.outTRS);
				boundCameraToManipulate->manipulate({}, &referenceFrame);

				boundCameraToManipulate->setMoveSpeedScale(pmSpeed);
				boundCameraToManipulate->setRotationSpeedScale(prSpeed);

				/*
				{
					static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);

					if (not enableActiveCameraMovement)
					{
						uint32_t vCount = {};

						boundCameraToManipulate->beginInputProcessing(m_nextPresentationTimestamp);
						{
							boundCameraToManipulate->process(nullptr, vCount);

							if (virtualEvents.size() < vCount)
								virtualEvents.resize(vCount);

							IGimbalController::SUpdateParameters params;
							params.imguizmoEvents = { { imguizmoModel.outDeltaTRS } };
							boundCameraToManipulate->process(virtualEvents.data(), vCount, params);
						}
						boundCameraToManipulate->endInputProcessing();

						// I start to think controller should be able to set sensitivity to scale magnitudes of generated events
						// in order for camera to not keep any magnitude scalars like move or rotation speed scales

						if (vCount)
						{
							const float pmSpeed = boundCameraToManipulate->getMoveSpeedScale();
							const float prSpeed = boundCameraToManipulate->getRotationSpeedScale();

							boundCameraToManipulate->setMoveSpeedScale(1);
							boundCameraToManipulate->setRotationSpeedScale(1);

							auto referenceFrame = getCastedMatrix<float64_t>(imguizmoModel.outTRS);
							boundCameraToManipulate->manipulate({ virtualEvents.data(), vCount }, &referenceFrame);

							boundCameraToManipulate->setMoveSpeedScale(pmSpeed);
							boundCameraToManipulate->setRotationSpeedScale(prSpeed);
						}
					}
				}
				*/
			}
			else
			{
				// for scene demo model full affine transformation without limits is assumed 
				m_model = float32_t3x4(hlsl::transpose(imguizmoModel.outTRS));
			}

}


