// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "gui/CSceneWindow.h"
#include "renderer/CRenderer.h"
#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include "portable-file-dialogs.h"
#include "ImGuizmo.h"
#include <bit>
#include <filesystem>

namespace nbl::this_example::gui
{

	void CSceneWindow::draw(const bool forceReposition)
	{
		if (!m_isOpen)
			return;

		// Position on the right side of the viewport
		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		if (viewport->WorkSize.x > 64.0f && viewport->WorkSize.y > 64.0f) // workaround because for some reason viewport size is wrong on first frame.
		{
			const float windowWidth = 320.0f;
			const ImGuiCond cond = forceReposition ? ImGuiCond_Always : ImGuiCond_Appearing;

			ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - windowWidth - 20, viewport->WorkPos.y + 20), cond);
			ImGui::SetNextWindowSize(ImVec2(windowWidth, viewport->WorkSize.y - 40), cond);
			ImGui::SetNextWindowSizeConstraints(ImVec2(280, 400), ImVec2(FLT_MAX, FLT_MAX));

			if (ImGui::Begin("Scene", &m_isOpen, ImGuiWindowFlags_NoCollapse))
			{
				drawLoadSection();

				ImGui::Separator();

				// Only show scene contents if a scene is loaded
				if (m_scene)
				{
					drawGlobalsSection();
					drawSensorsSection();
					drawEmittersSection();
					drawDebugProbeSection();

					ImGui::Separator();

					drawEditorSection();
				}
				else
				{
					ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No scene loaded");
				}

				}
			ImGui::End();
		}
	}

	void CSceneWindow::drawLoadSection()
	{
		// Load button
		if (ImGui::Button("Load Scene"))
		{
			// Get starting directory (extract parent dir if m_scenePath is a file)
			std::string startDir = ".";
			if (!m_scenePath.empty())
			{
				std::filesystem::path p(m_scenePath);
				if (std::filesystem::is_regular_file(p))
					startDir = p.parent_path().string();
				else if (std::filesystem::is_directory(p))
					startDir = m_scenePath;
			}

			pfd::open_file fileDialog("Choose Mitsuba Scene",
				startDir,
				{
					"Mitsuba Scene Files (*.xml)", "*.xml",
					"All Files", "*"
				}
			);

			auto result = fileDialog.result();
			if (!result.empty() && m_callbacks.onLoadRequested)
				m_callbacks.onLoadRequested(result[0]);
		}

		ImGui::SameLine();

		// Reload button - disabled if no scene loaded
		ImGui::BeginDisabled(m_scene == nullptr);
		if (ImGui::Button("Reload"))
		{
			if (m_callbacks.onReloadRequested)
				m_callbacks.onReloadRequested();
		}
		ImGui::EndDisabled();

		// Show current scene path
		if (!m_scenePath.empty())
		{
			ImGui::TextWrapped("Path: %s", m_scenePath.c_str());
		}
		else
		{
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No file loaded");
		}
	}

	void CSceneWindow::drawSensorsSection()
	{
		if (ImGui::CollapsingHeader("Sensors", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (!m_scene)
			{
				ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No scene");
				return;
			}

			const auto sensors = m_scene->getSensors();
			if (sensors.empty())
			{
				ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No sensors in scene");
				return;
			}

			// Table for sensors
			if (ImGui::BeginTable("SensorsTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
			{
				ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 30.0f);
				ImGui::TableSetupColumn("Resolution", ImGuiTableColumnFlags_WidthFixed, 90.0f);
				ImGui::TableSetupColumn("Crop", ImGuiTableColumnFlags_WidthFixed, 90.0f);
				ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableHeadersRow();

				for (size_t i = 0; i < sensors.size(); ++i)
				{
					const auto& sensor = sensors[i];
					const bool isSelected = (static_cast<int>(i) == m_selectedSensorIndex);

					ImGui::TableNextRow();

					// Highlight selected sensor
					if (isSelected)
						ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImVec4(0.2f, 0.4f, 0.6f, 0.5f)));

					// ID column (clickable)
					ImGui::TableNextColumn();
					char idLabel[32];
					snprintf(idLabel, sizeof(idLabel), "%zu", i);

					ImGui::PushID(static_cast<int>(i));
					if (ImGui::Selectable(idLabel, isSelected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick))
					{
						m_selectedSensorIndex = static_cast<int>(i);

						// Double-click triggers the callback
						if (ImGui::IsMouseDoubleClicked(0))
						{
							if (m_callbacks.onSensorSelected)
								m_callbacks.onSensorSelected(i);
						}
					}
					ImGui::PopID();

					// Resolution column
					ImGui::TableNextColumn();
					ImGui::Text("%ux%u", sensor.constants.width, sensor.constants.height);

					// Crop column
					ImGui::TableNextColumn();
					ImGui::Text("%dx%d", sensor.mutableDefaults.cropWidth, sensor.mutableDefaults.cropHeight);

					// Offset column
					ImGui::TableNextColumn();
					ImGui::Text("(%d, %d)", sensor.mutableDefaults.cropOffsetX, sensor.mutableDefaults.cropOffsetY);
				}

				ImGui::EndTable();
			}

			ImGui::Text("Total: %zu sensor(s)", sensors.size());

			ImGui::Separator();
			if (ImGui::DragFloat("Move Speed", &m_cameraMoveSpeed, 0.1f, 0.01f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
			{
				if (m_callbacks.onCameraMoveSpeedChanged)
					m_callbacks.onCameraMoveSpeedChanged(m_cameraMoveSpeed);
			}
		}
	}

	void CSceneWindow::drawGlobalsSection()
	{
		if (ImGui::CollapsingHeader("Globals"))
		{
			// Placeholder - will show scene-wide settings
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Scene globals (placeholder)");

			if (m_scene)
			{
				// Could show scene bounds, etc.
				ImGui::Text("Scene loaded");
			}
		}
	}

	void CSceneWindow::drawEmittersSection()
	{
		if (ImGui::CollapsingHeader("Emitters", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SliderFloat("Density", &m_emitterDensity, 0.f, 1.f, "%.2f");
			if (ImGui::IsItemDeactivatedAfterEdit() && m_callbacks.onEmitterDensityChanged)
				m_callbacks.onEmitterDensityChanged(m_emitterDensity);

			// Runtime A/B between the power-only alias table (O(1)) and the
			// orientation-aware stochastic light-cut tree (O(log N)). Pushed
			// in the next frame's push constant,no rebuild.
			if (ImGui::Checkbox("Use Alias NEE", &m_useAliasNEE) && m_callbacks.onUseAliasNEEChanged)
				m_callbacks.onUseAliasNEEChanged(m_useAliasNEE);
			ImGui::SameLine();
			ImGui::TextDisabled(m_useAliasNEE ? "(alias table, O(1))" : "(light-tree descent, O(log N))");

			// MIS mode: which Beauty pipeline variant (separate shader) to run. NEE-only and BxDF-only
			// are direct-lighting A/B anchors; Both is the full path tracer. Switching restarts
			// accumulation in the handler (the modes converge to different images).
			if (ImGui::Combo("MIS mode", &m_misMode, "NEE only\0BxDF only\0Both\0") && m_callbacks.onMisModeChanged)
				m_callbacks.onMisModeChanged(m_misMode);

			const auto& tree = m_scene->getLightTree();
			if (tree.numLeavesActual == 0)
			{
				ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.4f, 1.0f), "No emitters in scene");
				return;
			}

			// 4-ary tree depth: log_4(numLeavesPadded) = log_2(numLeavesPadded) / 2.
			const uint32_t depth = (std::bit_width(tree.numLeavesPadded) - 1u) / 2u;
			const auto& root = tree.nodes.front();
			const auto extent = root.bbox.getExtent();

			ImGui::Text("Emitters:    %u", tree.numLeavesActual);
			ImGui::Text("Padded to:   %u (Po4)", tree.numLeavesPadded);
			ImGui::Text("Tree nodes:  %zu", tree.nodes.size());
			ImGui::Text("Depth:       %u", depth);
			ImGui::Separator();
			ImGui::Text("Root power:  %.3g", root.power);
			ImGui::Text("Root bbox min: (%.2f, %.2f, %.2f)", root.bbox.minVx.x, root.bbox.minVx.y, root.bbox.minVx.z);
			ImGui::Text("Root bbox max: (%.2f, %.2f, %.2f)", root.bbox.maxVx.x, root.bbox.maxVx.y, root.bbox.maxVx.z);
			ImGui::Text("Root extent:   (%.2f, %.2f, %.2f)", extent.x, extent.y, extent.z);
		}
	}

	void CSceneWindow::drawEditorSection()
	{
		if (ImGui::CollapsingHeader("Editor"))
		{
			// Placeholder - will show properties of selected item
			if (m_selectedSensorIndex >= 0)
			{
				ImGui::Text("Selected: Sensor [%d]", m_selectedSensorIndex);
				ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Properties editor (placeholder)");
			}
			else
			{
				ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Select an item to edit");
			}
		}
	}

	void CSceneWindow::drawDebugProbeSection()
	{
		if (ImGui::CollapsingHeader("Debug Probe", ImGuiTreeNodeFlags_DefaultOpen))
		{
			bool changed = false;
			changed |= ImGui::DragFloat3("Point", m_probe, 0.02f);
			changed |= ImGui::DragFloat3("Normal", m_probeN, 0.02f, -1.f, 1.f);

			// ImGuizmo 3D drag: place a translation gizmo at the probe point. Needs the
			// caller (main.cpp) to push current view + proj matrices via setGizmoCameraMatrices.
			if (m_haveCameraMatrices)
			{
				const ImGuiViewport* vp = ImGui::GetMainViewport();
				ImGuizmo::SetOrthographic(false);
				ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
				ImGuizmo::SetRect(vp->WorkPos.x, vp->WorkPos.y, vp->WorkSize.x, vp->WorkSize.y);

				// Translation-only matrix (column-major float[16]) at the probe point.
				float gizmoMat[16] = {
					1.f, 0.f, 0.f, 0.f,
					0.f, 1.f, 0.f, 0.f,
					0.f, 0.f, 1.f, 0.f,
					m_probe[0], m_probe[1], m_probe[2], 1.f
				};
				if (ImGuizmo::Manipulate(m_viewMat, m_projMat, ImGuizmo::TRANSLATE, ImGuizmo::WORLD, gizmoMat))
				{
					m_probe[0] = gizmoMat[12];
					m_probe[1] = gizmoMat[13];
					m_probe[2] = gizmoMat[14];
					changed = true;
				}
			}

			if (changed && m_callbacks.onProbeChanged)
				m_callbacks.onProbeChanged(m_probe[0], m_probe[1], m_probe[2], m_probeN[0], m_probeN[1], m_probeN[2]);

			ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
				"Sliders / 3D gizmo move the NEE-pdf probe (debug.hlsl).");

			// Correctness invariant: the per-leaf NEE pdfs must sum to 1 (the
			// descent partitions probability across the whole tree). Green when
			// close, red otherwise, mirrors the screen-border tint in debug.hlsl.
			const float err = (m_probePdfSum >= 1.f) ? (m_probePdfSum - 1.f) : (1.f - m_probePdfSum);
			const ImVec4 col = (err < 0.05f) ? ImVec4(0.2f, 1.f, 0.3f, 1.f) : ImVec4(1.f, 0.3f, 0.2f, 1.f);
			ImGui::TextColored(col, "sum(pdf) = %.5f  (expect 1.0)", m_probePdfSum);
		}
	}

} // namespace nbl::this_example::gui
