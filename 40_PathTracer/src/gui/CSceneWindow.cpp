// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "gui/CSceneWindow.h"
#include "renderer/CRenderer.h"
#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include "portable-file-dialogs.h"
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
		if (ImGui::CollapsingHeader("Emitters"))
		{
			// Placeholder - will show lights/emitters
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Emitters list (placeholder)");
			ImGui::Text("Not yet implemented");
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

} // namespace nbl::this_example::gui
