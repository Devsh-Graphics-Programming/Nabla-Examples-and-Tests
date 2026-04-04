#include "app/App.hpp"

void App::DrawControlPanel()
{
			const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
			const float panelWidth = std::clamp(displaySize.x * 0.19f, 200.0f, displaySize.x * 0.25f);
			const float panelHeight = std::clamp(displaySize.y * 0.34f, 200.0f, displaySize.y * 0.50f);
			const ImVec2 panelSize = { panelWidth, panelHeight };
			const ImVec2 panelPos = { 0.0f, 0.0f };
			ImGui::SetNextWindowPos(panelPos, ImGuiCond_Always);
			ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 4.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(3.0f, 2.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 3.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 4.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(3.0f, 2.0f));

			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.06f, 0.08f, 0.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.12f, 0.16f, 0.44f));
			ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.64f, 0.72f, 0.84f, 0.55f));
			ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.16f, 0.19f, 0.24f, 0.54f));
			ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.26f, 0.32f, 0.40f, 0.64f));
			ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.30f, 0.36f, 0.45f, 0.70f));
			ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.14f, 0.18f, 0.24f, 0.60f));
			ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.24f, 0.30f, 0.40f, 0.70f));
			ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.28f, 0.36f, 0.46f, 0.78f));
			ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0.14f, 0.18f, 0.24f, 0.60f));
			ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.24f, 0.30f, 0.40f, 0.70f));
			ImGui::PushStyleColor(ImGuiCol_TabActive, ImVec4(0.20f, 0.26f, 0.36f, 0.78f));
			ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImVec4(0.12f, 0.14f, 0.18f, 0.50f));
			ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImVec4(0.16f, 0.18f, 0.22f, 0.50f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.98f, 0.99f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TextDisabled, ImVec4(0.82f, 0.86f, 0.90f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.54f, 0.60f, 0.70f, 0.80f));
			ImGui::PushStyleColor(ImGuiCol_SeparatorHovered, ImVec4(0.68f, 0.76f, 0.88f, 0.90f));
			ImGui::PushStyleColor(ImGuiCol_SeparatorActive, ImVec4(0.82f, 0.90f, 1.0f, 0.96f));

			ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);
			ImGui::SetNextWindowBgAlpha(0.0f);
			if (m_ciMode)
				ImGui::SetNextWindowFocus();
			ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

			const ImVec4 accent = ImVec4(0.60f, 0.82f, 1.0f, 1.0f);
			const ImVec4 good = ImVec4(0.45f, 0.90f, 0.60f, 1.0f);
			const ImVec4 bad = ImVec4(1.0f, 0.50f, 0.45f, 1.0f);
			const ImVec4 warn = ImVec4(0.95f, 0.80f, 0.45f, 1.0f);
			const ImVec4 muted = ImVec4(0.92f, 0.93f, 0.95f, 1.0f);
			const ImVec4 badgeText = ImVec4(0.10f, 0.11f, 0.13f, 1.0f);
			const ImVec4 keyBg = ImVec4(0.20f, 0.22f, 0.25f, 1.0f);
			const ImVec4 keyFg = ImVec4(0.92f, 0.94f, 0.96f, 1.0f);
			const ImGuiTableFlags tableFlags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_PadOuterX;
			const ImVec4 panelBg = ImVec4(0.03f, 0.04f, 0.05f, 0.50f);
			const ImVec4 panelEdge = ImVec4(0.62f, 0.70f, 0.84f, 0.60f);
			const ImVec4 panelStripe = ImVec4(0.28f, 0.56f, 0.90f, 0.70f);
			const ImVec4 panelShadow = ImVec4(0.0f, 0.0f, 0.0f, 0.12f);

			{
				const ImVec2 panelPos = ImGui::GetWindowPos();
				const ImVec2 panelSize = ImGui::GetWindowSize();
				auto* drawList = ImGui::GetWindowDrawList();
				drawList->AddRectFilled(ImVec2(panelPos.x + 2.0f, panelPos.y + 3.0f), ImVec2(panelPos.x + panelSize.x + 4.0f, panelPos.y + panelSize.y + 5.0f), ImGui::ColorConvertFloat4ToU32(panelShadow), 8.0f);
				drawList->AddRectFilled(panelPos, ImVec2(panelPos.x + panelSize.x, panelPos.y + panelSize.y), ImGui::ColorConvertFloat4ToU32(panelBg), 6.0f);
				drawList->AddRect(panelPos, ImVec2(panelPos.x + panelSize.x, panelPos.y + panelSize.y), ImGui::ColorConvertFloat4ToU32(panelEdge), 6.0f);
				drawList->AddRectFilled(panelPos, ImVec2(panelPos.x + 4.0f, panelPos.y + panelSize.y), ImGui::ColorConvertFloat4ToU32(panelStripe), 6.0f);
			}

			auto row = [&](const char* label, auto&& drawValue)
			{
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::TextUnformatted(label);
				ImGui::TableSetColumnIndex(1);
				drawValue();
			};

			auto metricMax = [&](const std::array<float, UiMetricSamples>& values, float minValue) -> float
			{
				float maxValue = minValue;
				for (const float v : values)
					maxValue = std::max(maxValue, v);
				return maxValue;
			};

			auto miniStat = [&](const char* id, const char* label, const ImVec4& color, const std::array<float, UiMetricSamples>& series, float minValue, auto&& drawValue)
			{
				ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.14f, 0.16f, 0.19f, 0.75f));
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
				if (ImGui::BeginChild(id, ImVec2(0, 56), true, ImGuiWindowFlags_NoScrollbar))
				{
					ImGui::TextDisabled("%s", label);
					ImGui::SetWindowFontScale(1.05f);
					drawValue();
					ImGui::SetWindowFontScale(1.0f);
					ImGui::PushStyleColor(ImGuiCol_PlotLines, color);
					const float maxValue = metricMax(series, minValue);
					ImGui::PlotLines("##plot", series.data(), static_cast<int>(UiMetricSamples), static_cast<int>(m_uiMetricIndex), nullptr, 0.0f, maxValue, ImVec2(0, 24));
					ImGui::PopStyleColor();
				}
				ImGui::EndChild();
				ImGui::PopStyleVar();
				ImGui::PopStyleColor();
			};

			auto calcPillWidth = [&](const char* label, const ImVec2& pad)
			{
				return ImGui::CalcTextSize(label).x + pad.x * 2.0f;
			};

			auto drawTogglePill = [&](const char* label, bool& value, const ImVec4& onCol, const ImVec4& offCol, const ImVec2& pad)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, value ? onCol : offCol);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, value ? onCol : offCol);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, value ? onCol : offCol);
				ImGui::PushStyleColor(ImGuiCol_Text, badgeText);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, pad);
				if (ImGui::Button(label))
					value = !value;
				ImGui::PopStyleVar();
				ImGui::PopStyleColor(4);
			};

			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
			if (ImGui::BeginChild("PanelHeader", ImVec2(0, 64), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
			{
				ImGui::Dummy(ImVec2(0.0f, 1.0f));
				ImGui::SetWindowFontScale(1.08f);
				ImGui::TextColored(accent, "Control Panel");
				ImGui::SetWindowFontScale(1.0f);
				{
					const ImVec2 badgePad = ImVec2(6.0f, 2.0f);
					const float gap = ImGui::GetStyle().ItemSpacing.x;
					const char* badgeWindow = useWindow ? "WINDOW" : "FULL";
					const char* badgeMove = enableActiveCameraMovement ? "MOVE ON" : "MOVE OFF";
					const char* badgeScript = m_scriptedInput.enabled ? (m_scriptedInput.exclusive ? "SCRIPT EXCL" : "SCRIPT") : "SCRIPT OFF";
					const float badgeRowWidth = calcPillWidth(badgeWindow, badgePad)
						+ gap + calcPillWidth(badgeMove, badgePad)
						+ gap + calcPillWidth(badgeScript, badgePad)
						+ (m_ciMode ? (gap + calcPillWidth("CI", badgePad)) : 0.0f);
					ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - badgeRowWidth) * 0.5f));

					DrawBadge(badgeWindow, accent, badgeText);
					ImGui::SameLine(0.0f, gap);
					DrawBadge(badgeMove, enableActiveCameraMovement ? good : bad, badgeText);
					ImGui::SameLine(0.0f, gap);
					DrawBadge(badgeScript, m_scriptedInput.enabled ? accent : ImVec4(0.35f, 0.36f, 0.38f, 1.0f), badgeText);
					if (m_ciMode)
					{
						ImGui::SameLine(0.0f, gap);
						DrawBadge("CI", warn, badgeText);
					}
				}

				ImGui::Dummy(ImVec2(0.0f, 2.0f));
				{
					const ImVec2 keyPad = ImVec2(4.0f, 1.0f);
					const float gap = ImGui::GetStyle().ItemSpacing.x;
					const float groupGap = gap * 2.0f;
					const float moveWidth = ImGui::CalcTextSize("Move").x + gap
						+ calcPillWidth("W", keyPad) + gap
						+ calcPillWidth("A", keyPad) + gap
						+ calcPillWidth("S", keyPad) + gap
						+ calcPillWidth("D", keyPad);
					const float lookWidth = ImGui::CalcTextSize("Look").x + gap + calcPillWidth("RMB", keyPad);
					const float zoomWidth = ImGui::CalcTextSize("Zoom").x + gap + calcPillWidth("MW", keyPad);
					const float rowWidth = moveWidth + groupGap + lookWidth + groupGap + zoomWidth;
					ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - rowWidth) * 0.5f));

					ImGui::TextDisabled("Move");
					ImGui::SameLine();
					DrawKeyHint("W", keyBg, keyFg);
					ImGui::SameLine();
					DrawKeyHint("A", keyBg, keyFg);
					ImGui::SameLine();
					DrawKeyHint("S", keyBg, keyFg);
					ImGui::SameLine();
					DrawKeyHint("D", keyBg, keyFg);

					ImGui::SameLine(0.0f, groupGap);
					ImGui::TextDisabled("Look");
					ImGui::SameLine();
					DrawKeyHint("RMB", keyBg, keyFg);

					ImGui::SameLine(0.0f, groupGap);
					ImGui::TextDisabled("Zoom");
					ImGui::SameLine();
					DrawKeyHint("MW", keyBg, keyFg);
				}

				ImGui::Dummy(ImVec2(0.0f, 2.0f));
				if (ImGui::BeginTable("HeaderMetrics", 3, ImGuiTableFlags_SizingStretchProp))
				{
					const float frameMs = std::max(0.0f, m_uiLastFrameMs);
					const float fps = frameMs > 0.0f ? (1000.0f / frameMs) : 0.0f;

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					miniStat("FrameStat", "Frame", accent, m_uiFrameMs, 16.0f, [&]
					{
						ImGui::TextColored(accent, "%.1f ms  %.0f fps", frameMs, fps);
					});

					ImGui::TableSetColumnIndex(1);
					miniStat("InputStat", "Input", accent, m_uiInputCounts, 4.0f, [&]
					{
						ImGui::TextColored(accent, "%u ev", m_uiLastInputEvents);
					});

					ImGui::TableSetColumnIndex(2);
					miniStat("VirtualStat", "Virtual", accent, m_uiVirtualCounts, 4.0f, [&]
					{
						ImGui::TextColored(accent, "%u ev", m_uiLastVirtualEvents);
					});
					ImGui::EndTable();
				}
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();

			ImGui::Spacing();

			{
				const ImVec2 togglePad = ImVec2(6.0f, 2.0f);
				const float gap = ImGui::GetStyle().ItemSpacing.x;
				const float rowWidth = calcPillWidth("WINDOW", togglePad)
					+ gap + calcPillWidth("STATUS", togglePad)
					+ gap + calcPillWidth("EVENT LOG", togglePad);
				ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - rowWidth) * 0.5f));
				drawTogglePill("WINDOW", useWindow, accent, ImVec4(0.35f, 0.36f, 0.38f, 1.0f), togglePad);
				DrawHoverHint("Toggle split render windows");
				ImGui::SameLine(0.0f, gap);
				drawTogglePill("STATUS", m_showHud, accent, ImVec4(0.35f, 0.36f, 0.38f, 1.0f), togglePad);
				DrawHoverHint("Show system and camera status panel");
				ImGui::SameLine(0.0f, gap);
				drawTogglePill("EVENT LOG", m_showEventLog, accent, ImVec4(0.35f, 0.36f, 0.38f, 1.0f), togglePad);
				DrawHoverHint("Show virtual event log");
			}

			ImGui::Separator();

			if (ImGui::BeginTabBar("ControlTabs"))
			{
				if (m_showHud && ImGui::BeginTabItem("Status"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("StatusPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						const ImVec4 cardTop = ImVec4(0.20f, 0.22f, 0.26f, 0.98f);
						const ImVec4 cardBottom = ImVec4(0.12f, 0.13f, 0.15f, 0.98f);
						const ImVec4 cardBorder = ImVec4(0.45f, 0.48f, 0.54f, 1.0f);

						DrawSectionHeader("SessionHeader", "Session", accent);
						if (BeginCard("SessionCard", CalcCardHeight(3), cardTop, cardBottom, cardBorder))
						{
							if (ImGui::BeginTable("SessionTable", 2, tableFlags))
							{
								ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 120.0f);
								ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
								row("Mode", [&] { DrawDot(accent); ImGui::TextColored(accent, "%s", useWindow ? "Window" : "Fullscreen"); });
								row("Active window", [&] { DrawDot(accent); ImGui::TextColored(accent, "%u", activeRenderWindowIx); });
								row("Movement", [&] { const ImVec4 c = enableActiveCameraMovement ? good : bad; DrawDot(c); ImGui::TextColored(c, "%s", enableActiveCameraMovement ? "Enabled" : "Disabled"); });
								ImGui::EndTable();
							}
						}
						EndCard();

						DrawSectionHeader("CameraHeader", "Camera", accent);

						auto* activeCamera = getActiveCamera();
						if (activeCamera)
						{
							const auto& gimbal = activeCamera->getGimbal();
							const auto pos = gimbal.getPosition();
							const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));

							if (BeginCard("CameraCard", CalcCardHeight(5), cardTop, cardBottom, cardBorder))
							{
								if (ImGui::BeginTable("CameraTable", 2, tableFlags))
								{
									ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 120.0f);
									ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
									row("Name", [&] { DrawDot(accent); ImGui::TextColored(muted, "%s", activeCamera->getIdentifier().data()); });
									row("Position", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.2f %.2f %.2f", pos.x, pos.y, pos.z); });
									row("Euler", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.1f %.1f %.1f", euler.x, euler.y, euler.z); });
									row("Move scale", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.4f", activeCamera->getMoveSpeedScale()); });
									row("Rotate scale", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.4f", activeCamera->getRotationSpeedScale()); });
									ImGui::EndTable();
								}
							}
							EndCard();
						}
						else
						{
							if (BeginCard("CameraCard", CalcCardHeight(2), cardTop, cardBottom, cardBorder))
								ImGui::TextDisabled("No active camera");
							EndCard();
						}

						DrawSectionHeader("ProjectionHeader", "Projection", accent);

						auto& binding = windowBindings[activeRenderWindowIx];
						auto& planar = m_planarProjections[binding.activePlanarIx];
						if (planar && binding.boundProjectionIx.has_value())
						{
							auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];
							const auto& params = projection.getParameters();
							if (BeginCard("ProjectionCard", CalcCardHeight(4), cardTop, cardBottom, cardBorder))
							{
								if (ImGui::BeginTable("ProjectionTable", 2, tableFlags))
								{
									ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 120.0f);
									ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
									row("Type", [&] { DrawDot(accent); ImGui::TextColored(muted, "%s", params.m_type == IPlanarProjection::CProjection::Perspective ? "Perspective" : "Orthographic"); });
									row("zNear", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.2f", params.m_zNear); });
									row("zFar", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.2f", params.m_zFar); });
									if (params.m_type == IPlanarProjection::CProjection::Perspective)
										row("Fov", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.1f", params.m_planar.perspective.fov); });
									else
										row("Ortho width", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.1f", params.m_planar.orthographic.orthoWidth); });
									ImGui::EndTable();
								}
							}
							EndCard();
						}
						else
						{
							if (BeginCard("ProjectionCard", CalcCardHeight(2), cardTop, cardBottom, cardBorder))
								ImGui::TextDisabled("No projection bound");
							EndCard();
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Projection"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("ProjectionPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						auto& active = windowBindings[activeRenderWindowIx];
						const auto activeRenderWindowIxString = std::to_string(activeRenderWindowIx);

						DrawSectionHeader("PlanarSelectHeader", "Planar Selection", accent);
						ImGui::Text("Active Render Window: %s", activeRenderWindowIxString.c_str());
						DrawHoverHint("Window that receives input and camera switching");
						{
							const size_t planarsCount = m_planarProjections.size();
							assert(planarsCount);

							std::vector<std::string> sbels(planarsCount);
							for (size_t i = 0; i < planarsCount; ++i)
								sbels[i] = "Planar " + std::to_string(i);

							std::vector<const char*> labels(planarsCount);
							for (size_t i = 0; i < planarsCount; ++i)
								labels[i] = sbels[i].c_str();

							int currentPlanarIx = static_cast<int>(active.activePlanarIx);
							if (ImGui::Combo("Active Planar", &currentPlanarIx, labels.data(), static_cast<int>(labels.size())))
							{
								active.activePlanarIx = static_cast<uint32_t>(currentPlanarIx);
								active.pickDefaultProjections(m_planarProjections[active.activePlanarIx]->getPlanarProjections());
							}
							DrawHoverHint("Select which camera the window renders");
						}

						assert(active.boundProjectionIx.has_value());
						assert(active.lastBoundPerspectivePresetProjectionIx.has_value());
						assert(active.lastBoundOrthoPresetProjectionIx.has_value());

						const auto activePlanarIxString = std::to_string(active.activePlanarIx);
						auto& planarBound = m_planarProjections[active.activePlanarIx];
						assert(planarBound);

						DrawSectionHeader("ProjectionParamsHeader", "Projection Parameters", accent);

						auto selectedProjectionType = planarBound->getPlanarProjections()[active.boundProjectionIx.value()].getParameters().m_type;
						{
							const char* labels[] = { "Perspective", "Orthographic" };
							int type = static_cast<int>(selectedProjectionType);

							if (ImGui::Combo("Projection Type", &type, labels, IM_ARRAYSIZE(labels)))
							{
								selectedProjectionType = static_cast<IPlanarProjection::CProjection::ProjectionType>(type);

								switch (selectedProjectionType)
								{
									case IPlanarProjection::CProjection::Perspective: active.boundProjectionIx = active.lastBoundPerspectivePresetProjectionIx.value(); break;
									case IPlanarProjection::CProjection::Orthographic: active.boundProjectionIx = active.lastBoundOrthoPresetProjectionIx.value(); break;
									default: active.boundProjectionIx = std::nullopt; assert(false); break;
								}
							}
							DrawHoverHint("Switch projection type for this planar");
						}

						auto getPresetName = [&](auto ix) -> std::string
						{
							switch (selectedProjectionType)
							{
								case IPlanarProjection::CProjection::Perspective: return "Perspective Projection Preset " + std::to_string(ix);
								case IPlanarProjection::CProjection::Orthographic: return "Orthographic Projection Preset " + std::to_string(ix);
								default: return "Unknown Projection Preset " + std::to_string(ix);
							}
						};

						bool updateBoundVirtualMaps = false;
						if (ImGui::BeginCombo("Projection Preset", getPresetName(active.boundProjectionIx.value()).c_str()))
						{
							auto& projections = planarBound->getPlanarProjections();

							for (uint32_t i = 0; i < projections.size(); ++i)
							{
								const auto& projection = projections[i];
								const auto& params = projection.getParameters();

								if (params.m_type != selectedProjectionType)
									continue;

								bool isSelected = (i == active.boundProjectionIx.value());

								if (ImGui::Selectable(getPresetName(i).c_str(), isSelected))
								{
									active.boundProjectionIx = i;
									updateBoundVirtualMaps |= true;

									switch (selectedProjectionType)
									{
										case IPlanarProjection::CProjection::Perspective: active.lastBoundPerspectivePresetProjectionIx = active.boundProjectionIx.value(); break;
										case IPlanarProjection::CProjection::Orthographic: active.lastBoundOrthoPresetProjectionIx = active.boundProjectionIx.value(); break;
										default: assert(false); break;
									}
								}

								if (isSelected)
									ImGui::SetItemDefaultFocus();
							}
							ImGui::EndCombo();
						}
						if (updateBoundVirtualMaps)
							syncWindowInputBinding(active);
						DrawHoverHint("Switch preset projection for this planar");

						auto* const boundCamera = planarBound->getCamera();
						auto& boundProjection = planarBound->getPlanarProjections()[active.boundProjectionIx.value()];
						assert(not boundProjection.isProjectionSingular());

						auto updateParameters = boundProjection.getParameters();

						if (useWindow)
							ImGui::Checkbox("Allow axes to flip##allowAxesToFlip", &active.allowGizmoAxesToFlip);
						DrawHoverHint("Allow ImGuizmo axes to flip based on view");

						if(useWindow)
							ImGui::Checkbox("Draw debug grid##drawDebugGrid", &active.enableDebugGridDraw);
						DrawHoverHint("Toggle debug grid in the render window");

						if (ImGui::RadioButton("LH", active.leftHandedProjection))
							active.leftHandedProjection = true;

						ImGui::SameLine();

						if (ImGui::RadioButton("RH", not active.leftHandedProjection))
							active.leftHandedProjection = false;
						DrawHoverHint("Toggle left or right handed projection");

						updateParameters.m_zNear = std::clamp(updateParameters.m_zNear, 0.1f, 100.f);
						updateParameters.m_zFar = std::clamp(updateParameters.m_zFar, 110.f, 10000.f);

						ImGui::SliderFloat("zNear", &updateParameters.m_zNear, 0.1f, 100.f, "%.2f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Near clip plane");
						ImGui::SliderFloat("zFar", &updateParameters.m_zFar, 110.f, 10000.f, "%.1f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Far clip plane");

						switch (selectedProjectionType)
						{
							case IPlanarProjection::CProjection::Perspective:
							{
								ImGui::SliderFloat("Fov", &updateParameters.m_planar.perspective.fov, 20.f, 150.f, "%.1f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Perspective field of view");
								boundProjection.setPerspective(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.perspective.fov);
							} break;

							case IPlanarProjection::CProjection::Orthographic:
							{
								ImGui::SliderFloat("Ortho width", &updateParameters.m_planar.orthographic.orthoWidth, 1.f, 30.f, "%.1f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Orthographic width");
								boundProjection.setOrthographic(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.orthographic.orthoWidth);
							} break;

							default: break;
						}

						DrawSectionHeader("CursorHeader", "Cursor Behaviour", accent);
						if (ImGui::TreeNodeEx("Cursor Behaviour"))
						{
							ImGui::Checkbox("Capture OS cursor in move mode", &captureCursorInMoveMode);
							DrawHoverHint("When disabled the app never warps or clamps system cursor");
							if (captureCursorInMoveMode)
							{
								if (ImGui::RadioButton("Clamp to the window", !resetCursorToCenter))
									resetCursorToCenter = false;
								if (ImGui::RadioButton("Reset to the window center", resetCursorToCenter))
									resetCursorToCenter = true;
							}
							else
							{
								ImGui::TextDisabled("Cursor lock disabled");
							}
							ImGui::TreePop();
						}

						if (enableActiveCameraMovement)
							ImGui::TextColored(good, "Bound Camera Movement: Enabled");
						else
							ImGui::TextColored(bad, "Bound Camera Movement: Disabled");

						ImGui::Separator();

						DrawSectionHeader("BoundCameraHeader", "Bound Camera", accent);
						const auto flags = ImGuiTreeNodeFlags_DefaultOpen;
						if (ImGui::TreeNodeEx("Bound Camera", flags))
						{
							ImGui::Text("Type: %s", boundCamera->getIdentifier().data());
							ImGui::Text("Object Ix: %s", std::to_string(active.activePlanarIx + 1u).c_str());
							ImGui::Separator();
							{
								ICamera::SphericalTargetState sphericalState;
								const bool isOrbitLike = boundCamera->tryGetSphericalTargetState(sphericalState);

								float moveSpeed = boundCamera->getMoveSpeedScale();
								float rotationSpeed = boundCamera->getRotationSpeedScale();

								ImGui::SliderFloat("Move speed factor", &moveSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Scale translation speed for this camera");

								if (boundCamera->getAllowedVirtualEvents() & CVirtualGimbalEvent::Rotate)
									ImGui::SliderFloat("Rotate speed factor", &rotationSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Scale rotation speed for this camera");

								boundCamera->setMotionScales(moveSpeed, rotationSpeed);

								if (isOrbitLike)
								{
									float distance = sphericalState.distance;
									ImGui::SliderFloat("Distance", &distance, sphericalState.minDistance, sphericalState.maxDistance, "%.4f", ImGuiSliderFlags_Logarithmic);
									DrawHoverHint("Current orbit distance");
									boundCamera->trySetSphericalDistance(distance);
								}
							}

							if (ImGui::TreeNodeEx("World Data", flags))
							{
								auto& gimbal = boundCamera->getGimbal();
								const auto position = getCastedVector<float32_t>(gimbal.getPosition());
								const auto& orientation = gimbal.getOrientation();
								const auto viewMatrix = getCastedMatrix<float32_t>(gimbal.getViewMatrix());

								addMatrixTable("Position", ("PositionTable_" + activePlanarIxString).c_str(), 1, 3, &position[0], false);
								addMatrixTable("Orientation (Quaternion)", ("OrientationTable_" + activePlanarIxString).c_str(), 1, 4, &orientation[0], false);
								addMatrixTable("View Matrix", ("ViewMatrixTable_" + activePlanarIxString).c_str(), 3, 4, &viewMatrix[0][0], false);
								ImGui::TreePop();
							}

							if (ImGui::TreeNodeEx("Virtual Event Mappings", flags))
							{
								syncWindowInputBinding(active);
								if (displayKeyMappingsAndVirtualStatesInline(&active.inputBinding))
									syncWindowInputBindingToProjection(active);
								ImGui::TreePop();
							}

							ImGui::TreePop();
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Camera"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("CameraPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						DrawSectionHeader("CameraInputHeader", "Input", accent);
						ImGui::Checkbox("Mirror input to all cameras", &m_cameraControls.mirrorInput);
						DrawHoverHint("Apply keyboard and mouse input to every camera");
						ImGui::Checkbox("World translate", &m_cameraControls.worldTranslate);
						DrawHoverHint("Translate in world space instead of camera space");
						ImGui::SliderFloat("Keyboard scale", &m_cameraControls.keyboardScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Scale keyboard movement magnitudes");
						ImGui::SliderFloat("Mouse move scale", &m_cameraControls.mouseMoveScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Scale mouse move magnitudes");
						ImGui::SliderFloat("Mouse scroll scale", &m_cameraControls.mouseScrollScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Scale mouse wheel magnitudes");
						ImGui::SliderFloat("Translate scale", &m_cameraControls.translationScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Overall translation scale for virtual events");
						ImGui::SliderFloat("Rotate scale", &m_cameraControls.rotationScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Overall rotation scale for virtual events");

						DrawSectionHeader("CameraConstraintsHeader", "Constraints", accent);
						ImGui::Checkbox("Enable constraints", &m_cameraConstraints.enabled);
						DrawHoverHint("Enable or disable all camera constraints");
						ImGui::Checkbox("Clamp distance", &m_cameraConstraints.clampDistance);
						DrawHoverHint("Clamp orbit distance to min/max");
						ImGui::SliderFloat("Min distance", &m_cameraConstraints.minDistance, 0.01f, 1000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Minimum orbit distance");
						ImGui::SliderFloat("Max distance", &m_cameraConstraints.maxDistance, 0.01f, 10000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Maximum orbit distance");
						ImGui::Separator();
						ImGui::Checkbox("Clamp pitch", &m_cameraConstraints.clampPitch);
						DrawHoverHint("Clamp pitch angle");
						ImGui::SliderFloat("Pitch min", &m_cameraConstraints.pitchMinDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Minimum pitch in degrees");
						ImGui::SliderFloat("Pitch max", &m_cameraConstraints.pitchMaxDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Maximum pitch in degrees");
						ImGui::Checkbox("Clamp yaw", &m_cameraConstraints.clampYaw);
						DrawHoverHint("Clamp yaw angle");
						ImGui::SliderFloat("Yaw min", &m_cameraConstraints.yawMinDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Minimum yaw in degrees");
						ImGui::SliderFloat("Yaw max", &m_cameraConstraints.yawMaxDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Maximum yaw in degrees");
						ImGui::Checkbox("Clamp roll", &m_cameraConstraints.clampRoll);
						DrawHoverHint("Clamp roll angle");
						ImGui::SliderFloat("Roll min", &m_cameraConstraints.rollMinDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Minimum roll in degrees");
						ImGui::SliderFloat("Roll max", &m_cameraConstraints.rollMaxDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Maximum roll in degrees");

						DrawSectionHeader("OrbitHeader", "Orbit Target", accent);

						auto* activeCamera = getActiveCamera();
						ICamera::SphericalTargetState orbitState;
						const bool hasOrbitTarget = activeCamera && activeCamera->tryGetSphericalTargetState(orbitState);
						if (hasOrbitTarget)
						{
							auto target = getCastedVector<float32_t>(orbitState.target);
							if (ImGui::InputFloat3("Target", &target[0]))
								activeCamera->trySetSphericalTarget(getCastedVector<float64_t>(target));

							if (ImGui::Button("Target model"))
							{
								const auto targetPos = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
								activeCamera->trySetSphericalTarget(float64_t3(targetPos.x, targetPos.y, targetPos.z));
							}
							DrawHoverHint("Set orbit target to the model position");
							ImGui::SameLine();
							if (ImGui::Button("Target origin"))
								activeCamera->trySetSphericalTarget(float64_t3(0.0));
							DrawHoverHint("Set orbit target to world origin");
						}
						if (!hasOrbitTarget)
						{
							ImGui::TextDisabled("Active camera is not orbit.");
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Presets"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("PresetsPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						DrawSectionHeader("PresetsHeader", "Presets", accent);
						ImGui::InputText("Preset name", m_presetName, IM_ARRAYSIZE(m_presetName));
						if (ImGui::Button("Add preset"))
						{
							auto* activeCamera = getActiveCamera();
							m_presets.emplace_back(capturePreset(activeCamera, m_presetName));
							m_selectedPresetIx = static_cast<int>(m_presets.size()) - 1;
						}
						DrawHoverHint("Store current camera as a preset");
						ImGui::SameLine();
						if (ImGui::Button("Clear presets"))
						{
							m_presets.clear();
							m_selectedPresetIx = -1;
						}
						DrawHoverHint("Remove all presets");

						if (!m_presets.empty())
						{
							auto* activeCamera = getActiveCamera();
							const char* presetFilterLabels[] = { "All", "Exact", "Best-effort" };
							int presetFilterIx = static_cast<int>(m_presetFilterMode);
							if (ImGui::Combo("Visibility", &presetFilterIx, presetFilterLabels, IM_ARRAYSIZE(presetFilterLabels)))
								m_presetFilterMode = static_cast<PresetFilterMode>(presetFilterIx);
							DrawHoverHint("Filter presets for the active camera using exact or best-effort compatibility");

							std::vector<int> filteredPresetIndices;
							filteredPresetIndices.reserve(m_presets.size());
							for (size_t i = 0; i < m_presets.size(); ++i)
							{
								if (presetMatchesFilter(activeCamera, m_presets[i]))
									filteredPresetIndices.push_back(static_cast<int>(i));
							}

							if (filteredPresetIndices.empty())
							{
								ImGui::TextDisabled("No presets match the current filter.");
							}
							else
							{
								if (m_selectedPresetIx < 0 ||
									std::find(filteredPresetIndices.begin(), filteredPresetIndices.end(), m_selectedPresetIx) == filteredPresetIndices.end())
								{
									m_selectedPresetIx = filteredPresetIndices.front();
								}

								int selectedFilteredPresetIx = 0;
								for (int i = 0; i < static_cast<int>(filteredPresetIndices.size()); ++i)
								{
									if (filteredPresetIndices[i] == m_selectedPresetIx)
									{
										selectedFilteredPresetIx = i;
										break;
									}
								}

							std::vector<const char*> names;
								names.reserve(filteredPresetIndices.size());
								for (const auto presetIx : filteredPresetIndices)
									names.push_back(m_presets[static_cast<size_t>(presetIx)].name.c_str());

								if (ImGui::ListBox("Preset list", &selectedFilteredPresetIx, names.data(), static_cast<int>(names.size()), 6))
									m_selectedPresetIx = filteredPresetIndices[static_cast<size_t>(selectedFilteredPresetIx)];

								if (m_selectedPresetIx >= 0 && static_cast<size_t>(m_selectedPresetIx) < m_presets.size())
								{
									const auto& preset = m_presets[static_cast<size_t>(m_selectedPresetIx)];
									const auto presetUi = analyzePresetForUi(activeCamera, preset);
									const ImVec4 compatibilityColor = !presetUi.hasActiveCamera ? bad : (presetUi.exact() ? good : warn);

									ImGui::TextDisabled("Preset source");
									ImGui::SameLine();
									ImGui::TextColored(muted, "%s", getCameraTypeLabel(presetUi.goal.sourceKind).data());
									ImGui::TextDisabled("Goal state");
									ImGui::SameLine();
									ImGui::TextColored(muted, "%s", describeGoalStateMask(presetUi.goal.sourceGoalStateMask).c_str());
									ImGui::TextDisabled("Policy");
									ImGui::SameLine();
									ImGui::TextColored(presetUi.canApply ? compatibilityColor : bad, "%s", presetUi.policyLabel.c_str());
									ImGui::TextDisabled("Compatibility");
									ImGui::SameLine();
									ImGui::TextColored(compatibilityColor, "%s", presetUi.compatibilityLabel.c_str());

									DrawBadge(presetUi.exact() ? "EXACT" : "BEST-EFFORT", presetUi.exact() ? good : warn, badgeText);
									if (presetUi.dropsGoalState())
									{
										ImGui::SameLine();
										DrawBadge("DROPS STATE", warn, badgeText);
									}
									else if (presetUi.usesSharedStateOnly())
									{
										ImGui::SameLine();
										DrawBadge("SHARED STATE", accent, badgeText);
									}
									if (!presetUi.canApply)
									{
										ImGui::SameLine();
										DrawBadge("BLOCKED", bad, badgeText);
									}

									if (!presetUi.canApply)
										ImGui::BeginDisabled();
									if (ImGui::Button("Apply preset"))
										applyPresetFromUi(activeCamera, preset);
									if (!presetUi.canApply)
										ImGui::EndDisabled();
									DrawHoverHint(presetUi.canApply ?
										"Apply selected preset to the active camera" :
										"Apply is blocked because there is no active camera or the preset goal is invalid");
									ImGui::SameLine();
									if (ImGui::Button("Remove preset"))
									{
										m_presets.erase(m_presets.begin() + m_selectedPresetIx);
										m_selectedPresetIx = -1;
									}
									DrawHoverHint("Remove selected preset");
								}
							}
						}

						if (m_manualPresetApplyBanner.visible())
						{
							const ImVec4 resultColor = m_manualPresetApplyBanner.succeeded ? (m_manualPresetApplyBanner.approximate ? warn : good) : bad;
							ImGui::TextColored(resultColor, "%s", m_manualPresetApplyBanner.summary.c_str());
						}

						DrawSectionHeader("PresetsStorageHeader", "Storage", accent);
						ImGui::InputText("Preset file", m_presetPath, IM_ARRAYSIZE(m_presetPath));
						if (ImGui::Button("Save presets"))
						{
							if (!savePresetsToFile(system::path(m_presetPath)))
								m_logger->log("Failed to save presets to \"%s\".", ILogger::ELL_ERROR, m_presetPath);
						}
						DrawHoverHint("Save presets to JSON file");
						ImGui::SameLine();
						if (ImGui::Button("Load presets"))
						{
							if (!loadPresetsFromFile(system::path(m_presetPath)))
								m_logger->log("Failed to load presets from \"%s\".", ILogger::ELL_ERROR, m_presetPath);
						}
						DrawHoverHint("Load presets from JSON file");
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Playback"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("PlaybackPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						DrawSectionHeader("PlaybackHeader", "Playback", accent);
						ImGui::Checkbox("Loop", &m_playback.loop);
						DrawHoverHint("Loop playback when it reaches the end");
						ImGui::Checkbox("Override input", &m_playback.overrideInput);
						DrawHoverHint("Ignore manual input during playback");
						ImGui::Checkbox("Affect all cameras", &m_playbackAffectsAll);
						DrawHoverHint("Apply playback to all cameras");
						ImGui::SliderFloat("Speed", &m_playback.speed, 0.1f, 4.f, "%.2f");
						DrawHoverHint("Playback speed multiplier");

						if (ImGui::Button(m_playback.playing ? "Pause" : "Play"))
							m_playback.playing = !m_playback.playing;
						DrawHoverHint("Start or pause playback");
						ImGui::SameLine();
						if (ImGui::Button("Stop"))
						{
							m_playback.playing = false;
							m_playback.time = 0.f;
							applyPlaybackAtTime(m_playback.time);
						}
						DrawHoverHint("Stop playback and reset time");

						if (!m_keyframes.empty())
						{
							const float duration = m_keyframes.back().time;
							if (ImGui::SliderFloat("Time", &m_playback.time, 0.f, duration, "%.3f"))
								applyPlaybackAtTime(m_playback.time);
						}
						if (m_playbackApplyBanner.visible())
						{
							const ImVec4 playbackColor = m_playbackApplyBanner.succeeded ? (m_playbackApplyBanner.approximate ? warn : good) : bad;
							ImGui::TextColored(playbackColor, "%s", m_playbackApplyBanner.summary.c_str());
						}

						DrawSectionHeader("KeyframesHeader", "Keyframes", accent);
						ImGui::InputFloat("New keyframe time", &m_newKeyframeTime, 0.1f, 1.f, "%.3f");
						DrawHoverHint("Time value for new keyframe");
						if (ImGui::Button("Add keyframe"))
						{
							auto* activeCamera = getActiveCamera();
							CameraKeyframe keyframe;
							keyframe.time = m_newKeyframeTime;
							keyframe.preset = capturePreset(activeCamera, "Keyframe");
							m_keyframes.emplace_back(std::move(keyframe));
							std::sort(m_keyframes.begin(), m_keyframes.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
						}
						DrawHoverHint("Add keyframe from current camera");
						ImGui::SameLine();
						if (ImGui::Button("Clear keyframes"))
						{
							m_keyframes.clear();
							clearApplyStatusBanner(m_playbackApplyBanner);
						}
						DrawHoverHint("Remove all keyframes");

						if (!m_keyframes.empty())
						{
							if (ImGui::BeginChild("KeyframeList", ImVec2(0, 120), true))
							{
								for (size_t i = 0; i < m_keyframes.size(); ++i)
								{
									ImGui::Text("[%zu] t=%.3f", i, m_keyframes[i].time);
								}
							}
							ImGui::EndChild();
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Gizmo"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("GizmoPanel", ImVec2(0, 0), true))
					{
						DrawSectionHeader("GizmoHeader", "Gizmo", accent);
						TransformEditorContents();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (m_showEventLog && ImGui::BeginTabItem("Log"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("LogPanel", ImVec2(0, 0), true))
					{
						DrawSectionHeader("LogHeader", "Virtual Events", accent);
						ImGui::Checkbox("Auto-scroll", &m_logAutoScroll);
						ImGui::SameLine();
						ImGui::Checkbox("Wrap", &m_logWrap);
						ImGui::Separator();

						ImGuiWindowFlags logFlags = m_logWrap ? ImGuiWindowFlags_None : ImGuiWindowFlags_HorizontalScrollbar;
						if (ImGui::BeginChild("LogList", ImVec2(0, 0), false, logFlags))
						{
							const float scrollY = ImGui::GetScrollY();
							const float scrollMax = ImGui::GetScrollMaxY();
							const bool wasAtBottom = scrollY >= scrollMax - 5.0f;
							const size_t start = m_virtualEventLog.size() > 200 ? m_virtualEventLog.size() - 200 : 0;
							if (m_logWrap)
								ImGui::PushTextWrapPos(0.0f);
							for (size_t i = start; i < m_virtualEventLog.size(); ++i)
							{
								const auto& entry = m_virtualEventLog[i];
								ImGui::TextUnformatted(entry.line.c_str());
							}
							if (m_logWrap)
								ImGui::PopTextWrapPos();
							if (m_logAutoScroll && wasAtBottom && !m_virtualEventLog.empty())
								ImGui::SetScrollHereY(1.0f);
						}
						ImGui::EndChild();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				ImGui::EndTabBar();
			}

			ImGui::End();
			ImGui::PopStyleColor(19);
			ImGui::PopStyleVar(9);

}


