#include "app/App.hpp"
#include "camera/CCameraPersistence.hpp"

bool App::savePresetsToFile(const nbl::system::path& path)
{
	return nbl::system::savePresetCollectionToFile(path, std::span<const CameraPreset>(m_presets.data(), m_presets.size()));
}

bool App::loadPresetsFromFile(const nbl::system::path& path)
{
	return nbl::system::loadPresetCollectionFromFile(path, m_presets);
}

bool App::saveKeyframesToFile(const nbl::system::path& path)
{
	return nbl::system::saveKeyframeTrackToFile(path, m_keyframeTrack);
}

bool App::loadKeyframesFromFile(const nbl::system::path& path)
{
	if (!nbl::system::loadKeyframeTrackFromFile(path, m_keyframeTrack))
		return false;

	clampPlaybackTimeToKeyframes();
	if (m_keyframeTrack.keyframes.empty())
		clearApplyStatusBanner(m_playbackApplyBanner);
	return true;
}

void App::DrawControlPanel()
{
			const nbl::ui::SCameraControlPanelStyle panelStyle = {};
			const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
			const ImVec2 panelSize = nbl::ui::calcControlPanelWindowSize(displaySize, panelStyle);
			const ImVec2 panelPos = { 0.0f, 0.0f };
			ImGui::SetNextWindowPos(panelPos, ImGuiCond_Always);
			ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);

			nbl::ui::pushControlPanelWindowStyle(panelStyle);

			ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);
			ImGui::SetNextWindowBgAlpha(0.0f);
			if (m_ciMode)
				ImGui::SetNextWindowFocus();
			ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

			const ImVec4 accent = panelStyle.AccentColor;
			const ImVec4 good = panelStyle.GoodColor;
			const ImVec4 bad = panelStyle.BadColor;
			const ImVec4 warn = panelStyle.WarnColor;
			const ImVec4 muted = panelStyle.MutedColor;
			const ImVec4 badgeText = panelStyle.BadgeTextColor;
			const ImVec4 keyBg = panelStyle.KeyBackgroundColor;
			const ImVec4 keyFg = panelStyle.KeyTextColor;
			const ImGuiTableFlags tableFlags = panelStyle.SummaryTableFlags;
			const ImVec4 inactiveBadge = panelStyle.InactiveBadgeColor;
			const ImVec4 cardTop = panelStyle.CardTopColor;
			const ImVec4 cardBottom = panelStyle.CardBottomColor;
			const ImVec4 cardBorder = panelStyle.CardBorderColor;

			{
				const ImVec2 panelPos = ImGui::GetWindowPos();
				const ImVec2 panelSize = ImGui::GetWindowSize();
				if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
					nbl::ui::drawControlPanelWindowBackdrop(*drawList, panelPos, panelSize, panelStyle);
			}

			auto row = [&](const char* label, auto&& drawValue)
			{
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::TextUnformatted(label);
				ImGui::TableSetColumnIndex(1);
				drawValue();
			};

			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, panelStyle.CardChildRounding);
			if (ImGui::BeginChild("PanelHeader", ImVec2(0.0f, panelStyle.HeaderWindowHeight), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
			{
				ImGui::Dummy(ImVec2(0.0f, panelStyle.HeaderDummyY));
				ImGui::SetWindowFontScale(panelStyle.HeaderTitleFontScale);
				ImGui::TextColored(accent, "Control Panel");
				ImGui::SetWindowFontScale(1.0f);
				{
					const float gap = ImGui::GetStyle().ItemSpacing.x;
					std::array<nbl::ui::SCameraControlPanelBadgeData, 4> headerBadges = {{
						{ useWindow ? "WINDOW" : "FULL", accent },
						{ enableActiveCameraMovement ? "MOVE ON" : "MOVE OFF", enableActiveCameraMovement ? good : bad },
						{ m_scriptedInput.enabled ? (m_scriptedInput.exclusive ? "SCRIPT EXCL" : "SCRIPT") : "SCRIPT OFF", m_scriptedInput.enabled ? accent : inactiveBadge },
						{ "CI", warn }
					}};
					const size_t headerBadgeCount = m_ciMode ? headerBadges.size() : headerBadges.size() - 1u;
					nbl::ui::drawBadgeRow(std::span<const nbl::ui::SCameraControlPanelBadgeData>(headerBadges.data(), headerBadgeCount), badgeText, gap, panelStyle);
				}

				ImGui::Dummy(ImVec2(0.0f, panelStyle.HeaderGapSmall));
				{
					const float gap = ImGui::GetStyle().ItemSpacing.x;
					const float groupGap = gap * 2.0f;
					static constexpr std::array<const char*, 4u> MoveKeys = { "W", "A", "S", "D" };
					static constexpr std::array<const char*, 1u> LookKeys = { "RMB" };
					static constexpr std::array<const char*, 1u> ZoomKeys = { "MW" };
					const std::array<nbl::ui::SCameraControlPanelKeyHintGroup, 3u> keyHintGroups = {{
						{ "Move", MoveKeys },
						{ "Look", LookKeys },
						{ "Zoom", ZoomKeys }
					}};
					nbl::ui::drawKeyHintGroupRow(keyHintGroups, gap, groupGap, keyBg, keyFg, panelStyle);
				}

				ImGui::Dummy(ImVec2(0.0f, panelStyle.HeaderGapSmall));
				if (ImGui::BeginTable("HeaderMetrics", 3, ImGuiTableFlags_SizingStretchProp))
				{
					const float frameMs = std::max(0.0f, m_uiLastFrameMs);
					const float fps = frameMs > 0.0f ? (1000.0f / frameMs) : 0.0f;
					const std::array<nbl::ui::SCameraControlPanelMiniStatSpec, 3u> miniStats = {{
						{ "FrameStat", "Frame", accent, panelStyle.DefaultFrameMetricMin },
						{ "InputStat", "Input", accent, panelStyle.DefaultEventMetricMin },
						{ "VirtualStat", "Virtual", accent, panelStyle.DefaultEventMetricMin }
					}};

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					nbl::ui::drawMiniStat(miniStats[0], m_uiFrameMs, m_uiMetricIndex, [&]
					{
						ImGui::TextColored(accent, "%.1f ms  %.0f fps", frameMs, fps);
					}, panelStyle);

					ImGui::TableSetColumnIndex(1);
					nbl::ui::drawMiniStat(miniStats[1], m_uiInputCounts, m_uiMetricIndex, [&]
					{
						ImGui::TextColored(accent, "%u ev", m_uiLastInputEvents);
					}, panelStyle);

					ImGui::TableSetColumnIndex(2);
					nbl::ui::drawMiniStat(miniStats[2], m_uiVirtualCounts, m_uiMetricIndex, [&]
					{
						ImGui::TextColored(accent, "%u ev", m_uiLastVirtualEvents);
					}, panelStyle);
					ImGui::EndTable();
				}
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();

			ImGui::Spacing();

			{
				const ImVec2 togglePad = ImVec2(6.0f, 2.0f);
				const float gap = ImGui::GetStyle().ItemSpacing.x;
				const std::array<const char*, 3u> toggleLabels = { "WINDOW", "STATUS", "EVENT LOG" };
				float rowWidth = 0.0f;
				for (size_t i = 0; i < toggleLabels.size(); ++i)
				{
					if (i > 0u)
						rowWidth += gap;
					rowWidth += nbl::ui::calcPillWidth(toggleLabels[i], togglePad);
				}
				nbl::ui::centerControlPanelRow(rowWidth);
				nbl::ui::drawTogglePill("WINDOW", useWindow, accent, inactiveBadge, badgeText, togglePad);
				nbl::ui::drawHoverHint("Toggle split render windows");
				ImGui::SameLine(0.0f, gap);
				nbl::ui::drawTogglePill("STATUS", m_showHud, accent, inactiveBadge, badgeText, togglePad);
				nbl::ui::drawHoverHint("Show system and camera status panel");
				ImGui::SameLine(0.0f, gap);
				nbl::ui::drawTogglePill("EVENT LOG", m_showEventLog, accent, inactiveBadge, badgeText, togglePad);
				nbl::ui::drawHoverHint("Show virtual event log");
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
						nbl::ui::drawSectionHeader("SessionHeader", "Session", accent, panelStyle);
						if (nbl::ui::beginCard("SessionCard", nbl::ui::calcCameraControlPanelCardHeight(3, panelStyle), cardTop, cardBottom, cardBorder, panelStyle))
						{
							if (ImGui::BeginTable("SessionTable", 2, tableFlags))
							{
								ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
								ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
								row("Mode", [&] { nbl::ui::drawDot(accent, panelStyle); ImGui::TextColored(accent, "%s", useWindow ? "Window" : "Fullscreen"); });
								row("Active window", [&] { nbl::ui::drawDot(accent, panelStyle); ImGui::TextColored(accent, "%u", activeRenderWindowIx); });
								row("Movement", [&] { const ImVec4 c = enableActiveCameraMovement ? good : bad; nbl::ui::drawDot(c, panelStyle); ImGui::TextColored(c, "%s", enableActiveCameraMovement ? "Enabled" : "Disabled"); });
								ImGui::EndTable();
							}
						}
						nbl::ui::endCard();

						nbl::ui::drawSectionHeader("CameraHeader", "Camera", accent, panelStyle);

						auto* activeCamera = getActiveCamera();
						if (activeCamera)
						{
							const auto& gimbal = activeCamera->getGimbal();
							const auto pos = gimbal.getPosition();
							const auto euler = getQuaternionEulerDegrees(gimbal.getOrientation());

							if (nbl::ui::beginCard("CameraCard", nbl::ui::calcCameraControlPanelCardHeight(5, panelStyle), cardTop, cardBottom, cardBorder, panelStyle))
							{
								if (ImGui::BeginTable("CameraTable", 2, tableFlags))
								{
									ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
									ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
									row("Name", [&] { nbl::ui::drawDot(accent, panelStyle); ImGui::TextColored(muted, "%s", activeCamera->getIdentifier().data()); });
									row("Position", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.2f %.2f %.2f", pos.x, pos.y, pos.z); });
									row("Euler", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.1f %.1f %.1f", euler.x, euler.y, euler.z); });
									row("Move scale", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.4f", activeCamera->getMoveSpeedScale()); });
									row("Rotate scale", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.4f", activeCamera->getRotationSpeedScale()); });
									ImGui::EndTable();
								}
							}
							nbl::ui::endCard();
						}
						else
						{
							if (nbl::ui::beginCard("CameraCard", nbl::ui::calcCameraControlPanelCardHeight(2, panelStyle), cardTop, cardBottom, cardBorder, panelStyle))
								ImGui::TextDisabled("No active camera");
							nbl::ui::endCard();
						}

						nbl::ui::drawSectionHeader("ProjectionHeader", "Projection", accent, panelStyle);

						auto& binding = windowBindings[activeRenderWindowIx];
						auto& planar = m_planarProjections[binding.activePlanarIx];
						if (planar && binding.boundProjectionIx.has_value())
						{
							auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];
							const auto& params = projection.getParameters();
							if (nbl::ui::beginCard("ProjectionCard", nbl::ui::calcCameraControlPanelCardHeight(4, panelStyle), cardTop, cardBottom, cardBorder, panelStyle))
							{
								if (ImGui::BeginTable("ProjectionTable", 2, tableFlags))
								{
									ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
									ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
									row("Type", [&] { nbl::ui::drawDot(accent, panelStyle); ImGui::TextColored(muted, "%s", params.m_type == IPlanarProjection::CProjection::Perspective ? "Perspective" : "Orthographic"); });
									row("zNear", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.2f", params.m_zNear); });
									row("zFar", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.2f", params.m_zFar); });
									if (params.m_type == IPlanarProjection::CProjection::Perspective)
										row("Fov", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.1f", params.m_planar.perspective.fov); });
									else
										row("Ortho width", [&] { nbl::ui::drawDot(muted, panelStyle); ImGui::TextColored(muted, "%.1f", params.m_planar.orthographic.orthoWidth); });
									ImGui::EndTable();
								}
							}
							nbl::ui::endCard();
						}
						else
						{
							if (nbl::ui::beginCard("ProjectionCard", nbl::ui::calcCameraControlPanelCardHeight(2, panelStyle), cardTop, cardBottom, cardBorder, panelStyle))
								ImGui::TextDisabled("No projection bound");
							nbl::ui::endCard();
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

						nbl::ui::drawSectionHeader("PlanarSelectHeader", "Planar Selection", accent, panelStyle);
						ImGui::Text("Active Render Window: %s", activeRenderWindowIxString.c_str());
						nbl::ui::drawHoverHint("Window that receives input and camera switching");
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
							nbl::ui::drawHoverHint("Select which camera the window renders");
						}

						assert(active.boundProjectionIx.has_value());
						assert(active.lastBoundPerspectivePresetProjectionIx.has_value());
						assert(active.lastBoundOrthoPresetProjectionIx.has_value());

						const auto activePlanarIxString = std::to_string(active.activePlanarIx);
						auto& planarBound = m_planarProjections[active.activePlanarIx];
						assert(planarBound);

						nbl::ui::drawSectionHeader("ProjectionParamsHeader", "Projection Parameters", accent, panelStyle);

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
							nbl::ui::drawHoverHint("Switch projection type for this planar");
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
						nbl::ui::drawHoverHint("Switch preset projection for this planar");

						auto* const boundCamera = planarBound->getCamera();
						auto& boundProjection = planarBound->getPlanarProjections()[active.boundProjectionIx.value()];
						assert(not boundProjection.isProjectionSingular());

						auto updateParameters = boundProjection.getParameters();

						if (useWindow)
							ImGui::Checkbox("Allow axes to flip##allowAxesToFlip", &active.allowGizmoAxesToFlip);
						nbl::ui::drawHoverHint("Allow ImGuizmo axes to flip based on view");

						if(useWindow)
							ImGui::Checkbox("Draw debug grid##drawDebugGrid", &active.enableDebugGridDraw);
						nbl::ui::drawHoverHint("Toggle debug grid in the render window");

						if (ImGui::RadioButton("LH", active.leftHandedProjection))
							active.leftHandedProjection = true;

						ImGui::SameLine();

						if (ImGui::RadioButton("RH", not active.leftHandedProjection))
							active.leftHandedProjection = false;
						nbl::ui::drawHoverHint("Toggle left or right handed projection");

						updateParameters.m_zNear = std::clamp(updateParameters.m_zNear, 0.1f, 100.f);
						updateParameters.m_zFar = std::clamp(updateParameters.m_zFar, 110.f, 10000.f);

						ImGui::SliderFloat("zNear", &updateParameters.m_zNear, 0.1f, 100.f, "%.2f", ImGuiSliderFlags_Logarithmic);
						nbl::ui::drawHoverHint("Near clip plane");
						ImGui::SliderFloat("zFar", &updateParameters.m_zFar, 110.f, 10000.f, "%.1f", ImGuiSliderFlags_Logarithmic);
						nbl::ui::drawHoverHint("Far clip plane");

						switch (selectedProjectionType)
						{
							case IPlanarProjection::CProjection::Perspective:
							{
								ImGui::SliderFloat("Fov", &updateParameters.m_planar.perspective.fov, 20.f, 150.f, "%.1f", ImGuiSliderFlags_Logarithmic);
								nbl::ui::drawHoverHint("Perspective field of view");
								boundProjection.setPerspective(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.perspective.fov);
							} break;

							case IPlanarProjection::CProjection::Orthographic:
							{
								ImGui::SliderFloat("Ortho width", &updateParameters.m_planar.orthographic.orthoWidth, 1.f, 30.f, "%.1f", ImGuiSliderFlags_Logarithmic);
								nbl::ui::drawHoverHint("Orthographic width");
								boundProjection.setOrthographic(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.orthographic.orthoWidth);
							} break;

							default: break;
						}

						nbl::ui::drawSectionHeader("CursorHeader", "Cursor Behaviour", accent, panelStyle);
						if (ImGui::TreeNodeEx("Cursor Behaviour"))
						{
							ImGui::Checkbox("Capture OS cursor in move mode", &captureCursorInMoveMode);
							nbl::ui::drawHoverHint("When disabled the app never warps or clamps system cursor");
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

						nbl::ui::drawSectionHeader("BoundCameraHeader", "Bound Camera", accent, panelStyle);
						const auto flags = ImGuiTreeNodeFlags_DefaultOpen;
						if (ImGui::TreeNodeEx("Bound Camera", flags))
						{
							ImGui::Text("Type: %s", boundCamera->getIdentifier().data());
							ImGui::Text("Object Ix: %s", std::to_string(active.activePlanarIx + 2u).c_str());
							ImGui::Separator();
							{
								ICamera::SphericalTargetState sphericalState;
								const bool isOrbitLike = boundCamera->tryGetSphericalTargetState(sphericalState);

								float moveSpeed = boundCamera->getMoveSpeedScale();
								float rotationSpeed = boundCamera->getRotationSpeedScale();

								ImGui::SliderFloat("Move speed factor", &moveSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);
								nbl::ui::drawHoverHint("Scale translation speed for this camera");

								if (boundCamera->getAllowedVirtualEvents() & CVirtualGimbalEvent::Rotate)
									ImGui::SliderFloat("Rotate speed factor", &rotationSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);
								nbl::ui::drawHoverHint("Scale rotation speed for this camera");

								boundCamera->setMotionScales(moveSpeed, rotationSpeed);

								if (isOrbitLike)
								{
									float distance = sphericalState.distance;
									ImGui::SliderFloat("Distance", &distance, sphericalState.minDistance, sphericalState.maxDistance, "%.4f", ImGuiSliderFlags_Logarithmic);
									nbl::ui::drawHoverHint("Current orbit distance");
									boundCamera->trySetSphericalDistance(distance);
								}
							}

							if (ImGui::TreeNodeEx("World Data", flags))
							{
								auto& gimbal = boundCamera->getGimbal();
								const auto position = getCastedVector<float32_t>(gimbal.getPosition());
								const auto orientation = getCastedVector<float32_t>(gimbal.getOrientation().data);
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
						nbl::ui::drawSectionHeader("CameraInputHeader", "Input", accent, panelStyle);
						ImGui::Checkbox("Mirror input to all cameras", &m_cameraControls.mirrorInput);
						nbl::ui::drawHoverHint("Apply keyboard and mouse input to every camera");
						ImGui::Checkbox("World translate", &m_cameraControls.worldTranslate);
						nbl::ui::drawHoverHint("Translate in world space instead of camera space");
						ImGui::SliderFloat("Keyboard scale", &m_cameraControls.keyboardScale, 0.01f, 10.f, "%.2f");
						nbl::ui::drawHoverHint("Scale keyboard movement magnitudes");
						ImGui::SliderFloat("Mouse move scale", &m_cameraControls.mouseMoveScale, 0.01f, 10.f, "%.2f");
						nbl::ui::drawHoverHint("Scale mouse move magnitudes");
						ImGui::SliderFloat("Mouse scroll scale", &m_cameraControls.mouseScrollScale, 0.01f, 10.f, "%.2f");
						nbl::ui::drawHoverHint("Scale mouse wheel magnitudes");
						ImGui::SliderFloat("Translate scale", &m_cameraControls.translationScale, 0.01f, 10.f, "%.2f");
						nbl::ui::drawHoverHint("Overall translation scale for virtual events");
						ImGui::SliderFloat("Rotate scale", &m_cameraControls.rotationScale, 0.01f, 10.f, "%.2f");
						nbl::ui::drawHoverHint("Overall rotation scale for virtual events");

						nbl::ui::drawSectionHeader("CameraConstraintsHeader", "Constraints", accent, panelStyle);
						ImGui::Checkbox("Enable constraints", &m_cameraConstraints.enabled);
						nbl::ui::drawHoverHint("Enable or disable all camera constraints");
						ImGui::Checkbox("Clamp distance", &m_cameraConstraints.clampDistance);
						nbl::ui::drawHoverHint("Clamp orbit distance to min/max");
						ImGui::SliderFloat("Min distance", &m_cameraConstraints.minDistance, 0.01f, 1000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
						nbl::ui::drawHoverHint("Minimum orbit distance");
						ImGui::SliderFloat("Max distance", &m_cameraConstraints.maxDistance, 0.01f, 10000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
						nbl::ui::drawHoverHint("Maximum orbit distance");
						ImGui::Separator();
						ImGui::Checkbox("Clamp pitch", &m_cameraConstraints.clampPitch);
						nbl::ui::drawHoverHint("Clamp pitch angle");
						ImGui::SliderFloat("Pitch min", &m_cameraConstraints.pitchMinDeg, -180.f, 180.f, "%.1f");
						nbl::ui::drawHoverHint("Minimum pitch in degrees");
						ImGui::SliderFloat("Pitch max", &m_cameraConstraints.pitchMaxDeg, -180.f, 180.f, "%.1f");
						nbl::ui::drawHoverHint("Maximum pitch in degrees");
						ImGui::Checkbox("Clamp yaw", &m_cameraConstraints.clampYaw);
						nbl::ui::drawHoverHint("Clamp yaw angle");
						ImGui::SliderFloat("Yaw min", &m_cameraConstraints.yawMinDeg, -180.f, 180.f, "%.1f");
						nbl::ui::drawHoverHint("Minimum yaw in degrees");
						ImGui::SliderFloat("Yaw max", &m_cameraConstraints.yawMaxDeg, -180.f, 180.f, "%.1f");
						nbl::ui::drawHoverHint("Maximum yaw in degrees");
						ImGui::Checkbox("Clamp roll", &m_cameraConstraints.clampRoll);
						nbl::ui::drawHoverHint("Clamp roll angle");
						ImGui::SliderFloat("Roll min", &m_cameraConstraints.rollMinDeg, -180.f, 180.f, "%.1f");
						nbl::ui::drawHoverHint("Minimum roll in degrees");
						ImGui::SliderFloat("Roll max", &m_cameraConstraints.rollMaxDeg, -180.f, 180.f, "%.1f");
						nbl::ui::drawHoverHint("Maximum roll in degrees");

						nbl::ui::drawSectionHeader("OrbitHeader", "Orbit Target", accent, panelStyle);

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
							nbl::ui::drawHoverHint("Set orbit target to the model position");
							ImGui::SameLine();
							if (ImGui::Button("Target origin"))
								activeCamera->trySetSphericalTarget(float64_t3(0.0));
							nbl::ui::drawHoverHint("Set orbit target to world origin");
						}
						if (!hasOrbitTarget)
						{
							ImGui::TextDisabled("Active camera is not orbit.");
						}

						nbl::ui::drawSectionHeader("FollowHeader", "Follow Target", accent, panelStyle);
						auto* activeFollowConfig = getActiveFollowConfig();
						if (activeFollowConfig)
						{
							auto& followConfig = *activeFollowConfig;
							const bool prevFollowEnabled = followConfig.enabled;
							const auto prevFollowMode = followConfig.mode;
							ImGui::Checkbox("Enable follow", &followConfig.enabled);
							nbl::ui::drawHoverHint("Apply tracked-target follow to the active planar camera");

							const char* followModeLabels[] = {
								getCameraFollowModeLabel(ECameraFollowMode::Disabled),
								getCameraFollowModeLabel(ECameraFollowMode::OrbitTarget),
								getCameraFollowModeLabel(ECameraFollowMode::LookAtTarget),
								getCameraFollowModeLabel(ECameraFollowMode::KeepWorldOffset),
								getCameraFollowModeLabel(ECameraFollowMode::KeepLocalOffset)
							};
							int followModeIx = static_cast<int>(followConfig.mode);
							if (ImGui::Combo("Mode", &followModeIx, followModeLabels, IM_ARRAYSIZE(followModeLabels)))
								followConfig.mode = static_cast<ECameraFollowMode>(followModeIx);
							const bool followStateChanged = followConfig.enabled != prevFollowEnabled || followConfig.mode != prevFollowMode;
							if (followStateChanged && followConfig.enabled && nbl::core::cameraFollowModeUsesCapturedOffset(followConfig.mode))
								captureFollowOffsetsForPlanar(getActivePlanarIx());
							if (followStateChanged && followConfig.enabled)
								applyFollowToConfiguredCameras();

							auto trackedTarget = getCastedVector<float32_t>(m_followTarget.getGimbal().getPosition());
							if (ImGui::InputFloat3("Tracked target", &trackedTarget[0]))
								m_followTarget.setPosition(getCastedVector<float64_t>(trackedTarget));

							ImGui::Checkbox("Show target marker", &m_followTargetVisible);
							nbl::ui::drawHoverHint("Render the tracked target marker in the scene");

							if (ImGui::Button("Reset target"))
								resetFollowTargetToDefault();
							nbl::ui::drawHoverHint("Reset tracked target gimbal to the default world-space follow pose");
							ImGui::SameLine();
							if (ImGui::Button("Snap to model"))
								snapFollowTargetToModel();
							nbl::ui::drawHoverHint("Optionally snap tracked target gimbal to the model transform");
							ImGui::SameLine();
							if (ImGui::Button("Target origin"))
								m_followTarget.setPose(float64_t3(0.0), makeIdentityQuaternion<float64_t>());
							nbl::ui::drawHoverHint("Reset tracked target to identity at world origin");
							ImGui::SameLine();
							if (ImGui::Button("Capture current offset"))
								captureFollowOffsetsForPlanar(getActivePlanarIx());
							nbl::ui::drawHoverHint("Store current camera-to-target relation into the active follow config");

							if (cameraFollowModeUsesWorldOffset(followConfig.mode))
							{
								auto worldOffset = getCastedVector<float32_t>(followConfig.worldOffset);
								if (ImGui::InputFloat3("World offset", &worldOffset[0]))
									followConfig.worldOffset = getCastedVector<float64_t>(worldOffset);
							}
							if (cameraFollowModeUsesLocalOffset(followConfig.mode))
							{
								auto localOffset = getCastedVector<float32_t>(followConfig.localOffset);
								if (ImGui::InputFloat3("Local offset", &localOffset[0]))
									followConfig.localOffset = getCastedVector<float64_t>(localOffset);
							}
						}
						else
						{
							ImGui::TextDisabled("No active follow config.");
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
						nbl::ui::drawSectionHeader("PresetsHeader", "Presets", accent, panelStyle);
						ImGui::InputText("Preset name", m_presetName, IM_ARRAYSIZE(m_presetName));
						auto* activeCamera = getActiveCamera();
						const auto presetCaptureUi = analyzeCameraCaptureForUi(activeCamera);
						if (!presetCaptureUi.canCapture)
							ImGui::BeginDisabled();
						if (ImGui::Button("Add preset"))
						{
							CameraPreset preset;
							if (nbl::core::tryCapturePreset(m_cameraGoalSolver, activeCamera, m_presetName, preset))
							{
								m_presets.emplace_back(std::move(preset));
								m_selectedPresetIx = static_cast<int>(m_presets.size()) - 1;
							}
						}
						if (!presetCaptureUi.canCapture)
							ImGui::EndDisabled();
						nbl::ui::drawHoverHint(presetCaptureUi.canCapture ?
							"Store current camera as a preset" :
							"Preset capture is blocked because there is no active camera or the current goal state is invalid");
						ImGui::SameLine();
						if (ImGui::Button("Clear presets"))
						{
							m_presets.clear();
							m_selectedPresetIx = -1;
						}
						nbl::ui::drawHoverHint("Remove all presets");
						ImGui::TextDisabled("Capture");
						ImGui::SameLine();
						ImGui::TextColored(presetCaptureUi.canCapture ? good : bad, "%s", presetCaptureUi.policyLabel.c_str());

						if (!m_presets.empty())
						{
							const char* presetFilterLabels[] = {
								nbl::ui::getPresetApplyPresentationFilterLabel(PresetFilterMode::All),
								nbl::ui::getPresetApplyPresentationFilterLabel(PresetFilterMode::Exact),
								nbl::ui::getPresetApplyPresentationFilterLabel(PresetFilterMode::BestEffort)
							};
							int presetFilterIx = static_cast<int>(m_presetFilterMode);
							if (ImGui::Combo("Visibility", &presetFilterIx, presetFilterLabels, IM_ARRAYSIZE(presetFilterLabels)))
								m_presetFilterMode = static_cast<PresetFilterMode>(presetFilterIx);
							nbl::ui::drawHoverHint("Filter presets for the active camera using exact or best-effort compatibility");

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
									const ImVec4 compatibilityColor = !presetUi.hasCamera ? bad : (presetUi.exact() ? good : warn);

									ImGui::TextDisabled("Preset source");
									ImGui::SameLine();
									ImGui::TextColored(muted, "%s", presetUi.sourceKindLabel.c_str());
									ImGui::TextDisabled("Goal state");
									ImGui::SameLine();
									ImGui::TextColored(muted, "%s", presetUi.goalStateLabel.c_str());
									ImGui::TextDisabled("Policy");
									ImGui::SameLine();
									ImGui::TextColored(presetUi.canApply ? compatibilityColor : bad, "%s", presetUi.policyLabel.c_str());
									ImGui::TextDisabled("Compatibility");
									ImGui::SameLine();
									ImGui::TextColored(compatibilityColor, "%s", presetUi.compatibilityLabel.c_str());

									if (presetUi.badges.exact)
										nbl::ui::drawBadge("EXACT", good, badgeText, panelStyle);
									else if (presetUi.badges.bestEffort)
										nbl::ui::drawBadge("BEST-EFFORT", warn, badgeText, panelStyle);
									if (presetUi.badges.dropsState)
									{
										ImGui::SameLine();
										nbl::ui::drawBadge("DROPS STATE", warn, badgeText, panelStyle);
									}
									else if (presetUi.badges.sharedStateOnly)
									{
										ImGui::SameLine();
										nbl::ui::drawBadge("SHARED STATE", accent, badgeText, panelStyle);
									}
									if (presetUi.badges.blocked)
									{
										ImGui::SameLine();
										nbl::ui::drawBadge("BLOCKED", bad, badgeText, panelStyle);
									}

									if (!presetUi.canApply)
										ImGui::BeginDisabled();
									if (ImGui::Button("Apply preset"))
										applyPresetFromUi(activeCamera, preset);
									if (!presetUi.canApply)
										ImGui::EndDisabled();
									nbl::ui::drawHoverHint(presetUi.canApply ?
										"Apply selected preset to the active camera" :
										"Apply is blocked because there is no active camera or the preset goal is invalid");
									ImGui::SameLine();
									if (ImGui::Button("Remove preset"))
									{
										m_presets.erase(m_presets.begin() + m_selectedPresetIx);
										m_selectedPresetIx = -1;
									}
									nbl::ui::drawHoverHint("Remove selected preset");
								}
							}
						}

						if (m_manualPresetApplyBanner.visible())
						{
							const ImVec4 resultColor = m_manualPresetApplyBanner.succeeded ? (m_manualPresetApplyBanner.approximate ? warn : good) : bad;
							ImGui::TextColored(resultColor, "%s", m_manualPresetApplyBanner.summary.c_str());
						}

						nbl::ui::drawSectionHeader("PresetsStorageHeader", "Storage", accent, panelStyle);
						ImGui::InputText("Preset file", m_presetPath, IM_ARRAYSIZE(m_presetPath));
						if (ImGui::Button("Save presets"))
						{
							if (!savePresetsToFile(nbl::system::path(m_presetPath)))
								m_logger->log("Failed to save presets to \"%s\".", ILogger::ELL_ERROR, m_presetPath);
						}
						nbl::ui::drawHoverHint("Save presets to JSON file");
						ImGui::SameLine();
						if (ImGui::Button("Load presets"))
						{
							if (!loadPresetsFromFile(nbl::system::path(m_presetPath)))
								m_logger->log("Failed to load presets from \"%s\".", ILogger::ELL_ERROR, m_presetPath);
						}
						nbl::ui::drawHoverHint("Load presets from JSON file");
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
						auto* activeCamera = getActiveCamera();
						nbl::ui::drawSectionHeader("PlaybackHeader", "Playback", accent, panelStyle);
						ImGui::Checkbox("Loop", &m_playback.loop);
						nbl::ui::drawHoverHint("Loop playback when it reaches the end");
						ImGui::Checkbox("Override input", &m_playback.overrideInput);
						nbl::ui::drawHoverHint("Ignore manual input during playback");
						ImGui::Checkbox("Affect all cameras", &m_playbackAffectsAll);
						nbl::ui::drawHoverHint("Apply playback to all cameras");
						ImGui::SliderFloat("Speed", &m_playback.speed, 0.1f, 4.f, "%.2f");
						nbl::ui::drawHoverHint("Playback speed multiplier");

						if (ImGui::Button(m_playback.playing ? "Pause" : "Play"))
							m_playback.playing = !m_playback.playing;
						nbl::ui::drawHoverHint("Start or pause playback");
						ImGui::SameLine();
						if (ImGui::Button("Stop"))
						{
							nbl::core::resetPlaybackCursor(m_playback);
							applyPlaybackAtTime(m_playback.time);
						}
						nbl::ui::drawHoverHint("Stop playback and reset time");

						if (!m_keyframeTrack.keyframes.empty())
						{
							const float duration = nbl::core::getPlaybackTrackDuration(m_keyframeTrack);
							if (ImGui::SliderFloat("Time", &m_playback.time, 0.f, duration, "%.3f"))
								applyPlaybackAtTime(m_playback.time);
						}
						if (m_playbackApplyBanner.visible())
						{
							const ImVec4 playbackColor = m_playbackApplyBanner.succeeded ? (m_playbackApplyBanner.approximate ? warn : good) : bad;
							ImGui::TextColored(playbackColor, "%s", m_playbackApplyBanner.summary.c_str());
						}
						if (!m_keyframeTrack.keyframes.empty())
						{
							CameraPreset playbackPreviewPreset;
							if (tryBuildPlaybackPresetAtTime(m_playback.time, playbackPreviewPreset))
							{
								const auto playbackPreviewUi = analyzePresetForUi(activeCamera, playbackPreviewPreset);
								const ImVec4 previewColor = !playbackPreviewUi.hasCamera ? bad : (playbackPreviewUi.exact() ? good : warn);
								ImGui::TextDisabled("Preview");
								ImGui::SameLine();
								ImGui::TextColored(playbackPreviewUi.canApply ? previewColor : bad, "%s", playbackPreviewUi.policyLabel.c_str());
							}
						}

						nbl::ui::drawSectionHeader("KeyframesHeader", "Keyframes", accent, panelStyle);
						ImGui::InputFloat("New keyframe time", &m_newKeyframeTime, 0.1f, 1.f, "%.3f");
						nbl::ui::drawHoverHint("Time value for new keyframe");
						ImGui::SameLine();
						if (ImGui::Button("Use playback time"))
							m_newKeyframeTime = m_playback.time;
						nbl::ui::drawHoverHint("Set new keyframe time from current playback position");
						const auto keyframeCaptureUi = analyzeCameraCaptureForUi(activeCamera);
						if (!keyframeCaptureUi.canCapture)
							ImGui::BeginDisabled();
						if (ImGui::Button("Add keyframe"))
						{
							CameraKeyframe keyframe;
							const float authoredTime = std::max(0.f, m_newKeyframeTime);
							keyframe.time = authoredTime;
							m_newKeyframeTime = authoredTime;
							if (nbl::core::tryCapturePreset(m_cameraGoalSolver, activeCamera, "Keyframe", keyframe.preset))
							{
								m_keyframeTrack.keyframes.emplace_back(std::move(keyframe));
								sortKeyframesByTime();
								selectKeyframeNearestTime(authoredTime);
							}
						}
						if (!keyframeCaptureUi.canCapture)
							ImGui::EndDisabled();
						nbl::ui::drawHoverHint(keyframeCaptureUi.canCapture ?
							"Add keyframe from current camera" :
							"Keyframe capture is blocked because there is no active camera or the current goal state is invalid");
						ImGui::TextDisabled("Capture");
						ImGui::SameLine();
						ImGui::TextColored(keyframeCaptureUi.canCapture ? good : bad, "%s", keyframeCaptureUi.policyLabel.c_str());
						ImGui::SameLine();
						if (ImGui::Button("Clear keyframes"))
						{
							m_keyframeTrack = {};
							nbl::core::resetPlaybackCursor(m_playback);
							clearApplyStatusBanner(m_playbackApplyBanner);
						}
						nbl::ui::drawHoverHint("Remove all keyframes");

						if (!m_keyframeTrack.keyframes.empty())
						{
							normalizeSelectedKeyframe();
							if (ImGui::BeginChild("KeyframeList", ImVec2(0, 120), true))
							{
								for (size_t i = 0; i < m_keyframeTrack.keyframes.size(); ++i)
								{
									char label[128];
									snprintf(label, sizeof(label), "[%zu] t=%.3f  %s", i, m_keyframeTrack.keyframes[i].time, m_keyframeTrack.keyframes[i].preset.name.c_str());
									if (ImGui::Selectable(label, m_keyframeTrack.selectedKeyframeIx == static_cast<int>(i)))
										m_keyframeTrack.selectedKeyframeIx = static_cast<int>(i);
								}
							}
							ImGui::EndChild();

							if (auto* selectedKeyframe = getSelectedKeyframe())
							{
								const auto keyframeUi = analyzePresetForUi(activeCamera, selectedKeyframe->preset);
								const ImVec4 compatibilityColor = !keyframeUi.hasCamera ? bad : (keyframeUi.exact() ? good : warn);
								float selectedTime = selectedKeyframe->time;
								if (ImGui::InputFloat("Selected time", &selectedTime, 0.1f, 1.f, "%.3f"))
								{
									selectedTime = std::max(0.f, selectedTime);
									selectedKeyframe->time = selectedTime;
									sortKeyframesByTime();
									selectKeyframeNearestTime(selectedTime);
									clampPlaybackTimeToKeyframes();
								}
								nbl::ui::drawHoverHint("Edit selected keyframe time");

								ImGui::TextDisabled("Keyframe source");
								ImGui::SameLine();
								ImGui::TextColored(muted, "%s", keyframeUi.sourceKindLabel.c_str());
								ImGui::TextDisabled("Goal state");
								ImGui::SameLine();
								ImGui::TextColored(muted, "%s", keyframeUi.goalStateLabel.c_str());
								ImGui::TextDisabled("Policy");
								ImGui::SameLine();
								ImGui::TextColored(keyframeUi.canApply ? compatibilityColor : bad, "%s", keyframeUi.policyLabel.c_str());
								ImGui::TextDisabled("Compatibility");
								ImGui::SameLine();
								ImGui::TextColored(compatibilityColor, "%s", keyframeUi.compatibilityLabel.c_str());

								if (keyframeUi.badges.exact)
									nbl::ui::drawBadge("EXACT", good, badgeText, panelStyle);
								else if (keyframeUi.badges.bestEffort)
									nbl::ui::drawBadge("BEST-EFFORT", warn, badgeText, panelStyle);
								if (keyframeUi.badges.dropsState)
								{
									ImGui::SameLine();
									nbl::ui::drawBadge("DROPS STATE", warn, badgeText, panelStyle);
								}
								else if (keyframeUi.badges.sharedStateOnly)
								{
									ImGui::SameLine();
									nbl::ui::drawBadge("SHARED STATE", accent, badgeText, panelStyle);
								}
								if (keyframeUi.badges.blocked)
								{
									ImGui::SameLine();
									nbl::ui::drawBadge("BLOCKED", bad, badgeText, panelStyle);
								}

								if (!keyframeUi.canApply)
									ImGui::BeginDisabled();
								if (ImGui::Button("Apply selected"))
									applyPresetFromUi(activeCamera, selectedKeyframe->preset);
								if (!keyframeUi.canApply)
									ImGui::EndDisabled();
								nbl::ui::drawHoverHint(keyframeUi.canApply ?
									"Apply selected keyframe to the active camera" :
									"Apply is blocked because there is no active camera or the keyframe goal is invalid");
								ImGui::SameLine();
								if (!keyframeCaptureUi.canCapture)
									ImGui::BeginDisabled();
								if (ImGui::Button("Replace from camera"))
									replaceSelectedKeyframeFromCamera(activeCamera);
								if (!keyframeCaptureUi.canCapture)
									ImGui::EndDisabled();
								nbl::ui::drawHoverHint(keyframeCaptureUi.canCapture ?
									"Overwrite selected keyframe from the current active camera" :
									"Replace is blocked because there is no active camera or the current goal state is invalid");
								ImGui::SameLine();
								if (ImGui::Button("Jump to selected"))
								{
									m_playback.time = selectedKeyframe->time;
									applyPlaybackAtTime(m_playback.time);
								}
								nbl::ui::drawHoverHint("Set playback time to selected keyframe and preview it");
								ImGui::SameLine();
								if (ImGui::Button("Remove selected"))
								{
									m_keyframeTrack.keyframes.erase(m_keyframeTrack.keyframes.begin() + m_keyframeTrack.selectedKeyframeIx);
									normalizeSelectedKeyframe();
									clampPlaybackTimeToKeyframes();
									if (m_keyframeTrack.keyframes.empty())
										clearApplyStatusBanner(m_playbackApplyBanner);
								}
								nbl::ui::drawHoverHint("Remove selected keyframe");
							}

							nbl::ui::drawSectionHeader("KeyframesStorageHeader", "Keyframe Storage", accent, panelStyle);
							ImGui::InputText("Keyframe file", m_keyframePath, IM_ARRAYSIZE(m_keyframePath));
							if (ImGui::Button("Save keyframes"))
							{
								if (!saveKeyframesToFile(nbl::system::path(m_keyframePath)))
									m_logger->log("Failed to save keyframes to \"%s\".", ILogger::ELL_ERROR, m_keyframePath);
							}
							nbl::ui::drawHoverHint("Save keyframes to JSON file");
							ImGui::SameLine();
							if (ImGui::Button("Load keyframes"))
							{
								if (!loadKeyframesFromFile(nbl::system::path(m_keyframePath)))
									m_logger->log("Failed to load keyframes from \"%s\".", ILogger::ELL_ERROR, m_keyframePath);
							}
							nbl::ui::drawHoverHint("Load keyframes from JSON file");
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
						nbl::ui::drawSectionHeader("GizmoHeader", "Gizmo", accent, panelStyle);
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
						nbl::ui::drawSectionHeader("LogHeader", "Virtual Events", accent, panelStyle);
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
			nbl::ui::popControlPanelWindowStyle();

}


