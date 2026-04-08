#include "app/App.hpp"
#include "app/AppViewportBindingUtilities.hpp"

void App::imguiListen()
{
	ImGuiIO& io = ImGui::GetIO();
	if (m_ciMode)
	{
		io.IniFilename = nullptr;
		useWindow = true;
	}

	ImGuizmo::BeginFrame();

	SImResourceInfo info;
	info.samplerIx = static_cast<uint16_t>(nbl::ext::imgui::UI::DefaultSamplerIx::USER);

	if (useWindow)
		drawWindowedViewportWindows(io, info);
	else
		drawFullscreenViewportWindow(io, info);

	drawScriptVisualDebugOverlay(io.DisplaySize);
	DrawControlPanel();
	UpdateBoundCameraMovement();
	UpdateCursorVisibility();
	applyFollowToConfiguredCameras();
	refreshViewportBindingMatrices();
}

void App::drawWindowedViewportWindows(ImGuiIO& io, SImResourceInfo& info)
{
	syncVisualDebugWindowBindings();
	const bool hideSceneGizmos = enableActiveCameraMovement || (m_scriptedInput.enabled && m_scriptedInput.visualDebug);
	ImGuizmo::Enable(!hideSceneGizmos);

	size_t gizmoIx = 0u;
	size_t manipulationCounter = 0u;
	const std::optional<uint32_t> manipulatedObjectIx = ImGuizmo::IsUsingAny() ? std::optional<uint32_t>(getManipulatedObjectIx()) : std::nullopt;
	(void)manipulatedObjectIx;

	for (uint32_t windowIx = 0u; windowIx < windowBindings.size(); ++windowIx)
	{
		{
			const auto& rw = wInit.renderWindows[windowIx];
			const ImGuiCond windowCond = m_ciMode ? ImGuiCond_Always : ImGuiCond_Appearing;
			ImGui::SetNextWindowPos({ rw.iPos.x, rw.iPos.y }, windowCond);
			ImGui::SetNextWindowSize({ rw.iSize.x, rw.iSize.y }, windowCond);
		}
		ImGui::SetNextWindowSizeConstraints(ImVec2(0x45, 0x45), ImVec2(7680, 4320));

		nbl::ui::pushViewportWindowStyle();
		const std::string ident = "Render Window \"" + std::to_string(windowIx) + "\"";

		ImGui::Begin(ident.data(), nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
		const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
		const ImVec2 cursorPos = ImGui::GetCursorScreenPos();
		const nbl::ui::SViewportOverlayRect viewportRect = { cursorPos, contentRegionSize };

		if (ImGuiWindow* const window = ImGui::GetCurrentWindow())
		{
			const auto mousePos = ImGui::GetMousePos();
			const bool mouseInsideViewport =
				mousePos.x >= cursorPos.x &&
				mousePos.y >= cursorPos.y &&
				mousePos.x <= cursorPos.x + contentRegionSize.x &&
				mousePos.y <= cursorPos.y + contentRegionSize.y;
			if (mouseInsideViewport)
				window->Flags |= ImGuiWindowFlags_NoMove;
			else
				window->Flags &= ~ImGuiWindowFlags_NoMove;
		}

		auto& binding = windowBindings[windowIx];
		nbl::ui::SBoundViewportCameraState viewportState = {};
		const auto planarSpan = std::span<const smart_refctd_ptr<planar_projection_t>>(m_planarProjections.data(), m_planarProjections.size());
		if (!nbl::ui::tryBuildViewportBoundCameraState(planarSpan, binding, contentRegionSize, flipGizmoY, viewportState))
		{
			ImGui::End();
			nbl::ui::popViewportWindowStyle();
			continue;
		}

		auto* const planarViewCameraBound = viewportState.camera;
		auto& projection = *viewportState.projection;
		info.textureID = windowIx + 1u;

		ImGuizmo::AllowAxisFlip(binding.allowGizmoAxesToFlip);
		ImGuizmo::SetOrthographic(projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic);
		ImGuizmo::SetDrawlist();
		ImGui::Image(info, contentRegionSize);
		ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);

		if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
		{
			const char* projLabel = projection.getParameters().m_type == IPlanarProjection::CProjection::Perspective ? "Persp" : "Ortho";
			nbl::ui::SCameraViewportInfoOverlayData overlayData = {};
			overlayData.headline = "Planar " + std::to_string(binding.activePlanarIx) + " | " + projLabel + " | W" + std::to_string(windowIx);
			overlayData.description = std::string(getCameraTypeLabel(planarViewCameraBound)) + ": " + std::string(getCameraTypeDescription(planarViewCameraBound));
			overlayData.detail = "Frustum: active camera (hidden in owner view)";
			nbl::ui::drawViewportInfoOverlay(*drawList, viewportRect, overlayData);

			if (m_scriptedInput.enabled && m_scriptedInput.visualDebug && m_scriptedInput.visualFollowActive)
				nbl::ui::drawFollowTargetViewportOverlay(*drawList, viewportState.viewProjMatrix, m_followTarget, viewportRect);
		}

		const bool windowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
		const bool windowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
		if (!(m_scriptedInput.enabled && m_scriptedInput.exclusive))
		{
			if (!m_scriptedInput.enabled && windowHovered)
				activeRenderWindowIx = windowIx;
			else if (windowFocused)
				activeRenderWindowIx = windowIx;
		}

		if (!hideSceneGizmos)
		{
			for (uint32_t objectIx = 0u; objectIx < getManipulableObjectCount(); ++objectIx)
			{
				ImGuizmo::PushID(gizmoIx);
				++gizmoIx;

				const auto planarIx = getManipulableObjectPlanarIx(objectIx);
				const bool isFollowTarget = isManipulableObjectFollowTarget(objectIx);
				ICamera* const targetGimbalManipulationCamera = planarIx.has_value() ? m_planarProjections[planarIx.value()]->getCamera() : nullptr;
				if (targetGimbalManipulationCamera == planarViewCameraBound)
				{
					ImGuizmo::PopID();
					continue;
				}

				ImGuizmoModelM16InOut imguizmoModel;
				imguizmoModel.inTRS = getManipulableObjectTransform(objectIx);

				const float gizmoWorldRadius = isFollowTarget ? 0.35f : 0.22f;
				const auto gizmoWorldPos = getManipulableObjectWorldPosition(objectIx);
				const auto viewPos = mul(viewportState.viewMatrix, float32_t4(gizmoWorldPos, 1.0f));
				const float depth = std::max(0.001f, std::abs(viewPos.z));
				float gizmoSizeClip = 0.1f;
				if (projection.getParameters().m_type == IPlanarProjection::CProjection::Perspective)
					gizmoSizeClip = (gizmoWorldRadius * viewportState.projectionMatrix[1][1]) / depth;
				else
					gizmoSizeClip = gizmoWorldRadius * viewportState.projectionMatrix[1][1];
				ImGuizmo::SetGizmoSizeClipSpace(gizmoSizeClip);

				imguizmoModel.outTRS = imguizmoModel.inTRS;
				const bool success = ImGuizmo::Manipulate(
					&viewportState.imguizmoPlanar.view[0][0],
					&viewportState.imguizmoPlanar.projection[0][0],
					ImGuizmo::OPERATION::UNIVERSAL,
					mCurrentGizmoMode,
					&imguizmoModel.outTRS[0][0],
					&imguizmoModel.outDeltaTRS[0][0],
					useSnap ? &snap[0] : nullptr);

				if (success)
				{
					if (targetGimbalManipulationCamera)
					{
						const auto referenceFrame = getCastedMatrix<float64_t>(imguizmoModel.outTRS);
						bindManipulatedCamera(planarIx.value());
						nbl::core::applyReferenceFrameToCamera(targetGimbalManipulationCamera, referenceFrame);
						refreshFollowOffsetConfigForPlanar(planarIx.value());
					}
					else if (isFollowTarget)
					{
						setFollowTargetTransform(getCastedMatrix<float64_t>(imguizmoModel.outTRS));
						bindManipulatedFollowTarget();
						applyFollowToConfiguredCameras();
					}
					else
					{
						m_model = float32_t3x4(hlsl::transpose(imguizmoModel.outTRS));
						bindManipulatedModel();
					}
				}

				if (ImGuizmo::IsOver() && !ImGuizmo::IsUsingAny() && !enableActiveCameraMovement)
				{
					if (targetGimbalManipulationCamera && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
					{
						const uint32_t newPlanarIx = planarIx.value();
						if (newPlanarIx < m_planarProjections.size())
						{
							binding.activePlanarIx = newPlanarIx;
							binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());
							if (!(m_scriptedInput.enabled && m_scriptedInput.exclusive))
								activeRenderWindowIx = windowIx;
						}
					}
					else if (isFollowTarget && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
					{
						bindManipulatedFollowTarget();
					}
					else if (!targetGimbalManipulationCamera && !isFollowTarget && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
					{
						bindManipulatedModel();
					}

					const ImVec2 mousePos = ImGui::GetIO().MousePos;
					nbl::ui::beginHoverInfoOverlay("InfoOverlay", mousePos);

					std::string objectLabel;
					if (targetGimbalManipulationCamera)
						objectLabel = targetGimbalManipulationCamera->getIdentifier();
					else if (isFollowTarget)
						objectLabel = m_followTarget.getIdentifier();
					else
						objectLabel = "Geometry Creator Object";

					ImGui::Text("Identifier: %s", objectLabel.c_str());
					ImGui::Text("Object Ix: %u", objectIx);
					if (targetGimbalManipulationCamera)
					{
						ImGui::Separator();
						ImGui::TextDisabled("RMB: switch view to this camera");
						ImGui::TextDisabled("LMB drag: manipulate gizmo");
						ImGui::TextDisabled("SPACE: toggle move mode");
					}
					else if (isFollowTarget)
					{
						ImGui::Separator();
						ImGui::TextDisabled("RMB: select follow target");
						ImGui::TextDisabled("LMB drag: move or rotate tracked target");
						ImGui::TextDisabled("Enabled follow cameras update on the next frame");
					}

					nbl::ui::endHoverInfoOverlay();
				}

				ImGuizmo::PopID();
			}
		}

		ImGui::End();
		nbl::ui::popViewportWindowStyle();
	}

	if (windowBindings.size() > 1u)
	{
		const auto& topRw = wInit.renderWindows[0];
		const float splitY = topRw.iPos.y + topRw.iSize.y;
		const float gap = std::max(0.0f, wInit.renderWindows[1].iPos.y - splitY);
		ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
		ImGui::Begin("SplitOverlay", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
		if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
			nbl::ui::drawViewportSplitOverlay(*drawList, io.DisplaySize, splitY, gap);
		ImGui::End();
	}

	assert(manipulationCounter <= 1u);
}

void App::drawFullscreenViewportWindow(ImGuiIO& io, SImResourceInfo& info)
{
	info.textureID = 1u + activeRenderWindowIx;

	ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
	ImGui::SetNextWindowSize(io.DisplaySize);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, nbl::ui::SCameraViewportWindowStyle::WindowBackgroundColor);
	ImGui::Begin("FullScreenWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
	const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
	const ImVec2 cursorPos = ImGui::GetCursorScreenPos();

	nbl::ui::SBoundViewportCameraState viewportState = {};
	const auto planarSpan = std::span<const smart_refctd_ptr<planar_projection_t>>(m_planarProjections.data(), m_planarProjections.size());
	const bool viewportValid = nbl::ui::tryBuildViewportBoundCameraState(planarSpan, windowBindings[activeRenderWindowIx], contentRegionSize, false, viewportState);

	ImGui::Image(info, contentRegionSize);
	ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);
	if (viewportValid && m_scriptedInput.enabled && m_scriptedInput.visualDebug && m_scriptedInput.visualFollowActive)
	{
		if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
		{
			const nbl::ui::SViewportOverlayRect followRect = { cursorPos, contentRegionSize };
			nbl::ui::drawFollowTargetViewportOverlay(*drawList, viewportState.viewProjMatrix, m_followTarget, followRect);
		}
	}

	ImGui::End();
	ImGui::PopStyleColor(1);
}

void App::refreshViewportBindingMatrices()
{
	const auto planarSpan = std::span<const smart_refctd_ptr<planar_projection_t>>(m_planarProjections.data(), m_planarProjections.size());
	for (auto& binding : windowBindings)
	{
		nbl::ui::SBoundViewportCameraState viewportState = {};
		nbl::ui::tryBuildWindowBindingMatrices(planarSpan, binding, viewportState);
	}
}

