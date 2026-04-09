#include "app/App.hpp"

#include <format>

void App::drawControlPanelStatusTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	if (!nbl::ui::beginControlPanelTabChild("StatusPanel", panelStyle))
	{
		nbl::ui::endControlPanelTabChild();
		return;
	}

	ImGui::PushItemWidth(-1.0f);
	nbl::ui::drawSectionHeader("SessionHeader", "Session", panelStyle.AccentColor, panelStyle);
	if (nbl::ui::beginCard("SessionCard", nbl::ui::calcCameraControlPanelCardHeight(3, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
	{
		if (ImGui::BeginTable("SessionTable", 2, panelStyle.SummaryTableFlags))
		{
			ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
			ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
			const auto activeWindowText = std::to_string(m_viewports.activeRenderWindowIx);
			const std::array<nbl::ui::SCameraControlPanelStatusLineSpec, 3u> sessionRows = {{
				{ .label = "Mode", .value = m_viewports.useWindow ? "Window" : "Fullscreen", .dotColor = panelStyle.AccentColor, .valueColor = panelStyle.AccentColor },
				{ .label = "Active window", .value = activeWindowText, .dotColor = panelStyle.AccentColor, .valueColor = panelStyle.AccentColor },
				{ .label = "Movement", .value = m_viewports.enableActiveCameraMovement ? "Enabled" : "Disabled", .dotColor = m_viewports.enableActiveCameraMovement ? panelStyle.GoodColor : panelStyle.BadColor, .valueColor = m_viewports.enableActiveCameraMovement ? panelStyle.GoodColor : panelStyle.BadColor }
			}};
			for (const auto& row : sessionRows)
				nbl::ui::drawStatusLine(row, panelStyle);
			ImGui::EndTable();
		}
	}
	nbl::ui::endCard();

	nbl::ui::drawSectionHeader("CameraHeader", "Camera", panelStyle.AccentColor, panelStyle);
	if (auto* activeCamera = getActiveCamera())
	{
		const auto& gimbal = activeCamera->getGimbal();
		const auto pos = gimbal.getPosition();
		const auto euler = hlsl::CCameraMathUtilities::getCameraOrientationEulerDegrees(gimbal.getOrientation());

		if (nbl::ui::beginCard("CameraCard", nbl::ui::calcCameraControlPanelCardHeight(5, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
		{
			if (ImGui::BeginTable("CameraTable", 2, panelStyle.SummaryTableFlags))
			{
				ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
				ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
				const auto positionText = std::format("{:.2f} {:.2f} {:.2f}", pos.x, pos.y, pos.z);
				const auto eulerText = std::format("{:.1f} {:.1f} {:.1f}", euler.x, euler.y, euler.z);
				const auto moveScaleText = std::format("{:.4f}", activeCamera->getMoveSpeedScale());
				const auto rotateScaleText = std::format("{:.4f}", activeCamera->getRotationSpeedScale());
				const std::array<nbl::ui::SCameraControlPanelStatusLineSpec, 5u> cameraRows = {{
					{ .label = "Name", .value = activeCamera->getIdentifier(), .dotColor = panelStyle.AccentColor, .valueColor = panelStyle.MutedColor },
					{ .label = "Position", .value = positionText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor },
					{ .label = "Euler", .value = eulerText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor },
					{ .label = "Move scale", .value = moveScaleText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor },
					{ .label = "Rotate scale", .value = rotateScaleText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }
				}};
				for (const auto& row : cameraRows)
					nbl::ui::drawStatusLine(row, panelStyle);
				ImGui::EndTable();
			}
		}
		nbl::ui::endCard();
	}
	else if (nbl::ui::beginCard("CameraCard", nbl::ui::calcCameraControlPanelCardHeight(2, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
	{
		ImGui::TextDisabled("No active camera");
		nbl::ui::endCard();
	}

	nbl::ui::drawSectionHeader("ProjectionHeader", "Projection", panelStyle.AccentColor, panelStyle);
	auto& binding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
	auto& planar = m_planarProjections[binding.activePlanarIx];
	if (planar && binding.boundProjectionIx.has_value())
	{
		auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];
		const auto& params = projection.getParameters();
		if (nbl::ui::beginCard("ProjectionCard", nbl::ui::calcCameraControlPanelCardHeight(4, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
		{
			if (ImGui::BeginTable("ProjectionTable", 2, panelStyle.SummaryTableFlags))
			{
				ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
				ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
				const auto zNearText = std::format("{:.2f}", params.m_zNear);
				const auto zFarText = std::format("{:.2f}", params.m_zFar);
				const auto typeText = params.m_type == IPlanarProjection::CProjection::Perspective ? "Perspective" : "Orthographic";
				nbl::ui::drawStatusLine({ .label = "Type", .value = typeText, .dotColor = panelStyle.AccentColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				nbl::ui::drawStatusLine({ .label = "zNear", .value = zNearText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				nbl::ui::drawStatusLine({ .label = "zFar", .value = zFarText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				if (params.m_type == IPlanarProjection::CProjection::Perspective)
				{
					const auto fovText = std::format("{:.1f}", params.m_planar.perspective.fov);
					nbl::ui::drawStatusLine({ .label = "Fov", .value = fovText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				}
				else
				{
					const auto orthoWidthText = std::format("{:.1f}", params.m_planar.orthographic.orthoWidth);
					nbl::ui::drawStatusLine({ .label = "Ortho width", .value = orthoWidthText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				}
				ImGui::EndTable();
			}
		}
		nbl::ui::endCard();
	}
	else if (nbl::ui::beginCard("ProjectionCard", nbl::ui::calcCameraControlPanelCardHeight(2, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
	{
		ImGui::TextDisabled("No projection bound");
		nbl::ui::endCard();
	}

	ImGui::PopItemWidth();
	nbl::ui::endControlPanelTabChild();
}
