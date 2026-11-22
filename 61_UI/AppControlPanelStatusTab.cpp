#include "app/App.hpp"

#include <format>

void App::drawControlPanelStatusTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	if (!nbl::ui::CCameraControlPanelUiUtilities::beginControlPanelTabChild("StatusPanel", panelStyle))
	{
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}

	ImGui::PushItemWidth(-1.0f);
	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("SessionHeader", "Session", panelStyle.AccentColor, panelStyle);
	if (nbl::ui::CCameraControlPanelUiUtilities::beginCard("SessionCard", nbl::ui::CCameraControlPanelUiUtilities::calcCameraControlPanelCardHeight(3, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
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
				nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine(row, panelStyle);
			ImGui::EndTable();
		}
	}
	nbl::ui::CCameraControlPanelUiUtilities::endCard();

	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("CameraHeader", "Camera", panelStyle.AccentColor, panelStyle);
	if (auto* activeCamera = getActiveCamera())
	{
		const auto& gimbal = activeCamera->getGimbal();
		const auto pos = gimbal.getPosition();
		const auto euler = CCameraMathUtilities::getPitchYawRollDegrees(gimbal.getOrientation());

		if (nbl::ui::CCameraControlPanelUiUtilities::beginCard("CameraCard", nbl::ui::CCameraControlPanelUiUtilities::calcCameraControlPanelCardHeight(3, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
		{
			if (ImGui::BeginTable("CameraTable", 2, panelStyle.SummaryTableFlags))
			{
				ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
				ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
				const auto positionText = std::format("{:.2f} {:.2f} {:.2f}", pos.x, pos.y, pos.z);
				const auto eulerText = std::format("{:.1f} {:.1f} {:.1f}", euler.x, euler.y, euler.z);
				const std::array<nbl::ui::SCameraControlPanelStatusLineSpec, 3u> cameraRows = {{
					{ .label = "Name", .value = activeCamera->getIdentifier(), .dotColor = panelStyle.AccentColor, .valueColor = panelStyle.MutedColor },
					{ .label = "Position", .value = positionText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor },
					{ .label = "Euler", .value = eulerText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }
				}};
				for (const auto& row : cameraRows)
					nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine(row, panelStyle);
				ImGui::EndTable();
			}
		}
		nbl::ui::CCameraControlPanelUiUtilities::endCard();
	}
	else if (nbl::ui::CCameraControlPanelUiUtilities::beginCard("CameraCard", nbl::ui::CCameraControlPanelUiUtilities::calcCameraControlPanelCardHeight(2, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
	{
		ImGui::TextDisabled("No active camera");
		nbl::ui::CCameraControlPanelUiUtilities::endCard();
	}

	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("ProjectionHeader", "Projection", panelStyle.AccentColor, panelStyle);
	auto& binding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
	auto& planar = m_planarProjections[binding.activePlanarIx];
	if (planar && binding.boundProjectionIx.has_value())
	{
		auto& projection = planar->getProjections()[binding.boundProjectionIx.value()];
		const auto& params = projection.getParameters();
		if (nbl::ui::CCameraControlPanelUiUtilities::beginCard("ProjectionCard", nbl::ui::CCameraControlPanelUiUtilities::calcCameraControlPanelCardHeight(4, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
		{
			if (ImGui::BeginTable("ProjectionTable", 2, panelStyle.SummaryTableFlags))
			{
				ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, panelStyle.SummaryLabelColumnWidth);
				ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
				const auto zNearText = std::format("{:.2f}", params.zNear);
				const auto zFarText = std::format("{:.2f}", params.zFar);
				const auto typeText = params.kind == CPlanarProjection::EKind::Perspective ? "Perspective" : "Orthographic";
				nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine({ .label = "Type", .value = typeText, .dotColor = panelStyle.AccentColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine({ .label = "zNear", .value = zNearText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine({ .label = "zFar", .value = zFarText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				if (params.kind == CPlanarProjection::EKind::Perspective)
				{
					const auto fovText = std::format("{:.1f}", params.perspective.fov);
					nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine({ .label = "Fov", .value = fovText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				}
				else
				{
					const auto orthoWidthText = std::format("{:.1f}", params.orthographic.orthoWidth);
					nbl::ui::CCameraControlPanelUiUtilities::drawStatusLine({ .label = "Ortho width", .value = orthoWidthText, .dotColor = panelStyle.MutedColor, .valueColor = panelStyle.MutedColor }, panelStyle);
				}
				ImGui::EndTable();
			}
		}
		nbl::ui::CCameraControlPanelUiUtilities::endCard();
	}
	else if (nbl::ui::CCameraControlPanelUiUtilities::beginCard("ProjectionCard", nbl::ui::CCameraControlPanelUiUtilities::calcCameraControlPanelCardHeight(2, panelStyle), panelStyle.CardTopColor, panelStyle.CardBottomColor, panelStyle.CardBorderColor, panelStyle))
	{
		ImGui::TextDisabled("No projection bound");
		nbl::ui::CCameraControlPanelUiUtilities::endCard();
	}

	ImGui::PopItemWidth();
	nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
}
