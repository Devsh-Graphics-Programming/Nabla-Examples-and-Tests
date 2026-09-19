#include "app/App.hpp"
#include "app/AppProjectionControlPanelUiUtilities.hpp"

void App::drawControlPanelProjectionTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	if (!nbl::ui::CCameraControlPanelUiUtilities::beginControlPanelTabChild("ProjectionPanel", panelStyle))
	{
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}

	ImGui::PushItemWidth(-1.0f);
	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("PlanarSelectHeader", "Planar Selection", panelStyle.AccentColor, panelStyle);

	SActiveProjectionTabContext runtime = {};
	auto refreshRuntime = [&]() -> bool
	{
		return tryBuildActiveProjectionTabContext(runtime);
	};

	if (!nbl::ui::drawRenderWindowSelector(m_viewports.windowBindings.size(), m_viewports.activeRenderWindowIx, refreshRuntime))
	{
		ImGui::PopItemWidth();
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}
	nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Choose which render window the panel edits");

	if (!refreshRuntime())
	{
		ImGui::TextDisabled("No active viewport.");
		ImGui::PopItemWidth();
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}

	ImGui::Text("Editing: %s", runtime.activeRenderWindowIxString.c_str());
	nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Selected render window for planar and projection changes");

	assert(!m_planarProjections.empty());
	auto& binding = runtime.requireBinding();
	if (!nbl::ui::drawProjectionPlanarSelector(getPlanarProjectionSpan(), runtime, refreshRuntime))
	{
		ImGui::PopItemWidth();
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}
	nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Select which camera the window renders");

	assert(binding.boundProjectionIx.has_value());
	assert(binding.lastBoundPerspectivePresetProjectionIx.has_value());
	assert(binding.lastBoundOrthoPresetProjectionIx.has_value());

	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("ProjectionParamsHeader", "Projection Parameters", panelStyle.AccentColor, panelStyle);
	if (!nbl::ui::drawProjectionTypeSelector(getPlanarProjectionSpan(), runtime, refreshRuntime))
	{
		ImGui::PopItemWidth();
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}

	const auto selectedProjectionType = runtime.requirePlanar().getProjections()[binding.boundProjectionIx.value()].getParameters().kind;
	nbl::ui::drawProjectionPresetSelector(getPlanarProjectionSpan(), runtime, selectedProjectionType);
	nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Switch preset projection for this planar");

	auto& boundProjection = runtime.requirePlanar().getProjections()[binding.boundProjectionIx.value()];
	assert(!boundProjection.isProjectionSingular());
	nbl::ui::drawProjectionParameterControls(binding, boundProjection, m_viewports.useWindow);

	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("CursorHeader", "Cursor Behaviour", panelStyle.AccentColor, panelStyle);
	nbl::ui::drawCursorBehaviourControls(m_viewports.captureCursorInMoveMode, m_viewports.resetCursorToCenter);

	ImGui::TextColored(
		m_viewports.enableActiveCameraMovement ? panelStyle.GoodColor : panelStyle.BadColor,
		"Bound Camera Movement: %s",
		m_viewports.enableActiveCameraMovement ? "Enabled" : "Disabled");
	ImGui::Separator();

	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("BoundCameraHeader", "Bound Camera", panelStyle.AccentColor, panelStyle);
	nbl::ui::drawBoundCameraSection(
		runtime,
		binding.activePlanarIx,
		m_cameraController.binding,
		[this](const char* topText, const char* tableName, int rows, int columns, const float* pointer, bool withSeparator)
		{
			addMatrixTable(topText, tableName, rows, columns, pointer, withSeparator);
		},
		[this, &runtime]() { refreshCameraInputBinding(runtime.viewport.camera); });

	ImGui::PopItemWidth();
	nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
}
