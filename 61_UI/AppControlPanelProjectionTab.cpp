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
	SActiveProjectionTabContext runtime = {};
	if (!tryBuildActiveProjectionTabContext(runtime))
	{
		ImGui::TextDisabled("No active viewport.");
		ImGui::PopItemWidth();
		nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
		return;
	}

	nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("PlanarSelectHeader", "Planar Selection", panelStyle.AccentColor, panelStyle);
	ImGui::Text("Active Render Window: %s", runtime.activeRenderWindowIxString.c_str());
	nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Window that receives input and camera switching");

	auto refreshRuntime = [&]() -> bool
	{
		return tryBuildActiveProjectionTabContext(runtime);
	};

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

	const auto selectedProjectionType = runtime.requirePlanar().getPlanarProjections()[binding.boundProjectionIx.value()].getParameters().m_type;
	const bool updateBoundVirtualMaps = nbl::ui::drawProjectionPresetSelector(getPlanarProjectionSpan(), runtime, selectedProjectionType);
	if (updateBoundVirtualMaps)
		syncWindowInputBinding(binding);
	nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Switch preset projection for this planar");

	auto& boundProjection = runtime.requirePlanar().getPlanarProjections()[binding.boundProjectionIx.value()];
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
		[this](const char* topText, const char* tableName, int rows, int columns, const float* pointer, bool withSeparator)
		{
			addMatrixTable(topText, tableName, rows, columns, pointer, withSeparator);
		},
		[this](SWindowControlBinding& windowBinding) { syncWindowInputBinding(windowBinding); },
		[this](SWindowControlBinding& windowBinding) { syncWindowInputBindingToProjection(windowBinding); });

	ImGui::PopItemWidth();
	nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
}
