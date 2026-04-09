#include "app/App.hpp"
#include "app/AppResourceUtilities.hpp"

nbl::system::SCameraAppResourceContext App::getCameraAppResourceContext() const
{
	return m_system ?
		nbl::system::makeCameraAppResourceContext(*m_system, localInputCWD) :
		nbl::system::SCameraAppResourceContext{};
}

ICamera* App::getActiveCamera()
{
	const auto viewportState = tryGetActiveViewportRuntimeState();
	return viewportState.camera;
}

uint32_t App::getActivePlanarIx() const
{
	return m_viewports.activeRenderWindowIx < m_viewports.windowBindings.size() ?
		m_viewports.windowBindings[m_viewports.activeRenderWindowIx].activePlanarIx :
		SWindowControlBinding::InvalidPlanarIx;
}

SCameraFollowConfig* App::getActiveFollowConfig()
{
	const auto planarIx = getActivePlanarIx();
	if (planarIx >= m_sceneInteraction.planarFollowConfigs.size())
		return nullptr;
	return &m_sceneInteraction.planarFollowConfigs[planarIx];
}

const SCameraFollowConfig* App::getActiveFollowConfig() const
{
	const auto planarIx = getActivePlanarIx();
	if (planarIx >= m_sceneInteraction.planarFollowConfigs.size())
		return nullptr;
	return &m_sceneInteraction.planarFollowConfigs[planarIx];
}

SActiveViewportRuntimeState App::tryGetActiveViewportRuntimeState()
{
	SActiveViewportRuntimeState viewportState = {};
	nbl::ui::tryBuildActiveViewportRuntimeState(
		getPlanarProjectionSpan(),
		std::span<SWindowControlBinding>(m_viewports.windowBindings.data(), m_viewports.windowBindings.size()),
		m_viewports.activeRenderWindowIx,
		viewportState);
	return viewportState;
}

bool App::tryBuildActiveCameraInputContext(SActiveCameraInputContext& outContext)
{
	outContext = {};
	outContext.viewport = tryGetActiveViewportRuntimeState();
	return outContext.valid();
}

bool App::tryBuildActiveProjectionTabContext(SActiveProjectionTabContext& outContext)
{
	outContext = {};
	outContext.viewport = tryGetActiveViewportRuntimeState();
	if (!outContext.valid())
		return false;

	outContext.activeRenderWindowIxString = std::to_string(m_viewports.activeRenderWindowIx);
	outContext.activePlanarIxString = std::to_string(outContext.viewport.binding->activePlanarIx);
	return true;
}

bool App::tryBuildActiveScriptedCameraContext(SActiveScriptedCameraContext& outContext)
{
	outContext = {};
	outContext.viewport = tryGetActiveViewportRuntimeState();
	if (!outContext.valid())
		return false;

	outContext.followConfig = getActiveFollowConfig();
	const auto planarSpan = getPlanarProjectionSpan();
	outContext.hasProjectionContext = nbl::ui::tryBuildBindingProjectionContext(
		planarSpan,
		outContext.requireBinding(),
		outContext.projectionContext);
	return true;
}
