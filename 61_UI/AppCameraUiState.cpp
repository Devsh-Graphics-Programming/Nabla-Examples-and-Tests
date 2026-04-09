#include "app/App.hpp"
#include "app/AppResourceUtilities.hpp"

namespace
{

template<typename TContext>
inline bool tryBindActiveViewportContext(TContext& outContext, const SActiveViewportRuntimeState& viewportState)
{
	outContext = {};
	outContext.viewport = viewportState;
	return outContext.valid();
}

inline SCameraFollowConfig* tryGetViewportFollowConfig(
	std::span<SCameraFollowConfig> followConfigs,
	const SActiveViewportRuntimeState& viewportState)
{
	if (!viewportState.valid())
		return nullptr;

	const auto planarIx = viewportState.requireBinding().activePlanarIx;
	if (planarIx >= followConfigs.size())
		return nullptr;

	return &followConfigs[planarIx];
}

} // namespace

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
	return tryGetViewportFollowConfig(
		std::span<SCameraFollowConfig>(m_sceneInteraction.planarFollowConfigs.data(), m_sceneInteraction.planarFollowConfigs.size()),
		tryGetActiveViewportRuntimeState());
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
	return tryBindActiveViewportContext(outContext, tryGetActiveViewportRuntimeState());
}

bool App::tryBuildActiveProjectionTabContext(SActiveProjectionTabContext& outContext)
{
	if (!tryBindActiveViewportContext(outContext, tryGetActiveViewportRuntimeState()))
		return false;

	outContext.activeRenderWindowIxString = std::to_string(m_viewports.activeRenderWindowIx);
	outContext.activePlanarIxString = std::to_string(outContext.requireBinding().activePlanarIx);
	return true;
}

bool App::tryBuildActiveScriptedCameraContext(SActiveScriptedCameraContext& outContext)
{
	if (!tryBindActiveViewportContext(outContext, tryGetActiveViewportRuntimeState()))
		return false;

	outContext.followConfig = tryGetViewportFollowConfig(
		std::span<SCameraFollowConfig>(m_sceneInteraction.planarFollowConfigs.data(), m_sceneInteraction.planarFollowConfigs.size()),
		outContext.viewport);
	const auto planarSpan = getPlanarProjectionSpan();
	outContext.hasProjectionContext = nbl::ui::tryBuildBindingProjectionContext(
		planarSpan,
		outContext.requireBinding(),
		outContext.projectionContext);
	return true;
}
