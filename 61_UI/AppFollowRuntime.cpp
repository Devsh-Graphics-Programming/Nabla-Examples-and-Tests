#include "app/App.hpp"

namespace
{

inline float getScriptVisualDebugFps(const SScriptedInputRuntimeState& scriptedInput)
{
	return std::max(1.f, scriptedInput.visualTargetFps);
}

inline uint64_t computeScriptVisualDebugHoldFrames(const SScriptedInputRuntimeState& scriptedInput, const float fps)
{
	return static_cast<uint64_t>(std::round(std::max(0.f, scriptedInput.visualCameraHoldSeconds) * fps));
}

inline uint64_t computeElapsedFrames(const uint64_t currentFrame, const SScriptedVisualPlanarState& visualPlanar)
{
	return (currentFrame >= visualPlanar.startFrame) ? (currentFrame - visualPlanar.startFrame) : 0ull;
}

inline uint64_t computeProgressFrames(const uint64_t elapsedFrames, const uint64_t holdFrames)
{
	return holdFrames ? std::min(elapsedFrames, holdFrames) : elapsedFrames;
}

inline nbl::ui::SCameraScriptVisualDebugStatus buildScriptVisualDebugStatus(
	const ICamera& camera,
	const uint32_t planarIx,
	const size_t planarCount,
	const uint64_t absoluteFrame,
	const SScriptedInputRuntimeState& scriptedInput)
{
	const auto fps = getScriptVisualDebugFps(scriptedInput);
	const auto holdFrames = computeScriptVisualDebugHoldFrames(scriptedInput, fps);
	const auto elapsedFrames = computeElapsedFrames(absoluteFrame, scriptedInput.visualPlanar);

	nbl::ui::SCameraScriptVisualDebugStatus status = {};
    status.cameraLabel = CCameraTextUtilities::getCameraTypeLabel(&camera);
    status.cameraHint = CCameraTextUtilities::getCameraTypeDescription(&camera);
	status.cameraIndex = planarIx;
	status.cameraCount = static_cast<uint32_t>(planarCount);
	status.planarIndex = planarIx;
	status.hasHoldFrames = holdFrames > 0u;
	status.progressFrames = computeProgressFrames(elapsedFrames, holdFrames);
	status.holdFrames = holdFrames;
	status.targetFps = fps;
	status.absoluteFrame = absoluteFrame;
	status.segmentLabel = scriptedInput.visualPlanar.segmentLabel;
	status.followActive = scriptedInput.visualFollow.active;
    status.followModeDescription = nbl::ui::CCameraTextUtilities::getCameraFollowModeDescription(scriptedInput.visualFollow.mode);
	status.followLockValid = scriptedInput.visualFollow.lockValid;
	status.followLockAngleDeg = scriptedInput.visualFollow.lockAngleDeg;
	status.followTargetDistance = scriptedInput.visualFollow.targetDistance;
	status.followTargetCenterNdcRadius = scriptedInput.visualFollow.projectedTarget.radius;

	float dynamicFov = 0.0f;
	if (camera.tryGetDynamicPerspectiveFov(dynamicFov))
	{
		status.hasDynamicFov = true;
		status.dynamicFovDeg = dynamicFov;
	}

	return status;
}

inline float getFollowTargetMarkerScale(const SScriptedInputRuntimeState& scriptedInput)
{
	return (scriptedInput.enabled && scriptedInput.visualDebug) ?
		SCameraAppSceneDefaults::FollowTargetMarkerScaleVisualDebug :
		SCameraAppSceneDefaults::FollowTargetMarkerScale;
}

} // namespace

void App::setFollowTargetTransform(const float64_t4x4& transform)
{
	m_sceneInteraction.followTarget.trySetFromTransform(transform);
}

float32_t3x4 App::computeFollowTargetMarkerWorld() const
{
	return buildFollowTargetMarkerWorldTransform(
		m_sceneInteraction.followTarget,
		getFollowTargetMarkerScale(m_scriptedInput));
}

bool App::captureFollowOffsetsForPlanar(const uint32_t planarIx)
{
	if (planarIx >= m_planarProjections.size() || planarIx >= m_sceneInteraction.planarFollowConfigs.size())
		return false;

	auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
	return nbl::core::CCameraFollowUtilities::captureFollowOffsetsFromCamera(
		m_cameraGoalSolver,
		camera,
		m_sceneInteraction.followTarget,
		m_sceneInteraction.planarFollowConfigs[planarIx]);
}

bool App::followConfigUsesCapturedOffset(const SCameraFollowConfig& config) const
{
	return config.enabled && nbl::core::CCameraFollowUtilities::cameraFollowModeUsesCapturedOffset(config.mode);
}

void App::refreshFollowOffsetConfigForPlanar(const uint32_t planarIx)
{
	if (planarIx >= m_planarProjections.size() || planarIx >= m_sceneInteraction.planarFollowConfigs.size())
		return;

	auto& config = m_sceneInteraction.planarFollowConfigs[planarIx];
	if (!followConfigUsesCapturedOffset(config))
		return;

	auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
	if (!camera)
		return;

	nbl::core::CCameraFollowUtilities::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, m_sceneInteraction.followTarget, config);
}

void App::refreshFollowOffsetConfigsForCamera(ICamera* camera)
{
	if (!camera)
		return;

	for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size() && planarIx < m_sceneInteraction.planarFollowConfigs.size(); ++planarIx)
	{
		auto* planarCamera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
		if (planarCamera != camera)
			continue;
		refreshFollowOffsetConfigForPlanar(planarIx);
	}
}

void App::refreshAllFollowOffsetConfigs()
{
	for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size() && planarIx < m_sceneInteraction.planarFollowConfigs.size(); ++planarIx)
		refreshFollowOffsetConfigForPlanar(planarIx);
}

float64_t3 App::getDefaultFollowTargetPosition() const
{
	return SCameraAppSceneDefaults::DefaultFollowTargetPosition;
}

camera_quaternion_t<float64_t> App::getDefaultFollowTargetOrientation() const
{
	return SCameraAppSceneDefaults::DefaultFollowTargetOrientation;
}

void App::resetFollowTargetToDefault()
{
	m_sceneInteraction.followTarget.setPose(getDefaultFollowTargetPosition(), getDefaultFollowTargetOrientation());
}

void App::snapFollowTargetToModel()
{
	const auto modelTransform = hlsl::transpose(getMatrix3x4As4x4(m_sceneInteraction.model));
	setFollowTargetTransform(getCastedMatrix<float64_t>(modelTransform));
}

void App::applyFollowToConfiguredCameras(const bool allowDuringScriptedInput)
{
	if (m_scriptedInput.enabled && !allowDuringScriptedInput)
		return;
	if (m_sceneInteraction.planarFollowConfigs.size() != m_planarProjections.size())
		return;

	for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
	{
		auto& planar = m_planarProjections[planarIx];
		auto* camera = planar ? planar->getCamera() : nullptr;
		if (!camera)
			continue;

		const auto& config = m_sceneInteraction.planarFollowConfigs[planarIx];
		if (!config.enabled || config.mode == ECameraFollowMode::Disabled)
			continue;

		const auto result = nbl::core::CCameraFollowUtilities::applyFollowToCamera(m_cameraGoalSolver, camera, m_sceneInteraction.followTarget, config);
		if (!result.succeeded())
			continue;

		for (auto& projection : planar->getPlanarProjections())
			nbl::core::CCameraProjectionUtilities::syncDynamicPerspectiveProjection(camera, projection);
	}
}

bool App::isOrbitLikeCamera(ICamera* camera)
{
	return camera && camera->hasCapability(ICamera::SphericalTarget);
}

void App::syncVisualDebugWindowBindings()
{
	if (!m_scriptedInput.enabled)
		return;
	if (m_viewports.windowBindings.size() < 2u || m_planarProjections.empty())
		return;

	auto& perspectiveBinding = m_viewports.windowBindings[0u];
	if (perspectiveBinding.activePlanarIx >= m_planarProjections.size())
		return;

	auto& perspectivePlanar = m_planarProjections[perspectiveBinding.activePlanarIx];
	if (!perspectivePlanar)
		return;
	if (!nbl::ui::trySelectBindingProjectionType(
			getPlanarProjectionSpan(),
			perspectiveBinding,
			IPlanarProjection::CProjection::Perspective))
	{
		return;
	}

	auto& orthoBinding = m_viewports.windowBindings[1u];
	if (orthoBinding.activePlanarIx != perspectiveBinding.activePlanarIx)
	{
		if (!nbl::ui::trySelectBindingPlanar(
				getPlanarProjectionSpan(),
				orthoBinding,
				perspectiveBinding.activePlanarIx))
		{
			return;
		}
	}

	if (orthoBinding.activePlanarIx >= m_planarProjections.size())
		return;

	auto& orthoPlanar = m_planarProjections[orthoBinding.activePlanarIx];
	if (!orthoPlanar)
		return;

	nbl::ui::trySelectBindingProjectionType(
		getPlanarProjectionSpan(),
		orthoBinding,
		IPlanarProjection::CProjection::Orthographic);
}

void App::drawScriptVisualDebugOverlay(const ImVec2& displaySize)
{
	if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug))
		return;

	const auto viewportState = tryGetActiveViewportRuntimeState();
	if (!viewportState.valid())
		return;

	if (!m_scriptedInput.visualPlanar.valid)
	{
		m_scriptedInput.visualPlanar.valid = true;
		m_scriptedInput.visualPlanar.planarIx = viewportState.requireBinding().activePlanarIx;
		m_scriptedInput.visualPlanar.startFrame = m_realFrameIx;
	}

	const auto debugStatus = buildScriptVisualDebugStatus(
		viewportState.requireCamera(),
		viewportState.requireBinding().activePlanarIx,
		m_planarProjections.size(),
		m_realFrameIx,
		m_scriptedInput);

	nbl::ui::CCameraScriptVisualDebugOverlayUtilities::drawScriptVisualDebugOverlay(displaySize, nbl::ui::CCameraScriptVisualDebugOverlayUtilities::buildScriptVisualDebugOverlayData(debugStatus));
}
