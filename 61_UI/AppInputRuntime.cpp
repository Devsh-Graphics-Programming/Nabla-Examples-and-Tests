#include "app/App.hpp"

#include <unordered_set>

template<typename AddTarget>
inline void appendUniqueCameraInputTargets(
	std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
	std::span<const SWindowControlBinding> windowBindings,
	const SActiveViewportRuntimeState& activeViewport,
	const bool mirrorInput,
	AddTarget&& addTarget)
{
	if (!mirrorInput)
	{
		addTarget({
			.camera = activeViewport.camera,
			.planarIx = activeViewport.requireBinding().activePlanarIx
		});
		return;
	}

	std::unordered_set<const ICamera*> visited;
	for (const auto& windowBinding : windowBindings)
	{
		if (windowBinding.activePlanarIx >= planarProjections.size())
			continue;

		const auto& planarProjection = planarProjections[windowBinding.activePlanarIx];
		if (!planarProjection)
			continue;

		auto* target = planarProjection->getCamera();
		if (!target || !visited.insert(target).second)
			continue;

		addTarget({
			.camera = target,
			.planarIx = windowBinding.activePlanarIx
		});
	}
}

inline void applyControlsToCamera(
	ICamera* target,
	const uint32_t planarIx,
	const SCameraControls& controls,
	const bool worldTranslate,
	const CCameraGoalSolver& goalSolver,
	const SCameraConstraintSettings& cameraConstraints,
	const bool scriptedInputEnabled,
	auto&& refreshFollowOffsets,
	auto&& appendControlLog)
{
	if (!target)
		return;

	auto targetControls = controls;

	// FPS and Free apply `translate` in their own frame, so reading it as world-space means rotating it into
	// that frame first. The target-relative and path rigs each define `translate` their own way, so the toggle
	// leaves them alone.
	if (worldTranslate)
	{
		const auto kind = target->getKind();
		if (kind == ICamera::CameraKind::FPS || kind == ICamera::CameraKind::Free)
		{
			targetControls.translate = CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame<float64_t>(
				target->getGimbal().getOrientation(),
				targetControls.translate);
		}
	}

	// one frame is collected from the active camera's binding and may reach cameras of another kind, and a rig
	// refuses a whole frame that carries an axis it does not accept, so the rest is dropped per target
	targetControls = targetControls.masked(target->getAcceptedControls());
	if (targetControls.nonZeroAxes() == 0u)
		return;

	target->manipulate(targetControls);

	nbl::this_example::CCameraConstraintUtilities::applyCameraConstraints(goalSolver, target, cameraConstraints);
	if (!scriptedInputEnabled)
		refreshFollowOffsets(planarIx);
	appendControlLog(target, planarIx, targetControls);
}

void App::refreshCameraInputBinding(ICamera* camera)
{
	if (!camera)
		return;

	const auto kind = camera->getKind();
	m_cameraInputBindingKind = kind;
	auto& binding = m_cameraController.binding;
	binding = CCameraMouseKeyboardPresets::makeDefaultBinding(kind);

	// re-derived from a fresh default every time, so dragging a slider does not compound
	binding.scaleSensitivity(
		nbl::core::bitflag<ECameraControlAxis>(ECameraControlAxis::Translate) | ECameraControlAxis::Distance | ECameraControlAxis::Path,
		m_cameraControls.translateScale);
	binding.scaleSensitivity(ECameraControlAxis::Rotate, m_cameraControls.rotateScale);

	// an orbit-like rig looks only while the right button is held, which is what the old app-side filter did
	if (isOrbitLikeCamera(camera))
		binding.setMouseMovementGate(ECameraControlAxis::Rotate, ui::EMB_RIGHT_BUTTON);
}

void App::applyActiveCameraInput(
	std::span<const SKeyboardEvent> keyboardEvents,
	std::span<const SMouseEvent> mouseEvents,
	const bool skipCameraInput)
{
	if (!(m_viewports.enableActiveCameraMovement && !skipCameraInput))
		return;

	SActiveCameraInputContext inputContext = {};
	if (!tryBuildActiveCameraInputContext(inputContext))
		return;
	auto* camera = inputContext.viewport.camera;
	if (!camera)
		return;

	refreshCameraInputBinding(camera);

	// the controller consumes the frame window, so this runs once per frame however many cameras it then drives
	const auto controls = m_cameraController.collect(m_nextPresentationTimestamp, keyboardEvents, mouseEvents);
	if (controls.nonZeroAxes() == 0u)
		return;

	appendUniqueCameraInputTargets(
		getPlanarProjectionSpan(),
		std::span<const SWindowControlBinding>(m_viewports.windowBindings.data(), m_viewports.windowBindings.size()),
		inputContext.viewport,
		m_cameraControls.mirrorInput,
		[&](const SActiveCameraInputTarget& target)
		{
			if (!target.valid())
				return;

			applyControlsToCamera(
				target.camera,
				target.planarIx,
				controls,
				m_cameraControls.worldTranslate,
				m_cameraGoalSolver,
				m_cameraConstraints,
				m_scriptedInput.enabled,
				[this](const uint32_t ix) { refreshFollowOffsetConfigForPlanar(ix); },
				[this](ICamera* logCamera, const uint32_t ix, const SCameraControls& applied)
				{
					appendVirtualEventLog("input", "Keyboard/Mouse", ix, logCamera, applied);
				});
		});

	if (!m_scriptedInput.log)
		return;

	logScriptedVirtualEvents("input", controls);
	logScriptedCameraPose("input", camera);
}
