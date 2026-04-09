#include "app/App.hpp"

#include <unordered_set>

namespace
{

struct SCollectedCameraVirtualEvents final
{
	std::vector<CVirtualGimbalEvent> events = {};
	uint32_t keyboardVirtualEventCount = 0u;

	inline uint32_t totalCount() const
	{
		return static_cast<uint32_t>(events.size());
	}

	inline bool empty() const
	{
		return events.empty();
	}
};

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

inline std::span<const SMouseEvent> buildOrbitFilteredMouseInput(
	std::span<const SMouseEvent> mouseEvents,
	const bool orbitLookDown,
	std::vector<SMouseEvent>& filteredMouseEvents)
{
	if (orbitLookDown)
		return mouseEvents;

	filteredMouseEvents.clear();
	filteredMouseEvents.reserve(mouseEvents.size());
	for (const auto& event : mouseEvents)
	{
		if (event.type != ui::SMouseEvent::EET_MOVEMENT)
			filteredMouseEvents.emplace_back(event);
	}
	return { filteredMouseEvents.data(), filteredMouseEvents.size() };
}

inline void scaleCollectedVirtualEvents(
	SCollectedCameraVirtualEvents& virtualEvents,
	const CameraControlSettings& cameraControls)
{
	for (uint32_t i = 0u; i < virtualEvents.keyboardVirtualEventCount; ++i)
		virtualEvents.events[i].magnitude *= cameraControls.keyboardScale;

	nbl::core::CCameraManipulationUtilities::scaleVirtualEvents(
		virtualEvents.events,
		virtualEvents.totalCount(),
		cameraControls.translationScale,
		cameraControls.rotationScale);
}

template<typename SyncWindowInputBinding>
inline SCollectedCameraVirtualEvents collectActiveCameraVirtualEvents(
	SWindowControlBinding& binding,
	ICamera* camera,
	const std::span<const SKeyboardEvent> keyboardEvents,
	const std::span<const SMouseEvent> mouseEvents,
	const std::chrono::microseconds presentationTimestamp,
	const CameraControlSettings& cameraControls,
	const bool orbitLikeCamera,
	const bool orbitLookDown,
	SyncWindowInputBinding&& syncWindowInputBinding)
{
	SCollectedCameraVirtualEvents collectedVirtualEvents = {};
	if (!camera)
		return collectedVirtualEvents;

	syncWindowInputBinding(binding);
	auto& inputBinder = binding.inputBinding;

	std::vector<SMouseEvent> filteredOrbitMouseEvents;
	auto filteredMouseInput = mouseEvents;
	if (orbitLikeCamera)
		filteredMouseInput = buildOrbitFilteredMouseInput(mouseEvents, orbitLookDown, filteredOrbitMouseEvents);

	auto binderEvents = inputBinder.collectVirtualEvents(presentationTimestamp, {
		.keyboardEvents = keyboardEvents,
		.mouseEvents = filteredMouseInput
	});
	const uint32_t virtualEventCount = binderEvents.totalCount();
	if (!virtualEventCount)
		return collectedVirtualEvents;

	collectedVirtualEvents.keyboardVirtualEventCount = binderEvents.keyboardCount;
	collectedVirtualEvents.events.assign(
		binderEvents.events.begin(),
		binderEvents.events.begin() + virtualEventCount);
	scaleCollectedVirtualEvents(collectedVirtualEvents, cameraControls);
	return collectedVirtualEvents;
}

template<typename RefreshFollowOffsets, typename AppendVirtualEventLog>
inline void applyCollectedVirtualEventsToCamera(
	ICamera* target,
	const uint32_t planarIx,
	const SCollectedCameraVirtualEvents& collectedVirtualEvents,
	const bool worldTranslate,
	const nbl::core::CCameraGoalSolver& goalSolver,
	const SCameraConstraintSettings& cameraConstraints,
	const bool scriptedInputEnabled,
	RefreshFollowOffsets&& refreshFollowOffsets,
	AppendVirtualEventLog&& appendVirtualEventLog)
{
	if (!target || collectedVirtualEvents.empty())
		return;

	if (worldTranslate)
	{
		std::vector<CVirtualGimbalEvent> perCameraEvents = collectedVirtualEvents.events;
		uint32_t perCount = collectedVirtualEvents.totalCount();
		nbl::core::CCameraManipulationUtilities::remapTranslationEventsFromWorldToCameraLocal(target, perCameraEvents, perCount);
		if (perCount)
			target->manipulate({ perCameraEvents.data(), perCount });
	}
	else
	{
		target->manipulate({ collectedVirtualEvents.events.data(), collectedVirtualEvents.totalCount() });
	}

	nbl::core::CCameraManipulationUtilities::applyCameraConstraints(goalSolver, target, cameraConstraints);
	if (!scriptedInputEnabled)
		refreshFollowOffsets(planarIx);
	appendVirtualEventLog(target, planarIx, collectedVirtualEvents);
}

} // namespace

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
	auto& binding = *inputContext.viewport.binding;
	auto* camera = inputContext.viewport.camera;
	const bool orbitLookDown = ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
		(m_scriptedInput.enabled && (m_scriptedInput.scriptedMouseButtons.leftDown || m_scriptedInput.scriptedMouseButtons.rightDown));
	SCollectedCameraVirtualEvents virtualEvents = collectActiveCameraVirtualEvents(
		binding,
		camera,
		keyboardEvents,
		mouseEvents,
		m_nextPresentationTimestamp,
		m_cameraControls,
		isOrbitLikeCamera(camera),
		orbitLookDown,
		[this](SWindowControlBinding& windowBinding) { syncWindowInputBinding(windowBinding); });

	if (virtualEvents.empty())
		return;

	const auto applyVirtualEventsToCamera = [&](ICamera* target, const uint32_t planarIx) -> void
	{
		applyCollectedVirtualEventsToCamera(
			target,
			planarIx,
			virtualEvents,
			m_cameraControls.worldTranslate,
			m_cameraGoalSolver,
			m_cameraConstraints,
			m_scriptedInput.enabled,
			[this](const uint32_t ix) { refreshFollowOffsetConfigForPlanar(ix); },
			[this](ICamera* logCamera, const uint32_t ix, const SCollectedCameraVirtualEvents& collectedEvents)
			{
				appendVirtualEventLog("input", "Keyboard/Mouse", ix, logCamera, collectedEvents.events.data(), collectedEvents.totalCount());
			});
	};

	appendUniqueCameraInputTargets(
		getPlanarProjectionSpan(),
		std::span<const SWindowControlBinding>(m_viewports.windowBindings.data(), m_viewports.windowBindings.size()),
		inputContext.viewport,
		m_cameraControls.mirrorInput,
		[&](const SActiveCameraInputTarget& target)
		{
			if (!target.valid())
				return;
			applyVirtualEventsToCamera(target.camera, target.planarIx);
		});

	if (!m_scriptedInput.log)
		return;

	for (const auto& event : virtualEvents.events)
	{
		m_logger->log("[script] virtual %s magnitude=%.6f", ILogger::ELL_INFO, CVirtualGimbalEvent::virtualEventToString(event.type).data(), event.magnitude);
	}
	logScriptedCameraPose("input", camera);
}
