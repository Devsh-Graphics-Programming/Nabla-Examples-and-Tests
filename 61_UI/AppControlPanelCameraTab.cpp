#include "app/App.hpp"

void App::drawControlPanelCameraTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	using checkbox_spec_t = nbl::ui::SCameraControlPanelCheckboxSpec;
	using slider_spec_t = nbl::ui::SCameraControlPanelSliderSpec;

	if (!nbl::ui::beginControlPanelTabChild("CameraPanel", panelStyle))
	{
		nbl::ui::endControlPanelTabChild();
		return;
	}

	ImGui::PushItemWidth(-1.0f);
	nbl::ui::drawSectionHeader("CameraInputHeader", "Input", panelStyle.AccentColor, panelStyle);
	for (const auto& spec : {
		checkbox_spec_t{ .label = "Mirror input to all cameras", .value = &m_cameraControls.mirrorInput, .hint = "Apply keyboard and mouse input to every camera" },
		checkbox_spec_t{ .label = "World translate", .value = &m_cameraControls.worldTranslate, .hint = "Translate in world space instead of camera space" }
	})
	{
		nbl::ui::drawCheckboxWithHint(spec);
	}
	for (const auto& spec : {
		slider_spec_t{ .label = "Keyboard scale", .value = &m_cameraControls.keyboardScale, .minValue = SCameraAppControlPanelRangeDefaults::InputScaleMin, .maxValue = SCameraAppControlPanelRangeDefaults::InputScaleMax, .format = "%.2f", .hint = "Scale keyboard movement magnitudes" },
		slider_spec_t{ .label = "Mouse move scale", .value = &m_cameraControls.mouseMoveScale, .minValue = SCameraAppControlPanelRangeDefaults::InputScaleMin, .maxValue = SCameraAppControlPanelRangeDefaults::InputScaleMax, .format = "%.2f", .hint = "Scale mouse move magnitudes" },
		slider_spec_t{ .label = "Mouse scroll scale", .value = &m_cameraControls.mouseScrollScale, .minValue = SCameraAppControlPanelRangeDefaults::InputScaleMin, .maxValue = SCameraAppControlPanelRangeDefaults::InputScaleMax, .format = "%.2f", .hint = "Scale mouse wheel magnitudes" },
		slider_spec_t{ .label = "Translate scale", .value = &m_cameraControls.translationScale, .minValue = SCameraAppControlPanelRangeDefaults::InputScaleMin, .maxValue = SCameraAppControlPanelRangeDefaults::InputScaleMax, .format = "%.2f", .hint = "Overall translation scale for virtual events" },
		slider_spec_t{ .label = "Rotate scale", .value = &m_cameraControls.rotationScale, .minValue = SCameraAppControlPanelRangeDefaults::InputScaleMin, .maxValue = SCameraAppControlPanelRangeDefaults::InputScaleMax, .format = "%.2f", .hint = "Overall rotation scale for virtual events" }
	})
	{
		nbl::ui::drawSliderFloatWithHint(spec);
	}

	nbl::ui::drawSectionHeader("CameraConstraintsHeader", "Constraints", panelStyle.AccentColor, panelStyle);
	for (const auto& spec : {
		checkbox_spec_t{ .label = "Enable constraints", .value = &m_cameraConstraints.enabled, .hint = "Enable or disable all camera constraints" },
		checkbox_spec_t{ .label = "Clamp distance", .value = &m_cameraConstraints.clampDistance, .hint = "Clamp orbit distance to min/max" }
	})
	{
		nbl::ui::drawCheckboxWithHint(spec);
	}
	for (const auto& spec : {
		slider_spec_t{ .label = "Min distance", .value = &m_cameraConstraints.minDistance, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintDistanceMin, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintMinDistanceMax, .format = "%.3f", .flags = ImGuiSliderFlags_Logarithmic, .hint = "Minimum orbit distance" },
		slider_spec_t{ .label = "Max distance", .value = &m_cameraConstraints.maxDistance, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintDistanceMin, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintMaxDistanceMax, .format = "%.3f", .flags = ImGuiSliderFlags_Logarithmic, .hint = "Maximum orbit distance" }
	})
	{
		nbl::ui::drawSliderFloatWithHint(spec);
	}
	ImGui::Separator();
	for (const auto& spec : {
		checkbox_spec_t{ .label = "Clamp pitch", .value = &m_cameraConstraints.clampPitch, .hint = "Clamp pitch angle" },
		checkbox_spec_t{ .label = "Clamp yaw", .value = &m_cameraConstraints.clampYaw, .hint = "Clamp yaw angle" },
		checkbox_spec_t{ .label = "Clamp roll", .value = &m_cameraConstraints.clampRoll, .hint = "Clamp roll angle" }
	})
	{
		nbl::ui::drawCheckboxWithHint(spec);
	}
	for (const auto& spec : {
		slider_spec_t{ .label = "Pitch min", .value = &m_cameraConstraints.pitchMinDeg, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMinDeg, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMaxDeg, .format = "%.1f", .hint = "Minimum pitch in degrees" },
		slider_spec_t{ .label = "Pitch max", .value = &m_cameraConstraints.pitchMaxDeg, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMinDeg, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMaxDeg, .format = "%.1f", .hint = "Maximum pitch in degrees" },
		slider_spec_t{ .label = "Yaw min", .value = &m_cameraConstraints.yawMinDeg, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMinDeg, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMaxDeg, .format = "%.1f", .hint = "Minimum yaw in degrees" },
		slider_spec_t{ .label = "Yaw max", .value = &m_cameraConstraints.yawMaxDeg, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMinDeg, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMaxDeg, .format = "%.1f", .hint = "Maximum yaw in degrees" },
		slider_spec_t{ .label = "Roll min", .value = &m_cameraConstraints.rollMinDeg, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMinDeg, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMaxDeg, .format = "%.1f", .hint = "Minimum roll in degrees" },
		slider_spec_t{ .label = "Roll max", .value = &m_cameraConstraints.rollMaxDeg, .minValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMinDeg, .maxValue = SCameraAppControlPanelRangeDefaults::ConstraintAngleMaxDeg, .format = "%.1f", .hint = "Maximum roll in degrees" }
	})
	{
		nbl::ui::drawSliderFloatWithHint(spec);
	}

	nbl::ui::drawSectionHeader("OrbitHeader", "Orbit Target", panelStyle.AccentColor, panelStyle);
	auto* activeCamera = getActiveCamera();
	ICamera::SphericalTargetState orbitState;
	const bool hasOrbitTarget = activeCamera && activeCamera->tryGetSphericalTargetState(orbitState);
	if (hasOrbitTarget)
	{
		auto target = getCastedVector<float32_t>(orbitState.target);
		if (ImGui::InputFloat3("Target", &target[0]))
			activeCamera->trySetSphericalTarget(getCastedVector<float64_t>(target));

		if (nbl::ui::drawActionButtonWithHint("Target model", "Set orbit target to the model position"))
		{
			const auto targetPos = hlsl::transpose(getMatrix3x4As4x4(m_sceneInteraction.model))[3];
			activeCamera->trySetSphericalTarget(float64_t3(targetPos.x, targetPos.y, targetPos.z));
		}
		ImGui::SameLine();
		if (nbl::ui::drawActionButtonWithHint("Target origin", "Set orbit target to world origin"))
			activeCamera->trySetSphericalTarget(float64_t3(0.0));
	}
	else
	{
		ImGui::TextDisabled("Active camera is not orbit.");
	}

	nbl::ui::drawSectionHeader("FollowHeader", "Follow Target", panelStyle.AccentColor, panelStyle);
	if (auto* activeFollowConfig = getActiveFollowConfig())
	{
		auto& followConfig = *activeFollowConfig;
		const bool prevFollowEnabled = followConfig.enabled;
		const auto prevFollowMode = followConfig.mode;
		nbl::ui::drawCheckboxWithHint({ .label = "Enable follow", .value = &followConfig.enabled, .hint = "Apply tracked-target follow to the active planar camera" });

		const char* followModeLabels[] = {
			getCameraFollowModeLabel(ECameraFollowMode::Disabled),
			getCameraFollowModeLabel(ECameraFollowMode::OrbitTarget),
			getCameraFollowModeLabel(ECameraFollowMode::LookAtTarget),
			getCameraFollowModeLabel(ECameraFollowMode::KeepWorldOffset),
			getCameraFollowModeLabel(ECameraFollowMode::KeepLocalOffset)
		};
		int followModeIx = static_cast<int>(followConfig.mode);
		if (ImGui::Combo("Mode", &followModeIx, followModeLabels, IM_ARRAYSIZE(followModeLabels)))
			followConfig.mode = static_cast<ECameraFollowMode>(followModeIx);

		const bool followStateChanged = followConfig.enabled != prevFollowEnabled || followConfig.mode != prevFollowMode;
		if (followStateChanged && followConfig.enabled && nbl::core::CCameraFollowUtilities::cameraFollowModeUsesCapturedOffset(followConfig.mode))
			captureFollowOffsetsForPlanar(getActivePlanarIx());
		if (followStateChanged && followConfig.enabled)
			applyFollowToConfiguredCameras();

		auto trackedTarget = getCastedVector<float32_t>(m_sceneInteraction.followTarget.getGimbal().getPosition());
		if (ImGui::InputFloat3("Tracked target", &trackedTarget[0]))
			m_sceneInteraction.followTarget.setPosition(getCastedVector<float64_t>(trackedTarget));

		nbl::ui::drawCheckboxWithHint({ .label = "Show target marker", .value = &m_sceneInteraction.followTargetVisible, .hint = "Render the tracked target marker in the scene" });

		if (nbl::ui::drawActionButtonWithHint("Reset target", "Reset tracked target gimbal to the default world-space follow pose"))
			resetFollowTargetToDefault();
		ImGui::SameLine();
		if (nbl::ui::drawActionButtonWithHint("Snap to model", "Optionally snap tracked target gimbal to the model transform"))
			snapFollowTargetToModel();
		ImGui::SameLine();
		if (nbl::ui::drawActionButtonWithHint("Target origin", "Reset tracked target to identity at world origin"))
			m_sceneInteraction.followTarget.setPose(float64_t3(0.0), makeIdentityQuaternion<float64_t>());
		ImGui::SameLine();
		if (nbl::ui::drawActionButtonWithHint("Capture current offset", "Store current camera-to-target relation into the active follow config"))
			captureFollowOffsetsForPlanar(getActivePlanarIx());

		if (CCameraFollowUtilities::cameraFollowModeUsesWorldOffset(followConfig.mode))
		{
			auto worldOffset = getCastedVector<float32_t>(followConfig.worldOffset);
			if (ImGui::InputFloat3("World offset", &worldOffset[0]))
				followConfig.worldOffset = getCastedVector<float64_t>(worldOffset);
		}
		if (CCameraFollowUtilities::cameraFollowModeUsesLocalOffset(followConfig.mode))
		{
			auto localOffset = getCastedVector<float32_t>(followConfig.localOffset);
			if (ImGui::InputFloat3("Local offset", &localOffset[0]))
				followConfig.localOffset = getCastedVector<float64_t>(localOffset);
		}
	}
	else
	{
		ImGui::TextDisabled("No active follow config.");
	}

	ImGui::PopItemWidth();
	nbl::ui::endControlPanelTabChild();
}
