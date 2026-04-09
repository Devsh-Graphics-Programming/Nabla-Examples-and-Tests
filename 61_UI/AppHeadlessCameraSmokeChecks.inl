	struct SCameraSmokeResolvedState final
	{
		const CCameraGoalSolver& goalSolver;
		nbl::system::ISystem* system = nullptr;
		const SCameraSmokePresetInventory& initialPresets;
		ICamera* fpsCamera = nullptr;
		ICamera* orbitCamera = nullptr;
		ICamera* arcballCamera = nullptr;
		ICamera* turntableCamera = nullptr;
		ICamera* topDownCamera = nullptr;
		ICamera* isometricCamera = nullptr;
		ICamera* freeCamera = nullptr;
		ICamera* chaseCamera = nullptr;
		ICamera* dollyCamera = nullptr;
		ICamera* pathCamera = nullptr;
		ICamera* dollyZoomCamera = nullptr;
	};

	inline bool verifyCrossKindAndPresentationSmoke(
		const SCameraSmokeResolvedState& state,
		std::string& outError)
	{
		if (state.initialPresets.orbit.has_value() && state.initialPresets.chase.has_value())
		{
			if (!verifyExactCrossKindApply(state.goalSolver, state.orbitCamera, state.initialPresets.chase.value(), "Chase->Orbit", outError))
				return false;
			if (!verifyExactCrossKindApply(state.goalSolver, state.chaseCamera, state.initialPresets.orbit.value(), "Orbit->Chase", outError))
				return false;
		}

		if (state.initialPresets.orbit.has_value() && state.initialPresets.dolly.has_value())
		{
			if (!verifyExactCrossKindApply(state.goalSolver, state.orbitCamera, state.initialPresets.dolly.value(), "Dolly->Orbit", outError))
				return false;
			if (!verifyExactCrossKindApply(state.goalSolver, state.dollyCamera, state.initialPresets.orbit.value(), "Orbit->Dolly", outError))
				return false;
		}

		if (state.initialPresets.orbit.has_value() && state.initialPresets.path.has_value() && state.orbitCamera)
		{
			if (!verifyApproximateCrossKindApply(
					state.goalSolver,
					state.orbitCamera,
					state.initialPresets.path.value(),
					CCameraGoalSolver::SApplyResult::MissingPathState,
					"Path->Orbit",
					outError))
			{
				return false;
			}
		}

		if (state.initialPresets.orbit.has_value() && state.initialPresets.dollyZoom.has_value() && state.orbitCamera)
		{
			if (!verifyApproximateCrossKindApply(
					state.goalSolver,
					state.orbitCamera,
					state.initialPresets.dollyZoom.value(),
					CCameraGoalSolver::SApplyResult::MissingDynamicPerspectiveState,
					"DollyZoom->Orbit",
					outError))
			{
				return false;
			}
		}

		if (!state.initialPresets.orbit.has_value())
			return true;

		if (std::string_view(nbl::ui::CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::All)) != "All" ||
			std::string_view(nbl::ui::CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::Exact)) != "Exact" ||
			std::string_view(nbl::ui::CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::BestEffort)) != "Best-effort")
		{
			outError = "Presentation utilities smoke returned an unexpected filter label.";
			return false;
		}

		const auto blockedPresentation = nbl::ui::CCameraPresentationUtilities::analyzePresetPresentation(state.goalSolver, nullptr, state.initialPresets.orbit.value());
		if (blockedPresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
			blockedPresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
		{
			outError = "Presentation utilities smoke allowed a null-camera preset through an exactness filter.";
			return false;
		}
		if (blockedPresentation.sourceKindLabel.empty() || blockedPresentation.goalStateLabel.empty())
		{
			outError = "Presentation utilities smoke produced empty blocked presentation labels.";
			return false;
		}

		const auto blockedBadges = nbl::ui::CCameraPresentationUtilities::collectGoalApplyPresentationBadges(blockedPresentation);
		if (!blockedBadges.blocked || blockedBadges.exact || blockedBadges.bestEffort || blockedPresentation.badges.blocked != blockedBadges.blocked)
		{
			outError = "Presentation utilities smoke produced wrong blocked badge flags.";
			return false;
		}

		if (state.orbitCamera)
		{
			const auto exactPresentation = nbl::ui::CCameraPresentationUtilities::analyzePresetPresentation(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value());
			if (!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
				!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
				exactPresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
			{
				outError = "Presentation utilities smoke failed exact filtering.";
				return false;
			}

			const auto exactBadges = nbl::ui::CCameraPresentationUtilities::collectGoalApplyPresentationBadges(exactPresentation);
			if (!exactBadges.exact || exactBadges.bestEffort || exactBadges.dropsState || exactBadges.sharedStateOnly || exactBadges.blocked)
			{
				outError = "Presentation utilities smoke produced wrong exact badge flags.";
				return false;
			}
			if (exactPresentation.sourceKindLabel.empty() || exactPresentation.goalStateLabel.empty())
			{
				outError = "Presentation utilities smoke produced empty exact presentation labels.";
				return false;
			}

			const auto capturePresentation = nbl::ui::CCameraPresentationUtilities::analyzeCapturePresentation(state.goalSolver, state.orbitCamera);
			if (!capturePresentation.canCapture || capturePresentation.policyLabel.empty())
			{
				outError = "Presentation utilities smoke failed orbit capture presentation.";
				return false;
			}
		}

		if (state.initialPresets.path.has_value() && state.orbitCamera)
		{
			const auto approximatePresentation = nbl::ui::CCameraPresentationUtilities::analyzePresetPresentation(state.goalSolver, state.orbitCamera, state.initialPresets.path.value());
			if (!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
				approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
				!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
			{
				outError = "Presentation utilities smoke failed best-effort filtering.";
				return false;
			}

			const auto approximateBadges = nbl::ui::CCameraPresentationUtilities::collectGoalApplyPresentationBadges(approximatePresentation);
			if (approximateBadges.exact || !approximateBadges.bestEffort || !approximateBadges.dropsState || approximateBadges.sharedStateOnly || approximateBadges.blocked)
			{
				outError = "Presentation utilities smoke produced wrong best-effort badge flags.";
				return false;
			}
			if (approximatePresentation.sourceKindLabel.empty() || approximatePresentation.goalStateLabel.empty())
			{
				outError = "Presentation utilities smoke produced empty best-effort presentation labels.";
				return false;
			}
		}

		return true;
	}

	inline std::vector<CameraPreset> collectAvailableSmokePresets(const SCameraSmokePresetInventory& initialPresets)
	{
		std::vector<CameraPreset> sourcePresets;
		sourcePresets.reserve(5u);
		if (initialPresets.orbit.has_value())
			sourcePresets.push_back(initialPresets.orbit.value());
		if (initialPresets.chase.has_value())
			sourcePresets.push_back(initialPresets.chase.value());
		if (initialPresets.dolly.has_value())
			sourcePresets.push_back(initialPresets.dolly.value());
		if (initialPresets.path.has_value())
			sourcePresets.push_back(initialPresets.path.value());
		if (initialPresets.dollyZoom.has_value())
			sourcePresets.push_back(initialPresets.dollyZoom.value());
		return sourcePresets;
	}

	inline float chooseShiftedReferenceDistance(const ICamera::SphericalTargetState& state)
	{
		const float farther = std::min(state.maxDistance, state.distance + 1.25f);
		if (hlsl::abs(static_cast<double>(farther - state.distance)) > CameraTinyScalarEpsilon)
			return farther;

		const float nearer = std::max(state.minDistance, state.distance - 1.25f);
		return nearer;
	}

	inline bool tryBuildReferenceFrameFromTargetRelativeState(
		const nbl::core::SCameraTargetRelativeState& desiredState,
		hlsl::float64_t4x4& outReferenceFrame,
		nbl::core::CCameraGoal& outExpectedGoal)
	{
		outExpectedGoal = {};
		nbl::core::SCameraTargetRelativePose pose = {};
		if (!nbl::core::CCameraTargetRelativeUtilities::tryBuildTargetRelativePoseFromState(
				desiredState,
				nbl::core::SCameraTargetRelativeTraits::MinDistance,
				nbl::core::SCameraTargetRelativeTraits::DefaultMaxDistance,
				pose) ||
			!nbl::core::CCameraGoalUtilities::applyCanonicalTargetRelativeGoal(outExpectedGoal, desiredState))
		{
			return false;
		}

		outReferenceFrame = hlsl::CCameraMathUtilities::composeTransformMatrix(pose.position, pose.orientation);
		return true;
	}

	inline bool verifyReferenceFrameGoalApply(
		const SCameraSmokeResolvedState& state,
		ICamera* const camera,
		const nbl::core::SCameraTargetRelativeState& desiredState,
		std::string_view label,
		std::string& outError)
	{
		ICamera::SphericalTargetState baselineState = {};
		if (!camera->tryGetSphericalTargetState(baselineState))
		{
			outError = std::string(label) + " reference-frame smoke failed to capture the baseline spherical state.";
			return false;
		}

		const nbl::core::SCameraTargetRelativeState baselineTargetRelativeState = {
			.target = baselineState.target,
			.orbitUv = baselineState.orbitUv,
			.distance = baselineState.distance
		};

		hlsl::float64_t4x4 referenceFrame = hlsl::float64_t4x4(1.0);
		hlsl::float64_t4x4 baselineReferenceFrame = hlsl::float64_t4x4(1.0);
		nbl::core::CCameraGoal expectedGoal = {};
		nbl::core::CCameraGoal baselineGoal = {};
		if (!tryBuildReferenceFrameFromTargetRelativeState(desiredState, referenceFrame, expectedGoal) ||
			!tryBuildReferenceFrameFromTargetRelativeState(baselineTargetRelativeState, baselineReferenceFrame, baselineGoal))
		{
			outError = std::string(label) + " reference-frame smoke failed to build the projected reference pose.";
			return false;
		}

		if (!camera->manipulate({}, &referenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to apply the reference pose through manipulate({}, &referenceFrame).";
			return false;
		}

		ICamera::SphericalTargetState actualState = {};
			if (!camera->tryGetSphericalTargetState(actualState) ||
			!hlsl::CCameraMathUtilities::nearlyEqualVec3(
				actualState.target,
				desiredState.target,
				nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance) ||
			hlsl::CCameraMathUtilities::getWrappedAngleDistanceRadians(actualState.orbitUv.x, desiredState.orbitUv.x) >
				hlsl::radians(nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg) ||
			hlsl::CCameraMathUtilities::getWrappedAngleDistanceRadians(actualState.orbitUv.y, desiredState.orbitUv.y) >
				hlsl::radians(nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg) ||
			hlsl::abs(static_cast<double>(actualState.distance - desiredState.distance)) >
				nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance)
		{
			std::ostringstream oss;
			oss << label
				<< " reference-frame smoke produced the wrong spherical state:"
				<< " actual_target=(" << actualState.target.x << "," << actualState.target.y << "," << actualState.target.z << ")"
				<< " expected_target=(" << desiredState.target.x << "," << desiredState.target.y << "," << desiredState.target.z << ")"
				<< " actual_orbit=(" << actualState.orbitUv.x << "," << actualState.orbitUv.y << ")"
				<< " expected_orbit=(" << desiredState.orbitUv.x << "," << desiredState.orbitUv.y << ")"
				<< " actual_distance=" << actualState.distance
				<< " expected_distance=" << desiredState.distance;
			outError = oss.str();
			return false;
		}

		expectedGoal.hasTargetPosition = false;
		expectedGoal.hasDistance = false;
		expectedGoal.hasOrbitState = false;

		const auto capture = state.goalSolver.captureDetailed(camera);
		if (!capture.canUseGoal() ||
			!nbl::core::CCameraGoalUtilities::compareGoals(
				capture.goal,
				expectedGoal,
				nbl::system::SCameraSmokeComparisonThresholds::StrictPositionTolerance,
				nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
				nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance))
		{
			outError = std::string(label) + " reference-frame smoke produced the wrong projected goal: " +
				(capture.canUseGoal() ? nbl::core::CCameraGoalUtilities::describeGoalMismatch(capture.goal, expectedGoal) : std::string("goal_state=unavailable"));
			return false;
		}

		if (!camera->manipulate({}, &baselineReferenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to restore the baseline reference pose through manipulate({}, &referenceFrame).";
			return false;
		}

		return true;
	}

	inline bool verifyReferenceFramePoseApply(
		const SCameraSmokeResolvedState& state,
		ICamera* const camera,
		const hlsl::float64_t3& desiredPosition,
		const hlsl::camera_quaternion_t<hlsl::float64_t>& desiredOrientation,
		std::string_view label,
		std::string& outError)
	{
		const auto baselineCapture = state.goalSolver.captureDetailed(camera);
		if (!baselineCapture.canUseGoal())
		{
			outError = std::string(label) + " reference-frame smoke failed to capture the baseline pose.";
			return false;
		}

		nbl::core::CCameraGoal expectedGoal = {};
		expectedGoal.position = desiredPosition;
		expectedGoal.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(desiredOrientation);

		const auto baselineReferenceFrame = hlsl::CCameraMathUtilities::composeTransformMatrix(
			baselineCapture.goal.position,
			baselineCapture.goal.orientation);
		const auto referenceFrame = hlsl::CCameraMathUtilities::composeTransformMatrix(
			desiredPosition,
			expectedGoal.orientation);
		if (!camera->manipulate({}, &referenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to apply the rigid reference pose through manipulate({}, &referenceFrame).";
			return false;
		}

		const auto capture = state.goalSolver.captureDetailed(camera);
		if (!capture.canUseGoal() ||
			!nbl::core::CCameraGoalUtilities::compareGoals(
				capture.goal,
				expectedGoal,
				nbl::system::SCameraSmokeComparisonThresholds::StrictPositionTolerance,
				nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
				nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance))
		{
			outError = std::string(label) + " reference-frame smoke produced the wrong rigid goal: " +
				(capture.canUseGoal() ? nbl::core::CCameraGoalUtilities::describeGoalMismatch(capture.goal, expectedGoal) : std::string("goal_state=unavailable"));
			return false;
		}

		if (!camera->manipulate({}, &baselineReferenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to restore the baseline rigid pose through manipulate({}, &referenceFrame).";
			return false;
		}

		return true;
	}

	inline bool verifyReferenceFrameSupportSmoke(
		const SCameraSmokeResolvedState& state,
		std::string& outError)
	{
		if (state.fpsCamera)
		{
			if (!verifyReferenceFramePoseApply(
					state,
					state.fpsCamera,
					hlsl::float64_t3(2.5, -0.75, 4.0),
					hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(hlsl::float64_t3(-20.0, 35.0, 0.0)),
					"FPS",
					outError))
			{
				return false;
			}
		}

		if (state.freeCamera)
		{
			if (!verifyReferenceFramePoseApply(
					state,
					state.freeCamera,
					hlsl::float64_t3(-1.25, 0.5, 3.5),
					hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(hlsl::float64_t3(15.0, 45.0, 20.0)),
					"Free",
					outError))
			{
				return false;
			}
		}

		const auto verifySphericalReference = [&](ICamera* const camera, std::string_view label, const auto& mutateDesiredState) -> bool
		{
			if (!camera)
				return true;

			ICamera::SphericalTargetState baselineState = {};
			if (!camera->tryGetSphericalTargetState(baselineState))
			{
				outError = std::string(label) + " reference-frame smoke failed to query the baseline spherical state.";
				return false;
			}

			nbl::core::SCameraTargetRelativeState desiredState = {
				.target = baselineState.target,
				.orbitUv = baselineState.orbitUv,
				.distance = chooseShiftedReferenceDistance(baselineState)
			};
			mutateDesiredState(desiredState);
			return verifyReferenceFrameGoalApply(state, camera, desiredState, label, outError);
		};

		if (!verifySphericalReference(state.orbitCamera, "Orbit", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv += hlsl::float64_t2(0.45, -0.25);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.arcballCamera, "Arcball", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv += hlsl::float64_t2(0.35, 0.2);
				desiredState.orbitUv.y = std::clamp(
					desiredState.orbitUv.y,
					-static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::ArcballPitchLimitRad),
					static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::ArcballPitchLimitRad));
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.turntableCamera, "Turntable", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv += hlsl::float64_t2(-0.4, 0.18);
				desiredState.orbitUv.y = std::clamp(
					desiredState.orbitUv.y,
					-static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::TurntablePitchLimitRad),
					static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::TurntablePitchLimitRad));
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.topDownCamera, "TopDown", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv = hlsl::float64_t2(
					desiredState.orbitUv.x + 0.6,
					nbl::core::SCameraTargetRelativeRigDefaults::TopDownPitchRad);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.isometricCamera, "Isometric", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv = hlsl::float64_t2(
					nbl::core::SCameraTargetRelativeRigDefaults::IsometricYawRad,
					nbl::core::SCameraTargetRelativeRigDefaults::IsometricPitchRad);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.chaseCamera, "Chase", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv += hlsl::float64_t2(0.3, 0.15);
				desiredState.orbitUv.y = std::clamp(
					desiredState.orbitUv.y,
					static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::ChaseMinPitchRad),
					static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::ChaseMaxPitchRad));
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.dollyCamera, "Dolly", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv += hlsl::float64_t2(-0.3, -0.22);
				desiredState.orbitUv.y = std::clamp(
					desiredState.orbitUv.y,
					-static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::DollyPitchLimitRad),
					static_cast<double>(nbl::core::SCameraTargetRelativeRigDefaults::DollyPitchLimitRad));
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.dollyZoomCamera, "DollyZoom", [&](nbl::core::SCameraTargetRelativeState& desiredState)
			{
				desiredState.orbitUv += hlsl::float64_t2(0.28, -0.14);
			}))
		{
			return false;
		}

		if (state.pathCamera)
		{
			ICamera::PathState baselinePathState = {};
			ICamera::PathStateLimits pathLimits = {};
			ICamera::SphericalTargetState sphericalState = {};
			if (!state.pathCamera->tryGetPathState(baselinePathState) ||
				!state.pathCamera->tryGetPathStateLimits(pathLimits) ||
				!state.pathCamera->tryGetSphericalTargetState(sphericalState))
			{
				outError = "Path reference-frame smoke failed to query the baseline typed state.";
				return false;
			}

			ICamera::PathState desiredPathState = {};
			ICamera::PathState projectedPathState = {};
			const auto pathDelta = nbl::core::CCameraPathUtilities::makePathDeltaFromVirtualPathMotion(
				hlsl::float64_t3(0.8, 0.35, 1.1),
				hlsl::float64_t3(0.0, 0.0, 0.45));
			if (!nbl::core::CCameraPathUtilities::tryApplyPathStateDelta(
					baselinePathState,
					pathDelta,
					pathLimits,
					desiredPathState))
			{
				outError = "Path reference-frame smoke failed to build the desired typed path state.";
				return false;
			}

			nbl::core::SCameraCanonicalPathState canonicalPathState = {};
			nbl::core::SCameraCanonicalPathState baselineCanonicalPathState = {};
			nbl::core::CCameraGoal expectedGoal = {};
			if (!nbl::core::CCameraPathUtilities::tryBuildCanonicalPathState(
					sphericalState.target,
					desiredPathState,
					pathLimits,
					canonicalPathState) ||
				!nbl::core::CCameraPathUtilities::tryResolvePathState(
					sphericalState.target,
					canonicalPathState.pose.position,
					pathLimits,
					nullptr,
					projectedPathState) ||
				!nbl::core::CCameraPathUtilities::tryBuildCanonicalPathState(
					sphericalState.target,
					baselinePathState,
					pathLimits,
					baselineCanonicalPathState) ||
				!nbl::core::CCameraGoalUtilities::applyCanonicalPathGoalFields(
					expectedGoal,
					sphericalState.target,
					projectedPathState,
					pathLimits))
			{
				outError = "Path reference-frame smoke failed to build the canonical target-relative path pose.";
				return false;
			}

			const auto baselineReferenceFrame = hlsl::CCameraMathUtilities::composeTransformMatrix(
				baselineCanonicalPathState.pose.position,
				baselineCanonicalPathState.pose.orientation);
			const auto referenceFrame = hlsl::CCameraMathUtilities::composeTransformMatrix(
				canonicalPathState.pose.position,
				canonicalPathState.pose.orientation);
			if (!state.pathCamera->manipulate({}, &referenceFrame))
			{
				outError = "Path reference-frame smoke failed to apply the projected path pose through manipulate({}, &referenceFrame).";
				return false;
			}

			const auto capture = state.goalSolver.captureDetailed(state.pathCamera);
			if (!capture.canUseGoal() ||
				!nbl::core::CCameraGoalUtilities::compareGoals(
					capture.goal,
					expectedGoal,
					nbl::system::SCameraSmokeComparisonThresholds::StrictPositionTolerance,
					nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
					nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance))
			{
				outError = "Path reference-frame smoke produced the wrong projected goal: " +
					(capture.canUseGoal() ? nbl::core::CCameraGoalUtilities::describeGoalMismatch(capture.goal, expectedGoal) : std::string("goal_state=unavailable"));
				return false;
			}

			if (!state.pathCamera->manipulate({}, &baselineReferenceFrame))
			{
				outError = "Path reference-frame smoke failed to restore the baseline reference pose through manipulate({}, &referenceFrame).";
				return false;
			}
		}

		return true;
	}

	inline bool verifyPersistenceAndPlaybackSmoke(
		const SCameraSmokeResolvedState& state,
		std::string& outError)
	{
		auto sourcePresets = collectAvailableSmokePresets(state.initialPresets);
		if (sourcePresets.empty())
		{
			outError = "Preset persistence smoke failed to collect source presets.";
			return false;
		}

		const auto sourcePresetSpan = std::span<const CameraPreset>(sourcePresets.data(), sourcePresets.size());

		std::stringstream presetBuffer;
		if (!nbl::system::writePresetCollection(presetBuffer, sourcePresetSpan))
		{
			outError = "Preset persistence smoke failed to serialize preset collection.";
			return false;
		}

		std::vector<CameraPreset> loadedPresets;
		if (!nbl::system::readPresetCollection(presetBuffer, loadedPresets))
		{
			outError = "Preset persistence smoke failed to deserialize preset collection.";
			return false;
		}
		if (!nbl::core::CCameraPresetUtilities::comparePresetCollections(
				sourcePresetSpan,
				std::span<const CameraPreset>(loadedPresets.data(), loadedPresets.size()),
				SCameraSmokePersistenceThresholds::PositionTolerance,
				SCameraSmokePersistenceThresholds::AngularToleranceDeg,
				SCameraSmokePersistenceThresholds::ScalarTolerance))
		{
			outError = "Preset persistence smoke changed stream preset collection content.";
			return false;
		}

		CCameraKeyframeTrack sourceTrack;
		sourceTrack.keyframes.reserve(sourcePresets.size());
		for (size_t i = 0u; i < sourcePresets.size(); ++i)
		{
			nbl::core::CCameraKeyframe keyframe;
			keyframe.time = static_cast<float>(i) * 1.5f;
			keyframe.preset = sourcePresets[i];
			sourceTrack.keyframes.emplace_back(std::move(keyframe));
		}
		sourceTrack.selectedKeyframeIx = static_cast<int>(sourceTrack.keyframes.size()) - 1;

		std::stringstream keyframeBuffer;
		if (!nbl::system::writeKeyframeTrack(keyframeBuffer, sourceTrack))
		{
			outError = "Keyframe persistence smoke failed to serialize track.";
			return false;
		}

		CCameraKeyframeTrack loadedTrack;
		if (!nbl::system::readKeyframeTrack(keyframeBuffer, loadedTrack))
		{
			outError = "Keyframe persistence smoke failed to deserialize track.";
			return false;
		}
		if (!nbl::system::CCameraSmokeRegressionUtilities::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, loadedTrack))
		{
			outError = "Keyframe persistence smoke changed stream track content.";
			return false;
		}

		struct TempFileCleanup final
		{
			std::vector<std::filesystem::path> paths;

			~TempFileCleanup()
			{
				std::error_code ec;
				for (const auto& path : paths)
					std::filesystem::remove(path, ec);
			}
		} tempFiles;

		const auto uniqueSuffix = std::to_string(static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
		const auto tempDir = std::filesystem::temp_directory_path();
		const auto presetFile = tempDir / ("nabla_cameraz_presets_" + uniqueSuffix + ".json");
		const auto keyframeFile = tempDir / ("nabla_cameraz_keyframes_" + uniqueSuffix + ".json");
		tempFiles.paths = { presetFile, keyframeFile };

		if (!state.system)
		{
			outError = "Persistence smoke is missing a valid system interface.";
			return false;
		}

		auto& system = *state.system;

		if (!nbl::system::savePresetCollectionToFile(system, presetFile, sourcePresetSpan))
		{
			outError = "Preset persistence smoke failed to save preset collection file.";
			return false;
		}

		std::vector<CameraPreset> fileLoadedPresets;
		if (!nbl::system::loadPresetCollectionFromFile(system, presetFile, fileLoadedPresets))
		{
			outError = "Preset persistence smoke failed to load preset collection file.";
			return false;
		}
		if (!nbl::core::CCameraPresetUtilities::comparePresetCollections(
				sourcePresetSpan,
				std::span<const CameraPreset>(fileLoadedPresets.data(), fileLoadedPresets.size()),
				SCameraSmokePersistenceThresholds::PositionTolerance,
				SCameraSmokePersistenceThresholds::AngularToleranceDeg,
				SCameraSmokePersistenceThresholds::ScalarTolerance))
		{
			outError = "Preset persistence smoke changed file preset collection content.";
			return false;
		}

		if (!nbl::system::saveKeyframeTrackToFile(system, keyframeFile, sourceTrack))
		{
			outError = "Keyframe persistence smoke failed to save track file.";
			return false;
		}

		CCameraKeyframeTrack fileLoadedTrack;
		if (!nbl::system::loadKeyframeTrackFromFile(system, keyframeFile, fileLoadedTrack))
		{
			outError = "Keyframe persistence smoke failed to load track file.";
			return false;
		}
		if (!nbl::system::CCameraSmokeRegressionUtilities::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, fileLoadedTrack))
		{
			outError = "Keyframe persistence smoke changed file track content.";
			return false;
		}

		if (state.initialPresets.orbit.has_value() && state.initialPresets.dolly.has_value())
		{
			CCameraKeyframeTrack playbackTrack;
			{
				nbl::core::CCameraKeyframe keyframe;
				keyframe.time = 0.f;
				keyframe.preset = state.initialPresets.orbit.value();
				playbackTrack.keyframes.push_back(keyframe);
			}
			{
				nbl::core::CCameraKeyframe keyframe;
				keyframe.time = SCameraSmokePlaybackDefaults::EndKeyframeTime;
				keyframe.preset = state.initialPresets.dolly.value();
				playbackTrack.keyframes.push_back(keyframe);
			}

			CCameraPlaybackCursor cursor = {
				.playing = true,
				.loop = false,
				.speed = 1.f,
				.time = SCameraSmokePlaybackDefaults::MidPlaybackTime
			};

			const auto advanceToEnd = nbl::core::CCameraPlaybackTimelineUtilities::advancePlaybackCursor(cursor, playbackTrack, SCameraSmokePlaybackDefaults::AdvanceDt);
			if (!advanceToEnd.hasTrack || !advanceToEnd.changedTime || !advanceToEnd.reachedEnd || advanceToEnd.wrapped || !advanceToEnd.stopped)
			{
				outError = "Playback timeline smoke failed for non-loop end-of-track advance.";
				return false;
			}
			if (hlsl::abs(static_cast<double>(advanceToEnd.time - SCameraSmokePlaybackDefaults::EndKeyframeTime)) > CameraTinyScalarEpsilon)
			{
				outError = "Playback timeline smoke produced wrong end-of-track time.";
				return false;
			}

			nbl::core::CCameraPlaybackTimelineUtilities::resetPlaybackCursor(cursor, SCameraSmokePlaybackDefaults::ResetPlaybackTime);
			if (cursor.playing || hlsl::abs(static_cast<double>(cursor.time - SCameraSmokePlaybackDefaults::ResetPlaybackTime)) > CameraTinyScalarEpsilon)
			{
				outError = "Playback timeline smoke failed to reset cursor.";
				return false;
			}

			cursor.playing = true;
			cursor.loop = true;
			cursor.speed = 1.f;
			cursor.time = SCameraSmokePlaybackDefaults::MidPlaybackTime;
			const auto advanceLoop = nbl::core::CCameraPlaybackTimelineUtilities::advancePlaybackCursor(cursor, playbackTrack, SCameraSmokePlaybackDefaults::AdvanceDt);
			if (!advanceLoop.hasTrack || !advanceLoop.changedTime || !advanceLoop.wrapped || advanceLoop.stopped || advanceLoop.reachedEnd)
			{
				outError = "Playback timeline smoke failed for looped advance.";
				return false;
			}
			if (hlsl::abs(static_cast<double>(advanceLoop.time - SCameraSmokePlaybackDefaults::WrappedPlaybackTime)) > CameraTinyScalarEpsilon)
			{
				outError = "Playback timeline smoke produced wrong wrapped time.";
				return false;
			}

			cursor.time = SCameraSmokePlaybackDefaults::OvershootPlaybackTime;
			nbl::core::CCameraPlaybackTimelineUtilities::clampPlaybackCursorToTrack(playbackTrack, cursor);
			if (hlsl::abs(static_cast<double>(cursor.time - SCameraSmokePlaybackDefaults::EndKeyframeTime)) > CameraTinyScalarEpsilon)
			{
				outError = "Playback timeline smoke failed to clamp cursor time.";
				return false;
			}
		}

		return true;
	}

	inline bool verifySequenceCompileSmoke(
		const SCameraSmokeResolvedState& state,
		std::string& outError)
	{
		if (!state.initialPresets.orbit.has_value())
			return true;

		CCameraSequenceScript sequence;
		sequence.fps = SCameraSmokeSequenceDefaults::Fps;
		sequence.defaults.durationSeconds = SCameraSmokeSequenceDefaults::DurationSeconds;
		sequence.defaults.presentations = {
			{ .projection = IPlanarProjection::CProjection::Perspective, .leftHanded = true },
			{ .projection = IPlanarProjection::CProjection::Orthographic, .leftHanded = false }
		};
		sequence.defaults.captureFractions = { SCameraSmokeSequenceDefaults::CaptureFractions[0], SCameraSmokeSequenceDefaults::CaptureFractions[1], SCameraSmokeSequenceDefaults::CaptureFractions[2] };

		CCameraSequenceSegment segment;
		segment.name = "sequence_compile_smoke";
		segment.cameraKind = ICamera::CameraKind::Orbit;
		{
			CCameraSequenceKeyframe keyframe;
			keyframe.time = 0.f;
			keyframe.hasAbsolutePreset = true;
			keyframe.absolutePreset = state.initialPresets.orbit.value();
			segment.keyframes.push_back(keyframe);
		}
		for (const auto& [time, position] : {
				std::pair{ 0.0f, SCameraSmokeSequenceDefaults::TargetPositionA },
				std::pair{ SCameraSmokeSequenceDefaults::SecondKeyframeTime, SCameraSmokeSequenceDefaults::TargetPositionB },
				std::pair{ SCameraSmokeSequenceDefaults::SecondKeyframeTime, SCameraSmokeSequenceDefaults::TargetPositionC } })
		{
			nbl::core::CCameraSequenceTrackedTargetKeyframe keyframe;
			keyframe.time = time;
			keyframe.hasAbsolutePosition = true;
			keyframe.absolutePosition = position;
			segment.targetKeyframes.push_back(keyframe);
		}
		sequence.segments.push_back(segment);

		if (!nbl::core::CCameraSequenceScriptUtilities::sequenceScriptUsesMultiplePresentations(sequence))
		{
			outError = "Sequence compile smoke failed to detect multi-presentation authored defaults.";
			return false;
		}

		CCameraSequenceTrackedTargetPose referenceTrackedTargetPose = {};
		referenceTrackedTargetPose.position = SCameraAppSceneDefaults::DefaultFollowTargetPosition;
		referenceTrackedTargetPose.orientation = SCameraAppSceneDefaults::DefaultFollowTargetOrientation;

		nbl::core::CCameraSequenceCompiledSegment compiledSegment;
		std::string compileError;
		if (!nbl::core::CCameraSequenceScriptUtilities::compileSequenceSegmentFromReference(
				sequence,
				sequence.segments.front(),
				state.initialPresets.orbit.value(),
				referenceTrackedTargetPose,
				compiledSegment,
				&compileError))
		{
			outError = "Sequence compile smoke failed to compile a shared segment. " + compileError;
			return false;
		}

		if (compiledSegment.durationFrames != SCameraSmokeSequenceDefaults::DurationFrames ||
			compiledSegment.sampleTimes.size() != SCameraSmokeSequenceDefaults::DurationFrames)
		{
			outError = "Sequence compile smoke produced wrong sampled frame count.";
			return false;
		}
		if (compiledSegment.captureFrameOffsets != std::vector<uint64_t>(
				SCameraSmokeSequenceDefaults::CaptureFrameOffsets.begin(),
				SCameraSmokeSequenceDefaults::CaptureFrameOffsets.end()))
		{
			outError = "Sequence compile smoke produced wrong capture frame offsets.";
			return false;
		}
		if (compiledSegment.presentations.size() != 2u)
		{
			outError = "Sequence compile smoke lost authored presentations.";
			return false;
		}
		if (!compiledSegment.usesTrackedTargetTrack() || compiledSegment.trackedTargetTrack.keyframes.size() != 2u)
		{
			outError = "Sequence compile smoke failed to normalize tracked-target keyframes.";
			return false;
		}

		std::vector<nbl::core::CCameraSequenceCompiledFramePolicy> framePolicies;
		if (!nbl::core::CCameraSequenceScriptUtilities::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, true))
		{
			outError = "Sequence compile smoke failed to build shared frame policies.";
			return false;
		}
		if (framePolicies.size() != SCameraSmokeSequenceDefaults::DurationFrames)
		{
			outError = "Sequence compile smoke produced wrong frame-policy count.";
			return false;
		}
		if (!framePolicies[0].baseline || framePolicies[0].continuityStep || !framePolicies[0].capture)
		{
			outError = "Sequence compile smoke produced wrong first-frame policy.";
			return false;
		}
		if (!framePolicies[1].continuityStep || !framePolicies[1].followTargetLock || framePolicies[1].baseline)
		{
			outError = "Sequence compile smoke produced wrong continuity follow policy.";
			return false;
		}
		if (!framePolicies[4].capture || !framePolicies[7].capture)
		{
			outError = "Sequence compile smoke produced wrong capture milestone policy.";
			return false;
		}

		CCameraSequenceTrackedTargetPose poseAtOne;
		if (!nbl::core::CCameraSequenceScriptUtilities::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, 1.f, poseAtOne))
		{
			outError = "Sequence compile smoke failed to sample normalized tracked-target track.";
			return false;
		}
		if (length(poseAtOne.position - SCameraSmokeSequenceDefaults::TargetPositionC) > CameraTinyScalarEpsilon)
		{
			outError = "Sequence compile smoke did not keep the last authored target pose for duplicate keyframe time.";
			return false;
		}

		CCameraScriptedTimeline scriptedTimeline;
		std::string runtimeBuildError;
		if (!nbl::system::CCameraSequenceScriptedBuilderUtilities::appendCompiledSequenceSegmentToScriptedTimeline(
				scriptedTimeline,
				SCameraSmokeSequenceDefaults::StartFrame,
				compiledSegment,
				{
					.planarIx = SCameraSmokeSequenceDefaults::PlanarIx,
					.availableWindowCount = SCameraSmokeSequenceDefaults::AvailableWindowCount,
					.useWindow = true,
					.includeFollowTargetLock = true
				},
				&runtimeBuildError))
		{
			outError = "Sequence runtime builder smoke failed to append a compiled segment. " + runtimeBuildError;
			return false;
		}
		nbl::system::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(scriptedTimeline);

		if (scriptedTimeline.captureFrames != std::vector<uint64_t>(
				SCameraSmokeSequenceDefaults::CaptureFrames.begin(),
				SCameraSmokeSequenceDefaults::CaptureFrames.end()))
		{
			outError = "Sequence runtime builder smoke produced wrong capture frames.";
			return false;
		}

		size_t baselineChecks = 0u;
		size_t stepChecks = 0u;
		size_t followChecks = 0u;
		for (const auto& check : scriptedTimeline.checks)
		{
			switch (check.kind)
			{
				case CCameraScriptedInputCheck::Kind::Baseline:
					++baselineChecks;
					break;
				case CCameraScriptedInputCheck::Kind::GimbalStep:
					++stepChecks;
					break;
				case CCameraScriptedInputCheck::Kind::FollowTargetLock:
					++followChecks;
					break;
				default:
					break;
			}
		}
		if (baselineChecks != SCameraSmokeSequenceDefaults::BaselineCheckCount ||
			stepChecks != SCameraSmokeSequenceDefaults::ContinuityCheckCount ||
			followChecks != SCameraSmokeSequenceDefaults::FollowCheckCount)
		{
			outError = "Sequence runtime builder smoke produced wrong scripted check counts.";
			return false;
		}

		size_t runtimeNextEventIndex = 0u;
		CCameraScriptedFrameEvents runtimeBatch;
		nbl::system::CCameraScriptedFrameEventUtilities::dequeueScriptedFrameEvents(scriptedTimeline.events, runtimeNextEventIndex, SCameraSmokeSequenceDefaults::StartFrame, runtimeBatch);
		if (runtimeBatch.actions.size() != 10u || runtimeBatch.goals.size() != 1u ||
			runtimeBatch.trackedTargetTransforms.size() != 1u || runtimeBatch.segmentLabels.size() != 1u)
		{
			outError = "Sequence runtime builder smoke produced wrong first-frame batch.";
			return false;
		}
		if (runtimeBatch.actions.front().kind != CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow ||
			runtimeBatch.segmentLabels.front() != "sequence_compile_smoke")
		{
			outError = "Sequence runtime builder smoke lost first-frame scripted payload.";
			return false;
		}

		return true;
	}

	inline bool verifyRangeAndUtilitySmoke(
		const SCameraSmokeResolvedState& state,
		std::string& outError)
	{
		if (state.initialPresets.orbit.has_value() && state.orbitCamera)
		{
			std::array<ICamera*, 2u> exactTargets = { state.orbitCamera, nullptr };
			const auto exactSummary = nbl::core::CCameraPresetFlowUtilities::applyPresetToCameraRange(
				state.goalSolver,
				std::span<ICamera* const>(exactTargets.data(), exactTargets.size()),
				state.initialPresets.orbit.value());
			if (exactSummary.targetCount != 1u || exactSummary.successCount != 1u || exactSummary.approximateCount != 0u || exactSummary.failureCount != 0u)
			{
				outError = "Preset apply summary smoke failed for exact target range.";
				return false;
			}
		}

		if (state.initialPresets.path.has_value() && state.orbitCamera)
		{
			std::array<ICamera*, 1u> approximateTargets = { state.orbitCamera };
			const auto approximateSummary = nbl::core::CCameraPresetFlowUtilities::applyPresetToCameraRange(
				state.goalSolver,
				std::span<ICamera* const>(approximateTargets.data(), approximateTargets.size()),
				state.initialPresets.path.value());
			if (approximateSummary.targetCount != 1u || approximateSummary.successCount != 1u || approximateSummary.approximateCount != 1u || approximateSummary.failureCount != 0u)
			{
				outError = "Preset apply summary smoke failed for approximate target range.";
				return false;
			}
		}

		if (state.initialPresets.path.has_value() && state.pathCamera)
		{
			if (!restorePresetStrict(
					state.goalSolver,
					state.pathCamera,
					state.initialPresets.path.value(),
					"Path manipulation smoke failed to restore the baseline preset",
					outError))
			{
				return false;
			}

			ICamera::PathState baselinePathState = {};
			if (!state.pathCamera->tryGetPathState(baselinePathState))
			{
				outError = "Path manipulation smoke failed to read the baseline path state.";
				return false;
			}

			const hlsl::float64_t3 directTranslationMagnitude(1.5, 0.75, 2.0);
			const double directRollMagnitude = 0.5;
			const std::array<CVirtualGimbalEvent, 4u> directPathEvents = {{
				{ CVirtualGimbalEvent::MoveRight, directTranslationMagnitude.x },
				{ CVirtualGimbalEvent::MoveUp, directTranslationMagnitude.y },
				{ CVirtualGimbalEvent::MoveForward, directTranslationMagnitude.z },
				{ CVirtualGimbalEvent::RollRight, directRollMagnitude }
			}};

			if (!state.pathCamera->manipulate({ directPathEvents.data(), directPathEvents.size() }))
			{
				outError = "Path manipulation smoke failed to apply direct path virtual events.";
				return false;
			}

			ICamera::PathState manipulatedPathState = {};
			if (!state.pathCamera->tryGetPathState(manipulatedPathState))
			{
				outError = "Path manipulation smoke failed to read the manipulated path state.";
				return false;
			}

			ICamera::PathStateLimits activePathLimits = nbl::core::CCameraPathUtilities::makeDefaultPathLimits();
			state.pathCamera->tryGetPathStateLimits(activePathLimits);
			const auto expectedPathDelta = nbl::core::CCameraPathUtilities::makePathDeltaFromVirtualPathMotion(
				state.pathCamera->scaleVirtualTranslation(directTranslationMagnitude),
				state.pathCamera->scaleVirtualRotation(hlsl::float64_t3(0.0, 0.0, directRollMagnitude)));
			ICamera::PathState expectedPathState = {};
			if (!nbl::core::CCameraPathUtilities::tryApplyPathStateDelta(
					baselinePathState,
					expectedPathDelta,
					activePathLimits,
					expectedPathState) ||
				!nbl::core::CCameraPathUtilities::pathStatesNearlyEqual(
					manipulatedPathState,
					expectedPathState,
					nbl::core::SCameraPathDefaults::ExactComparisonThresholds))
			{
				outError = "Path manipulation smoke changed the default s/u/v/roll runtime mapping.";
				return false;
			}

			const auto movedCapture = state.goalSolver.captureDetailed(state.pathCamera);
			if (!movedCapture.canUseGoal())
			{
				outError = "Path manipulation smoke failed to capture the moved path goal.";
				return false;
			}

			if (!restorePresetStrict(
					state.goalSolver,
					state.pathCamera,
					state.initialPresets.path.value(),
					"Path manipulation smoke failed to reset the baseline preset before replay",
					outError))
			{
				return false;
			}

			std::vector<CVirtualGimbalEvent> replayEvents;
			if (!state.goalSolver.buildEvents(state.pathCamera, movedCapture.goal, replayEvents) || replayEvents.empty())
			{
				outError = "Path manipulation smoke failed to build replay virtual events for the moved path goal.";
				return false;
			}

			bool hasRollReplay = false;
			for (const auto& event : replayEvents)
			{
				if (event.type == CVirtualGimbalEvent::RollLeft || event.type == CVirtualGimbalEvent::RollRight)
				{
					hasRollReplay = true;
					break;
				}
			}
			if (!hasRollReplay)
			{
				outError = "Path manipulation smoke dropped the roll replay event for the moved path goal.";
				return false;
			}

			if (!state.pathCamera->manipulate({ replayEvents.data(), replayEvents.size() }))
			{
				outError = "Path manipulation smoke failed to replay path virtual events onto the baseline camera.";
				return false;
			}

			const auto replayCapture = state.goalSolver.captureDetailed(state.pathCamera);
			if (!replayCapture.canUseGoal() ||
				!nbl::core::CCameraGoalUtilities::compareGoals(
					replayCapture.goal,
					movedCapture.goal,
					nbl::system::SCameraSmokeComparisonThresholds::StrictPositionTolerance,
					nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
					nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance))
			{
				outError = "Path manipulation smoke failed the goal -> events -> manipulate replay roundtrip.";
				return false;
			}

			if (!restorePresetStrict(
					state.goalSolver,
					state.pathCamera,
					state.initialPresets.path.value(),
					"Path manipulation smoke failed to restore the baseline preset after replay",
					outError))
			{
				return false;
			}

			if (!state.initialPresets.path->goal.hasTargetPosition)
			{
				outError = "Path manipulation smoke is missing the baseline path target state for custom path-limit validation.";
				return false;
			}

			const auto defaultPathModel = nbl::core::CCameraPathUtilities::makeDefaultPathModel();
			nbl::core::CPathCamera::path_model_t incompletePathModel = {};
			incompletePathModel.resolveState = defaultPathModel.resolveState;

			ICamera::PathStateLimits customPathLimits = {
				.minU = 2.0,
				.minDistance = 2.0,
				.maxDistance = 3.0
			};
			auto customPathCamera = nbl::core::make_smart_refctd_ptr<nbl::core::CPathCamera>(
				state.initialPresets.path->goal.position,
				state.initialPresets.path->goal.targetPosition,
				std::move(incompletePathModel),
				customPathLimits);

			const auto& customPathModel = customPathCamera->getPathModel();
			if (!customPathModel.resolveState || !customPathModel.controlLaw || !customPathModel.integrate || !customPathModel.evaluate || !customPathModel.updateDistance)
			{
				outError = "Path manipulation smoke left a partially initialized path model active after constructor fallback.";
				return false;
			}

			ICamera::PathStateLimits resolvedPathLimits = {};
			if (!customPathCamera->tryGetPathStateLimits(resolvedPathLimits) ||
				hlsl::abs(resolvedPathLimits.minU - customPathLimits.minU) > CameraTinyScalarEpsilon ||
				hlsl::abs(resolvedPathLimits.minDistance - customPathLimits.minDistance) > CameraTinyScalarEpsilon ||
				hlsl::abs(resolvedPathLimits.maxDistance - customPathLimits.maxDistance) > CameraTinyScalarEpsilon)
			{
				outError = "Path manipulation smoke failed to expose custom per-camera path limits.";
				return false;
			}

			ICamera::SphericalTargetState customSphericalState = {};
			if (!customPathCamera->tryGetSphericalTargetState(customSphericalState) ||
				hlsl::abs(static_cast<double>(customSphericalState.minDistance) - resolvedPathLimits.minDistance) > CameraTinyScalarEpsilon ||
				hlsl::abs(static_cast<double>(customSphericalState.maxDistance) - resolvedPathLimits.maxDistance) > CameraTinyScalarEpsilon)
			{
				outError = "Path manipulation smoke failed to surface path limits through spherical target state.";
				return false;
			}

			ICamera::PathState customBaselinePathState = {};
			if (!customPathCamera->tryGetPathState(customBaselinePathState))
			{
				outError = "Path manipulation smoke failed to capture the custom path-camera baseline state.";
				return false;
			}

			const double customBaselineDistance = hlsl::CCameraMathUtilities::getPathDistance(customBaselinePathState.u, customBaselinePathState.v);
			if (customBaselineDistance + CameraTinyScalarEpsilon < resolvedPathLimits.minDistance ||
				customBaselineDistance - CameraTinyScalarEpsilon > resolvedPathLimits.maxDistance)
			{
				outError = "Path manipulation smoke failed to clamp the constructor-resolved path state to custom limits.";
				return false;
			}

			if (!customPathCamera->manipulate({ directPathEvents.data(), directPathEvents.size() }))
			{
				outError = "Path manipulation smoke failed to apply direct virtual events on the custom-limits path camera.";
				return false;
			}

			ICamera::PathState customManipulatedPathState = {};
			if (!customPathCamera->tryGetPathState(customManipulatedPathState))
			{
				outError = "Path manipulation smoke failed to read the manipulated custom-limits path state.";
				return false;
			}

			ICamera::PathState expectedCustomPathState = {};
			if (!nbl::core::CCameraPathUtilities::tryApplyPathStateDelta(
					customBaselinePathState,
					nbl::core::CCameraPathUtilities::makePathDeltaFromVirtualPathMotion(
						customPathCamera->scaleVirtualTranslation(directTranslationMagnitude),
						customPathCamera->scaleVirtualRotation(hlsl::float64_t3(0.0, 0.0, directRollMagnitude))),
					resolvedPathLimits,
					expectedCustomPathState) ||
				!nbl::core::CCameraPathUtilities::pathStatesNearlyEqual(
					customManipulatedPathState,
					expectedCustomPathState,
					nbl::core::SCameraPathDefaults::ExactComparisonThresholds))
			{
				outError = "Path manipulation smoke failed the custom-limits default runtime mapping check.";
				return false;
			}

			const auto customMovedCapture = state.goalSolver.captureDetailed(customPathCamera.get());
			if (!customMovedCapture.canUseGoal())
			{
				outError = "Path manipulation smoke failed to capture the moved custom-limits path goal.";
				return false;
			}

			if (!customPathCamera->trySetPathState(customBaselinePathState))
			{
				outError = "Path manipulation smoke failed to restore the custom-limits baseline path state.";
				return false;
			}

			std::vector<CVirtualGimbalEvent> customReplayEvents;
			if (!state.goalSolver.buildEvents(customPathCamera.get(), customMovedCapture.goal, customReplayEvents) || customReplayEvents.empty())
			{
				outError = "Path manipulation smoke failed to build replay events for the custom-limits path goal.";
				return false;
			}

			if (!customPathCamera->manipulate({ customReplayEvents.data(), customReplayEvents.size() }))
			{
				outError = "Path manipulation smoke failed to replay events on the custom-limits path camera.";
				return false;
			}

			const auto customReplayCapture = state.goalSolver.captureDetailed(customPathCamera.get());
			if (!customReplayCapture.canUseGoal() ||
				!nbl::core::CCameraGoalUtilities::compareGoals(
					customReplayCapture.goal,
					customMovedCapture.goal,
					nbl::system::SCameraSmokeComparisonThresholds::StrictPositionTolerance,
					nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
					nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance))
			{
				outError = "Path manipulation smoke failed the custom-limits goal replay roundtrip.";
				return false;
			}
		}

		{
			std::vector<CVirtualGimbalEvent> scaledEvents(3u);
			scaledEvents[0].type = CVirtualGimbalEvent::MoveForward;
			scaledEvents[0].magnitude = 2.0;
			scaledEvents[1].type = CVirtualGimbalEvent::PanRight;
			scaledEvents[1].magnitude = 3.0;
			scaledEvents[2].type = CVirtualGimbalEvent::ScaleXInc;
			scaledEvents[2].magnitude = 4.0;
			nbl::core::CCameraManipulationUtilities::scaleVirtualEvents(scaledEvents, static_cast<uint32_t>(scaledEvents.size()), 0.5f, 2.0f);
			if (hlsl::abs(scaledEvents[0].magnitude - 1.0) > SCameraSmokeUtilityThresholds::VirtualEventScale ||
				hlsl::abs(scaledEvents[1].magnitude - 6.0) > SCameraSmokeUtilityThresholds::VirtualEventScale ||
				hlsl::abs(scaledEvents[2].magnitude - 4.0) > SCameraSmokeUtilityThresholds::VirtualEventScale)
			{
				outError = "Camera manipulation utilities smoke failed for virtual-event scaling.";
				return false;
			}
		}

		{
			const auto findEventMagnitude = [](const auto& events, const CVirtualGimbalEvent::VirtualEventType type) -> std::optional<double>
			{
				for (const auto& event : events)
				{
					if (event.type == type)
						return event.magnitude;
				}
				return std::nullopt;
			};

			const auto sumEventMagnitude = [](const auto& events, const CVirtualGimbalEvent::VirtualEventType type) -> double
			{
				double sum = 0.0;
				for (const auto& event : events)
				{
					if (event.type == type)
						sum += event.magnitude;
				}
				return sum;
			};

			const auto frameStepSeconds = std::chrono::duration<double>(SCameraSmokeInputDefaults::EventStep).count();
			const auto expectedKeyboardMagnitude =
				frameStepSeconds * nbl::ui::CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond;

			nbl::ui::CGimbalInputBinder inputBinder;
			nbl::ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(
				inputBinder,
				ICamera::CameraKind::FPS,
				CVirtualGimbalEvent::All);

			const auto keyboardEvents = collectKeyboardVirtualEvents(inputBinder, nbl::ui::E_KEY_CODE::EKC_W);
			const auto keyboardMagnitude = findEventMagnitude(keyboardEvents, CVirtualGimbalEvent::MoveForward);
			if (!keyboardMagnitude.has_value() ||
				hlsl::abs(*keyboardMagnitude - expectedKeyboardMagnitude) > SCameraSmokeUtilityThresholds::VirtualEventScale)
			{
				outError = "Input binding smoke produced the wrong held-key magnitude for default FPS WASD.";
				return false;
			}

			inputBinder.clearBindingLayout();
			nbl::ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(
				inputBinder,
				ICamera::CameraKind::FPS,
				CVirtualGimbalEvent::All);

			const auto moveEvent = buildMovementSmokeMouseEvent();
			const std::array<SMouseEvent, 1u> moveEvents = { moveEvent };
			const auto mouseEvents = collectMouseVirtualEvents(inputBinder, { moveEvents.data(), moveEvents.size() });
			const auto panMagnitude = findEventMagnitude(mouseEvents, CVirtualGimbalEvent::PanRight);
			const auto tiltMagnitude = findEventMagnitude(mouseEvents, CVirtualGimbalEvent::TiltDown);
			if (!panMagnitude.has_value() ||
				!tiltMagnitude.has_value() ||
				hlsl::abs(*panMagnitude - static_cast<double>(SCameraSmokeInputDefaults::RelativeMouseMove)) > SCameraSmokeUtilityThresholds::VirtualEventScale ||
				hlsl::abs(*tiltMagnitude - hlsl::abs(static_cast<double>(SCameraSmokeInputDefaults::RelativeMouseMoveY))) > SCameraSmokeUtilityThresholds::VirtualEventScale)
			{
				outError = "Input binding smoke produced the wrong relative-mouse magnitudes for default FPS look.";
				return false;
			}

			inputBinder.clearBindingLayout();
			nbl::ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(
				inputBinder,
				ICamera::CameraKind::Orbit,
				CVirtualGimbalEvent::All);

			const auto scrollEvent = buildScrollSmokeMouseEvent();
			const std::array<SMouseEvent, 1u> scrollEvents = { scrollEvent };
			const auto mouseScrollEvents = collectMouseVirtualEvents(inputBinder, { scrollEvents.data(), scrollEvents.size() });
			const auto scrollForwardMagnitude = sumEventMagnitude(mouseScrollEvents, CVirtualGimbalEvent::MoveForward);
			if (hlsl::abs(scrollForwardMagnitude - static_cast<double>(SCameraSmokeInputDefaults::VerticalScroll + SCameraSmokeInputDefaults::HorizontalScroll)) > SCameraSmokeUtilityThresholds::VirtualEventScale)
			{
				outError = "Input binding smoke produced the wrong scroll magnitude for default orbit zoom.";
				return false;
			}

			nbl::ui::CGimbalBindingLayoutStorage customLayout;
			customLayout.updateKeyboardMapping([&](auto& map)
				{
					map[nbl::ui::E_KEY_CODE::EKC_W] = nbl::ui::IGimbalBindingLayout::CHashInfo(CVirtualGimbalEvent::MoveForward, 7.5);
				});
			inputBinder.copyBindingLayoutFrom(customLayout);

			const auto customKeyboardEvents = collectKeyboardVirtualEvents(inputBinder, nbl::ui::E_KEY_CODE::EKC_W);
			const auto customKeyboardMagnitude = findEventMagnitude(customKeyboardEvents, CVirtualGimbalEvent::MoveForward);
			const auto expectedCustomKeyboardMagnitude = frameStepSeconds * 7.5;
			if (!customKeyboardMagnitude.has_value() ||
				hlsl::abs(*customKeyboardMagnitude - expectedCustomKeyboardMagnitude) > SCameraSmokeUtilityThresholds::VirtualEventScale)
			{
				outError = "Input binding smoke failed to preserve binding-scale metadata through layout copies.";
				return false;
			}
		}

		if (state.initialPresets.free.has_value() && state.freeCamera)
		{
			CameraPreset orientedPreset = state.initialPresets.free.value();
			orientedPreset.goal.orientation = hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(SCameraSmokeManipulationDefaults::FreeOrientationYawDeg);
			const auto orientResult = nbl::core::CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.freeCamera, orientedPreset);
			if (!orientResult.succeeded() || !nbl::system::CCameraSmokeRegressionUtilities::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.freeCamera, orientedPreset))
			{
				outError = "Camera manipulation utilities smoke failed to orient Free camera before translation remap.";
				return false;
			}

			std::vector<CVirtualGimbalEvent> worldTranslationEvents(3u);
			worldTranslationEvents[0].type = CVirtualGimbalEvent::MoveRight;
			worldTranslationEvents[0].magnitude = SCameraSmokeManipulationDefaults::WorldTranslationDelta.x;
			worldTranslationEvents[1].type = CVirtualGimbalEvent::MoveUp;
			worldTranslationEvents[1].magnitude = SCameraSmokeManipulationDefaults::WorldTranslationDelta.y;
			worldTranslationEvents[2].type = CVirtualGimbalEvent::MoveForward;
			worldTranslationEvents[2].magnitude = SCameraSmokeManipulationDefaults::WorldTranslationDelta.z;
			uint32_t remappedCount = static_cast<uint32_t>(worldTranslationEvents.size());
			nbl::core::CCameraManipulationUtilities::remapTranslationEventsFromWorldToCameraLocal(state.freeCamera, worldTranslationEvents, remappedCount);
			if (remappedCount == 0u)
			{
				outError = "Camera manipulation utilities smoke produced empty translation remap.";
				return false;
			}

			if (!state.freeCamera->manipulate({ worldTranslationEvents.data(), remappedCount }))
			{
				outError = "Camera manipulation utilities smoke failed to apply remapped translation.";
				return false;
			}

			const auto remappedPosition = state.freeCamera->getGimbal().getPosition();
			const auto positionDelta = remappedPosition - orientedPreset.goal.position;
			if (!hlsl::CCameraMathUtilities::nearlyEqualVec3(positionDelta, SCameraSmokeManipulationDefaults::WorldTranslationDelta, SCameraSmokeUtilityThresholds::PositionWriteback))
			{
				outError = "Camera manipulation utilities smoke changed world-space translation semantics.";
				return false;
			}

			CameraPreset pitchPreset = state.initialPresets.free.value();
			pitchPreset.goal.orientation = hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(SCameraSmokeManipulationDefaults::FreePitchClampSourceDeg);
			const auto pitchResult = nbl::core::CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.freeCamera, pitchPreset);
			if (!pitchResult.succeeded())
			{
				outError = "Camera manipulation utilities smoke failed to prepare Free camera pitch clamp.";
				return false;
			}

			SCameraConstraintSettings freeConstraints = {
				.enabled = true,
				.clampPitch = true,
				.pitchMinDeg = SCameraSmokeManipulationDefaults::PitchMinDeg,
				.pitchMaxDeg = SCameraSmokeManipulationDefaults::PitchMaxDeg
			};
			if (!nbl::core::CCameraManipulationUtilities::applyCameraConstraints(state.goalSolver, state.freeCamera, freeConstraints))
			{
				outError = "Camera manipulation utilities smoke failed to clamp Free camera orientation.";
				return false;
			}

			const auto freeEulerDeg = hlsl::CCameraMathUtilities::getCameraOrientationEulerDegrees(state.freeCamera->getGimbal().getOrientation());
			if (hlsl::abs(static_cast<double>(freeEulerDeg.x - SCameraSmokeManipulationDefaults::PitchMaxDeg)) > SCameraSmokeManipulationDefaults::PitchAppliedToleranceDeg)
			{
				outError = "Camera manipulation utilities smoke produced wrong clamped Free camera pitch.";
				return false;
			}

			const auto restoreFree = nbl::core::CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.freeCamera, state.initialPresets.free.value());
			if (!restoreFree.succeeded() || !nbl::system::CCameraSmokeRegressionUtilities::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.freeCamera, state.initialPresets.free.value()))
			{
				outError = "Camera manipulation utilities smoke failed to restore Free camera baseline.";
				return false;
			}
		}

		if (!verifyReferenceFrameSupportSmoke(state, outError))
			return false;

		if (state.initialPresets.orbit.has_value() && state.orbitCamera && state.initialPresets.orbit->goal.hasDistance)
		{
			CameraPreset farOrbitPreset = state.initialPresets.orbit.value();
			farOrbitPreset.goal.distance = state.initialPresets.orbit->goal.distance + SCameraSmokeManipulationDefaults::OrbitDistanceDelta;
			const auto farOrbitResult = nbl::core::CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.orbitCamera, farOrbitPreset);
			if (!farOrbitResult.succeeded())
			{
				outError = "Camera manipulation utilities smoke failed to prepare Orbit distance clamp.";
				return false;
			}

			SCameraConstraintSettings orbitConstraints = {
				.enabled = true,
				.clampDistance = true,
				.minDistance = std::max(
					SCameraSmokeManipulationDefaults::MinDistanceClampFloor,
					state.initialPresets.orbit->goal.distance * SCameraSmokeManipulationDefaults::OrbitClampMinScale),
				.maxDistance = state.initialPresets.orbit->goal.distance * SCameraSmokeManipulationDefaults::OrbitClampMaxScale
			};
			if (!nbl::core::CCameraManipulationUtilities::applyCameraConstraints(state.goalSolver, state.orbitCamera, orbitConstraints))
			{
				outError = "Camera manipulation utilities smoke failed to clamp Orbit distance.";
				return false;
			}

			ICamera::SphericalTargetState clampedOrbitState;
			if (!state.orbitCamera->tryGetSphericalTargetState(clampedOrbitState) ||
				hlsl::abs(static_cast<double>(clampedOrbitState.distance - orbitConstraints.maxDistance)) > SCameraSmokeUtilityThresholds::DynamicPerspectiveDelta)
			{
				outError = "Camera manipulation utilities smoke produced wrong clamped Orbit distance.";
				return false;
			}

			const auto restoreOrbit = nbl::core::CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value());
			if (!restoreOrbit.succeeded() || !nbl::system::CCameraSmokeRegressionUtilities::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value()))
			{
				outError = "Camera manipulation utilities smoke failed to restore Orbit baseline.";
				return false;
			}
		}

		if (state.initialPresets.dollyZoom.has_value() && state.dollyZoomCamera)
		{
			float dynamicFov = 0.0f;
			if (!state.dollyZoomCamera->tryGetDynamicPerspectiveFov(dynamicFov))
			{
				outError = "Camera projection utilities smoke failed to query DollyZoom dynamic FOV.";
				return false;
			}

			auto perspectiveProjection = IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Perspective>(
				SCameraSmokeManipulationDefaults::PerspectiveNearPlane,
				SCameraSmokeManipulationDefaults::PerspectiveFarPlane,
				SCameraSmokeManipulationDefaults::PerspectiveFovDeg);
			if (!nbl::core::CCameraProjectionUtilities::syncDynamicPerspectiveProjection(state.dollyZoomCamera, perspectiveProjection))
			{
				outError = "Camera projection utilities smoke failed to sync dynamic perspective projection.";
				return false;
			}
			if (hlsl::abs(static_cast<double>(perspectiveProjection.getParameters().m_planar.perspective.fov - dynamicFov)) > SCameraSmokeUtilityThresholds::DynamicPerspectiveDelta)
			{
				outError = "Camera projection utilities smoke produced wrong dynamic perspective FOV.";
				return false;
			}

			auto orthographicProjection = IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Orthographic>(
				SCameraSmokeManipulationDefaults::PerspectiveNearPlane,
				SCameraSmokeManipulationDefaults::PerspectiveFarPlane,
				SCameraSmokeManipulationDefaults::OrthoExtent);
			if (nbl::core::CCameraProjectionUtilities::syncDynamicPerspectiveProjection(state.dollyZoomCamera, orthographicProjection))
			{
				outError = "Camera projection utilities smoke unexpectedly synced orthographic projection.";
				return false;
			}
		}

        if (CCameraTextUtilities::getCameraTypeLabel(ICamera::CameraKind::DollyZoom) != "Dolly Zoom")
		{
			outError = "Camera text utilities smoke failed for Dolly Zoom label.";
			return false;
		}
        if (CCameraTextUtilities::getCameraTypeDescription(ICamera::CameraKind::Path) != std::string(nbl::core::SCameraPathRigMetadata::KindDescription))
        {
			outError = "Camera text utilities smoke failed for Path description.";
			return false;
		}
        if (CCameraTextUtilities::describeGoalStateMask(ICamera::GoalStateNone) != "Pose only")
		{
			outError = "Camera text utilities smoke failed for empty goal-state description.";
			return false;
		}
        if (CCameraTextUtilities::describeGoalStateMask(ICamera::GoalStateSphericalTarget | ICamera::GoalStateDynamicPerspective) != "Spherical target, Dynamic perspective")
		{
			outError = "Camera text utilities smoke failed for combined goal-state description.";
			return false;
		}

		CCameraGoalSolver::SApplyResult defaultApplyResult;
        const auto applyResultText = CCameraTextUtilities::describeApplyResult(defaultApplyResult);
		if (applyResultText.find("status=Unsupported") == std::string::npos || applyResultText.find("events=0") == std::string::npos)
		{
			outError = "Camera text utilities smoke failed for apply-result description.";
			return false;
		}

		SCameraPresetApplySummary summary;
		summary.targetCount = 2u;
		summary.successCount = 2u;
		summary.approximateCount = 1u;
        const auto summaryText = nbl::ui::CCameraTextUtilities::describePresetApplySummary(summary, "none");
		if (summaryText.find("targets=2") == std::string::npos || summaryText.find("approximate=1") == std::string::npos)
		{
			outError = "Camera text utilities smoke failed for preset-apply summary description.";
			return false;
		}

		return true;
	}

	template<typename TMakeDefaultFollowConfig, typename TVerifyMarkerAlignment, typename TVerifyOffsetRecapture>
	inline bool verifyFollowSmoke(
		const SCameraSmokeResolvedState& state,
		std::span<const smart_refctd_ptr<ICamera>> cameras,
		std::span<const smart_refctd_ptr<planar_projection_t>> planarSpan,
		TMakeDefaultFollowConfig&& makeDefaultFollowConfig,
		TVerifyMarkerAlignment&& verifyMarkerAlignment,
		TVerifyOffsetRecapture&& verifyOffsetRecapture,
		std::string& outError)
	{
		CTrackedTarget trackedTarget(
			SCameraSmokeFollowScenario::InitialTargetPosition,
			SCameraSmokeFollowScenario::InitialTargetOrientation,
			"Smoke Target");

		const auto& movedTrackedTargetPosition = SCameraSmokeFollowScenario::MovedTargetPosition;
		const auto& movedTrackedTargetOrientation = SCameraSmokeFollowScenario::MovedTargetOrientation;

		if (state.orbitCamera)
		{
			const auto baselinePreset = nbl::core::CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.orbitCamera, "orbit-follow-baseline");
			SCameraFollowConfig followConfig = {};
			followConfig.enabled = true;
			followConfig.mode = ECameraFollowMode::OrbitTarget;

			if (!validateFollowScenario(state.goalSolver, planarSpan, state.orbitCamera, trackedTarget, followConfig, "orbit follow", outError))
				return false;
			if (!verifyMarkerAlignment(trackedTarget, "orbit follow", outError))
				return false;

			if (!restorePresetStrict(state.goalSolver, state.orbitCamera, baselinePreset, "Orbit follow smoke failed to restore the baseline preset", outError))
				return false;

			followConfig.mode = ECameraFollowMode::KeepWorldOffset;
			followConfig.worldOffset = SCameraSmokeFollowScenario::OrbitWorldOffset;
			trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

			if (!validateFollowScenario(state.goalSolver, planarSpan, state.orbitCamera, trackedTarget, followConfig, "orbit keep-world-offset follow", outError))
				return false;
			if (!restorePresetStrict(state.goalSolver, state.orbitCamera, baselinePreset, "Orbit keep-world-offset smoke failed to restore the baseline preset", outError))
				return false;
		}

		for (const auto& cameraRef : cameras)
		{
			auto* defaultFollowCamera = cameraRef.get();
			if (!defaultFollowCamera)
				continue;

			auto followConfig = makeDefaultFollowConfig(defaultFollowCamera);
			if (!followConfig.enabled || followConfig.mode == ECameraFollowMode::Disabled)
				continue;

			const auto label = std::string(defaultFollowCamera->getIdentifier()) + " default follow";
			const auto baselinePreset = nbl::core::CCameraPresetFlowUtilities::capturePreset(state.goalSolver, defaultFollowCamera, label + " baseline");

			trackedTarget.setPose(
				SCameraSmokeFollowScenario::InitialTargetPosition,
				SCameraSmokeFollowScenario::InitialTargetOrientation);
			if ((nbl::core::CCameraFollowUtilities::cameraFollowModeUsesLocalOffset(followConfig.mode) || nbl::core::CCameraFollowUtilities::cameraFollowModeUsesWorldOffset(followConfig.mode)) &&
				!nbl::core::CCameraFollowUtilities::captureFollowOffsetsFromCamera(state.goalSolver, defaultFollowCamera, trackedTarget, followConfig))
			{
				outError = "Default follow smoke failed to capture offsets for camera \"" + std::string(defaultFollowCamera->getIdentifier()) + "\".";
				return false;
			}

			trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

			if (!validateFollowScenario(state.goalSolver, planarSpan, defaultFollowCamera, trackedTarget, followConfig, label, outError))
				return false;
			if (!verifyMarkerAlignment(trackedTarget, label, outError))
				return false;

			if (!restorePresetStrict(
					state.goalSolver,
					defaultFollowCamera,
					baselinePreset,
					"Default follow smoke failed to restore the baseline preset for camera \"" + std::string(defaultFollowCamera->getIdentifier()) + "\"",
					outError))
			{
				return false;
			}
		}

		if (state.freeCamera)
		{
			const auto baselinePreset = nbl::core::CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.freeCamera, "free-follow-baseline");
			SCameraFollowConfig followConfig = {};
			followConfig.enabled = true;
			followConfig.mode = ECameraFollowMode::LookAtTarget;

			if (!validateFollowScenario(state.goalSolver, planarSpan, state.freeCamera, trackedTarget, followConfig, "free look-at follow", outError))
				return false;
			if (!verifyMarkerAlignment(trackedTarget, "free look-at follow", outError))
				return false;

			if (!restorePresetStrict(state.goalSolver, state.freeCamera, baselinePreset, "Free follow smoke failed to restore the baseline preset", outError))
				return false;

			followConfig.mode = ECameraFollowMode::KeepWorldOffset;
			followConfig.worldOffset = SCameraSmokeFollowScenario::FreeWorldOffset;
			trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

			if (!validateFollowScenario(state.goalSolver, planarSpan, state.freeCamera, trackedTarget, followConfig, "free keep-world-offset follow", outError))
				return false;
			if (!restorePresetStrict(state.goalSolver, state.freeCamera, baselinePreset, "Free keep-world-offset smoke failed to restore the baseline preset", outError))
				return false;
		}

		if (state.chaseCamera)
		{
			const auto baselinePreset = nbl::core::CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.chaseCamera, "chase-follow-baseline");
			SCameraFollowConfig followConfig = {};
			followConfig.enabled = true;
			followConfig.mode = ECameraFollowMode::KeepLocalOffset;
			if (!nbl::core::CCameraFollowUtilities::captureFollowOffsetsFromCamera(state.goalSolver, state.chaseCamera, trackedTarget, followConfig))
			{
				outError = "Chase follow smoke failed to capture local offset.";
				return false;
			}

			trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

			if (!validateFollowScenario(state.goalSolver, planarSpan, state.chaseCamera, trackedTarget, followConfig, "chase local-offset follow", outError))
				return false;
			if (!verifyMarkerAlignment(trackedTarget, "chase local-offset follow", outError))
				return false;

			if (!restorePresetStrict(state.goalSolver, state.chaseCamera, baselinePreset, "Chase follow smoke failed to restore the baseline preset", outError))
				return false;
		}

		if (!verifyOffsetRecapture(state.chaseCamera, trackedTarget, "chase follow recapture", outError))
			return false;
		if (!verifyOffsetRecapture(state.dollyCamera, trackedTarget, "dolly follow recapture", outError))
			return false;

		return true;
	}

