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
					CCameraGoalSolver::SApplyResult::EIssue::MissingPathState,
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
					CCameraGoalSolver::SApplyResult::EIssue::MissingDynamicPerspectiveState,
					"DollyZoom->Orbit",
					outError))
			{
				return false;
			}
		}

		if (!state.initialPresets.orbit.has_value())
			return true;

		if (std::string_view(CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::All)) != "All" ||
			std::string_view(CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::Exact)) != "Exact" ||
			std::string_view(CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::BestEffort)) != "Best-effort")
		{
			outError = "Presentation utilities smoke returned an unexpected filter label.";
			return false;
		}

		const auto blockedPresentation = CCameraPresentationUtilities::analyzePresetPresentation(state.goalSolver, nullptr, state.initialPresets.orbit.value());
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

		const auto blockedBadges = CCameraPresentationUtilities::collectGoalApplyPresentationBadges(blockedPresentation);
		if (!blockedBadges.blocked || blockedBadges.exact || blockedBadges.bestEffort || blockedPresentation.badges.blocked != blockedBadges.blocked)
		{
			outError = "Presentation utilities smoke produced wrong blocked badge flags.";
			return false;
		}

		if (state.orbitCamera)
		{
			const auto exactPresentation = CCameraPresentationUtilities::analyzePresetPresentation(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value());
			if (!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
				!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
				exactPresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
			{
				outError = "Presentation utilities smoke failed exact filtering.";
				return false;
			}

			const auto exactBadges = CCameraPresentationUtilities::collectGoalApplyPresentationBadges(exactPresentation);
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

			const auto capturePresentation = CCameraPresentationUtilities::analyzeCapturePresentation(state.goalSolver, state.orbitCamera);
			if (!capturePresentation.canCapture || capturePresentation.policyLabel.empty())
			{
				outError = "Presentation utilities smoke failed orbit capture presentation.";
				return false;
			}
		}

		if (state.initialPresets.path.has_value() && state.orbitCamera)
		{
			const auto approximatePresentation = CCameraPresentationUtilities::analyzePresetPresentation(state.goalSolver, state.orbitCamera, state.initialPresets.path.value());
			if (!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
				approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
				!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
			{
				outError = "Presentation utilities smoke failed best-effort filtering.";
				return false;
			}

			const auto approximateBadges = CCameraPresentationUtilities::collectGoalApplyPresentationBadges(approximatePresentation);
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
		const nbl::ext::cameras::STargetOrbit& desiredState,
		hlsl::float64_t4x4& outReferenceFrame,
		CCameraGoal& outExpectedGoal)
	{
		outExpectedGoal = {};
		nbl::ext::cameras::SCameraRigPose pose = {};
		if (!nbl::ext::cameras::CCameraMathUtilities::tryBuildPoseFromOrbit(
				desiredState,
				nbl::ext::cameras::ICamera::DefaultMinTargetDistance,
				nbl::ext::cameras::ICamera::DefaultMaxTargetDistance,
				pose) ||
			!CCameraGoalUtilities::applyCanonicalTargetRelativeGoal(outExpectedGoal, desiredState))
		{
			return false;
		}

		outReferenceFrame = CCameraMathUtilities::composeTransformMatrix(pose.position, pose.orientation);
		return true;
	}

	inline bool verifyReferenceFrameGoalApply(
		const SCameraSmokeResolvedState& state,
		ICamera* const camera,
		const nbl::ext::cameras::STargetOrbit& desiredState,
		std::string_view label,
		std::string& outError)
	{
		ICamera::SphericalTargetState baselineState = {};
		if (!camera->tryGetSphericalTargetState(baselineState))
		{
			outError = std::string(label) + " reference-frame smoke failed to capture the baseline spherical state.";
			return false;
		}

		const nbl::ext::cameras::STargetOrbit baselineTargetRelativeState = {
			.target = baselineState.target,
			.angles = baselineState.orbitUv,
			.distance = baselineState.distance
		};

		hlsl::float64_t4x4 referenceFrame = hlsl::float64_t4x4(1.0);
		hlsl::float64_t4x4 baselineReferenceFrame = hlsl::float64_t4x4(1.0);
		CCameraGoal expectedGoal = {};
		CCameraGoal baselineGoal = {};
		if (!tryBuildReferenceFrameFromTargetRelativeState(desiredState, referenceFrame, expectedGoal) ||
			!tryBuildReferenceFrameFromTargetRelativeState(baselineTargetRelativeState, baselineReferenceFrame, baselineGoal))
		{
			outError = std::string(label) + " reference-frame smoke failed to build the projected reference pose.";
			return false;
		}

		if (!camera->setPose(referenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to apply the reference pose through setPose(referenceFrame).";
			return false;
		}

		ICamera::SphericalTargetState actualState = {};
			if (!camera->tryGetSphericalTargetState(actualState) ||
			!CCameraMathUtilities::nearlyEqualVec3(
				actualState.target,
				desiredState.target,
				SCameraSmokeComparisonThresholds::StrictScalarTolerance) ||
			CCameraMathUtilities::getWrappedAngleDistanceRadians(actualState.orbitUv.x, desiredState.angles.x) >
				hlsl::radians(SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg) ||
			CCameraMathUtilities::getWrappedAngleDistanceRadians(actualState.orbitUv.y, desiredState.angles.y) >
				hlsl::radians(SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg) ||
			hlsl::abs(static_cast<double>(actualState.distance - desiredState.distance)) >
				SCameraSmokeComparisonThresholds::StrictScalarTolerance)
		{
			std::ostringstream oss;
			oss << label
				<< " reference-frame smoke produced the wrong spherical state:"
				<< " actual_target=(" << actualState.target.x << "," << actualState.target.y << "," << actualState.target.z << ")"
				<< " expected_target=(" << desiredState.target.x << "," << desiredState.target.y << "," << desiredState.target.z << ")"
				<< " actual_orbit=(" << actualState.orbitUv.x << "," << actualState.orbitUv.y << ")"
				<< " expected_orbit=(" << desiredState.angles.x << "," << desiredState.angles.y << ")"
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
			!CCameraGoalUtilities::compareGoals(
				capture.goal,
				expectedGoal,
				SCameraSmokeComparisonThresholds::StrictPositionTolerance,
				SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
				SCameraSmokeComparisonThresholds::StrictScalarTolerance))
		{
			outError = std::string(label) + " reference-frame smoke produced the wrong projected goal: " +
				(capture.canUseGoal() ? CCameraGoalUtilities::describeGoalMismatch(capture.goal, expectedGoal) : std::string("goal_state=unavailable"));
			return false;
		}

		if (!camera->setPose(baselineReferenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to restore the baseline reference pose through setPose(referenceFrame).";
			return false;
		}

		return true;
	}

	inline bool verifyReferenceFramePoseApply(
		const SCameraSmokeResolvedState& state,
		ICamera* const camera,
		const hlsl::float64_t3& desiredPosition,
		const hlsl::math::quaternion<hlsl::float64_t>& desiredOrientation,
		std::string_view label,
		std::string& outError)
	{
		const auto baselineCapture = state.goalSolver.captureDetailed(camera);
		if (!baselineCapture.canUseGoal())
		{
			outError = std::string(label) + " reference-frame smoke failed to capture the baseline pose.";
			return false;
		}

		CCameraGoal expectedGoal = {};
		expectedGoal.position = desiredPosition;
		expectedGoal.orientation = hlsl::normalize(desiredOrientation);

		const auto baselineReferenceFrame = CCameraMathUtilities::composeTransformMatrix(
			baselineCapture.goal.position,
			baselineCapture.goal.orientation);
		const auto referenceFrame = CCameraMathUtilities::composeTransformMatrix(
			desiredPosition,
			expectedGoal.orientation);
		if (!camera->setPose(referenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to apply the rigid reference pose through setPose(referenceFrame).";
			return false;
		}

		const auto capture = state.goalSolver.captureDetailed(camera);
		if (!capture.canUseGoal() ||
			!CCameraGoalUtilities::compareGoals(
				capture.goal,
				expectedGoal,
				SCameraSmokeComparisonThresholds::StrictPositionTolerance,
				SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
				SCameraSmokeComparisonThresholds::StrictScalarTolerance))
		{
			outError = std::string(label) + " reference-frame smoke produced the wrong rigid goal: " +
				(capture.canUseGoal() ? CCameraGoalUtilities::describeGoalMismatch(capture.goal, expectedGoal) : std::string("goal_state=unavailable"));
			return false;
		}

		if (!camera->setPose(baselineReferenceFrame))
		{
			outError = std::string(label) + " reference-frame smoke failed to restore the baseline rigid pose through setPose(referenceFrame).";
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
					hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(hlsl::radians(35.0), hlsl::radians(-20.0), hlsl::radians(0.0)),
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
					hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(hlsl::radians(45.0), hlsl::radians(15.0), hlsl::radians(20.0)),
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

			nbl::ext::cameras::STargetOrbit desiredState = {
				.target = baselineState.target,
				.angles = baselineState.orbitUv,
				.distance = chooseShiftedReferenceDistance(baselineState)
			};
			mutateDesiredState(desiredState);
			return verifyReferenceFrameGoalApply(state, camera, desiredState, label, outError);
		};

		if (!verifySphericalReference(state.orbitCamera, "Orbit", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles += hlsl::float64_t2(0.45, -0.25);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.arcballCamera, "Arcball", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles += hlsl::float64_t2(0.35, 0.2);
				desiredState.angles.y = std::clamp(
					desiredState.angles.y,
					CArcballCamera::MinPitch,
					CArcballCamera::MaxPitch);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.turntableCamera, "Turntable", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles += hlsl::float64_t2(-0.4, 0.18);
				desiredState.angles.y = std::clamp(
					desiredState.angles.y,
					CTurntableCamera::MinPitch,
					CTurntableCamera::MaxPitch);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.topDownCamera, "TopDown", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles = hlsl::float64_t2(
					desiredState.angles.x + 0.6,
					CTopDownCamera::TopDownPitch);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.isometricCamera, "Isometric", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles = hlsl::float64_t2(
					CIsometricCamera::IsoYaw,
					CIsometricCamera::IsoPitch);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.chaseCamera, "Chase", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles += hlsl::float64_t2(0.3, 0.15);
				desiredState.angles.y = std::clamp(
					desiredState.angles.y,
					CChaseCamera::MinPitch,
					CChaseCamera::MaxPitch);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.dollyCamera, "Dolly", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles += hlsl::float64_t2(-0.3, -0.22);
				desiredState.angles.y = std::clamp(
					desiredState.angles.y,
					CDollyCamera::MinPitch,
					CDollyCamera::MaxPitch);
			}))
		{
			return false;
		}

		if (!verifySphericalReference(state.dollyZoomCamera, "DollyZoom", [&](nbl::ext::cameras::STargetOrbit& desiredState)
			{
				desiredState.angles += hlsl::float64_t2(0.28, -0.14);
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
			nbl::ext::cameras::SCameraPathDelta pathDelta = {};
			pathDelta.u = 0.8;
			pathDelta.v = 0.35;
			pathDelta.s = 1.1;
			pathDelta.roll = 0.45;
			if (!nbl::ext::cameras::CCameraPathUtilities::tryApplyPathStateDelta(
					baselinePathState,
					pathDelta,
					pathLimits,
					desiredPathState))
			{
				outError = "Path reference-frame smoke failed to build the desired typed path state.";
				return false;
			}

			nbl::ext::cameras::SCameraCanonicalPathState canonicalPathState = {};
			nbl::ext::cameras::SCameraCanonicalPathState baselineCanonicalPathState = {};
			CCameraGoal expectedGoal = {};
			if (!nbl::ext::cameras::CCameraPathUtilities::tryBuildCanonicalPathState(
					sphericalState.target,
					desiredPathState,
					pathLimits,
					canonicalPathState) ||
				!nbl::ext::cameras::CCameraPathUtilities::tryResolvePathState(
					sphericalState.target,
					canonicalPathState.pose.position,
					pathLimits,
					nullptr,
					projectedPathState) ||
				!nbl::ext::cameras::CCameraPathUtilities::tryBuildCanonicalPathState(
					sphericalState.target,
					baselinePathState,
					pathLimits,
					baselineCanonicalPathState) ||
				!CCameraGoalUtilities::applyCanonicalPathGoalFields(
					expectedGoal,
					sphericalState.target,
					projectedPathState,
					pathLimits))
			{
				outError = "Path reference-frame smoke failed to build the canonical target-relative path pose.";
				return false;
			}

			const auto baselineReferenceFrame = CCameraMathUtilities::composeTransformMatrix(
				baselineCanonicalPathState.pose.position,
				baselineCanonicalPathState.pose.orientation);
			const auto referenceFrame = CCameraMathUtilities::composeTransformMatrix(
				canonicalPathState.pose.position,
				canonicalPathState.pose.orientation);
			if (!state.pathCamera->setPose(referenceFrame))
			{
				outError = "Path reference-frame smoke failed to apply the projected path pose through setPose(referenceFrame).";
				return false;
			}

			const auto capture = state.goalSolver.captureDetailed(state.pathCamera);
			if (!capture.canUseGoal() ||
				!CCameraGoalUtilities::compareGoals(
					capture.goal,
					expectedGoal,
					SCameraSmokeComparisonThresholds::StrictPositionTolerance,
					SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
					SCameraSmokeComparisonThresholds::StrictScalarTolerance))
			{
				outError = "Path reference-frame smoke produced the wrong projected goal: " +
					(capture.canUseGoal() ? CCameraGoalUtilities::describeGoalMismatch(capture.goal, expectedGoal) : std::string("goal_state=unavailable"));
				return false;
			}

			if (!state.pathCamera->setPose(baselineReferenceFrame))
			{
				outError = "Path reference-frame smoke failed to restore the baseline reference pose through setPose(referenceFrame).";
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

		const auto presetText = CCameraPersistenceUtilities::serializePresetCollection(sourcePresetSpan);
		if (presetText.empty())
		{
			outError = "Preset persistence smoke failed to serialize preset collection.";
			return false;
		}

		std::vector<CameraPreset> loadedPresets;
		if (!CCameraPersistenceUtilities::deserializePresetCollection(presetText, loadedPresets))
		{
			outError = "Preset persistence smoke failed to deserialize preset collection.";
			return false;
		}
		if (!CCameraPresetUtilities::comparePresetCollections(
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
			CCameraKeyframe keyframe;
			keyframe.time = static_cast<float>(i) * 1.5f;
			keyframe.preset = sourcePresets[i];
			sourceTrack.keyframes.emplace_back(std::move(keyframe));
		}
		sourceTrack.selectedKeyframeIx = static_cast<int>(sourceTrack.keyframes.size()) - 1;

		const auto keyframeText = CCameraKeyframeTrackPersistenceUtilities::serializeKeyframeTrack(sourceTrack);
		if (keyframeText.empty())
		{
			outError = "Keyframe persistence smoke failed to serialize track.";
			return false;
		}

		CCameraKeyframeTrack loadedTrack;
		if (!CCameraKeyframeTrackPersistenceUtilities::deserializeKeyframeTrack(keyframeText, loadedTrack))
		{
			outError = "Keyframe persistence smoke failed to deserialize track.";
			return false;
		}
		if (!CCameraSmokeRegressionUtilities::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, loadedTrack))
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

		if (!CCameraPersistenceUtilities::savePresetCollectionToFile(system, presetFile, sourcePresetSpan))
		{
			outError = "Preset persistence smoke failed to save preset collection file.";
			return false;
		}

		std::vector<CameraPreset> fileLoadedPresets;
		if (!CCameraPersistenceUtilities::loadPresetCollectionFromFile(system, presetFile, fileLoadedPresets))
		{
			outError = "Preset persistence smoke failed to load preset collection file.";
			return false;
		}
		if (!CCameraPresetUtilities::comparePresetCollections(
				sourcePresetSpan,
				std::span<const CameraPreset>(fileLoadedPresets.data(), fileLoadedPresets.size()),
				SCameraSmokePersistenceThresholds::PositionTolerance,
				SCameraSmokePersistenceThresholds::AngularToleranceDeg,
				SCameraSmokePersistenceThresholds::ScalarTolerance))
		{
			outError = "Preset persistence smoke changed file preset collection content.";
			return false;
		}

		if (!CCameraKeyframeTrackPersistenceUtilities::saveKeyframeTrackToFile(system, keyframeFile, sourceTrack))
		{
			outError = "Keyframe persistence smoke failed to save track file.";
			return false;
		}

		CCameraKeyframeTrack fileLoadedTrack;
		if (!CCameraKeyframeTrackPersistenceUtilities::loadKeyframeTrackFromFile(system, keyframeFile, fileLoadedTrack))
		{
			outError = "Keyframe persistence smoke failed to load track file.";
			return false;
		}
		if (!CCameraSmokeRegressionUtilities::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, fileLoadedTrack))
		{
			outError = "Keyframe persistence smoke changed file track content.";
			return false;
		}

		if (state.initialPresets.orbit.has_value() && state.initialPresets.dolly.has_value())
		{
			CCameraKeyframeTrack playbackTrack;
			{
				CCameraKeyframe keyframe;
				keyframe.time = 0.f;
				keyframe.preset = state.initialPresets.orbit.value();
				playbackTrack.keyframes.push_back(keyframe);
			}
			{
				CCameraKeyframe keyframe;
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

			const auto advanceToEnd = CCameraPlaybackTimelineUtilities::advancePlaybackCursor(cursor, playbackTrack, SCameraSmokePlaybackDefaults::AdvanceDt);
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

			CCameraPlaybackTimelineUtilities::resetPlaybackCursor(cursor, SCameraSmokePlaybackDefaults::ResetPlaybackTime);
			if (cursor.playing || hlsl::abs(static_cast<double>(cursor.time - SCameraSmokePlaybackDefaults::ResetPlaybackTime)) > CameraTinyScalarEpsilon)
			{
				outError = "Playback timeline smoke failed to reset cursor.";
				return false;
			}

			cursor.playing = true;
			cursor.loop = true;
			cursor.speed = 1.f;
			cursor.time = SCameraSmokePlaybackDefaults::MidPlaybackTime;
			const auto advanceLoop = CCameraPlaybackTimelineUtilities::advancePlaybackCursor(cursor, playbackTrack, SCameraSmokePlaybackDefaults::AdvanceDt);
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
			CCameraPlaybackTimelineUtilities::clampPlaybackCursorToTrack(playbackTrack, cursor);
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
			{ .projection = CPlanarProjection::EKind::Perspective, .leftHanded = true },
			{ .projection = CPlanarProjection::EKind::Orthographic, .leftHanded = false }
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
			CCameraSequenceTrackedTargetKeyframe keyframe;
			keyframe.time = time;
			keyframe.hasAbsolutePosition = true;
			keyframe.absolutePosition = position;
			segment.targetKeyframes.push_back(keyframe);
		}
		sequence.segments.push_back(segment);

		if (!CCameraSequenceScriptUtilities::sequenceScriptUsesMultiplePresentations(sequence))
		{
			outError = "Sequence compile smoke failed to detect multi-presentation authored defaults.";
			return false;
		}

		CCameraSequenceTrackedTargetPose referenceTrackedTargetPose = {};
		referenceTrackedTargetPose.position = SCameraAppSceneDefaults::DefaultFollowTargetPosition;
		referenceTrackedTargetPose.orientation = SCameraAppSceneDefaults::DefaultFollowTargetOrientation;

		CCameraSequenceCompiledSegment compiledSegment;
		std::string compileError;
		if (!CCameraSequenceScriptUtilities::compileSequenceSegmentFromReference(
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

		std::vector<CCameraSequenceCompiledFramePolicy> framePolicies;
		if (!CCameraSequenceScriptUtilities::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, true))
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
		if (!CCameraSequenceScriptUtilities::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, 1.f, poseAtOne))
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
		std::vector<nbl::this_example::CCameraScriptedActionEvent> actionEvents;
		std::string runtimeBuildError;
		if (!CCameraSequenceScriptedBuilderUtilities::appendCompiledSequenceSegmentToScriptedTimeline(
				scriptedTimeline,
				actionEvents,
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
		CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(scriptedTimeline);
		nbl::this_example::CCameraScriptedActionUtilities::finalizeActionEvents(actionEvents);

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
		size_t runtimeNextActionIndex = 0u;
		CCameraScriptedFrameEvents runtimeBatch;
		std::vector<nbl::this_example::CCameraScriptedActionEvent> runtimeActions;
		CCameraScriptedFrameEventUtilities::dequeueScriptedFrameEvents(scriptedTimeline.events, runtimeNextEventIndex, SCameraSmokeSequenceDefaults::StartFrame, runtimeBatch);
		nbl::this_example::CCameraScriptedActionUtilities::dequeueFrameActions(actionEvents, runtimeNextActionIndex, SCameraSmokeSequenceDefaults::StartFrame, runtimeActions);
		if (runtimeActions.size() != 10u || runtimeBatch.goals.size() != 1u ||
			runtimeBatch.trackedTargetTransforms.size() != 1u || runtimeBatch.segmentLabels.size() != 1u)
		{
			outError = "Sequence runtime builder smoke produced wrong first-frame batch.";
			return false;
		}
		if (!nbl::this_example::CCameraScriptedActionUtilities::hasCode(runtimeActions.front(), nbl::this_example::ECameraScriptedActionCode::SetActiveRenderWindow) ||
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
			const auto exactSummary = CCameraPresetFlowUtilities::applyPresetToCameraRange(
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
			const auto approximateSummary = CCameraPresetFlowUtilities::applyPresetToCameraRange(
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

			SCameraControls directPathControls = {};
			directPathControls.path.u = 1.5;
			directPathControls.path.v = 0.75;
			directPathControls.path.s = 2.0;
			directPathControls.path.roll = 0.5;

			if (!state.pathCamera->manipulate(directPathControls))
			{
				outError = "Path manipulation smoke failed to apply direct path controls.";
				return false;
			}

			ICamera::PathState manipulatedPathState = {};
			if (!state.pathCamera->tryGetPathState(manipulatedPathState))
			{
				outError = "Path manipulation smoke failed to read the manipulated path state.";
				return false;
			}

			ICamera::PathStateLimits activePathLimits = nbl::ext::cameras::CCameraPathUtilities::makeDefaultPathLimits();
			state.pathCamera->tryGetPathStateLimits(activePathLimits);
			nbl::ext::cameras::SCameraPathDelta expectedPathDelta = {};
			expectedPathDelta.s = directPathControls.path.s;
			expectedPathDelta.u = directPathControls.path.u;
			expectedPathDelta.v = directPathControls.path.v;
			expectedPathDelta.roll = directPathControls.path.roll;
			ICamera::PathState expectedPathState = {};
			if (!nbl::ext::cameras::CCameraPathUtilities::tryApplyPathStateDelta(
					baselinePathState,
					expectedPathDelta,
					activePathLimits,
					expectedPathState) ||
				!nbl::ext::cameras::CCameraPathUtilities::pathStatesNearlyEqual(
					manipulatedPathState,
					expectedPathState,
					nbl::ext::cameras::SCameraPathDefaults::ExactComparisonThresholds))
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

			SCameraControls replayControls = {};
			if (!state.goalSolver.buildControls(state.pathCamera, movedCapture.goal, replayControls) || replayControls.nonZeroAxes() == 0u)
			{
				outError = "Path manipulation smoke failed to build replay controls for the moved path goal.";
				return false;
			}

			if (replayControls.path.roll == 0.0)
			{
				outError = "Path manipulation smoke dropped the roll replay axis for the moved path goal.";
				return false;
			}

			if (!state.pathCamera->manipulate(replayControls))
			{
				outError = "Path manipulation smoke failed to replay path controls onto the baseline camera.";
				return false;
			}

			const auto replayCapture = state.goalSolver.captureDetailed(state.pathCamera);
			if (!replayCapture.canUseGoal() ||
				!CCameraGoalUtilities::compareGoals(
					replayCapture.goal,
					movedCapture.goal,
					SCameraSmokeComparisonThresholds::StrictPositionTolerance,
					SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
					SCameraSmokeComparisonThresholds::StrictScalarTolerance))
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

			const auto defaultPathModel = nbl::ext::cameras::CCameraPathUtilities::makeDefaultPathModel();
			nbl::ext::cameras::CPathCamera::path_model_t incompletePathModel = {};
			incompletePathModel.resolveState = defaultPathModel.resolveState;

			ICamera::PathStateLimits customPathLimits = {
				.minU = 2.0,
				.minDistance = 2.0,
				.maxDistance = 3.0
			};
			auto customPathCamera = nbl::core::make_smart_refctd_ptr<nbl::ext::cameras::CPathCamera>(
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

			const double customBaselineDistance = hlsl::length(hlsl::float64_t2(customBaselinePathState.u, customBaselinePathState.v));
			if (customBaselineDistance + CameraTinyScalarEpsilon < resolvedPathLimits.minDistance ||
				customBaselineDistance - CameraTinyScalarEpsilon > resolvedPathLimits.maxDistance)
			{
				outError = "Path manipulation smoke failed to clamp the constructor-resolved path state to custom limits.";
				return false;
			}

			if (!customPathCamera->manipulate(directPathControls))
			{
				outError = "Path manipulation smoke failed to apply direct controls on the custom-limits path camera.";
				return false;
			}

			ICamera::PathState customManipulatedPathState = {};
			if (!customPathCamera->tryGetPathState(customManipulatedPathState))
			{
				outError = "Path manipulation smoke failed to read the manipulated custom-limits path state.";
				return false;
			}

			ICamera::PathState expectedCustomPathState = {};
			if (!nbl::ext::cameras::CCameraPathUtilities::tryApplyPathStateDelta(
					customBaselinePathState,
					expectedPathDelta,
					resolvedPathLimits,
					expectedCustomPathState) ||
				!nbl::ext::cameras::CCameraPathUtilities::pathStatesNearlyEqual(
					customManipulatedPathState,
					expectedCustomPathState,
					nbl::ext::cameras::SCameraPathDefaults::ExactComparisonThresholds))
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

			SCameraControls customReplayControls = {};
			if (!state.goalSolver.buildControls(customPathCamera.get(), customMovedCapture.goal, customReplayControls) || customReplayControls.nonZeroAxes() == 0u)
			{
				outError = "Path manipulation smoke failed to build replay controls for the custom-limits path goal.";
				return false;
			}

			if (!customPathCamera->manipulate(customReplayControls))
			{
				outError = "Path manipulation smoke failed to replay controls on the custom-limits path camera.";
				return false;
			}

			const auto customReplayCapture = state.goalSolver.captureDetailed(customPathCamera.get());
			if (!customReplayCapture.canUseGoal() ||
				!CCameraGoalUtilities::compareGoals(
					customReplayCapture.goal,
					customMovedCapture.goal,
					SCameraSmokeComparisonThresholds::StrictPositionTolerance,
					SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
					SCameraSmokeComparisonThresholds::StrictScalarTolerance))
			{
				outError = "Path manipulation smoke failed the custom-limits goal replay roundtrip.";
				return false;
			}
		}

		if (state.fpsCamera)
		{
			const auto baselinePreset = CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.fpsCamera, "fps-translation-baseline");
			if (!restorePresetStrict(state.goalSolver, state.fpsCamera, baselinePreset, "FPS translation smoke failed to restore baseline before test", outError))
				return false;

			SCameraControls forwardControls = {};
			forwardControls.translate.z = 2.0;

			const auto baselinePosition = state.fpsCamera->getGimbal().getPosition();
			const auto baselineForward = state.fpsCamera->getGimbal().getForward();
			const auto expectedPositionDelta = baselineForward * forwardControls.translate.z;
			if (!state.fpsCamera->manipulate(forwardControls))
			{
				outError = "FPS translation smoke failed to apply the forward control.";
				return false;
			}

			const auto actualPositionDelta = state.fpsCamera->getGimbal().getPosition() - baselinePosition;
			if (!CCameraMathUtilities::nearlyEqualVec3(actualPositionDelta, expectedPositionDelta, SCameraSmokeUtilityThresholds::PositionWriteback))
			{
				outError = "FPS translation smoke did not move one world unit per unit of camera-local translate.";
				return false;
			}

			if (!restorePresetStrict(state.goalSolver, state.fpsCamera, baselinePreset, "FPS translation smoke failed to restore baseline after test", outError))
				return false;
		}

		if (state.freeCamera)
		{
			const auto freeBaselinePreset = CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.freeCamera, "free-manipulation-baseline");
			if (!restorePresetStrict(state.goalSolver, state.freeCamera, freeBaselinePreset, "Free manipulation smoke failed to restore baseline before test", outError))
				return false;

			{
				SCameraControls translationControls = {};
				translationControls.translate.z = 2.0;

				const auto baselinePosition = state.freeCamera->getGimbal().getPosition();
				const auto baselineForward = state.freeCamera->getGimbal().getForward();
				const auto expectedPositionDelta = baselineForward * translationControls.translate.z;
				if (!state.freeCamera->manipulate(translationControls))
				{
					outError = "Free translation smoke failed to apply the forward control.";
					return false;
				}

				const auto actualPositionDelta = state.freeCamera->getGimbal().getPosition() - baselinePosition;
				if (!CCameraMathUtilities::nearlyEqualVec3(actualPositionDelta, expectedPositionDelta, SCameraSmokeUtilityThresholds::PositionWriteback))
				{
					outError = "Free translation smoke did not move one world unit per unit of camera-local translate.";
					return false;
				}

				if (!restorePresetStrict(state.goalSolver, state.freeCamera, freeBaselinePreset, "Free translation smoke failed to restore baseline after translation test", outError))
					return false;
			}

			{
				SCameraControls rotationControls = {};
				rotationControls.rotate.y = 2.0;

				const auto baselineForward = state.freeCamera->getGimbal().getForward();
				const auto baselineUp = state.freeCamera->getGimbal().getUp();
				const auto expectedForward = hlsl::normalize(
					hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(hlsl::normalize(baselineUp), rotationControls.rotate.y)
				).transformVector(baselineForward, true);
				if (!state.freeCamera->manipulate(rotationControls))
				{
					outError = "Free rotation smoke failed to apply the yaw control.";
					return false;
				}

				const auto actualForward = state.freeCamera->getGimbal().getForward();
				if (!CCameraMathUtilities::nearlyEqualVec3(actualForward, expectedForward, SCameraSmokeUtilityThresholds::PositionWriteback))
				{
					outError = "Free rotation smoke did not yaw one radian per unit of rotate.y about the rig's own up axis.";
					return false;
				}

				if (!restorePresetStrict(state.goalSolver, state.freeCamera, freeBaselinePreset, "Free rotation smoke failed to restore baseline after rotation test", outError))
					return false;
			}

			CameraPreset pitchPreset = state.initialPresets.free.value();
			const auto& pitchClampSourceDeg = SCameraSmokeManipulationDefaults::FreePitchClampSourceDeg;
			pitchPreset.goal.orientation = hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(
				hlsl::radians(pitchClampSourceDeg.y),
				hlsl::radians(pitchClampSourceDeg.x),
				hlsl::radians(pitchClampSourceDeg.z));
			const auto pitchResult = CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.freeCamera, pitchPreset);
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
			if (!nbl::this_example::CCameraConstraintUtilities::applyCameraConstraints(state.goalSolver, state.freeCamera, freeConstraints))
			{
				outError = "Camera manipulation utilities smoke failed to clamp Free camera orientation.";
				return false;
			}

			const auto freeEulerDeg = CCameraMathUtilities::getPitchYawRollDegrees(state.freeCamera->getGimbal().getOrientation());
			if (hlsl::abs(static_cast<double>(freeEulerDeg.x - SCameraSmokeManipulationDefaults::PitchMaxDeg)) > SCameraSmokeManipulationDefaults::PitchAppliedToleranceDeg)
			{
				outError = "Camera manipulation utilities smoke produced wrong clamped Free camera pitch.";
				return false;
			}

			const auto restoreFree = CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.freeCamera, state.initialPresets.free.value());
			if (!restoreFree.succeeded() || !CCameraSmokeRegressionUtilities::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.freeCamera, state.initialPresets.free.value()))
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
			const auto farOrbitResult = CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.orbitCamera, farOrbitPreset);
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
			if (!nbl::this_example::CCameraConstraintUtilities::applyCameraConstraints(state.goalSolver, state.orbitCamera, orbitConstraints))
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

			const auto restoreOrbit = CCameraPresetFlowUtilities::applyPresetDetailed(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value());
			if (!restoreOrbit.succeeded() || !CCameraSmokeRegressionUtilities::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value()))
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

			auto perspectiveProjection = CPlanarProjection::createPerspective(
				SCameraSmokeManipulationDefaults::PerspectiveNearPlane,
				SCameraSmokeManipulationDefaults::PerspectiveFarPlane,
				SCameraSmokeManipulationDefaults::PerspectiveFovDeg);
			if (!CCameraProjectionUtilities::syncDynamicPerspectiveProjection(state.dollyZoomCamera, perspectiveProjection))
			{
				outError = "Camera projection utilities smoke failed to sync dynamic perspective projection.";
				return false;
			}
			if (hlsl::abs(static_cast<double>(perspectiveProjection.getParameters().perspective.fov - dynamicFov)) > SCameraSmokeUtilityThresholds::DynamicPerspectiveDelta)
			{
				outError = "Camera projection utilities smoke produced wrong dynamic perspective FOV.";
				return false;
			}

			auto orthographicProjection = CPlanarProjection::createOrthographic(
				SCameraSmokeManipulationDefaults::PerspectiveNearPlane,
				SCameraSmokeManipulationDefaults::PerspectiveFarPlane,
				SCameraSmokeManipulationDefaults::OrthoExtent);
			if (CCameraProjectionUtilities::syncDynamicPerspectiveProjection(state.dollyZoomCamera, orthographicProjection))
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
        if (CCameraTextUtilities::getCameraTypeDescription(ICamera::CameraKind::Path) != std::string(nbl::ext::cameras::SCameraPathRigMetadata::KindDescription))
        {
			outError = "Camera text utilities smoke failed for Path description.";
			return false;
		}
        if (CCameraTextUtilities::describeGoalStateMask(ICamera::GoalStateNone) != "Pose only")
		{
			outError = "Camera text utilities smoke failed for empty goal-state description.";
			return false;
		}
        const ICamera::goal_state_flags_t combinedGoalStateMask = ICamera::goal_state_flags_t(ICamera::GoalStateSphericalTarget) | ICamera::goal_state_flags_t(ICamera::GoalStateDynamicPerspective);
        if (CCameraTextUtilities::describeGoalStateMask(combinedGoalStateMask) != "Spherical target, Dynamic perspective")
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
        const auto summaryText = CCameraTextUtilities::describePresetApplySummary(summary, "none");
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
			const auto baselinePreset = CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.orbitCamera, "orbit-follow-baseline");
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
			followConfig.offset = SCameraSmokeFollowScenario::OrbitWorldOffset;
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
			if (!followConfig.enabled || followConfig.mode == ECameraFollowMode::Unknown)
				continue;

			const auto label = std::string(defaultFollowCamera->getIdentifier()) + " default follow";
			const auto baselinePreset = CCameraPresetFlowUtilities::capturePreset(state.goalSolver, defaultFollowCamera, label + " baseline");

			trackedTarget.setPose(
				SCameraSmokeFollowScenario::InitialTargetPosition,
				SCameraSmokeFollowScenario::InitialTargetOrientation);
			if (CCameraFollowUtilities::cameraFollowModeUsesCapturedOffset(followConfig.mode) &&
				!CCameraFollowUtilities::captureFollowOffsetsFromCamera(state.goalSolver, defaultFollowCamera, trackedTarget, followConfig))
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
			const auto baselinePreset = CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.freeCamera, "free-follow-baseline");
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
			followConfig.offset = SCameraSmokeFollowScenario::FreeWorldOffset;
			trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

			if (!validateFollowScenario(state.goalSolver, planarSpan, state.freeCamera, trackedTarget, followConfig, "free keep-world-offset follow", outError))
				return false;
			if (!restorePresetStrict(state.goalSolver, state.freeCamera, baselinePreset, "Free keep-world-offset smoke failed to restore the baseline preset", outError))
				return false;
		}

		if (state.chaseCamera)
		{
			const auto baselinePreset = CCameraPresetFlowUtilities::capturePreset(state.goalSolver, state.chaseCamera, "chase-follow-baseline");
			SCameraFollowConfig followConfig = {};
			followConfig.enabled = true;
			followConfig.mode = ECameraFollowMode::KeepLocalOffset;
			if (!CCameraFollowUtilities::captureFollowOffsetsFromCamera(state.goalSolver, state.chaseCamera, trackedTarget, followConfig))
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

