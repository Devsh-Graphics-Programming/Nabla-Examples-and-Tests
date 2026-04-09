namespace
{
	using camera_json_t = nlohmann::json;
	using CameraPreset = nbl::core::CCameraPreset;
	constexpr double CameraTinyScalarEpsilon = nbl::system::SCameraSmokeComparisonThresholds::TinyScalarEpsilon;
	constexpr nbl::system::SCameraFollowRegressionThresholds CameraFollowRegressionThresholds = {};

	struct SCameraSmokePersistenceThresholds final
	{
		static constexpr double PositionTolerance = nbl::system::SCameraSmokeComparisonThresholds::StrictPositionTolerance;
		static constexpr double AngularToleranceDeg = nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg;
		static constexpr double ScalarTolerance = nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance;
	};

	struct SCameraSmokeInputDefaults final
	{
		static constexpr auto EventStep = std::chrono::microseconds(16667);
		static constexpr int32_t RelativeMouseMove = 12;
		static constexpr int32_t RelativeMouseMoveY = -8;
		static constexpr int32_t VerticalScroll = 4;
		static constexpr int32_t HorizontalScroll = 2;
	};

	struct SCameraSmokeScriptedCheckDefaults final
	{
		static inline const float64_t3 OrbitCameraPosition = float64_t3(0.0, 1.5, -6.0);
		static inline const float64_t3 OrbitCameraTarget = float64_t3(0.0, 0.0, 0.0);
		static inline const float64_t3 InitialTrackedTargetPosition = float64_t3(2.0, 0.5, -1.5);
		static inline const camera_quaternion_t<float64_t> InitialTrackedTargetOrientation =
			makeQuaternionFromAxisAngle(float64_t3(0.0, 1.0, 0.0), hlsl::radians(35.0));
		static constexpr uint64_t BaselineFrame = 1u;
		static constexpr uint64_t StepFrame = 2u;
		static constexpr uint64_t FollowLockFrame = 3u;
		static constexpr float PositionTolerance = 2.0f;
		static constexpr float MinPositionDelta = 0.005f;
		static constexpr float AngularToleranceDeg = 45.0f;
		static constexpr float MinAngularDeltaDeg = 0.05f;
		static constexpr double StepEventMagnitude = 12.0;
	};

	struct SCameraSmokeFollowScenario final
	{
		static inline const float64_t3 InitialTargetPosition = float64_t3(2.25, -0.75, 1.25);
		static inline const camera_quaternion_t<float64_t> InitialTargetOrientation =
			makeQuaternionFromEulerRadians(float64_t3(0.18, -0.22, 0.41));
		static inline const float64_t3 MovedTargetPosition = float64_t3(-1.5, 0.5, 2.25);
		static inline const camera_quaternion_t<float64_t> MovedTargetOrientation =
			makeQuaternionFromEulerRadians(float64_t3(-0.12, 0.35, 0.27));
		static inline const float64_t3 OrbitWorldOffset = float64_t3(4.0, -1.5, 2.0);
		static inline const float64_t3 FreeWorldOffset = float64_t3(5.0, -2.0, 1.5);
		static constexpr double OrbitRecaptureDeltaDeg = 18.0;
		static constexpr float OrbitRecaptureDistanceDelta = 0.75f;
	};

	struct SCameraSmokeManipulationDefaults final
	{
		static inline const float64_t3 WorldTranslationDelta = float64_t3(1.25, 0.5, 2.0);
		static inline const float64_t3 FreeOrientationYawDeg = float64_t3(0.0, 90.0, 0.0);
		static inline const float64_t3 FreePitchClampSourceDeg = float64_t3(60.0, 0.0, 0.0);
		static constexpr float PitchMinDeg = -15.0f;
		static constexpr float PitchMaxDeg = 15.0f;
		static constexpr double PitchAppliedToleranceDeg = 0.1;
		static constexpr float MinDistanceClampFloor = 0.1f;
		static constexpr float OrbitClampMinScale = 0.5f;
		static constexpr float OrbitClampMaxScale = 0.75f;
		static constexpr float OrbitDistanceDelta = 10.0f;
		static constexpr float PerspectiveNearPlane = 0.1f;
		static constexpr float PerspectiveFarPlane = 100.0f;
		static constexpr float PerspectiveFovDeg = 60.0f;
		static constexpr float OrthoExtent = 10.0f;
	};

	struct SCameraSmokeDynamicPerspectiveDefaults final
	{
		static constexpr float BaseFovDeltaDeg = 7.5f;
		static constexpr float BaseFovMinDeg = 10.0f;
		static constexpr float BaseFovMaxDeg = 150.0f;
		static constexpr float ReferenceDistanceDelta = 1.25f;
		static constexpr float ReferenceDistanceMin = 0.1f;
	};

	struct SCameraSmokeUtilityThresholds final
	{
		static constexpr double PositionWriteback = nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance;
		static constexpr double DynamicPerspectiveDelta = nbl::system::SCameraSmokeComparisonThresholds::StrictScalarTolerance;
		static constexpr double VirtualEventScale = CameraTinyScalarEpsilon;
	};

	struct SCameraSmokePresetMutationDefaults final
	{
		static inline const float64_t3 TargetOffset = float64_t3(0.5, -0.25, 0.75);
		static constexpr double DirectEventMagnitude = 1.0;
	};

	struct SCameraSmokeSequenceDefaults final
	{
		static constexpr float Fps = 4.0f;
		static constexpr float DurationSeconds = 2.0f;
		static constexpr std::array<float, 3u> CaptureFractions = { 0.0f, 0.5f, 1.0f };
		static constexpr float SecondKeyframeTime = 1.0f;
		static constexpr uint64_t DurationFrames = 8ull;
		static constexpr std::array<uint64_t, 3u> CaptureFrameOffsets = { 0ull, 4ull, 7ull };
		static constexpr uint32_t AvailableWindowCount = 2u;
		static constexpr uint32_t PlanarIx = 5u;
		static constexpr uint64_t StartFrame = 11u;
		static constexpr std::array<uint64_t, 3u> CaptureFrames = { 11ull, 15ull, 18ull };
		static constexpr size_t BaselineCheckCount = 1u;
		static constexpr size_t ContinuityCheckCount = 7u;
		static constexpr size_t FollowCheckCount = 7u;
		static inline const float64_t3 TargetPositionA = float64_t3(1.0, 2.0, 3.0);
		static inline const float64_t3 TargetPositionB = float64_t3(4.0, 5.0, 6.0);
		static inline const float64_t3 TargetPositionC = float64_t3(7.0, 8.0, 9.0);
	};

	struct SCameraSmokePlaybackDefaults final
	{
		static constexpr float EndKeyframeTime = 2.0f;
		static constexpr float MidPlaybackTime = 1.5f;
		static constexpr float ResetPlaybackTime = 1.25f;
		static constexpr float OvershootPlaybackTime = 9.0f;
		static constexpr float AdvanceDt = 1.0f;
		static constexpr float WrappedPlaybackTime = 0.5f;
	};

	struct SCameraSmokeRuntimeDefaults final
	{
		static constexpr uint64_t ActionFrame = 3u;
		static constexpr uint64_t FollowFrame = 4u;
		static constexpr int32_t ActivePlanarValue = 4;
		static inline constexpr std::string_view SegmentLabel = "segment-three";
		static inline const float64_t3 GoalPosition = float64_t3(1.0, 2.0, 3.0);
		static inline const float64_t3 TrackedTargetPosition = float64_t3(7.0, 8.0, 9.0);
	};

	struct SCameraSmokeRuntimeParserDefaults final
	{
		static inline constexpr std::string_view CapturePrefix = "parser_smoke";
		static constexpr double KeyboardScale = 2.0;
		static constexpr double RotationScale = 0.5;
		static constexpr uint64_t EventFrame = 2u;
		static constexpr uint64_t StepFrame = 3u;
		static constexpr int32_t ActivePlanarValue = 3;
		static constexpr float MinPositionDelta = 0.01f;
		static constexpr float MaxPositionDelta = 1.0f;
	};

	struct SCameraSmokePresetInventory final
	{
		std::optional<nbl::core::CCameraPreset> orbit = std::nullopt;
		std::optional<nbl::core::CCameraPreset> free = std::nullopt;
		std::optional<nbl::core::CCameraPreset> chase = std::nullopt;
		std::optional<nbl::core::CCameraPreset> dolly = std::nullopt;
		std::optional<nbl::core::CCameraPreset> path = std::nullopt;
		std::optional<nbl::core::CCameraPreset> dollyZoom = std::nullopt;
	};

	struct SCameraSmokeCameraInventory final
	{
		ICamera* orbit = nullptr;
		ICamera* free = nullptr;
		ICamera* chase = nullptr;
		ICamera* dolly = nullptr;
		ICamera* dollyZoom = nullptr;
	};

	enum class EPresetComparePolicy : uint8_t
	{
		None,
		DefaultThresholds,
		StrictThresholds
	};

	inline bool reportHeadlessCameraSmokeFailure(App& app, const std::string& message)
	{
		std::cerr << "[headless-camera-smoke][fail] " << message << std::endl;
		(void)app;
		return false;
	}

	inline std::vector<CVirtualGimbalEvent> collectKeyboardVirtualEvents(
		CGimbalInputBinder& inputBinder,
		const ui::E_KEY_CODE keyCode)
	{
		static std::chrono::microseconds smokeTimestamp = std::chrono::microseconds::zero();
		smokeTimestamp += SCameraSmokeInputDefaults::EventStep;
		const auto pressTs = smokeTimestamp;

		SKeyboardEvent pressEvent(pressTs);
		pressEvent.keyCode = keyCode;
		pressEvent.action = SKeyboardEvent::ECA_PRESSED;
		pressEvent.window = nullptr;

		inputBinder.collectVirtualEvents(pressTs, { .keyboardEvents = { &pressEvent, 1u } });

		smokeTimestamp += SCameraSmokeInputDefaults::EventStep;
		const auto sampleTs = smokeTimestamp;
		return inputBinder.collectVirtualEvents(sampleTs).events;
	}

	inline std::vector<CVirtualGimbalEvent> collectMouseVirtualEvents(
		CGimbalInputBinder& inputBinder,
		std::span<const SMouseEvent> mouseEvents)
	{
		static std::chrono::microseconds smokeTimestamp = std::chrono::microseconds::zero();
		smokeTimestamp += SCameraSmokeInputDefaults::EventStep;
		return inputBinder.collectVirtualEvents(smokeTimestamp, { .mouseEvents = mouseEvents }).events;
	}

	inline std::vector<SMouseEvent> filterOrbitMouseEvents(
		ICamera* const camera,
		std::span<const SMouseEvent> input,
		const bool orbitLookDown)
	{
		if (!(camera && camera->hasCapability(ICamera::SphericalTarget)))
			return std::vector<SMouseEvent>(input.begin(), input.end());

		std::vector<SMouseEvent> filtered;
		filtered.reserve(input.size());
		for (const auto& event : input)
		{
			if (event.type == ui::SMouseEvent::EET_MOVEMENT && !orbitLookDown)
				continue;
			filtered.emplace_back(event);
		}
		return filtered;
	}

	inline SMouseEvent buildMovementSmokeMouseEvent()
	{
		SMouseEvent event(SCameraSmokeInputDefaults::EventStep);
		event.window = nullptr;
		event.type = ui::SMouseEvent::EET_MOVEMENT;
		event.movementEvent.relativeMovementX = SCameraSmokeInputDefaults::RelativeMouseMove;
		event.movementEvent.relativeMovementY = SCameraSmokeInputDefaults::RelativeMouseMoveY;
		return event;
	}

	inline SMouseEvent buildScrollSmokeMouseEvent()
	{
		SMouseEvent event(SCameraSmokeInputDefaults::EventStep);
		event.window = nullptr;
		event.type = ui::SMouseEvent::EET_SCROLL;
		event.scrollEvent.verticalScroll = SCameraSmokeInputDefaults::VerticalScroll;
		event.scrollEvent.horizontalScroll = SCameraSmokeInputDefaults::HorizontalScroll;
		return event;
	}

	inline void buildDirectManipulationEvents(
		const uint32_t allowedEvents,
		std::vector<CVirtualGimbalEvent>& outEvents)
	{
		outEvents.clear();
		outEvents.reserve(3u);

		const auto appendEvent = [&](const CVirtualGimbalEvent::VirtualEventType type)
		{
			outEvents.emplace_back(CVirtualGimbalEvent{
				.type = type,
				.magnitude = SCameraSmokePresetMutationDefaults::DirectEventMagnitude
			});
		};

		const auto tryAppendFirstAllowedEvent = [&](const std::span<const CVirtualGimbalEvent::VirtualEventType> candidates) -> bool
		{
			for (const auto event : candidates)
			{
				if ((allowedEvents & event) != event)
					continue;
				if (std::find_if(outEvents.begin(), outEvents.end(), [&](const CVirtualGimbalEvent& existing) { return existing.type == event; }) != outEvents.end())
					continue;

				appendEvent(event);
				return true;
			}
			return false;
		};

		static constexpr std::array PreferredTranslationEvents = {
			CVirtualGimbalEvent::MoveForward,
			CVirtualGimbalEvent::MoveRight,
			CVirtualGimbalEvent::MoveUp,
			CVirtualGimbalEvent::MoveLeft,
			CVirtualGimbalEvent::MoveDown,
			CVirtualGimbalEvent::MoveBackward
		};
		static constexpr std::array PreferredRotationEvents = {
			CVirtualGimbalEvent::PanRight,
			CVirtualGimbalEvent::TiltUp,
			CVirtualGimbalEvent::RollRight
		};

		const bool appendedTranslation = tryAppendFirstAllowedEvent(PreferredTranslationEvents);
		const bool appendedRotation = tryAppendFirstAllowedEvent(PreferredRotationEvents);
		if (appendedTranslation && !appendedRotation)
			tryAppendFirstAllowedEvent(PreferredTranslationEvents);

		if (!outEvents.empty())
			return;

		for (const auto event : CVirtualGimbalEvent::VirtualEventsTypeTable)
		{
			if ((allowedEvents & event) != event)
				continue;

			appendEvent(event);
			return;
		}
	}

	inline ICamera* findCameraByKind(
		const std::span<const smart_refctd_ptr<ICamera>> cameras,
		const ICamera::CameraKind kind)
	{
		for (const auto& cameraRef : cameras)
		{
			auto* const camera = cameraRef.get();
			if (camera && camera->getKind() == kind)
				return camera;
		}
		return nullptr;
	}

	inline uint32_t expectedMissingGoalStateMaskForIssue(const CCameraGoalSolver::SApplyResult::EIssue issue)
	{
		switch (issue)
		{
			case CCameraGoalSolver::SApplyResult::MissingPathState:
				return ICamera::GoalStatePath;
			case CCameraGoalSolver::SApplyResult::MissingDynamicPerspectiveState:
				return ICamera::GoalStateDynamicPerspective;
			case CCameraGoalSolver::SApplyResult::MissingSphericalTargetState:
				return ICamera::GoalStateSphericalTarget;
			default:
				return ICamera::GoalStateNone;
		}
	}

	inline void storeInitialPresetForKind(
		const ICamera::CameraKind kind,
		const CameraPreset& preset,
		SCameraSmokePresetInventory& inventory)
	{
		switch (kind)
		{
			case ICamera::CameraKind::Orbit:
				inventory.orbit = preset;
				break;
			case ICamera::CameraKind::Free:
				inventory.free = preset;
				break;
			case ICamera::CameraKind::Chase:
				inventory.chase = preset;
				break;
			case ICamera::CameraKind::Dolly:
				inventory.dolly = preset;
				break;
			case ICamera::CameraKind::Path:
				inventory.path = preset;
				break;
			case ICamera::CameraKind::DollyZoom:
				inventory.dollyZoom = preset;
				break;
			default:
				break;
		}
	}

	inline SCameraSmokeCameraInventory collectSmokeCameras(const std::span<const smart_refctd_ptr<ICamera>> cameras)
	{
		return {
			.orbit = findCameraByKind(cameras, ICamera::CameraKind::Orbit),
			.free = findCameraByKind(cameras, ICamera::CameraKind::Free),
			.chase = findCameraByKind(cameras, ICamera::CameraKind::Chase),
			.dolly = findCameraByKind(cameras, ICamera::CameraKind::Dolly),
			.dollyZoom = findCameraByKind(cameras, ICamera::CameraKind::DollyZoom)
		};
	}

	inline bool cameraMatchesPreset(
		const CCameraGoalSolver& goalSolver,
		ICamera* const camera,
		const CameraPreset& preset,
		const EPresetComparePolicy comparePolicy)
	{
		switch (comparePolicy)
		{
			case EPresetComparePolicy::None:
				return true;
			case EPresetComparePolicy::DefaultThresholds:
				return nbl::system::comparePresetToCameraStateWithDefaultThresholds(goalSolver, camera, preset);
			case EPresetComparePolicy::StrictThresholds:
				return nbl::system::comparePresetToCameraStateWithStrictThresholds(goalSolver, camera, preset);
			default:
				return false;
		}
	}

	inline std::string buildPresetSmokeMismatchMessage(
		std::string_view prefix,
		const CCameraGoalSolver& goalSolver,
		ICamera* const camera,
		const CameraPreset& preset)
	{
		return std::string(prefix) + " " + nbl::core::describePresetCameraMismatch(goalSolver, camera, preset);
	}

	inline bool applyPresetAndValidate(
		const CCameraGoalSolver& goalSolver,
		ICamera* const camera,
		const CameraPreset& preset,
		const EPresetComparePolicy comparePolicy,
		const bool requireChanged,
		const bool requireExact,
		const std::string_view failurePrefix,
		std::string& outError)
	{
		const auto applyResult = nbl::core::applyPresetDetailed(goalSolver, camera, preset);
		if (!applyResult.succeeded() ||
			(requireChanged && !applyResult.changed()) ||
			(requireExact && !applyResult.exact))
		{
            outError = std::string(failurePrefix) + ". " + CCameraTextUtilities::describeApplyResult(applyResult);
			return false;
		}

		if (!cameraMatchesPreset(goalSolver, camera, preset, comparePolicy))
		{
			outError = buildPresetSmokeMismatchMessage(failurePrefix, goalSolver, camera, preset);
			return false;
		}

		return true;
	}

	inline bool restorePresetStrict(
		const CCameraGoalSolver& goalSolver,
		ICamera* const camera,
		const CameraPreset& preset,
		const std::string_view failurePrefix,
		std::string& outError)
	{
		const auto restoreResult = nbl::core::applyPresetDetailed(goalSolver, camera, preset);
		if (restoreResult.succeeded() && nbl::system::comparePresetToCameraStateWithStrictThresholds(goalSolver, camera, preset))
			return true;

        outError = std::string(failurePrefix) + ". " + CCameraTextUtilities::describeApplyResult(restoreResult);
		if (camera)
			outError += " " + nbl::core::describePresetCameraMismatch(goalSolver, camera, preset);
		return false;
	}

	inline bool buildAndValidateFollowTargetContract(
		const CCameraGoalSolver& solver,
		std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const SCameraFollowConfig& followConfig,
		nbl::system::SCameraFollowApplyValidationResult& outResult,
		std::string* const outError)
	{
		std::string regressionError;
        if (nbl::system::CCameraFollowRegressionUtilities::buildApplyAndValidateFollowTargetContract(
                solver,
                camera,
                trackedTarget,
				followConfig,
				outResult,
				&regressionError,
				nullptr))
		{
			nbl::system::SCameraProjectionContext projectionContext = {};
			if (!nbl::ui::tryBuildCameraProjectionContext(planarProjections, camera, projectionContext))
				return true;

			nbl::system::SCameraFollowRegressionResult postApplyRegression = {};
            if (!nbl::system::CCameraFollowRegressionUtilities::validateFollowTargetContract(
                    camera,
                    trackedTarget,
                    followConfig,
					outResult.goal,
					postApplyRegression,
					&regressionError,
					&projectionContext,
					CameraFollowRegressionThresholds))
			{
				if (outError)
					*outError = regressionError;
				return false;
			}

			outResult.regression = postApplyRegression;
			return true;
		}

		if (outError)
			*outError = regressionError;
		return false;
	}

	inline SCameraFollowVisualMetrics buildFollowVisualMetricsForCamera(
		const std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const SCameraFollowConfig& followConfig);

	inline bool verifyFollowVisualMetrics(
		const std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const SCameraFollowConfig& followConfig,
		const char* const label,
		std::string* const outError)
	{
		const auto metrics = buildFollowVisualMetricsForCamera(planarProjections, camera, trackedTarget, followConfig);
		nbl::system::SCameraProjectionContext projectionContext = {};
		const bool expectsProjectedMetrics = nbl::ui::tryBuildCameraProjectionContext(planarProjections, camera, projectionContext);
		if (!metrics.active)
		{
			if (outError)
				*outError = std::string("Follow visual metrics smoke was inactive for ") + label + ".";
			return false;
		}
		if (nbl::core::CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(followConfig.mode) && !metrics.lockValid)
		{
			if (outError)
				*outError = std::string("Follow visual metrics smoke was missing lock metrics for ") + label + ".";
			return false;
		}
		if (expectsProjectedMetrics && !metrics.projectedValid)
		{
			if (outError)
				*outError = std::string("Follow visual metrics smoke was missing projected metrics for ") + label + ".";
			return false;
		}
		if (metrics.projectedValid && metrics.projectedTarget.radius > CameraFollowRegressionThresholds.projectedNdcTolerance)
		{
			if (outError)
			{
				const auto targetPosition = trackedTarget.getGimbal().getPosition();
				const auto cameraPosition = camera ? camera->getGimbal().getPosition() : float64_t3(0.0);
				const auto viewMatrix = camera ? hlsl::getMatrix3x4As4x4(camera->getGimbal().getViewMatrix()) : float64_t4x4(1.0);
				const auto targetView = hlsl::mul(viewMatrix, float64_t4(targetPosition, 1.0));
				std::ostringstream oss;
				oss << "Follow visual metrics smoke had projected center error for " << label
					<< ". ndc=(" << metrics.projectedTarget.ndc.x << ", " << metrics.projectedTarget.ndc.y << ")"
					<< " radius=" << metrics.projectedTarget.radius
					<< " lock_deg=" << metrics.lockAngleDeg
					<< " target_distance=" << metrics.targetDistance
					<< " camera_pos=(" << cameraPosition.x << ", " << cameraPosition.y << ", " << cameraPosition.z << ")"
					<< " target_pos=(" << targetPosition.x << ", " << targetPosition.y << ", " << targetPosition.z << ")"
					<< " target_view=(" << targetView.x << ", " << targetView.y << ", " << targetView.z << ", " << targetView.w << ")";
				*outError = oss.str();
			}
			return false;
		}
		return true;
	}

	inline bool validateFollowScenario(
		const CCameraGoalSolver& goalSolver,
		std::span<const smart_refctd_ptr<planar_projection_t>> planarSpan,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const SCameraFollowConfig& followConfig,
		const std::string_view label,
		std::string& outError)
	{
		nbl::system::SCameraFollowApplyValidationResult followResult = {};
		std::string followError;
		if (!buildAndValidateFollowTargetContract(
				goalSolver,
				planarSpan,
				camera,
				trackedTarget,
				followConfig,
				followResult,
				&followError))
		{
			outError = std::string("Follow smoke contract failed for ") + std::string(label) + ". " + followError;
			return false;
		}
		if (!verifyFollowVisualMetrics(planarSpan, camera, trackedTarget, followConfig, label.data(), &followError))
		{
			outError = followError;
			return false;
		}
		return true;
	}

	inline bool runPerCameraPresetAndBindingSmoke(
		const CCameraGoalSolver& goalSolver,
		const std::span<const smart_refctd_ptr<ICamera>> cameras,
		SCameraSmokePresetInventory& initialPresets,
		std::string& outError)
	{
		for (const auto& cameraRef : cameras)
		{
			auto* const camera = cameraRef.get();
			if (!camera)
			{
				outError = "Null camera instance.";
				return false;
			}

			CGimbalInputBinder inputBinder;
            CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(inputBinder, *camera);

			const std::string cameraIdentifier(camera->getIdentifier());
			const auto initialPreset = nbl::core::capturePreset(goalSolver, camera, "smoke-initial");
            const auto initialCompatibility = nbl::core::CCameraGoalAnalysisUtilities::analyzePresetApply(goalSolver, camera, initialPreset).compatibility;
			if (!initialCompatibility.exact || initialCompatibility.missingGoalStateMask != ICamera::GoalStateNone)
			{
				outError = "Preset compatibility smoke failed for camera \"" + cameraIdentifier +
                    "\". missing=" + CCameraTextUtilities::describeGoalStateMask(initialCompatibility.missingGoalStateMask);
				return false;
			}

			storeInitialPresetForKind(camera->getKind(), initialPreset, initialPresets);

			if (!nbl::core::applyPreset(goalSolver, camera, initialPreset))
			{
				outError = "Preset no-op smoke failed for camera \"" + cameraIdentifier + "\".";
				return false;
			}

			if (initialPreset.goal.hasTargetPosition)
			{
				CameraPreset shiftedPreset = initialPreset;
				shiftedPreset.goal.targetPosition += SCameraSmokePresetMutationDefaults::TargetOffset;

				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						shiftedPreset,
						EPresetComparePolicy::None,
						true,
						true,
						"Preset target apply smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}

				ICamera::SphericalTargetState shiftedState;
				if (!camera->tryGetSphericalTargetState(shiftedState) ||
					!hlsl::nearlyEqualVec3(shiftedState.target, shiftedPreset.goal.targetPosition, SCameraSmokeUtilityThresholds::PositionWriteback))
				{
					outError = "Preset target writeback smoke failed for camera \"" + cameraIdentifier + "\".";
					return false;
				}

				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						initialPreset,
						EPresetComparePolicy::DefaultThresholds,
						false,
						true,
						"Preset restore smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}

				ICamera::SphericalTargetState restoredState;
				if (!camera->tryGetSphericalTargetState(restoredState) ||
					!hlsl::nearlyEqualVec3(restoredState.target, initialPreset.goal.targetPosition, SCameraSmokeUtilityThresholds::PositionWriteback))
				{
					outError = "Preset target restore smoke failed for camera \"" + cameraIdentifier + "\".";
					return false;
				}
			}

			if (initialPreset.goal.hasDynamicPerspectiveState)
			{
				CameraPreset shiftedPreset = initialPreset;
				shiftedPreset.goal.dynamicPerspectiveState.baseFov =
					std::clamp(
						initialPreset.goal.dynamicPerspectiveState.baseFov + SCameraSmokeDynamicPerspectiveDefaults::BaseFovDeltaDeg,
						SCameraSmokeDynamicPerspectiveDefaults::BaseFovMinDeg,
						SCameraSmokeDynamicPerspectiveDefaults::BaseFovMaxDeg);
				if (hlsl::abs(static_cast<double>(
						shiftedPreset.goal.dynamicPerspectiveState.baseFov -
						initialPreset.goal.dynamicPerspectiveState.baseFov)) < SCameraSmokeUtilityThresholds::DynamicPerspectiveDelta)
				{
					shiftedPreset.goal.dynamicPerspectiveState.baseFov =
						std::max(
							SCameraSmokeDynamicPerspectiveDefaults::BaseFovMinDeg,
							initialPreset.goal.dynamicPerspectiveState.baseFov - SCameraSmokeDynamicPerspectiveDefaults::BaseFovDeltaDeg);
				}
				shiftedPreset.goal.dynamicPerspectiveState.referenceDistance =
					std::max(
						SCameraSmokeDynamicPerspectiveDefaults::ReferenceDistanceMin,
						initialPreset.goal.dynamicPerspectiveState.referenceDistance + SCameraSmokeDynamicPerspectiveDefaults::ReferenceDistanceDelta);

				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						shiftedPreset,
						EPresetComparePolicy::StrictThresholds,
						true,
						false,
						"Preset dynamic perspective apply smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}

				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						initialPreset,
						EPresetComparePolicy::StrictThresholds,
						false,
						false,
						"Preset dynamic perspective restore smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}
			}

			const uint32_t allowed = camera->getAllowedVirtualEvents();
			std::vector<CVirtualGimbalEvent> directEvents;
			buildDirectManipulationEvents(allowed, directEvents);
			if (directEvents.empty())
			{
				outError = "No allowed virtual events for camera \"" + cameraIdentifier + "\".";
				return false;
			}

			nbl::system::SCameraManipulationDelta directDelta = {};
			if (!nbl::system::tryManipulateCameraAndMeasureDelta(camera, { directEvents.data(), directEvents.size() }, directDelta, CameraTinyScalarEpsilon))
			{
				outError = "Direct manipulate smoke failed for camera \"" + cameraIdentifier + "\".";
				return false;
			}

			{
				const auto modifiedPreset = nbl::core::capturePreset(goalSolver, camera, "smoke-direct");
				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						initialPreset,
						EPresetComparePolicy::StrictThresholds,
						false,
						false,
						"Preset restore from direct smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}

				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						modifiedPreset,
						EPresetComparePolicy::StrictThresholds,
						true,
						false,
						"Preset apply from direct smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}

				if (!applyPresetAndValidate(
						goalSolver,
						camera,
						initialPreset,
						EPresetComparePolicy::StrictThresholds,
						false,
						false,
						"Preset final restore smoke failed for camera \"" + cameraIdentifier + "\"",
						outError))
				{
					return false;
				}
			}

			bool keyboardOk = false;
			nbl::system::SCameraManipulationDelta keyboardDelta = {};
			for (const auto key : nbl::ui::SCameraInputBindingPhysicalGroups::KeyboardProbeCodes)
			{
                CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(inputBinder, *camera);
				auto keyboardEvents = collectKeyboardVirtualEvents(inputBinder, key);
				if (keyboardEvents.empty())
					continue;
				if (nbl::system::tryManipulateCameraAndMeasureDelta(camera, { keyboardEvents.data(), keyboardEvents.size() }, keyboardDelta, CameraTinyScalarEpsilon))
				{
					keyboardOk = true;
					break;
				}
			}
			if (!keyboardOk)
			{
				outError = "Keyboard binding smoke failed for camera \"" + cameraIdentifier + "\".";
				return false;
			}

            const auto& mousePreset = CCameraInputBindingUtilities::getDefaultCameraMouseMappingPreset(*camera);
            const bool hasMoveMapping = nbl::ui::CCameraInputBindingUtilities::hasMouseRelativeMovementBinding(mousePreset);
            const bool hasScrollMapping = nbl::ui::CCameraInputBindingUtilities::hasMouseScrollBinding(mousePreset);

			nbl::system::SCameraManipulationDelta mouseMoveDelta = {};
			if (hasMoveMapping)
			{
				const auto moveEv = buildMovementSmokeMouseEvent();
				const std::array<SMouseEvent, 1u> rawMove = { moveEv };
				auto filteredMoveLookDown = filterOrbitMouseEvents(camera, rawMove, true);
				auto filteredMoveLookUp = filterOrbitMouseEvents(camera, rawMove, false);
				const bool hasBlockedMovement = std::any_of(filteredMoveLookUp.begin(), filteredMoveLookUp.end(), [](const SMouseEvent& ev) { return ev.type == ui::SMouseEvent::EET_MOVEMENT; });
				if (camera->hasCapability(ICamera::SphericalTarget) && hasBlockedMovement)
				{
					outError = "Orbit mouse movement gate failed for camera \"" + cameraIdentifier + "\".";
					return false;
				}

                CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(inputBinder, *camera);
				auto mouseMoveEvents = collectMouseVirtualEvents(inputBinder, { filteredMoveLookDown.data(), filteredMoveLookDown.size() });
				if (mouseMoveEvents.empty())
				{
					outError = "Mouse move virtual events missing for camera \"" + cameraIdentifier + "\".";
					return false;
				}
				if (!nbl::system::tryManipulateCameraAndMeasureDelta(camera, { mouseMoveEvents.data(), mouseMoveEvents.size() }, mouseMoveDelta, CameraTinyScalarEpsilon))
				{
					outError = "Mouse move binding smoke failed for camera \"" + cameraIdentifier + "\".";
					return false;
				}
			}

			nbl::system::SCameraManipulationDelta mouseScrollDelta = {};
			if (hasScrollMapping)
			{
				const auto scrollEv = buildScrollSmokeMouseEvent();
				const std::array<SMouseEvent, 1u> rawScroll = { scrollEv };
				auto filteredScroll = filterOrbitMouseEvents(camera, rawScroll, false);

                CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(inputBinder, *camera);
				auto mouseScrollEvents = collectMouseVirtualEvents(inputBinder, { filteredScroll.data(), filteredScroll.size() });
				if (mouseScrollEvents.empty())
				{
					outError = "Mouse scroll virtual events missing for camera \"" + cameraIdentifier + "\".";
					return false;
				}
				if (!nbl::system::tryManipulateCameraAndMeasureDelta(camera, { mouseScrollEvents.data(), mouseScrollEvents.size() }, mouseScrollDelta, CameraTinyScalarEpsilon))
				{
					outError = "Mouse scroll binding smoke failed for camera \"" + cameraIdentifier + "\".";
					return false;
				}
			}

			std::cout << "[headless-camera-smoke][pass] " << cameraIdentifier
				<< " direct_pos_delta=" << directDelta.position
				<< " direct_rot_delta_deg=" << directDelta.rotationDeg
				<< " kb_pos_delta=" << keyboardDelta.position
				<< " kb_rot_delta_deg=" << keyboardDelta.rotationDeg
				<< " mouse_move_pos_delta=" << mouseMoveDelta.position
				<< " mouse_move_rot_delta_deg=" << mouseMoveDelta.rotationDeg
				<< " mouse_scroll_pos_delta=" << mouseScrollDelta.position
				<< " mouse_scroll_rot_delta_deg=" << mouseScrollDelta.rotationDeg
				<< std::endl;
		}

		return true;
	}

	inline bool verifyApproximateCrossKindApply(
		const CCameraGoalSolver& goalSolver,
		ICamera* const targetCamera,
		const CameraPreset& sourcePreset,
		const CCameraGoalSolver::SApplyResult::EIssue expectedIssue,
		const char* const label,
		std::string& outError)
	{
		if (!targetCamera)
			return true;

		const uint32_t expectedMissingGoalStateMask = expectedMissingGoalStateMaskForIssue(expectedIssue);
        const auto compatibility = nbl::core::CCameraGoalAnalysisUtilities::analyzePresetApply(goalSolver, targetCamera, sourcePreset).compatibility;
		if (compatibility.exact || compatibility.missingGoalStateMask != expectedMissingGoalStateMask)
		{
			outError = std::string("Cross-kind preset compatibility smoke failed for ") + label +
                ". missing=" + CCameraTextUtilities::describeGoalStateMask(compatibility.missingGoalStateMask);
			return false;
		}

		const auto baselinePreset = nbl::core::capturePreset(goalSolver, targetCamera, std::string(label) + "-baseline");
		const auto applyResult = nbl::core::applyPresetDetailed(goalSolver, targetCamera, sourcePreset);
		if (!applyResult.succeeded() || !applyResult.approximate() || !applyResult.hasIssue(expectedIssue))
		{
            outError = std::string("Cross-kind preset smoke failed for ") + label + ". " + CCameraTextUtilities::describeApplyResult(applyResult);
			return false;
		}

		return applyPresetAndValidate(
			goalSolver,
			targetCamera,
			baselinePreset,
			EPresetComparePolicy::StrictThresholds,
			false,
			false,
			std::string("Cross-kind preset restore smoke failed for ") + label,
			outError);
	}

	inline bool verifyExactCrossKindApply(
		const CCameraGoalSolver& goalSolver,
		ICamera* const targetCamera,
		const CameraPreset& sourcePreset,
		const char* const label,
		std::string& outError)
	{
		if (!targetCamera)
			return true;

        const auto compatibility = nbl::core::CCameraGoalAnalysisUtilities::analyzePresetApply(goalSolver, targetCamera, sourcePreset).compatibility;
		if (!compatibility.exact || compatibility.missingGoalStateMask != ICamera::GoalStateNone)
		{
			outError = std::string("Exact cross-kind preset compatibility smoke failed for ") + label +
                ". missing=" + CCameraTextUtilities::describeGoalStateMask(compatibility.missingGoalStateMask);
			return false;
		}

		const auto baselinePreset = nbl::core::capturePreset(goalSolver, targetCamera, std::string(label) + "-baseline");
		if (!applyPresetAndValidate(
				goalSolver,
				targetCamera,
				sourcePreset,
				EPresetComparePolicy::StrictThresholds,
				false,
				true,
				std::string("Exact cross-kind preset smoke failed for ") + label,
				outError))
		{
			return false;
		}

		return applyPresetAndValidate(
			goalSolver,
			targetCamera,
			baselinePreset,
			EPresetComparePolicy::StrictThresholds,
			false,
			true,
			std::string("Exact cross-kind preset restore smoke failed for ") + label,
			outError);
	}

	inline camera_json_t makeScriptedRuntimeParserSmokeJson()
	{
		camera_json_t json = {
			{ "enabled", true },
			{ "capture_prefix", SCameraSmokeRuntimeParserDefaults::CapturePrefix },
			{ "camera_controls", {
				{ "keyboard_scale", SCameraSmokeRuntimeParserDefaults::KeyboardScale },
				{ "rotation_scale", SCameraSmokeRuntimeParserDefaults::RotationScale }
			} },
			{ "events", camera_json_t::array({
				camera_json_t{
					{ "frame", SCameraSmokeRuntimeParserDefaults::EventFrame },
					{ "type", "action" },
					{ "action", "set_active_planar" },
					{ "value", SCameraSmokeRuntimeParserDefaults::ActivePlanarValue }
				},
				camera_json_t{
					{ "frame", SCameraSmokeRuntimeParserDefaults::EventFrame },
					{ "type", "keyboard" },
					{ "key", "W" },
					{ "action", "pressed" },
					{ "capture", true }
				}
			}) },
			{ "checks", camera_json_t::array({
				camera_json_t{
					{ "frame", SCameraSmokeRuntimeParserDefaults::EventFrame },
					{ "kind", "baseline" }
				},
				camera_json_t{
					{ "frame", SCameraSmokeRuntimeParserDefaults::StepFrame },
					{ "kind", "gimbal_step" },
					{ "min_pos_delta", SCameraSmokeRuntimeParserDefaults::MinPositionDelta },
					{ "max_pos_delta", SCameraSmokeRuntimeParserDefaults::MaxPositionDelta }
				}
			}) }
		};
		return json;
	}

	inline SCameraFollowVisualMetrics buildFollowVisualMetricsForCamera(
		const std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const SCameraFollowConfig& followConfig)
	{
		nbl::system::SCameraProjectionContext projectionContext = {};
		const bool hasProjectionContext = nbl::ui::tryBuildCameraProjectionContext(planarProjections, camera, projectionContext);
        return nbl::system::CCameraFollowRegressionUtilities::buildFollowVisualMetrics(
            camera,
            trackedTarget,
            &followConfig,
			hasProjectionContext ? &projectionContext : nullptr);
	}

	inline float32_t3x4 buildFollowTargetMarkerWorldForSmoke(const CTrackedTarget& trackedTarget)
	{
		return buildFollowTargetMarkerWorldTransform(
			trackedTarget,
			SCameraAppSceneDefaults::FollowTargetMarkerScale);
	}

	inline bool verifyFollowTargetContractForSmoke(
		const CCameraGoalSolver& goalSolver,
		const std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const SCameraFollowConfig& followConfig,
		const CCameraGoal& followGoal,
		const std::string_view label,
		std::string& outError)
	{
		nbl::system::SCameraFollowRegressionResult regression = {};
		std::string regressionError;
		nbl::system::SCameraProjectionContext projectionContext = {};
		const bool hasProjectionContext = nbl::ui::tryBuildCameraProjectionContext(planarProjections, camera, projectionContext);
        if (nbl::system::CCameraFollowRegressionUtilities::validateFollowTargetContract(
                camera,
                trackedTarget,
                followConfig,
				followGoal,
				regression,
				&regressionError,
				hasProjectionContext ? &projectionContext : nullptr,
				CameraFollowRegressionThresholds))
		{
			return true;
		}

		outError = std::string("Follow smoke contract failed for ") + std::string(label) + ". " + regressionError;
		return false;
	}

	inline bool verifyFollowTargetMarkerAlignmentForSmoke(
		const CTrackedTarget& trackedTarget,
		const std::string_view label,
		std::string& outError)
	{
		const auto markerWorld = buildFollowTargetMarkerWorldForSmoke(trackedTarget);
		const auto markerTransform = hlsl::transpose(getMatrix3x4As4x4(markerWorld));
		const auto markerPosition = getCastedVector<float64_t>(float32_t3(markerTransform[3]));
		const auto positionDelta = markerPosition - trackedTarget.getGimbal().getPosition();
		const auto errorLength = length(positionDelta);
		if (hlsl::isFiniteScalar(errorLength) && errorLength <= CameraTinyScalarEpsilon)
			return true;

		outError = std::string("Follow target marker alignment smoke failed for ") + std::string(label) + ".";
		return false;
	}

	inline bool verifyOffsetFollowRecaptureForSmoke(
		const CCameraGoalSolver& goalSolver,
		const std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
		ICamera* const camera,
		const CTrackedTarget& trackedTarget,
		const std::string_view label,
		std::string& outError)
	{
		if (!camera)
			return true;

		const auto baselinePreset = nbl::core::capturePreset(goalSolver, camera, std::string(label) + " baseline");
		SCameraFollowConfig followConfig = {};
		followConfig.enabled = true;
		followConfig.mode = ECameraFollowMode::KeepLocalOffset;

		if (!nbl::core::CCameraFollowUtilities::captureFollowOffsetsFromCamera(goalSolver, camera, trackedTarget, followConfig))
		{
			outError = std::string("Follow recapture smoke failed to capture initial offset for ") + std::string(label) + ".";
			return false;
		}

		const auto initialApply = nbl::core::CCameraFollowUtilities::applyFollowToCamera(goalSolver, camera, trackedTarget, followConfig);
		if (!initialApply.succeeded())
		{
			outError = std::string("Follow recapture smoke failed to apply initial follow for ") + std::string(label) + ".";
			return false;
		}

		auto editedPreset = nbl::core::capturePreset(goalSolver, camera, std::string(label) + " edited");
		if (!editedPreset.goal.hasOrbitState)
		{
			outError = std::string("Follow recapture smoke missing orbit state for ") + std::string(label) + ".";
			return false;
		}

        editedPreset.goal.orbitUv.x = hlsl::wrapAngleRad(
            editedPreset.goal.orbitUv.x + hlsl::radians(SCameraSmokeFollowScenario::OrbitRecaptureDeltaDeg));
		editedPreset.goal.orbitDistance = std::clamp(
			editedPreset.goal.orbitDistance + SCameraSmokeFollowScenario::OrbitRecaptureDistanceDelta,
			CSphericalTargetCamera::MinDistance,
			CSphericalTargetCamera::MaxDistance);
		editedPreset.goal = nbl::core::CCameraGoalUtilities::canonicalizeGoal(editedPreset.goal);
		if (!nbl::core::CCameraGoalUtilities::isGoalFinite(editedPreset.goal))
		{
			outError = std::string("Follow recapture smoke produced a non-finite edited goal for ") + std::string(label) + ".";
			return false;
		}

		const auto editedApply = nbl::core::applyPresetDetailed(goalSolver, camera, editedPreset);
		if (!editedApply.succeeded() || !editedApply.changed())
		{
			outError = std::string("Follow recapture smoke failed to apply edited preset for ") + std::string(label) +
                ". " + CCameraTextUtilities::describeApplyResult(editedApply);
			return false;
		}

		const auto reachedEditedPreset = nbl::core::capturePreset(goalSolver, camera, std::string(label) + " reached");

		if (!nbl::core::CCameraFollowUtilities::captureFollowOffsetsFromCamera(goalSolver, camera, trackedTarget, followConfig))
		{
			outError = std::string("Follow recapture smoke failed to recapture offset for ") + std::string(label) + ".";
			return false;
		}

		CCameraGoal recapturedGoal = {};
		if (!nbl::core::CCameraFollowUtilities::tryBuildFollowGoal(goalSolver, camera, trackedTarget, followConfig, recapturedGoal))
		{
			outError = std::string("Follow recapture smoke failed to rebuild follow goal for ") + std::string(label) + ".";
			return false;
		}

		const auto recapturedApply = nbl::core::CCameraFollowUtilities::applyFollowToCamera(goalSolver, camera, trackedTarget, followConfig);
		if (!recapturedApply.succeeded())
		{
			outError = std::string("Follow recapture smoke failed to apply recaptured follow for ") + std::string(label) +
                ". " + CCameraTextUtilities::describeApplyResult(recapturedApply);
			return false;
		}

		if (!nbl::system::comparePresetToCameraStateWithStrictThresholds(goalSolver, camera, reachedEditedPreset))
		{
			outError = std::string("Follow recapture smoke mismatch for ") + std::string(label) + ". " +
				nbl::core::describePresetCameraMismatch(goalSolver, camera, reachedEditedPreset);
			return false;
		}

		if (!verifyFollowTargetContractForSmoke(goalSolver, planarProjections, camera, trackedTarget, followConfig, recapturedGoal, label, outError))
			return false;

		return restorePresetStrict(
			goalSolver,
			camera,
			baselinePreset,
			"Follow recapture smoke failed to restore baseline for " + std::string(label),
			outError);
	}

	inline bool verifyScriptedRuntimeFrameBatch(std::string* const outError)
	{
		CCameraScriptedTimeline timeline = {};
		nbl::system::appendScriptedActionEvent(
			timeline,
			SCameraSmokeRuntimeDefaults::ActionFrame,
			CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar,
			SCameraSmokeRuntimeDefaults::ActivePlanarValue);
		{
			CCameraGoal goal = {};
			goal.position = SCameraSmokeRuntimeDefaults::GoalPosition;
			nbl::system::appendScriptedGoalEvent(timeline, SCameraSmokeRuntimeDefaults::ActionFrame, goal, true);
		}
		nbl::system::appendScriptedSegmentLabelEvent(
			timeline,
			SCameraSmokeRuntimeDefaults::ActionFrame,
			std::string(SCameraSmokeRuntimeDefaults::SegmentLabel));
		{
			float64_t4x4 transform = float64_t4x4(1.0);
			transform[3] = float64_t4(SCameraSmokeRuntimeDefaults::TrackedTargetPosition, 1.0);
			nbl::system::appendScriptedTrackedTargetTransformEvent(timeline, SCameraSmokeRuntimeDefaults::FollowFrame, transform);
		}

		size_t nextEventIndex = 0u;
		CCameraScriptedFrameEvents batch;
		nbl::system::dequeueScriptedFrameEvents(timeline.events, nextEventIndex, SCameraSmokeRuntimeDefaults::ActionFrame, batch);
		if (nextEventIndex != 3u || batch.actions.size() != 1u || batch.goals.size() != 1u ||
			batch.segmentLabels.size() != 1u || !batch.mouse.empty() || !batch.keyboard.empty())
		{
			if (outError)
				*outError = "Scripted runtime frame batch smoke failed for frame 3.";
			return false;
		}
		if (batch.actions.front().kind != CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar ||
			batch.actions.front().value != SCameraSmokeRuntimeDefaults::ActivePlanarValue ||
			batch.segmentLabels.front() != SCameraSmokeRuntimeDefaults::SegmentLabel)
		{
			if (outError)
				*outError = "Scripted runtime frame batch payload smoke failed for frame 3.";
			return false;
		}

		nbl::system::dequeueScriptedFrameEvents(timeline.events, nextEventIndex, SCameraSmokeRuntimeDefaults::FollowFrame, batch);
		if (nextEventIndex != timeline.events.size() || batch.trackedTargetTransforms.size() != 1u ||
			!batch.actions.empty() || !batch.goals.empty())
		{
			if (outError)
				*outError = "Scripted runtime frame batch smoke failed for frame 4.";
			return false;
		}
		const auto trackedTargetPosition = float64_t3(batch.trackedTargetTransforms.front().transform[3]);
		if (!hlsl::nearlyEqualVec3(trackedTargetPosition, SCameraSmokeRuntimeDefaults::TrackedTargetPosition, CameraTinyScalarEpsilon))
		{
			if (outError)
				*outError = "Scripted runtime tracked-target payload smoke failed.";
			return false;
		}

		return true;
	}

	inline bool verifyScriptedRuntimeParser(std::string* const outError)
	{
		nbl::system::CCameraScriptedInputParseResult parsed;
		std::string parseError;
		const std::string scriptText = makeScriptedRuntimeParserSmokeJson().dump();
		if (!nbl::system::readCameraScriptedInput(scriptText, parsed, &parseError))
		{
			if (outError)
				*outError = "Scripted runtime parser smoke failed to parse low-level runtime payload. " + parseError;
			return false;
		}
		if (!parsed.enabled ||
			parsed.capturePrefix != SCameraSmokeRuntimeParserDefaults::CapturePrefix ||
			!parsed.cameraControls.hasKeyboardScale ||
			!parsed.cameraControls.hasRotationScale)
		{
			if (outError)
				*outError = "Scripted runtime parser smoke lost top-level metadata.";
			return false;
		}
		if (parsed.timeline.events.size() != 2u || parsed.timeline.checks.size() != 2u || parsed.timeline.captureFrames.size() != 1u)
		{
			if (outError)
				*outError = "Scripted runtime parser smoke produced wrong payload counts.";
			return false;
		}
		if (parsed.timeline.captureFrames.front() != SCameraSmokeRuntimeParserDefaults::EventFrame)
		{
			if (outError)
				*outError = "Scripted runtime parser smoke produced wrong capture frame.";
			return false;
		}

		size_t nextEventIndex = 0u;
		CCameraScriptedFrameEvents batch;
		nbl::system::dequeueScriptedFrameEvents(parsed.timeline.events, nextEventIndex, SCameraSmokeRuntimeParserDefaults::EventFrame, batch);
		if (batch.actions.size() != 1u ||
			batch.keyboard.size() != 1u ||
			batch.actions.front().value != SCameraSmokeRuntimeParserDefaults::ActivePlanarValue)
		{
			if (outError)
				*outError = "Scripted runtime parser smoke produced wrong frame-two batch.";
			return false;
		}
		if (parsed.timeline.checks.front().kind != CCameraScriptedInputCheck::Kind::Baseline ||
			parsed.timeline.checks.back().kind != CCameraScriptedInputCheck::Kind::GimbalStep)
		{
			if (outError)
				*outError = "Scripted runtime parser smoke produced wrong check kinds.";
			return false;
		}

		return true;
	}

	inline bool verifyScriptedCheckRunner(const CCameraGoalSolver& goalSolver, std::string* const outError)
	{
		auto orbitCamera = core::make_smart_refctd_ptr<COrbitCamera>(
			SCameraSmokeScriptedCheckDefaults::OrbitCameraPosition,
			SCameraSmokeScriptedCheckDefaults::OrbitCameraTarget);
		CTrackedTarget trackedTarget(
			SCameraSmokeScriptedCheckDefaults::InitialTrackedTargetPosition,
			SCameraSmokeScriptedCheckDefaults::InitialTrackedTargetOrientation);

		CCameraScriptedTimeline timeline = {};
		nbl::system::appendScriptedBaselineCheck(timeline, SCameraSmokeScriptedCheckDefaults::BaselineFrame);
		nbl::system::appendScriptedGimbalStepCheck(
			timeline,
			SCameraSmokeScriptedCheckDefaults::StepFrame,
			true,
			SCameraSmokeScriptedCheckDefaults::PositionTolerance,
			SCameraSmokeScriptedCheckDefaults::MinPositionDelta,
			true,
			SCameraSmokeScriptedCheckDefaults::AngularToleranceDeg,
			SCameraSmokeScriptedCheckDefaults::MinAngularDeltaDeg);
		nbl::system::appendScriptedFollowTargetLockCheck(
			timeline,
			SCameraSmokeScriptedCheckDefaults::FollowLockFrame,
			CameraFollowRegressionThresholds.lockAngleToleranceDeg,
			CameraFollowRegressionThresholds.projectedNdcTolerance);

		CCameraScriptedCheckRuntimeState state = {};
		{
			const auto frameResult = evaluateScriptedChecksForFrame(
				timeline.checks,
				state,
				{
					.frame = SCameraSmokeScriptedCheckDefaults::BaselineFrame,
					.camera = orbitCamera.get()
				});
			if (frameResult.hadFailures || state.nextCheckIndex != 1u || !state.baseline.valid || !state.step.valid)
			{
				const auto& gimbal = orbitCamera->getGimbal();
				const auto pos = gimbal.getPosition();
				const auto orientation = gimbal.getOrientation();
				const auto basis = gimbal.getOrthonornalMatrix();
				const auto eulerDeg = hlsl::getCameraOrientationEulerDegrees(gimbal.getOrientation());
				std::ostringstream oss;
				oss << std::fixed << std::setprecision(6)
					<< "Scripted check runner baseline smoke failed."
					<< " nextCheckIndex=" << state.nextCheckIndex
					<< " baselineValid=" << state.baseline.valid
					<< " stepValid=" << state.step.valid
					<< " pos=(" << pos.x << ", " << pos.y << ", " << pos.z << ")"
					<< " quat=(" << orientation.data.x << ", " << orientation.data.y << ", " << orientation.data.z << ", " << orientation.data.w << ")"
					<< " basis_x=(" << basis[0].x << ", " << basis[0].y << ", " << basis[0].z << ")"
					<< " basis_y=(" << basis[1].x << ", " << basis[1].y << ", " << basis[1].z << ")"
					<< " basis_z=(" << basis[2].x << ", " << basis[2].y << ", " << basis[2].z << ")"
					<< " euler_deg=(" << eulerDeg.x << ", " << eulerDeg.y << ", " << eulerDeg.z << ")";
				if (!frameResult.logs.empty())
					oss << ' ' << frameResult.logs.front().text;
				if (outError)
					*outError = oss.str();
				return false;
			}
		}

		{
			CVirtualGimbalEvent stepEvent = {};
			stepEvent.type = CVirtualGimbalEvent::MoveRight;
			stepEvent.magnitude = SCameraSmokeScriptedCheckDefaults::StepEventMagnitude;
			if (!orbitCamera->manipulate({ &stepEvent, 1u }))
			{
				if (outError)
					*outError = "Scripted check runner smoke failed to manipulate the camera for step validation.";
				return false;
			}

			const auto frameResult = evaluateScriptedChecksForFrame(
				timeline.checks,
				state,
				{
					.frame = SCameraSmokeScriptedCheckDefaults::StepFrame,
					.camera = orbitCamera.get()
				});
			if (frameResult.hadFailures || state.nextCheckIndex != 2u)
			{
				if (outError)
					*outError = std::string("Scripted check runner step smoke failed. ") +
						(!frameResult.logs.empty() ? frameResult.logs.front().text : std::string("missing log details"));
				return false;
			}
		}

		SCameraFollowConfig followConfig = {};
		followConfig.enabled = true;
		followConfig.mode = ECameraFollowMode::OrbitTarget;
		CCameraGoal followGoal = {};
		if (!nbl::core::CCameraFollowUtilities::applyFollowToCamera(goalSolver, orbitCamera.get(), trackedTarget, followConfig, &followGoal).succeeded())
		{
			if (outError)
				*outError = "Scripted check runner smoke failed to apply follow before follow-lock validation.";
			return false;
		}

		{
			const auto frameResult = evaluateScriptedChecksForFrame(
				timeline.checks,
				state,
				{
					.frame = SCameraSmokeScriptedCheckDefaults::FollowLockFrame,
					.camera = orbitCamera.get(),
					.trackedTarget = &trackedTarget,
					.followConfig = &followConfig,
					.goalSolver = &goalSolver
				});
			if (frameResult.hadFailures || state.nextCheckIndex != timeline.checks.size())
			{
				const auto details = !frameResult.logs.empty() ? frameResult.logs.front().text : std::string("missing log details");
				const auto& gimbal = orbitCamera->getGimbal();
				const auto cameraPos = gimbal.getPosition();
				const auto cameraForward = gimbal.getZAxis();
				const auto targetPos = trackedTarget.getGimbal().getPosition();
				const auto desiredForward = normalize(targetPos - cameraPos);
				camera_quaternion_t<float64_t> desiredOrientation = makeIdentityQuaternion<float64_t>();
				if (!nbl::hlsl::tryBuildLookAtOrientation(
						cameraPos,
						targetPos,
						float64_t3(0.0, 1.0, 0.0),
						desiredOrientation))
				{
					if (outError)
						*outError = "Scripted check runner follow-lock smoke failed to build desired look-at orientation.";
					return false;
				}
				const auto desiredBasis = getQuaternionBasisMatrix(desiredOrientation);
				const auto desiredRight = desiredBasis[0];
				const auto desiredUp = desiredBasis[1];
				const auto goalRightVec = normalizeQuaternion(followGoal.orientation).transformVector(float64_t3(1.0, 0.0, 0.0), true);
				const auto goalUpVec = normalizeQuaternion(followGoal.orientation).transformVector(float64_t3(0.0, 1.0, 0.0), true);
				const auto goalForwardVec = normalizeQuaternion(followGoal.orientation).transformVector(float64_t3(0.0, 0.0, 1.0), true);
				const auto goalBasis = getQuaternionBasisMatrix(followGoal.orientation);
				float lockAngle = 0.0f;
				double targetDistance = 0.0;
				const bool hasLockMetrics = nbl::core::CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(gimbal, trackedTarget, lockAngle, &targetDistance);
				std::ostringstream oss;
				oss << std::fixed << std::setprecision(6)
					<< "Scripted check runner follow-lock smoke failed. " << details
					<< " camera_pos=(" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")"
					<< " camera_forward=(" << cameraForward.x << ", " << cameraForward.y << ", " << cameraForward.z << ")"
					<< " target_pos=(" << targetPos.x << ", " << targetPos.y << ", " << targetPos.z << ")"
					<< " desired_forward=(" << desiredForward.x << ", " << desiredForward.y << ", " << desiredForward.z << ")"
					<< " desired_right=(" << desiredRight.x << ", " << desiredRight.y << ", " << desiredRight.z << ")"
					<< " desired_up=(" << desiredUp.x << ", " << desiredUp.y << ", " << desiredUp.z << ")"
					<< " goal_pos=(" << followGoal.position.x << ", " << followGoal.position.y << ", " << followGoal.position.z << ")"
					<< " goal_quat=(" << followGoal.orientation.data.x << ", " << followGoal.orientation.data.y << ", "
					<< followGoal.orientation.data.z << ", " << followGoal.orientation.data.w << ")"
					<< " goal_right_vec=(" << goalRightVec.x << ", " << goalRightVec.y << ", " << goalRightVec.z << ")"
					<< " goal_up_vec=(" << goalUpVec.x << ", " << goalUpVec.y << ", " << goalUpVec.z << ")"
					<< " goal_forward_vec=(" << goalForwardVec.x << ", " << goalForwardVec.y << ", " << goalForwardVec.z << ")"
					<< " goal_basis_x=(" << goalBasis[0].x << ", " << goalBasis[0].y << ", " << goalBasis[0].z << ")"
					<< " goal_basis_y=(" << goalBasis[1].x << ", " << goalBasis[1].y << ", " << goalBasis[1].z << ")"
					<< " goal_basis_z=(" << goalBasis[2].x << ", " << goalBasis[2].y << ", " << goalBasis[2].z << ")";
				if (hasLockMetrics)
					oss << " lock_angle_deg=" << lockAngle << " target_distance=" << targetDistance;
				if (outError)
					*outError = oss.str();
				return false;
			}
		}

		return true;
	}

