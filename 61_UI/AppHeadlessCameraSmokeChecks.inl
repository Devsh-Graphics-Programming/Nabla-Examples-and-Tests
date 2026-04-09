	struct SCameraSmokeResolvedState final
	{
		const CCameraGoalSolver& goalSolver;
		nbl::system::ISystem* system = nullptr;
		const SCameraSmokePresetInventory& initialPresets;
		ICamera* orbitCamera = nullptr;
		ICamera* freeCamera = nullptr;
		ICamera* chaseCamera = nullptr;
		ICamera* dollyCamera = nullptr;
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

		if (std::string_view(nbl::ui::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::All)) != "All" ||
			std::string_view(nbl::ui::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::Exact)) != "Exact" ||
			std::string_view(nbl::ui::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::BestEffort)) != "Best-effort")
		{
			outError = "Presentation utilities smoke returned an unexpected filter label.";
			return false;
		}

		const auto blockedPresentation = nbl::ui::analyzePresetPresentation(state.goalSolver, nullptr, state.initialPresets.orbit.value());
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

		const auto blockedBadges = nbl::ui::collectGoalApplyPresentationBadges(blockedPresentation);
		if (!blockedBadges.blocked || blockedBadges.exact || blockedBadges.bestEffort || blockedPresentation.badges.blocked != blockedBadges.blocked)
		{
			outError = "Presentation utilities smoke produced wrong blocked badge flags.";
			return false;
		}

		if (state.orbitCamera)
		{
			const auto exactPresentation = nbl::ui::analyzePresetPresentation(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value());
			if (!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
				!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
				exactPresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
			{
				outError = "Presentation utilities smoke failed exact filtering.";
				return false;
			}

			const auto exactBadges = nbl::ui::collectGoalApplyPresentationBadges(exactPresentation);
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

			const auto capturePresentation = nbl::ui::analyzeCapturePresentation(state.goalSolver, state.orbitCamera);
			if (!capturePresentation.canCapture || capturePresentation.policyLabel.empty())
			{
				outError = "Presentation utilities smoke failed orbit capture presentation.";
				return false;
			}
		}

		if (state.initialPresets.path.has_value() && state.orbitCamera)
		{
			const auto approximatePresentation = nbl::ui::analyzePresetPresentation(state.goalSolver, state.orbitCamera, state.initialPresets.path.value());
			if (!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
				approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
				!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
			{
				outError = "Presentation utilities smoke failed best-effort filtering.";
				return false;
			}

			const auto approximateBadges = nbl::ui::collectGoalApplyPresentationBadges(approximatePresentation);
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
		if (!nbl::core::comparePresetCollections(
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
		if (!nbl::system::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, loadedTrack))
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
		if (!nbl::core::comparePresetCollections(
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
		if (!nbl::system::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, fileLoadedTrack))
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

			const auto advanceToEnd = nbl::core::advancePlaybackCursor(cursor, playbackTrack, SCameraSmokePlaybackDefaults::AdvanceDt);
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

			nbl::core::resetPlaybackCursor(cursor, SCameraSmokePlaybackDefaults::ResetPlaybackTime);
			if (cursor.playing || hlsl::abs(static_cast<double>(cursor.time - SCameraSmokePlaybackDefaults::ResetPlaybackTime)) > CameraTinyScalarEpsilon)
			{
				outError = "Playback timeline smoke failed to reset cursor.";
				return false;
			}

			cursor.playing = true;
			cursor.loop = true;
			cursor.speed = 1.f;
			cursor.time = SCameraSmokePlaybackDefaults::MidPlaybackTime;
			const auto advanceLoop = nbl::core::advancePlaybackCursor(cursor, playbackTrack, SCameraSmokePlaybackDefaults::AdvanceDt);
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
			nbl::core::clampPlaybackCursorToTrack(playbackTrack, cursor);
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

		if (!nbl::core::sequenceScriptUsesMultiplePresentations(sequence))
		{
			outError = "Sequence compile smoke failed to detect multi-presentation authored defaults.";
			return false;
		}

		const CCameraSequenceTrackedTargetPose referenceTrackedTargetPose = {
			.position = SCameraAppSceneDefaults::DefaultFollowTargetPosition,
			.orientation = SCameraAppSceneDefaults::DefaultFollowTargetOrientation
		};

		nbl::core::CCameraSequenceCompiledSegment compiledSegment;
		std::string compileError;
		if (!nbl::core::compileSequenceSegmentFromReference(
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
		if (!nbl::core::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, true))
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
		if (!nbl::core::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, 1.f, poseAtOne))
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
		if (!nbl::system::appendCompiledSequenceSegmentToScriptedTimeline(
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
		nbl::system::finalizeScriptedTimeline(scriptedTimeline);

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
		nbl::system::dequeueScriptedFrameEvents(scriptedTimeline.events, runtimeNextEventIndex, SCameraSmokeSequenceDefaults::StartFrame, runtimeBatch);
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
			const auto exactSummary = nbl::core::applyPresetToCameraRange(
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
			const auto approximateSummary = nbl::core::applyPresetToCameraRange(
				state.goalSolver,
				std::span<ICamera* const>(approximateTargets.data(), approximateTargets.size()),
				state.initialPresets.path.value());
			if (approximateSummary.targetCount != 1u || approximateSummary.successCount != 1u || approximateSummary.approximateCount != 1u || approximateSummary.failureCount != 0u)
			{
				outError = "Preset apply summary smoke failed for approximate target range.";
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
			nbl::core::scaleVirtualEvents(scaledEvents, static_cast<uint32_t>(scaledEvents.size()), 0.5f, 2.0f);
			if (hlsl::abs(scaledEvents[0].magnitude - 1.0) > SCameraSmokeUtilityThresholds::VirtualEventScale ||
				hlsl::abs(scaledEvents[1].magnitude - 6.0) > SCameraSmokeUtilityThresholds::VirtualEventScale ||
				hlsl::abs(scaledEvents[2].magnitude - 4.0) > SCameraSmokeUtilityThresholds::VirtualEventScale)
			{
				outError = "Camera manipulation utilities smoke failed for virtual-event scaling.";
				return false;
			}
		}

		if (state.initialPresets.free.has_value() && state.freeCamera)
		{
			CameraPreset orientedPreset = state.initialPresets.free.value();
			orientedPreset.goal.orientation = hlsl::makeQuaternionFromEulerDegreesYXZ(SCameraSmokeManipulationDefaults::FreeOrientationYawDeg);
			const auto orientResult = nbl::core::applyPresetDetailed(state.goalSolver, state.freeCamera, orientedPreset);
			if (!orientResult.succeeded() || !nbl::system::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.freeCamera, orientedPreset))
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
			nbl::core::remapTranslationEventsFromWorldToCameraLocal(state.freeCamera, worldTranslationEvents, remappedCount);
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
			if (!hlsl::nearlyEqualVec3(positionDelta, SCameraSmokeManipulationDefaults::WorldTranslationDelta, SCameraSmokeUtilityThresholds::PositionWriteback))
			{
				outError = "Camera manipulation utilities smoke changed world-space translation semantics.";
				return false;
			}

			CameraPreset pitchPreset = state.initialPresets.free.value();
			pitchPreset.goal.orientation = hlsl::makeQuaternionFromEulerDegreesYXZ(SCameraSmokeManipulationDefaults::FreePitchClampSourceDeg);
			const auto pitchResult = nbl::core::applyPresetDetailed(state.goalSolver, state.freeCamera, pitchPreset);
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
			if (!nbl::core::applyCameraConstraints(state.goalSolver, state.freeCamera, freeConstraints))
			{
				outError = "Camera manipulation utilities smoke failed to clamp Free camera orientation.";
				return false;
			}

			const auto freeEulerDeg = hlsl::getCameraOrientationEulerDegrees(state.freeCamera->getGimbal().getOrientation());
			if (hlsl::abs(static_cast<double>(freeEulerDeg.x - SCameraSmokeManipulationDefaults::PitchMaxDeg)) > SCameraSmokeManipulationDefaults::PitchAppliedToleranceDeg)
			{
				outError = "Camera manipulation utilities smoke produced wrong clamped Free camera pitch.";
				return false;
			}

			const auto restoreFree = nbl::core::applyPresetDetailed(state.goalSolver, state.freeCamera, state.initialPresets.free.value());
			if (!restoreFree.succeeded() || !nbl::system::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.freeCamera, state.initialPresets.free.value()))
			{
				outError = "Camera manipulation utilities smoke failed to restore Free camera baseline.";
				return false;
			}
		}

		if (state.initialPresets.orbit.has_value() && state.orbitCamera && state.initialPresets.orbit->goal.hasDistance)
		{
			CameraPreset farOrbitPreset = state.initialPresets.orbit.value();
			farOrbitPreset.goal.distance = state.initialPresets.orbit->goal.distance + SCameraSmokeManipulationDefaults::OrbitDistanceDelta;
			const auto farOrbitResult = nbl::core::applyPresetDetailed(state.goalSolver, state.orbitCamera, farOrbitPreset);
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
			if (!nbl::core::applyCameraConstraints(state.goalSolver, state.orbitCamera, orbitConstraints))
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

			const auto restoreOrbit = nbl::core::applyPresetDetailed(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value());
			if (!restoreOrbit.succeeded() || !nbl::system::comparePresetToCameraStateWithStrictThresholds(state.goalSolver, state.orbitCamera, state.initialPresets.orbit.value()))
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
			if (!nbl::core::syncDynamicPerspectiveProjection(state.dollyZoomCamera, perspectiveProjection))
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
			if (nbl::core::syncDynamicPerspectiveProjection(state.dollyZoomCamera, orthographicProjection))
			{
				outError = "Camera projection utilities smoke unexpectedly synced orthographic projection.";
				return false;
			}
		}

		if (getCameraTypeLabel(ICamera::CameraKind::DollyZoom) != "Dolly Zoom")
		{
			outError = "Camera text utilities smoke failed for Dolly Zoom label.";
			return false;
		}
		if (getCameraTypeDescription(ICamera::CameraKind::Path) != std::string(nbl::core::SCameraPathDefaults::Description))
		{
			outError = "Camera text utilities smoke failed for Path description.";
			return false;
		}
		if (describeGoalStateMask(ICamera::GoalStateNone) != "Pose only")
		{
			outError = "Camera text utilities smoke failed for empty goal-state description.";
			return false;
		}
		if (describeGoalStateMask(ICamera::GoalStateSphericalTarget | ICamera::GoalStateDynamicPerspective) != "Spherical target, Dynamic perspective")
		{
			outError = "Camera text utilities smoke failed for combined goal-state description.";
			return false;
		}

		CCameraGoalSolver::SApplyResult defaultApplyResult;
		const auto applyResultText = describeApplyResult(defaultApplyResult);
		if (applyResultText.find("status=Unsupported") == std::string::npos || applyResultText.find("events=0") == std::string::npos)
		{
			outError = "Camera text utilities smoke failed for apply-result description.";
			return false;
		}

		SCameraPresetApplySummary summary;
		summary.targetCount = 2u;
		summary.successCount = 2u;
		summary.approximateCount = 1u;
		const auto summaryText = nbl::ui::describePresetApplySummary(summary, "none");
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
			const auto baselinePreset = nbl::core::capturePreset(state.goalSolver, state.orbitCamera, "orbit-follow-baseline");
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

			auto followConfig = nbl::core::makeDefaultFollowConfig(defaultFollowCamera);
			if (!followConfig.enabled || followConfig.mode == ECameraFollowMode::Disabled)
				continue;

			const auto label = std::string(defaultFollowCamera->getIdentifier()) + " default follow";
			const auto baselinePreset = nbl::core::capturePreset(state.goalSolver, defaultFollowCamera, label + " baseline");

			trackedTarget.setPose(
				SCameraSmokeFollowScenario::InitialTargetPosition,
				SCameraSmokeFollowScenario::InitialTargetOrientation);
			if ((nbl::core::cameraFollowModeUsesLocalOffset(followConfig.mode) || nbl::core::cameraFollowModeUsesWorldOffset(followConfig.mode)) &&
				!nbl::core::captureFollowOffsetsFromCamera(state.goalSolver, defaultFollowCamera, trackedTarget, followConfig))
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
			const auto baselinePreset = nbl::core::capturePreset(state.goalSolver, state.freeCamera, "free-follow-baseline");
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
			const auto baselinePreset = nbl::core::capturePreset(state.goalSolver, state.chaseCamera, "chase-follow-baseline");
			SCameraFollowConfig followConfig = {};
			followConfig.enabled = true;
			followConfig.mode = ECameraFollowMode::KeepLocalOffset;
			if (!nbl::core::captureFollowOffsetsFromCamera(state.goalSolver, state.chaseCamera, trackedTarget, followConfig))
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
}
