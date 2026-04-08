#include "app/App.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <sstream>
#include <span>
#include <string_view>
#include <vector>

#include "app/AppCameraConfigUtilities.hpp"
#include "app/AppResourceUtilities.hpp"
#include "camera/CCameraPersistence.hpp"
#include "camera/CCameraScriptedRuntimePersistence.hpp"
#include "camera/CCameraSmokeRegressionUtilities.hpp"

namespace
{
	using camera_json_t = nbl::system::camera_json_t;

	constexpr double CameraTinyScalarEpsilon = nbl::system::SCameraSmokeComparisonThresholds::TinyScalarEpsilon;
	constexpr nbl::system::SCameraFollowRegressionThresholds CameraFollowRegressionThresholds = {};
}

bool App::onAppInitialized(smart_refctd_ptr<ISystem>&& system)
{
			argparse::ArgumentParser program("Virtual camera event system demo");

			program.add_argument<std::string>("--file")
				.help("Path to json file with camera inputs");
			program.add_argument("--ci")
				.help("Run in CI mode: capture a screenshot after a few frames and exit.")
				.default_value(false)
				.implicit_value(true);
			program.add_argument<std::string>("--script")
				.help("Path to json file with scripted input events");
			program.add_argument("--script-log")
				.help("Log scripted input and virtual events.")
				.default_value(false)
				.implicit_value(true);
			program.add_argument("--script-visual-debug")
				.help("Enable scripted visual debug overlay and fixed frame pacing.")
				.default_value(false)
				.implicit_value(true);
			program.add_argument("--no-screenshots")
				.help("Disable CI and scripted screenshot captures.")
				.default_value(false)
				.implicit_value(true);
			program.add_argument("--headless-camera-smoke")
				.help("Run a headless camera-only smoke test and exit after initialization.")
				.default_value(false)
				.implicit_value(true);

			try
			{
				program.parse_args({ argv.data(), argv.data() + argv.size() });
			}
			catch (const std::exception& err)
			{
				std::cerr << err.what() << std::endl << program;
				return false;
			}

			m_headlessCameraSmokeMode = program.get<bool>("--headless-camera-smoke");
			if (m_headlessCameraSmokeMode)
			{
				auto fail = [&](const std::string& msg) -> bool
				{
					std::cerr << "[headless-camera-smoke][fail] " << msg << std::endl;
					m_headlessCameraSmokePassed = false;
					return false;
				};

				if (!asset_base_t::onAppInitialized(std::move(system)))
					return fail("Failed to initialize mounted resources for headless camera smoke.");

				auto configPath = [&]() -> std::filesystem::path
				{
					if (program.is_used("--file"))
					{
						std::filesystem::path path = program.get<std::string>("--file");
						if (path.is_relative())
							path = localInputCWD / path;
						return path.lexically_normal();
					}
					return std::filesystem::path(nbl::system::SCameraAppResourcePaths::DefaultCameraConfigRelativePath);
				}();

				camera_json_t j;
				std::string jsonError;
				if (!nbl::system::loadJsonFromPath(*m_system, configPath, j, &jsonError))
					return fail(jsonError);

				std::vector<smart_refctd_ptr<ICamera>> cameras;
				if (!nbl::system::tryLoadCameraCollectionFromJson(j, jsonError, cameras))
					return fail(jsonError);

				auto comparePresetToCameraDefault = [&](ICamera* camera, const CameraPreset& preset) -> bool
				{
					return nbl::system::comparePresetToCameraStateWithDefaultThresholds(m_cameraGoalSolver, camera, preset);
				};

				auto comparePresetToCameraStrict = [&](ICamera* camera, const CameraPreset& preset) -> bool
				{
					return nbl::system::comparePresetToCameraStateWithStrictThresholds(m_cameraGoalSolver, camera, preset);
				};

				auto describePresetMismatch = [&](ICamera* camera, const CameraPreset& preset) -> std::string
				{
					return nbl::core::describePresetCameraMismatch(m_cameraGoalSolver, camera, preset);
				};

				auto tryBuildFollowViewProjForCamera = [&](ICamera* camera, float32_t4x4& outViewProjMatrix) -> bool
				{
					if (!camera)
						return false;

					for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
					{
						auto& planar = m_planarProjections[planarIx];
						if (!planar || planar->getCamera() != camera)
							continue;

						const auto& projections = planar->getPlanarProjections();
						if (projections.empty())
							return false;

						uint32_t projectionIx = 0u;
						for (uint32_t ix = 0u; ix < projections.size(); ++ix)
						{
							if (projections[ix].getParameters().m_type == IPlanarProjection::CProjection::Perspective)
							{
								projectionIx = ix;
								break;
							}
						}

						const auto viewMatrix = getMatrix3x4As4x4(getCastedMatrix<float32_t>(camera->getGimbal().getViewMatrix()));
						const auto projectionMatrix = getCastedMatrix<float32_t>(projections[projectionIx].getProjectionMatrix());
						outViewProjMatrix = mul(projectionMatrix, viewMatrix);
						return true;
					}

					return false;
				};

				auto buildFollowVisualMetricsForCamera = [&](ICamera* camera, const CTrackedTarget& trackedTarget,
					const SCameraFollowConfig& followConfig) -> SCameraFollowVisualMetrics
				{
					float32_t4x4 viewProjMatrix = float32_t4x4(1.0f);
					const bool hasViewProjMatrix = tryBuildFollowViewProjForCamera(camera, viewProjMatrix);
					return nbl::system::buildFollowVisualMetrics(
						camera,
						trackedTarget,
						&followConfig,
						hasViewProjMatrix ? &viewProjMatrix : nullptr);
				};

				auto buildAndValidateFollowTargetContract = [&](ICamera* camera, const CTrackedTarget& trackedTarget,
					const SCameraFollowConfig& followConfig, const char* label, nbl::system::SCameraFollowApplyValidationResult& outResult) -> bool
				{
					std::string regressionError;
					float32_t4x4 viewProjMatrix = float32_t4x4(1.0f);
					const bool hasViewProjMatrix = tryBuildFollowViewProjForCamera(camera, viewProjMatrix);
					if (!nbl::system::buildApplyAndValidateFollowTargetContract(
						m_cameraGoalSolver,
						camera,
						trackedTarget,
						followConfig,
						outResult,
						&regressionError,
						hasViewProjMatrix ? &viewProjMatrix : nullptr))
					{
						return fail(std::string("Follow smoke contract failed for ") + label + ". " + regressionError);
					}
					return true;
				};

				auto verifyFollowVisualMetrics = [&](ICamera* camera, const CTrackedTarget& trackedTarget,
					const SCameraFollowConfig& followConfig, const char* label) -> bool
				{
					const auto metrics = buildFollowVisualMetricsForCamera(camera, trackedTarget, followConfig);
					float32_t4x4 viewProjMatrix = float32_t4x4(1.0f);
					const bool expectsProjectedMetrics = tryBuildFollowViewProjForCamera(camera, viewProjMatrix);
					if (!metrics.active)
						return fail(std::string("Follow visual metrics smoke was inactive for ") + label + ".");
					if (nbl::core::cameraFollowModeLocksViewToTarget(followConfig.mode) && !metrics.lockValid)
						return fail(std::string("Follow visual metrics smoke was missing lock metrics for ") + label + ".");
					if (expectsProjectedMetrics && !metrics.projectedValid)
						return fail(std::string("Follow visual metrics smoke was missing projected metrics for ") + label + ".");
					if (metrics.projectedValid && metrics.projectedNdcRadius > CameraFollowRegressionThresholds.projectedNdcTolerance)
						return fail(std::string("Follow visual metrics smoke had projected center error for ") + label + ".");
					return true;
				};

				auto verifyScriptedRuntimeFrameBatch = [&]() -> bool
				{
					CCameraScriptedTimeline timeline = {};
					nbl::system::appendScriptedActionEvent(timeline, 3u, CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar, 4);
					{
						CCameraGoal goal = {};
						goal.position = float64_t3(1.0, 2.0, 3.0);
						nbl::system::appendScriptedGoalEvent(timeline, 3u, goal, true);
					}
					nbl::system::appendScriptedSegmentLabelEvent(timeline, 3u, "segment-three");
					{
						float64_t4x4 transform = float64_t4x4(1.0);
						transform[3] = float64_t4(7.0, 8.0, 9.0, 1.0);
						nbl::system::appendScriptedTrackedTargetTransformEvent(timeline, 4u, transform);
					}

					size_t nextEventIndex = 0u;
					CCameraScriptedFrameEvents batch;
					nbl::system::dequeueScriptedFrameEvents(timeline.events, nextEventIndex, 3u, batch);
					if (nextEventIndex != 3u || batch.actions.size() != 1u || batch.goals.size() != 1u ||
						batch.segmentLabels.size() != 1u || !batch.mouse.empty() || !batch.keyboard.empty())
					{
						return fail("Scripted runtime frame batch smoke failed for frame 3.");
					}
					if (batch.actions.front().kind != CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar ||
						batch.actions.front().value != 4 || batch.segmentLabels.front() != "segment-three")
					{
						return fail("Scripted runtime frame batch payload smoke failed for frame 3.");
					}

					nbl::system::dequeueScriptedFrameEvents(timeline.events, nextEventIndex, 4u, batch);
					if (nextEventIndex != timeline.events.size() || batch.trackedTargetTransforms.size() != 1u ||
						!batch.actions.empty() || !batch.goals.empty())
					{
						return fail("Scripted runtime frame batch smoke failed for frame 4.");
					}
					const auto trackedTargetPosition = float64_t3(batch.trackedTargetTransforms.front().transform[3]);
					if (!hlsl::nearlyEqualVec3(trackedTargetPosition, float64_t3(7.0, 8.0, 9.0), CameraTinyScalarEpsilon))
						return fail("Scripted runtime tracked-target payload smoke failed.");

					return true;
				};

				auto verifyScriptedRuntimeParser = [&]() -> bool
				{
					std::stringstream script;
					script << R"json({
  "enabled": true,
  "capture_prefix": "parser_smoke",
  "camera_controls": {
    "keyboard_scale": 2.0,
    "rotation_scale": 0.5
  },
  "events": [
    {
      "frame": 2,
      "type": "action",
      "action": "set_active_planar",
      "value": 3
    },
    {
      "frame": 2,
      "type": "keyboard",
      "key": "W",
      "action": "pressed",
      "capture": true
    }
  ],
  "checks": [
    {
      "frame": 2,
      "kind": "baseline"
    },
    {
      "frame": 3,
      "kind": "gimbal_step",
      "min_pos_delta": 0.01,
      "max_pos_delta": 1.0
    }
  ]
})json";

					nbl::system::CCameraScriptedInputParseResult parsed;
					std::string parseError;
					if (!nbl::system::readCameraScriptedInput(script, parsed, &parseError))
						return fail("Scripted runtime parser smoke failed to parse low-level runtime payload. " + parseError);
					if (!parsed.enabled || parsed.capturePrefix != "parser_smoke" || !parsed.cameraControls.hasKeyboardScale || !parsed.cameraControls.hasRotationScale)
						return fail("Scripted runtime parser smoke lost top-level metadata.");
					if (parsed.timeline.events.size() != 2u || parsed.timeline.checks.size() != 2u || parsed.timeline.captureFrames.size() != 1u)
						return fail("Scripted runtime parser smoke produced wrong payload counts.");
					if (parsed.timeline.captureFrames.front() != 2u)
						return fail("Scripted runtime parser smoke produced wrong capture frame.");

					size_t nextEventIndex = 0u;
					CCameraScriptedFrameEvents batch;
					nbl::system::dequeueScriptedFrameEvents(parsed.timeline.events, nextEventIndex, 2u, batch);
					if (batch.actions.size() != 1u || batch.keyboard.size() != 1u || batch.actions.front().value != 3)
						return fail("Scripted runtime parser smoke produced wrong frame-two batch.");
					if (parsed.timeline.checks.front().kind != CCameraScriptedInputCheck::Kind::Baseline ||
						parsed.timeline.checks.back().kind != CCameraScriptedInputCheck::Kind::GimbalStep)
					{
						return fail("Scripted runtime parser smoke produced wrong check kinds.");
					}

					return true;
				};

				auto verifyScriptedCheckRunner = [&]() -> bool
				{
					auto orbitCamera = core::make_smart_refctd_ptr<COrbitCamera>(float64_t3(0.0, 1.5, -6.0), float64_t3(0.0, 0.0, 0.0));
					CTrackedTarget trackedTarget(
						float64_t3(2.0, 0.5, -1.5),
						makeQuaternionFromAxisAngle(float64_t3(0.0, 1.0, 0.0), hlsl::radians(35.0)));

					CCameraScriptedTimeline timeline = {};
					nbl::system::appendScriptedBaselineCheck(timeline, 1u);
					nbl::system::appendScriptedGimbalStepCheck(timeline, 2u, true, 2.0f, 0.005f, true, 45.0f, 0.05f);
					nbl::system::appendScriptedFollowTargetLockCheck(
						timeline,
						3u,
						CameraFollowRegressionThresholds.lockAngleToleranceDeg,
						CameraFollowRegressionThresholds.projectedNdcTolerance);

					CCameraScriptedCheckRuntimeState state = {};
					{
						const auto frameResult = evaluateScriptedChecksForFrame(
							timeline.checks,
							state,
							{
								.frame = 1u,
								.camera = orbitCamera.get()
							});
						if (frameResult.hadFailures || state.nextCheckIndex != 1u || !state.baselineValid || !state.stepValid)
						{
							const auto& gimbal = orbitCamera->getGimbal();
							const auto pos = gimbal.getPosition();
							const auto orientation = gimbal.getOrientation();
							const auto basis = gimbal.getOrthonornalMatrix();
							const auto eulerDeg = getQuaternionEulerDegrees(gimbal.getOrientation());
							std::ostringstream oss;
							oss << std::fixed << std::setprecision(6)
								<< "Scripted check runner baseline smoke failed."
								<< " nextCheckIndex=" << state.nextCheckIndex
								<< " baselineValid=" << state.baselineValid
								<< " stepValid=" << state.stepValid
								<< " pos=(" << pos.x << ", " << pos.y << ", " << pos.z << ")"
								<< " quat=(" << orientation.data.x << ", " << orientation.data.y << ", " << orientation.data.z << ", " << orientation.data.w << ")"
								<< " basis_x=(" << basis[0].x << ", " << basis[0].y << ", " << basis[0].z << ")"
								<< " basis_y=(" << basis[1].x << ", " << basis[1].y << ", " << basis[1].z << ")"
								<< " basis_z=(" << basis[2].x << ", " << basis[2].y << ", " << basis[2].z << ")"
								<< " euler_deg=(" << eulerDeg.x << ", " << eulerDeg.y << ", " << eulerDeg.z << ")";
							if (!frameResult.logs.empty())
								oss << ' ' << frameResult.logs.front().text;
							return fail(oss.str());
						}
					}

					{
						CVirtualGimbalEvent stepEvent = {};
						stepEvent.type = CVirtualGimbalEvent::MoveRight;
						stepEvent.magnitude = 12.0;
						if (!orbitCamera->manipulate({ &stepEvent, 1u }))
							return fail("Scripted check runner smoke failed to manipulate the camera for step validation.");

						const auto frameResult = evaluateScriptedChecksForFrame(
							timeline.checks,
							state,
							{
								.frame = 2u,
								.camera = orbitCamera.get()
							});
						if (frameResult.hadFailures || state.nextCheckIndex != 2u)
						{
							const auto details = !frameResult.logs.empty() ? frameResult.logs.front().text : std::string("missing log details");
							return fail(std::string("Scripted check runner step smoke failed. ") + details);
						}
					}

					SCameraFollowConfig followConfig = {};
					followConfig.enabled = true;
					followConfig.mode = ECameraFollowMode::OrbitTarget;
					CCameraGoal followGoal = {};
					if (!nbl::core::applyFollowToCamera(m_cameraGoalSolver, orbitCamera.get(), trackedTarget, followConfig, &followGoal).succeeded())
						return fail("Scripted check runner smoke failed to apply follow before follow-lock validation.");

					{
						const auto frameResult = evaluateScriptedChecksForFrame(
							timeline.checks,
							state,
							{
								.frame = 3u,
								.camera = orbitCamera.get(),
								.trackedTarget = &trackedTarget,
								.followConfig = &followConfig,
								.goalSolver = &m_cameraGoalSolver
							});
						if (frameResult.hadFailures || state.nextCheckIndex != timeline.checks.size())
						{
							const auto details = !frameResult.logs.empty() ? frameResult.logs.front().text : std::string("missing log details");
							const auto& gimbal = orbitCamera->getGimbal();
							const auto cameraPos = gimbal.getPosition();
							const auto cameraForward = gimbal.getZAxis();
							const auto targetPos = trackedTarget.getGimbal().getPosition();
							const auto desiredForward = normalize(targetPos - cameraPos);
							const auto desiredRight = normalize(cross(float64_t3(0.0, 1.0, 0.0), desiredForward));
							const auto desiredUp = normalize(cross(desiredForward, desiredRight));
							const auto goalRightVec = normalizeQuaternion(followGoal.orientation).transformVector(float64_t3(1.0, 0.0, 0.0), true);
							const auto goalUpVec = normalizeQuaternion(followGoal.orientation).transformVector(float64_t3(0.0, 1.0, 0.0), true);
							const auto goalForwardVec = normalizeQuaternion(followGoal.orientation).transformVector(float64_t3(0.0, 0.0, 1.0), true);
							const auto goalBasis = getQuaternionBasisMatrix(followGoal.orientation);
							float lockAngle = 0.0f;
							double targetDistance = 0.0;
							const bool hasLockMetrics = nbl::core::tryComputeFollowTargetLockMetrics(gimbal, trackedTarget, lockAngle, &targetDistance);
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
							return fail(oss.str());
						}
					}

					return true;
				};

				auto verifyFollowTargetContract = [&](ICamera* camera, const CTrackedTarget& trackedTarget,
					const SCameraFollowConfig& followConfig, const CCameraGoal& followGoal, const char* label) -> bool
				{
					nbl::system::SCameraFollowRegressionResult regression = {};
					std::string regressionError;
					float32_t4x4 viewProjMatrix = float32_t4x4(1.0f);
					const bool hasViewProjMatrix = tryBuildFollowViewProjForCamera(camera, viewProjMatrix);
					if (!nbl::system::validateFollowTargetContract(
						camera,
						trackedTarget,
						followConfig,
						followGoal,
						regression,
						&regressionError,
						hasViewProjMatrix ? &viewProjMatrix : nullptr,
						CameraFollowRegressionThresholds))
					{
						return fail(std::string("Follow smoke contract failed for ") + label + ". " + regressionError);
					}
					return true;
				};

				auto verifyFollowTargetMarkerAlignment = [&](const CTrackedTarget& trackedTarget, const char* label) -> bool
				{
					m_followTarget.setPose(trackedTarget.getGimbal().getPosition(), trackedTarget.getGimbal().getOrientation());
					const auto markerWorld = computeFollowTargetMarkerWorld();
					const auto markerTransform = hlsl::transpose(getMatrix3x4As4x4(markerWorld));
					const auto markerPosition = getCastedVector<float64_t>(float32_t3(markerTransform[3]));
					const auto positionDelta = markerPosition - trackedTarget.getGimbal().getPosition();
					const auto errorLength = length(positionDelta);
					if (!std::isfinite(errorLength) || errorLength > CameraTinyScalarEpsilon)
					{
						return fail(std::string("Follow target marker alignment smoke failed for ") + label + ".");
					}
					return true;
				};

				auto verifyOffsetFollowRecapture = [&](ICamera* camera, const CTrackedTarget& trackedTarget, const char* label) -> bool
				{
					if (!camera)
						return true;

					const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, camera, std::string(label) + " baseline");
					SCameraFollowConfig followConfig = {};
					followConfig.enabled = true;
					followConfig.mode = ECameraFollowMode::KeepLocalOffset;

					if (!nbl::core::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, trackedTarget, followConfig))
						return fail(std::string("Follow recapture smoke failed to capture initial offset for ") + label + ".");

					const auto initialApply = nbl::core::applyFollowToCamera(m_cameraGoalSolver, camera, trackedTarget, followConfig);
					if (!initialApply.succeeded())
						return fail(std::string("Follow recapture smoke failed to apply initial follow for ") + label + ".");

					auto editedPreset = nbl::core::capturePreset(m_cameraGoalSolver, camera, std::string(label) + " edited");
					if (!editedPreset.goal.hasOrbitState)
						return fail(std::string("Follow recapture smoke missing orbit state for ") + label + ".");

					editedPreset.goal.orbitU = hlsl::wrapAngleRad(editedPreset.goal.orbitU + hlsl::radians(18.0));
					editedPreset.goal.orbitDistance = std::clamp(editedPreset.goal.orbitDistance + 0.75f, CSphericalTargetCamera::MinDistance, CSphericalTargetCamera::MaxDistance);
					editedPreset.goal = nbl::core::canonicalizeGoal(editedPreset.goal);
					if (!nbl::core::isGoalFinite(editedPreset.goal))
						return fail(std::string("Follow recapture smoke produced a non-finite edited goal for ") + label + ".");

					const auto editedApply = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, editedPreset);
					if (!editedApply.succeeded() || !editedApply.changed())
					{
						return fail(std::string("Follow recapture smoke failed to apply edited preset for ") + label +
							". " + describeApplyResult(editedApply));
					}

					const auto reachedEditedPreset = nbl::core::capturePreset(m_cameraGoalSolver, camera, std::string(label) + " reached");

					if (!nbl::core::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, trackedTarget, followConfig))
						return fail(std::string("Follow recapture smoke failed to recapture offset for ") + label + ".");

					CCameraGoal recapturedGoal = {};
					if (!nbl::core::tryBuildFollowGoal(m_cameraGoalSolver, camera, trackedTarget, followConfig, recapturedGoal))
						return fail(std::string("Follow recapture smoke failed to rebuild follow goal for ") + label + ".");

					const auto recapturedApply = nbl::core::applyFollowToCamera(m_cameraGoalSolver, camera, trackedTarget, followConfig);
					if (!recapturedApply.succeeded())
					{
						return fail(std::string("Follow recapture smoke failed to apply recaptured follow for ") + label +
							". " + describeApplyResult(recapturedApply));
					}

					if (!nbl::core::comparePresetToCameraState(m_cameraGoalSolver, camera, reachedEditedPreset, 5e-6, nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg, 5e-6))
						return fail(std::string("Follow recapture smoke mismatch for ") + label + ". " + describePresetMismatch(camera, reachedEditedPreset));
					if (!verifyFollowTargetContract(camera, trackedTarget, followConfig, recapturedGoal, label))
						return false;

					const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, baselinePreset);
					if (!restoreResult.succeeded() || !comparePresetToCameraStrict(camera, baselinePreset))
					{
						return fail(std::string("Follow recapture smoke failed to restore baseline for ") + label +
							". " + describeApplyResult(restoreResult) + " " + describePresetMismatch(camera, baselinePreset));
					}

					return true;
				};

				if (!verifyScriptedRuntimeFrameBatch())
					return false;
				if (!verifyScriptedRuntimeParser())
					return false;
				if (!verifyScriptedCheckRunner())
					return false;

				auto collectKeyboardVirtualEvents = [&](CGimbalInputBinder& inputBinder, const ui::E_KEY_CODE keyCode) -> std::vector<CVirtualGimbalEvent>
				{
					static std::chrono::microseconds smokeTimestamp = std::chrono::microseconds::zero();
					smokeTimestamp += std::chrono::microseconds(16667);
					const auto pressTs = smokeTimestamp;

					SKeyboardEvent pressEvent(pressTs);
					pressEvent.keyCode = keyCode;
					pressEvent.action = SKeyboardEvent::ECA_PRESSED;
					pressEvent.window = nullptr;

					inputBinder.collectVirtualEvents(pressTs, { .keyboardEvents = { &pressEvent, 1u } });

					smokeTimestamp += std::chrono::microseconds(16667);
					const auto sampleTs = smokeTimestamp;
					return inputBinder.collectVirtualEvents(sampleTs).events;
				};

				auto collectMouseVirtualEvents = [&](CGimbalInputBinder& inputBinder, std::span<const SMouseEvent> mouseEvents) -> std::vector<CVirtualGimbalEvent>
				{
					static std::chrono::microseconds smokeTimestamp = std::chrono::microseconds::zero();
					smokeTimestamp += std::chrono::microseconds(16667);
					const auto ts = smokeTimestamp;
					return inputBinder.collectVirtualEvents(ts, { .mouseEvents = mouseEvents }).events;
				};

				auto filterOrbitMouseEvents = [&](ICamera* camera, std::span<const SMouseEvent> input, bool orbitLookDown) -> std::vector<SMouseEvent>
				{
					if (!isOrbitLikeCamera(camera))
						return std::vector<SMouseEvent>(input.begin(), input.end());

					std::vector<SMouseEvent> filtered;
					filtered.reserve(input.size());
					for (const auto& ev : input)
					{
						if (ev.type == ui::SMouseEvent::EET_MOVEMENT && !orbitLookDown)
							continue;
						filtered.emplace_back(ev);
					}
					return filtered;
				};

				const std::array<ui::E_KEY_CODE, 12u> keyboardCandidates = {
					ui::E_KEY_CODE::EKC_W,
					ui::E_KEY_CODE::EKC_A,
					ui::E_KEY_CODE::EKC_S,
					ui::E_KEY_CODE::EKC_D,
					ui::E_KEY_CODE::EKC_Q,
					ui::E_KEY_CODE::EKC_E,
					ui::E_KEY_CODE::EKC_I,
					ui::E_KEY_CODE::EKC_J,
					ui::E_KEY_CODE::EKC_K,
					ui::E_KEY_CODE::EKC_L,
					ui::E_KEY_CODE::EKC_U,
					ui::E_KEY_CODE::EKC_O
				};

				CameraPreset initialOrbitPreset;
				CameraPreset initialFreePreset;
				CameraPreset initialChasePreset;
				CameraPreset initialDollyPreset;
				CameraPreset initialPathPreset;
				CameraPreset initialDollyZoomPreset;
				bool hasOrbitPreset = false;
				bool hasFreePreset = false;
				bool hasChasePreset = false;
				bool hasDollyPreset = false;
				bool hasPathPreset = false;
				bool hasDollyZoomPreset = false;

				for (const auto& cameraRef : cameras)
				{
					auto* camera = cameraRef.get();
					if (!camera)
						return fail("Null camera instance.");

					CGimbalInputBinder inputBinder;
					applyDefaultCameraInputBindingPreset(inputBinder, *camera);

					const auto initialPreset = nbl::core::capturePreset(m_cameraGoalSolver, camera, "smoke-initial");
					const auto initialCompatibility = analyzePresetCompatibility(camera, initialPreset);
					if (!initialCompatibility.exact || initialCompatibility.missingGoalStateMask != ICamera::GoalStateNone)
						return fail("Preset compatibility smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". missing=" + describeGoalStateMask(initialCompatibility.missingGoalStateMask));
					switch (camera->getKind())
					{
						case ICamera::CameraKind::Orbit:
							initialOrbitPreset = initialPreset;
							hasOrbitPreset = true;
							break;
						case ICamera::CameraKind::Free:
							initialFreePreset = initialPreset;
							hasFreePreset = true;
							break;
						case ICamera::CameraKind::Chase:
							initialChasePreset = initialPreset;
							hasChasePreset = true;
							break;
						case ICamera::CameraKind::Dolly:
							initialDollyPreset = initialPreset;
							hasDollyPreset = true;
							break;
						case ICamera::CameraKind::Path:
							initialPathPreset = initialPreset;
							hasPathPreset = true;
							break;
						case ICamera::CameraKind::DollyZoom:
							initialDollyZoomPreset = initialPreset;
							hasDollyZoomPreset = true;
							break;
						default:
							break;
					}
					if (!nbl::core::applyPreset(m_cameraGoalSolver, camera, initialPreset))
						return fail("Preset no-op smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

					if (initialPreset.goal.hasTargetPosition)
					{
						CameraPreset shiftedPreset = initialPreset;
						shiftedPreset.goal.targetPosition += float64_t3(0.5, -0.25, 0.75);

						const auto shiftedResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, shiftedPreset);
						if (!shiftedResult.succeeded() || !shiftedResult.changed() || !shiftedResult.exact)
							return fail("Preset target apply smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(shiftedResult));

						ICamera::SphericalTargetState shiftedState;
						if (!camera->tryGetSphericalTargetState(shiftedState) || !hlsl::nearlyEqualVec3(shiftedState.target, shiftedPreset.goal.targetPosition, 1e-9))
							return fail("Preset target writeback smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

						const auto restoredResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, initialPreset);
						if (!restoredResult.succeeded() || !restoredResult.exact)
							return fail("Preset restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(restoredResult));

						ICamera::SphericalTargetState restoredState;
						if (!camera->tryGetSphericalTargetState(restoredState) || !hlsl::nearlyEqualVec3(restoredState.target, initialPreset.goal.targetPosition, 1e-9))
							return fail("Preset target restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

						if (!comparePresetToCameraDefault(camera, initialPreset))
							return fail("Preset restore mismatch smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, initialPreset));
					}

					if (initialPreset.goal.hasDynamicPerspectiveState)
					{
						CameraPreset shiftedPreset = initialPreset;
						shiftedPreset.goal.dynamicPerspectiveState.baseFov =
							std::clamp(initialPreset.goal.dynamicPerspectiveState.baseFov + 7.5f, 10.0f, 150.0f);
						if (std::abs(static_cast<double>(
							shiftedPreset.goal.dynamicPerspectiveState.baseFov - initialPreset.goal.dynamicPerspectiveState.baseFov)) < 1e-6)
						{
							shiftedPreset.goal.dynamicPerspectiveState.baseFov =
								std::max(10.0f, initialPreset.goal.dynamicPerspectiveState.baseFov - 7.5f);
						}
						shiftedPreset.goal.dynamicPerspectiveState.referenceDistance =
							std::max(0.1f, initialPreset.goal.dynamicPerspectiveState.referenceDistance + 1.25f);

						const auto shiftedResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, shiftedPreset);
						if (!shiftedResult.succeeded() || !shiftedResult.changed() || !comparePresetToCameraStrict(camera, shiftedPreset))
							return fail("Preset dynamic perspective apply smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(shiftedResult) + " " + describePresetMismatch(camera, shiftedPreset));

						const auto restoredResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, initialPreset);
						if (!restoredResult.succeeded() || !comparePresetToCameraStrict(camera, initialPreset))
							return fail("Preset dynamic perspective restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(restoredResult) + " " + describePresetMismatch(camera, initialPreset));
					}

					const uint32_t allowed = camera->getAllowedVirtualEvents();
					std::vector<CVirtualGimbalEvent> directEvents;
					directEvents.reserve(3u);
					auto pushDirectEvent = [&](const CVirtualGimbalEvent::VirtualEventType type, const double magnitude) -> void
					{
						CVirtualGimbalEvent ev;
						ev.type = type;
						ev.magnitude = magnitude;
						directEvents.emplace_back(ev);
					};
					if (allowed & CVirtualGimbalEvent::MoveForward)
						pushDirectEvent(CVirtualGimbalEvent::MoveForward, 1.0);
					else if (allowed & CVirtualGimbalEvent::MoveRight)
						pushDirectEvent(CVirtualGimbalEvent::MoveRight, 1.0);
					else if (allowed & CVirtualGimbalEvent::MoveUp)
						pushDirectEvent(CVirtualGimbalEvent::MoveUp, 1.0);
					if (allowed & CVirtualGimbalEvent::PanRight)
						pushDirectEvent(CVirtualGimbalEvent::PanRight, 1.0);
					else if (allowed & CVirtualGimbalEvent::TiltUp)
						pushDirectEvent(CVirtualGimbalEvent::TiltUp, 1.0);
					else if (allowed & CVirtualGimbalEvent::RollRight)
						pushDirectEvent(CVirtualGimbalEvent::RollRight, 1.0);
					if (directEvents.empty())
					{
						for (const auto event : CVirtualGimbalEvent::VirtualEventsTypeTable)
						{
							if (allowed & event)
							{
								pushDirectEvent(event, 1.0);
								break;
							}
						}
					}
					if (directEvents.empty())
						return fail("No allowed virtual events for camera \"" + std::string(camera->getIdentifier()) + "\".");

					nbl::system::SCameraManipulationDelta directDelta = {};
					if (!nbl::system::tryManipulateCameraAndMeasureDelta(camera, { directEvents.data(), directEvents.size() }, directDelta, CameraTinyScalarEpsilon))
						return fail("Direct manipulate smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");
					{
						const auto modifiedPreset = nbl::core::capturePreset(m_cameraGoalSolver, camera, "smoke-direct");
						const auto restoreInitial = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, initialPreset);
						if (!restoreInitial.succeeded() || !nbl::core::comparePresetToCameraState(m_cameraGoalSolver, camera, initialPreset, 1e-3, nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg, 1e-4))
							return fail("Preset restore from direct smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, initialPreset));

						const auto applyModified = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, modifiedPreset);
						if (!applyModified.succeeded() || !applyModified.changed() || !nbl::core::comparePresetToCameraState(m_cameraGoalSolver, camera, modifiedPreset, 1e-3, nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg, 1e-4))
							return fail("Preset apply from direct smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, modifiedPreset));

						const auto restoreAgain = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, initialPreset);
						if (!restoreAgain.succeeded() || !nbl::core::comparePresetToCameraState(m_cameraGoalSolver, camera, initialPreset, 1e-3, nbl::system::SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg, 1e-4))
							return fail("Preset final restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, initialPreset));
					}

					bool keyboardOk = false;
					nbl::system::SCameraManipulationDelta keyboardDelta = {};
					for (const auto key : keyboardCandidates)
					{
						applyDefaultCameraInputBindingPreset(inputBinder, *camera);
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
						return fail("Keyboard binding smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

					const auto& mousePreset = getDefaultCameraMouseMappingPreset(*camera);
					const bool hasMoveMapping =
						mousePreset.find(ui::EMC_RELATIVE_POSITIVE_MOVEMENT_X) != mousePreset.end() ||
						mousePreset.find(ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_X) != mousePreset.end() ||
						mousePreset.find(ui::EMC_RELATIVE_POSITIVE_MOVEMENT_Y) != mousePreset.end() ||
						mousePreset.find(ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y) != mousePreset.end();
					const bool hasScrollMapping =
						mousePreset.find(ui::EMC_VERTICAL_POSITIVE_SCROLL) != mousePreset.end() ||
						mousePreset.find(ui::EMC_VERTICAL_NEGATIVE_SCROLL) != mousePreset.end() ||
						mousePreset.find(ui::EMC_HORIZONTAL_POSITIVE_SCROLL) != mousePreset.end() ||
						mousePreset.find(ui::EMC_HORIZONTAL_NEGATIVE_SCROLL) != mousePreset.end();

					nbl::system::SCameraManipulationDelta mouseMoveDelta = {};
					if (hasMoveMapping)
					{
						SMouseEvent moveEv(std::chrono::microseconds(16667));
						moveEv.window = nullptr;
						moveEv.type = ui::SMouseEvent::EET_MOVEMENT;
						moveEv.movementEvent.relativeMovementX = 12;
						moveEv.movementEvent.relativeMovementY = -8;

						const std::array<SMouseEvent, 1u> rawMove = { moveEv };
						auto filteredMoveLookDown = filterOrbitMouseEvents(camera, rawMove, true);
						auto filteredMoveLookUp = filterOrbitMouseEvents(camera, rawMove, false);
						const bool hasBlockedMovement = std::any_of(filteredMoveLookUp.begin(), filteredMoveLookUp.end(), [](const SMouseEvent& ev) { return ev.type == ui::SMouseEvent::EET_MOVEMENT; });
						if (isOrbitLikeCamera(camera) && hasBlockedMovement)
							return fail("Orbit mouse movement gate failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

						applyDefaultCameraInputBindingPreset(inputBinder, *camera);
						auto mouseMoveEvents = collectMouseVirtualEvents(inputBinder, { filteredMoveLookDown.data(), filteredMoveLookDown.size() });
						if (mouseMoveEvents.empty())
							return fail("Mouse move virtual events missing for camera \"" + std::string(camera->getIdentifier()) + "\".");
						if (!nbl::system::tryManipulateCameraAndMeasureDelta(camera, { mouseMoveEvents.data(), mouseMoveEvents.size() }, mouseMoveDelta, CameraTinyScalarEpsilon))
							return fail("Mouse move binding smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");
					}

					nbl::system::SCameraManipulationDelta mouseScrollDelta = {};
					if (hasScrollMapping)
					{
						SMouseEvent scrollEv(std::chrono::microseconds(16667));
						scrollEv.window = nullptr;
						scrollEv.type = ui::SMouseEvent::EET_SCROLL;
						scrollEv.scrollEvent.verticalScroll = 4;
						scrollEv.scrollEvent.horizontalScroll = 2;
						const std::array<SMouseEvent, 1u> rawScroll = { scrollEv };
						auto filteredScroll = filterOrbitMouseEvents(camera, rawScroll, false);

						applyDefaultCameraInputBindingPreset(inputBinder, *camera);
						auto mouseScrollEvents = collectMouseVirtualEvents(inputBinder, { filteredScroll.data(), filteredScroll.size() });
						if (mouseScrollEvents.empty())
							return fail("Mouse scroll virtual events missing for camera \"" + std::string(camera->getIdentifier()) + "\".");
						if (!nbl::system::tryManipulateCameraAndMeasureDelta(camera, { mouseScrollEvents.data(), mouseScrollEvents.size() }, mouseScrollDelta, CameraTinyScalarEpsilon))
							return fail("Mouse scroll binding smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");
					}

					std::cout << "[headless-camera-smoke][pass] " << camera->getIdentifier()
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

				auto findCameraByKind = [&](const ICamera::CameraKind kind) -> ICamera*
				{
					for (const auto& cameraRef : cameras)
					{
						auto* candidate = cameraRef.get();
						if (candidate && candidate->getKind() == kind)
							return candidate;
					}
					return nullptr;
				};

				auto verifyApproximateCrossKindApply = [&](ICamera* targetCamera, const CameraPreset& sourcePreset,
					const CCameraGoalSolver::SApplyResult::EIssue expectedIssue, const char* label) -> bool
				{
					if (!targetCamera)
						return true;

					uint32_t expectedMissingGoalStateMask = ICamera::GoalStateNone;
					switch (expectedIssue)
					{
						case CCameraGoalSolver::SApplyResult::MissingPathState:
							expectedMissingGoalStateMask = ICamera::GoalStatePath;
							break;
						case CCameraGoalSolver::SApplyResult::MissingDynamicPerspectiveState:
							expectedMissingGoalStateMask = ICamera::GoalStateDynamicPerspective;
							break;
						case CCameraGoalSolver::SApplyResult::MissingSphericalTargetState:
							expectedMissingGoalStateMask = ICamera::GoalStateSphericalTarget;
							break;
						default:
							break;
					}

					const auto compatibility = analyzePresetCompatibility(targetCamera, sourcePreset);
					if (compatibility.exact || compatibility.missingGoalStateMask != expectedMissingGoalStateMask)
					{
						return fail(std::string("Cross-kind preset compatibility smoke failed for ") + label +
							". missing=" + describeGoalStateMask(compatibility.missingGoalStateMask));
					}

					const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, targetCamera, std::string(label) + "-baseline");
					const auto applyResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, targetCamera, sourcePreset);
					if (!applyResult.succeeded() || !applyResult.approximate() || !applyResult.hasIssue(expectedIssue))
						return fail(std::string("Cross-kind preset smoke failed for ") + label + ". " + describeApplyResult(applyResult));

					const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, targetCamera, baselinePreset);
					if (!restoreResult.succeeded() || !comparePresetToCameraStrict(targetCamera, baselinePreset))
						return fail(std::string("Cross-kind preset restore smoke failed for ") + label + ". " + describeApplyResult(restoreResult) + " " + describePresetMismatch(targetCamera, baselinePreset));

					return true;
				};

				auto verifyExactCrossKindApply = [&](ICamera* targetCamera, const CameraPreset& sourcePreset, const char* label) -> bool
				{
					if (!targetCamera)
						return true;

					const auto compatibility = analyzePresetCompatibility(targetCamera, sourcePreset);
					if (!compatibility.exact || compatibility.missingGoalStateMask != ICamera::GoalStateNone)
					{
						return fail(std::string("Exact cross-kind preset compatibility smoke failed for ") + label +
							". missing=" + describeGoalStateMask(compatibility.missingGoalStateMask));
					}

					const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, targetCamera, std::string(label) + "-baseline");
					const auto applyResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, targetCamera, sourcePreset);
					if (!applyResult.succeeded() || !applyResult.exact || !comparePresetToCameraStrict(targetCamera, sourcePreset))
					{
						return fail(std::string("Exact cross-kind preset smoke failed for ") + label + ". " +
							describeApplyResult(applyResult) + " " + describePresetMismatch(targetCamera, sourcePreset));
					}

					const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, targetCamera, baselinePreset);
					if (!restoreResult.succeeded() || !restoreResult.exact || !comparePresetToCameraStrict(targetCamera, baselinePreset))
					{
						return fail(std::string("Exact cross-kind preset restore smoke failed for ") + label + ". " +
							describeApplyResult(restoreResult) + " " + describePresetMismatch(targetCamera, baselinePreset));
					}

					return true;
				};

				ICamera* orbitCamera = findCameraByKind(ICamera::CameraKind::Orbit);
				ICamera* freeCamera = findCameraByKind(ICamera::CameraKind::Free);
				ICamera* chaseCamera = findCameraByKind(ICamera::CameraKind::Chase);
				ICamera* dollyCamera = findCameraByKind(ICamera::CameraKind::Dolly);
				ICamera* dollyZoomCamera = findCameraByKind(ICamera::CameraKind::DollyZoom);

				{
					CTrackedTarget trackedTarget(
						float64_t3(2.25, -0.75, 1.25),
						makeQuaternionFromEulerRadians(float64_t3(0.18, -0.22, 0.41)),
						"Smoke Target");

					const auto movedTrackedTargetPosition = float64_t3(-1.5, 0.5, 2.25);
					const auto movedTrackedTargetOrientation = makeQuaternionFromEulerRadians(float64_t3(-0.12, 0.35, 0.27));

					if (orbitCamera)
					{
						const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, orbitCamera, "orbit-follow-baseline");
						SCameraFollowConfig followConfig = {};
						followConfig.enabled = true;
						followConfig.mode = ECameraFollowMode::OrbitTarget;

						nbl::system::SCameraFollowApplyValidationResult followResult = {};
						if (!buildAndValidateFollowTargetContract(orbitCamera, trackedTarget, followConfig, "orbit follow", followResult))
							return false;
						if (!verifyFollowVisualMetrics(orbitCamera, trackedTarget, followConfig, "orbit follow"))
							return false;
						if (!verifyFollowTargetMarkerAlignment(trackedTarget, "orbit follow"))
							return false;

						const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, orbitCamera, baselinePreset);
						if (!restoreResult.succeeded() || !comparePresetToCameraStrict(orbitCamera, baselinePreset))
							return fail("Orbit follow smoke failed to restore the baseline preset.");

						followConfig.mode = ECameraFollowMode::KeepWorldOffset;
						followConfig.worldOffset = float64_t3(4.0, -1.5, 2.0);
						trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

						nbl::system::SCameraFollowApplyValidationResult worldOffsetResult = {};
						if (!buildAndValidateFollowTargetContract(orbitCamera, trackedTarget, followConfig, "orbit keep-world-offset follow", worldOffsetResult))
							return false;
						if (!verifyFollowVisualMetrics(orbitCamera, trackedTarget, followConfig, "orbit keep-world-offset follow"))
							return false;

						const auto restoreWorldOffsetResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, orbitCamera, baselinePreset);
						if (!restoreWorldOffsetResult.succeeded() || !comparePresetToCameraStrict(orbitCamera, baselinePreset))
							return fail("Orbit keep-world-offset smoke failed to restore the baseline preset.");
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
						const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, defaultFollowCamera, label + " baseline");

						trackedTarget.setPose(float64_t3(2.25, -0.75, 1.25), makeQuaternionFromEulerRadians(float64_t3(0.18, -0.22, 0.41)));
						if ((nbl::core::cameraFollowModeUsesLocalOffset(followConfig.mode) || nbl::core::cameraFollowModeUsesWorldOffset(followConfig.mode)) &&
							!nbl::core::captureFollowOffsetsFromCamera(m_cameraGoalSolver, defaultFollowCamera, trackedTarget, followConfig))
						{
							return fail("Default follow smoke failed to capture offsets for camera \"" + std::string(defaultFollowCamera->getIdentifier()) + "\".");
						}

						trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

						nbl::system::SCameraFollowApplyValidationResult defaultFollowResult = {};
						if (!buildAndValidateFollowTargetContract(defaultFollowCamera, trackedTarget, followConfig, label.c_str(), defaultFollowResult))
							return false;
						if (!verifyFollowVisualMetrics(defaultFollowCamera, trackedTarget, followConfig, label.c_str()))
							return false;
						if (!verifyFollowTargetMarkerAlignment(trackedTarget, label.c_str()))
							return false;

						const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, defaultFollowCamera, baselinePreset);
						if (!restoreResult.succeeded() || !comparePresetToCameraStrict(defaultFollowCamera, baselinePreset))
							return fail("Default follow smoke failed to restore the baseline preset for camera \"" + std::string(defaultFollowCamera->getIdentifier()) + "\".");
					}

					if (freeCamera)
					{
						const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, freeCamera, "free-follow-baseline");
						SCameraFollowConfig followConfig = {};
						followConfig.enabled = true;
						followConfig.mode = ECameraFollowMode::LookAtTarget;

						nbl::system::SCameraFollowApplyValidationResult lookAtResult = {};
						if (!buildAndValidateFollowTargetContract(freeCamera, trackedTarget, followConfig, "free look-at follow", lookAtResult))
							return false;
						if (!verifyFollowVisualMetrics(freeCamera, trackedTarget, followConfig, "free look-at follow"))
							return false;
						if (!verifyFollowTargetMarkerAlignment(trackedTarget, "free look-at follow"))
							return false;

						const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, freeCamera, baselinePreset);
						if (!restoreResult.succeeded() || !comparePresetToCameraStrict(freeCamera, baselinePreset))
							return fail("Free follow smoke failed to restore the baseline preset.");

						followConfig.mode = ECameraFollowMode::KeepWorldOffset;
						followConfig.worldOffset = float64_t3(5.0, -2.0, 1.5);
						trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

						nbl::system::SCameraFollowApplyValidationResult keepWorldResult = {};
						if (!buildAndValidateFollowTargetContract(freeCamera, trackedTarget, followConfig, "free keep-world-offset follow", keepWorldResult))
							return false;
						if (!verifyFollowVisualMetrics(freeCamera, trackedTarget, followConfig, "free keep-world-offset follow"))
							return false;

						const auto restoreWorldOffsetResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, freeCamera, baselinePreset);
						if (!restoreWorldOffsetResult.succeeded() || !comparePresetToCameraStrict(freeCamera, baselinePreset))
							return fail("Free keep-world-offset smoke failed to restore the baseline preset.");
					}

					if (chaseCamera)
					{
						const auto baselinePreset = nbl::core::capturePreset(m_cameraGoalSolver, chaseCamera, "chase-follow-baseline");
						SCameraFollowConfig followConfig = {};
						followConfig.enabled = true;
						followConfig.mode = ECameraFollowMode::KeepLocalOffset;
						if (!nbl::core::captureFollowOffsetsFromCamera(m_cameraGoalSolver, chaseCamera, trackedTarget, followConfig))
							return fail("Chase follow smoke failed to capture local offset.");

						trackedTarget.setPose(movedTrackedTargetPosition, movedTrackedTargetOrientation);

						nbl::system::SCameraFollowApplyValidationResult localOffsetResult = {};
						if (!buildAndValidateFollowTargetContract(chaseCamera, trackedTarget, followConfig, "chase local-offset follow", localOffsetResult))
							return false;
						if (!verifyFollowVisualMetrics(chaseCamera, trackedTarget, followConfig, "chase local-offset follow"))
							return false;
						if (!verifyFollowTargetMarkerAlignment(trackedTarget, "chase local-offset follow"))
							return false;

						const auto restoreResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, chaseCamera, baselinePreset);
						if (!restoreResult.succeeded() || !comparePresetToCameraStrict(chaseCamera, baselinePreset))
							return fail("Chase follow smoke failed to restore the baseline preset.");
					}

					if (!verifyOffsetFollowRecapture(chaseCamera, trackedTarget, "chase follow recapture"))
						return false;
					if (!verifyOffsetFollowRecapture(dollyCamera, trackedTarget, "dolly follow recapture"))
						return false;
				}

				if (hasOrbitPreset && hasChasePreset)
				{
					if (!verifyExactCrossKindApply(orbitCamera, initialChasePreset, "Chase->Orbit"))
						return false;
					if (!verifyExactCrossKindApply(chaseCamera, initialOrbitPreset, "Orbit->Chase"))
						return false;
				}

				if (hasOrbitPreset && hasDollyPreset)
				{
					if (!verifyExactCrossKindApply(orbitCamera, initialDollyPreset, "Dolly->Orbit"))
						return false;
					if (!verifyExactCrossKindApply(dollyCamera, initialOrbitPreset, "Orbit->Dolly"))
						return false;
				}

				if (hasOrbitPreset && hasPathPreset && orbitCamera)
				{
					if (!verifyApproximateCrossKindApply(
						orbitCamera,
						initialPathPreset,
						CCameraGoalSolver::SApplyResult::MissingPathState,
						"Path->Orbit"))
					{
						return false;
					}
				}

				if (hasOrbitPreset && hasDollyZoomPreset && orbitCamera)
				{
					if (!verifyApproximateCrossKindApply(
						orbitCamera,
						initialDollyZoomPreset,
						CCameraGoalSolver::SApplyResult::MissingDynamicPerspectiveState,
						"DollyZoom->Orbit"))
					{
						return false;
					}
				}

				if (hasOrbitPreset)
				{
					if (std::string_view(nbl::ui::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::All)) != "All" ||
						std::string_view(nbl::ui::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::Exact)) != "Exact" ||
						std::string_view(nbl::ui::getPresetApplyPresentationFilterLabel(EPresetApplyPresentationFilter::BestEffort)) != "Best-effort")
					{
						return fail("Presentation utilities smoke returned an unexpected filter label.");
					}

					const auto blockedPresentation = nbl::ui::analyzePresetPresentation(m_cameraGoalSolver, nullptr, initialOrbitPreset);
					if (blockedPresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
						blockedPresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
					{
						return fail("Presentation utilities smoke allowed a null-camera preset through an exactness filter.");
					}
					if (blockedPresentation.sourceKindLabel.empty() || blockedPresentation.goalStateLabel.empty())
						return fail("Presentation utilities smoke produced empty blocked presentation labels.");

					const auto blockedBadges = nbl::ui::collectGoalApplyPresentationBadges(blockedPresentation);
					if (!blockedBadges.blocked || blockedBadges.exact || blockedBadges.bestEffort || blockedPresentation.badges.blocked != blockedBadges.blocked)
						return fail("Presentation utilities smoke produced wrong blocked badge flags.");

					if (orbitCamera)
					{
						const auto exactPresentation = nbl::ui::analyzePresetPresentation(m_cameraGoalSolver, orbitCamera, initialOrbitPreset);
						if (!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
							!exactPresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
							exactPresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
						{
							return fail("Presentation utilities smoke failed exact filtering.");
						}

						const auto exactBadges = nbl::ui::collectGoalApplyPresentationBadges(exactPresentation);
						if (!exactBadges.exact || exactBadges.bestEffort || exactBadges.dropsState || exactBadges.sharedStateOnly || exactBadges.blocked)
							return fail("Presentation utilities smoke produced wrong exact badge flags.");
						if (exactPresentation.sourceKindLabel.empty() || exactPresentation.goalStateLabel.empty())
							return fail("Presentation utilities smoke produced empty exact presentation labels.");

						const auto capturePresentation = nbl::ui::analyzeCapturePresentation(m_cameraGoalSolver, orbitCamera);
						if (!capturePresentation.canCapture || capturePresentation.policyLabel.empty())
							return fail("Presentation utilities smoke failed orbit capture presentation.");
					}
				}

				if (hasOrbitPreset && hasPathPreset && orbitCamera)
				{
					const auto approximatePresentation = nbl::ui::analyzePresetPresentation(m_cameraGoalSolver, orbitCamera, initialPathPreset);
					if (!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::All) ||
						approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::Exact) ||
						!approximatePresentation.matchesFilter(EPresetApplyPresentationFilter::BestEffort))
					{
						return fail("Presentation utilities smoke failed best-effort filtering.");
					}

					const auto approximateBadges = nbl::ui::collectGoalApplyPresentationBadges(approximatePresentation);
					if (approximateBadges.exact || !approximateBadges.bestEffort || !approximateBadges.dropsState || approximateBadges.sharedStateOnly || approximateBadges.blocked)
						return fail("Presentation utilities smoke produced wrong best-effort badge flags.");
					if (approximatePresentation.sourceKindLabel.empty() || approximatePresentation.goalStateLabel.empty())
						return fail("Presentation utilities smoke produced empty best-effort presentation labels.");
				}

				{
					std::vector<CameraPreset> sourcePresets;
					if (hasOrbitPreset)
						sourcePresets.push_back(initialOrbitPreset);
					if (hasChasePreset)
						sourcePresets.push_back(initialChasePreset);
					if (hasDollyPreset)
						sourcePresets.push_back(initialDollyPreset);
					if (hasPathPreset)
						sourcePresets.push_back(initialPathPreset);
					if (hasDollyZoomPreset)
						sourcePresets.push_back(initialDollyZoomPreset);

					if (sourcePresets.empty())
						return fail("Preset persistence smoke failed to collect source presets.");

					std::stringstream presetBuffer;
					if (!nbl::system::writePresetCollection(presetBuffer, std::span<const CameraPreset>(sourcePresets.data(), sourcePresets.size())))
						return fail("Preset persistence smoke failed to serialize preset collection.");

					std::vector<CameraPreset> loadedPresets;
					if (!nbl::system::readPresetCollection(presetBuffer, loadedPresets))
						return fail("Preset persistence smoke failed to deserialize preset collection.");
					if (!nbl::core::comparePresetCollections(
						std::span<const CameraPreset>(sourcePresets.data(), sourcePresets.size()),
						std::span<const CameraPreset>(loadedPresets.data(), loadedPresets.size()),
						1e-6, 0.1, 1e-6))
					{
						return fail("Preset persistence smoke changed stream preset collection content.");
					}

					CCameraKeyframeTrack sourceTrack;
					sourceTrack.keyframes.reserve(sourcePresets.size());
					for (size_t i = 0u; i < sourcePresets.size(); ++i)
					{
						CameraKeyframe keyframe;
						keyframe.time = static_cast<float>(i) * 1.5f;
						keyframe.preset = sourcePresets[i];
						sourceTrack.keyframes.emplace_back(std::move(keyframe));
					}
					sourceTrack.selectedKeyframeIx = static_cast<int>(sourceTrack.keyframes.size()) - 1;

					std::stringstream keyframeBuffer;
					if (!nbl::system::writeKeyframeTrack(keyframeBuffer, sourceTrack))
						return fail("Keyframe persistence smoke failed to serialize track.");

					CCameraKeyframeTrack loadedTrack;
					if (!nbl::system::readKeyframeTrack(keyframeBuffer, loadedTrack))
						return fail("Keyframe persistence smoke failed to deserialize track.");
					if (!nbl::system::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, loadedTrack))
						return fail("Keyframe persistence smoke changed stream track content.");

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

					if (!nbl::system::savePresetCollectionToFile(presetFile, std::span<const CameraPreset>(sourcePresets.data(), sourcePresets.size())))
						return fail("Preset persistence smoke failed to save preset collection file.");

					std::vector<CameraPreset> fileLoadedPresets;
					if (!nbl::system::loadPresetCollectionFromFile(presetFile, fileLoadedPresets))
						return fail("Preset persistence smoke failed to load preset collection file.");
					if (!nbl::core::comparePresetCollections(
						std::span<const CameraPreset>(sourcePresets.data(), sourcePresets.size()),
						std::span<const CameraPreset>(fileLoadedPresets.data(), fileLoadedPresets.size()),
						1e-6, 0.1, 1e-6))
					{
						return fail("Preset persistence smoke changed file preset collection content.");
					}

					if (!nbl::system::saveKeyframeTrackToFile(keyframeFile, sourceTrack))
						return fail("Keyframe persistence smoke failed to save track file.");

					CCameraKeyframeTrack fileLoadedTrack;
					if (!nbl::system::loadKeyframeTrackFromFile(keyframeFile, fileLoadedTrack))
						return fail("Keyframe persistence smoke failed to load track file.");
					if (!nbl::system::compareKeyframeTrackContentWithStrictThresholds(sourceTrack, fileLoadedTrack))
						return fail("Keyframe persistence smoke changed file track content.");
				}

				if (hasOrbitPreset && hasDollyPreset)
				{
					CCameraKeyframeTrack playbackTrack;
					{
						CameraKeyframe a;
						a.time = 0.f;
						a.preset = initialOrbitPreset;
						playbackTrack.keyframes.push_back(a);
					}
					{
						CameraKeyframe b;
						b.time = 2.f;
						b.preset = initialDollyPreset;
						playbackTrack.keyframes.push_back(b);
					}

					CCameraPlaybackCursor cursor;
					cursor.playing = true;
					cursor.loop = false;
					cursor.speed = 1.f;
					cursor.time = 1.5f;

					const auto advanceToEnd = nbl::core::advancePlaybackCursor(cursor, playbackTrack, 1.0);
					if (!advanceToEnd.hasTrack || !advanceToEnd.changedTime || !advanceToEnd.reachedEnd || !advanceToEnd.stopped || advanceToEnd.wrapped)
						return fail("Playback timeline smoke failed for non-loop end-of-track advance.");
					if (std::abs(static_cast<double>(advanceToEnd.time - 2.f)) > 1e-6)
						return fail("Playback timeline smoke produced wrong end-of-track time.");

					nbl::core::resetPlaybackCursor(cursor, 1.25f);
					if (cursor.playing || std::abs(static_cast<double>(cursor.time - 1.25f)) > 1e-6)
						return fail("Playback timeline smoke failed to reset cursor.");

					cursor.playing = true;
					cursor.loop = true;
					cursor.speed = 1.f;
					cursor.time = 1.5f;
					const auto advanceLoop = nbl::core::advancePlaybackCursor(cursor, playbackTrack, 1.0);
					if (!advanceLoop.hasTrack || !advanceLoop.changedTime || !advanceLoop.wrapped || advanceLoop.stopped || advanceLoop.reachedEnd)
						return fail("Playback timeline smoke failed for looped advance.");
					if (std::abs(static_cast<double>(advanceLoop.time - 0.5f)) > 1e-6)
						return fail("Playback timeline smoke produced wrong wrapped time.");

					cursor.time = 9.f;
					nbl::core::clampPlaybackCursorToTrack(playbackTrack, cursor);
					if (std::abs(static_cast<double>(cursor.time - 2.f)) > 1e-6)
						return fail("Playback timeline smoke failed to clamp cursor time.");
				}

				if (hasOrbitPreset)
				{
					CCameraSequenceScript sequence;
					sequence.fps = 4.f;
					sequence.defaults.durationSeconds = 2.f;
					sequence.defaults.presentations = {
						{ .projection = IPlanarProjection::CProjection::Perspective, .leftHanded = true },
						{ .projection = IPlanarProjection::CProjection::Orthographic, .leftHanded = false }
					};
					sequence.defaults.captureFractions = { 0.f, 0.5f, 1.f };

					CCameraSequenceSegment segment;
					segment.name = "sequence_compile_smoke";
					segment.cameraKind = ICamera::CameraKind::Orbit;
					{
						CCameraSequenceKeyframe keyframe;
						keyframe.time = 0.f;
						keyframe.hasAbsolutePreset = true;
						keyframe.absolutePreset = initialOrbitPreset;
						segment.keyframes.push_back(keyframe);
					}
					{
						nbl::core::CCameraSequenceTrackedTargetKeyframe keyframe;
						keyframe.time = 0.f;
						keyframe.hasAbsolutePosition = true;
						keyframe.absolutePosition = float64_t3(1.0, 2.0, 3.0);
						segment.targetKeyframes.push_back(keyframe);
					}
					{
						nbl::core::CCameraSequenceTrackedTargetKeyframe keyframe;
						keyframe.time = 1.f;
						keyframe.hasAbsolutePosition = true;
						keyframe.absolutePosition = float64_t3(4.0, 5.0, 6.0);
						segment.targetKeyframes.push_back(keyframe);
					}
					{
						nbl::core::CCameraSequenceTrackedTargetKeyframe keyframe;
						keyframe.time = 1.f;
						keyframe.hasAbsolutePosition = true;
						keyframe.absolutePosition = float64_t3(7.0, 8.0, 9.0);
						segment.targetKeyframes.push_back(keyframe);
					}
					sequence.segments.push_back(segment);

					if (!nbl::core::sequenceScriptUsesMultiplePresentations(sequence))
						return fail("Sequence compile smoke failed to detect multi-presentation authored defaults.");

					const CCameraSequenceTrackedTargetPose referenceTrackedTargetPose = {
						.position = getDefaultFollowTargetPosition(),
						.orientation = getDefaultFollowTargetOrientation()
					};

					nbl::core::CCameraSequenceCompiledSegment compiledSegment;
					std::string compileError;
					if (!nbl::core::compileSequenceSegmentFromReference(
						sequence,
						sequence.segments.front(),
						initialOrbitPreset,
						referenceTrackedTargetPose,
						compiledSegment,
						&compileError))
					{
						return fail("Sequence compile smoke failed to compile a shared segment. " + compileError);
					}

					if (compiledSegment.durationFrames != 8ull || compiledSegment.sampleTimes.size() != 8u)
						return fail("Sequence compile smoke produced wrong sampled frame count.");
					if (compiledSegment.captureFrameOffsets.size() != 3u ||
						compiledSegment.captureFrameOffsets[0] != 0ull ||
						compiledSegment.captureFrameOffsets[1] != 4ull ||
						compiledSegment.captureFrameOffsets[2] != 7ull)
					{
						return fail("Sequence compile smoke produced wrong capture frame offsets.");
					}
					if (compiledSegment.presentations.size() != 2u)
						return fail("Sequence compile smoke lost authored presentations.");
					if (!compiledSegment.usesTrackedTargetTrack() || compiledSegment.trackedTargetTrack.keyframes.size() != 2u)
						return fail("Sequence compile smoke failed to normalize tracked-target keyframes.");

					std::vector<nbl::core::CCameraSequenceCompiledFramePolicy> framePolicies;
					if (!nbl::core::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, true))
						return fail("Sequence compile smoke failed to build shared frame policies.");
					if (framePolicies.size() != 8u)
						return fail("Sequence compile smoke produced wrong frame-policy count.");
					if (!framePolicies[0].baseline || framePolicies[0].continuityStep || !framePolicies[0].capture)
						return fail("Sequence compile smoke produced wrong first-frame policy.");
					if (!framePolicies[1].continuityStep || !framePolicies[1].followTargetLock || framePolicies[1].baseline)
						return fail("Sequence compile smoke produced wrong continuity follow policy.");
					if (!framePolicies[4].capture || !framePolicies[7].capture)
						return fail("Sequence compile smoke produced wrong capture milestone policy.");

					CCameraSequenceTrackedTargetPose poseAtOne;
					if (!nbl::core::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, 1.f, poseAtOne))
						return fail("Sequence compile smoke failed to sample normalized tracked-target track.");
					if (length(poseAtOne.position - float64_t3(7.0, 8.0, 9.0)) > 1e-9)
						return fail("Sequence compile smoke did not keep the last authored target pose for duplicate keyframe time.");

					CCameraScriptedTimeline scriptedTimeline;
					std::string runtimeBuildError;
					if (!nbl::system::appendCompiledSequenceSegmentToScriptedTimeline(
						scriptedTimeline,
						11u,
						compiledSegment,
						{
							.planarIx = 5u,
							.availableWindowCount = 2u,
							.useWindow = true,
							.includeFollowTargetLock = true
						},
						&runtimeBuildError))
					{
						return fail("Sequence runtime builder smoke failed to append a compiled segment. " + runtimeBuildError);
					}
					nbl::system::finalizeScriptedTimeline(scriptedTimeline);

					if (scriptedTimeline.captureFrames.size() != 3u ||
						scriptedTimeline.captureFrames[0] != 11ull ||
						scriptedTimeline.captureFrames[1] != 15ull ||
						scriptedTimeline.captureFrames[2] != 18ull)
					{
						return fail("Sequence runtime builder smoke produced wrong capture frames.");
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
					if (baselineChecks != 1u || stepChecks != 7u || followChecks != 7u)
						return fail("Sequence runtime builder smoke produced wrong scripted check counts.");

					size_t runtimeNextEventIndex = 0u;
					CCameraScriptedFrameEvents runtimeBatch;
					nbl::system::dequeueScriptedFrameEvents(scriptedTimeline.events, runtimeNextEventIndex, 11u, runtimeBatch);
					if (runtimeBatch.actions.size() != 10u || runtimeBatch.goals.size() != 1u ||
						runtimeBatch.trackedTargetTransforms.size() != 1u || runtimeBatch.segmentLabels.size() != 1u)
					{
						return fail("Sequence runtime builder smoke produced wrong first-frame batch.");
					}
					if (runtimeBatch.actions.front().kind != CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow ||
						runtimeBatch.segmentLabels.front() != "sequence_compile_smoke")
					{
						return fail("Sequence runtime builder smoke lost first-frame scripted payload.");
					}
				}

				if (hasOrbitPreset && orbitCamera)
				{
					std::array<ICamera*, 2u> exactTargets = { orbitCamera, nullptr };
					const auto exactSummary = nbl::core::applyPresetToCameraRange(
						m_cameraGoalSolver,
						std::span<ICamera* const>(exactTargets.data(), exactTargets.size()),
						initialOrbitPreset);
					if (exactSummary.targetCount != 1u || exactSummary.successCount != 1u || exactSummary.approximateCount != 0u || exactSummary.failureCount != 0u)
						return fail("Preset apply summary smoke failed for exact target range.");
				}

				if (hasPathPreset && orbitCamera)
				{
					std::array<ICamera*, 1u> approximateTargets = { orbitCamera };
					const auto approximateSummary = nbl::core::applyPresetToCameraRange(
						m_cameraGoalSolver,
						std::span<ICamera* const>(approximateTargets.data(), approximateTargets.size()),
						initialPathPreset);
					if (approximateSummary.targetCount != 1u || approximateSummary.successCount != 1u || approximateSummary.approximateCount != 1u || approximateSummary.failureCount != 0u)
						return fail("Preset apply summary smoke failed for approximate target range.");
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
					if (std::abs(scaledEvents[0].magnitude - 1.0) > 1e-9 ||
						std::abs(scaledEvents[1].magnitude - 6.0) > 1e-9 ||
						std::abs(scaledEvents[2].magnitude - 4.0) > 1e-9)
					{
						return fail("Camera manipulation utilities smoke failed for virtual-event scaling.");
					}
				}

				if (hasFreePreset && freeCamera)
				{
					CameraPreset orientedPreset = initialFreePreset;
					orientedPreset.goal.orientation = makeQuaternionFromEulerDegrees(float64_t3(0.0, 90.0, 0.0));
					const auto orientResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, freeCamera, orientedPreset);
					if (!orientResult.succeeded() || !comparePresetToCameraStrict(freeCamera, orientedPreset))
						return fail("Camera manipulation utilities smoke failed to orient Free camera before translation remap.");

					std::vector<CVirtualGimbalEvent> worldTranslationEvents(3u);
					worldTranslationEvents[0].type = CVirtualGimbalEvent::MoveRight;
					worldTranslationEvents[0].magnitude = 1.25;
					worldTranslationEvents[1].type = CVirtualGimbalEvent::MoveUp;
					worldTranslationEvents[1].magnitude = 0.5;
					worldTranslationEvents[2].type = CVirtualGimbalEvent::MoveForward;
					worldTranslationEvents[2].magnitude = 2.0;
					uint32_t remappedCount = static_cast<uint32_t>(worldTranslationEvents.size());
					nbl::core::remapTranslationEventsFromWorldToCameraLocal(freeCamera, worldTranslationEvents, remappedCount);
					if (remappedCount == 0u)
						return fail("Camera manipulation utilities smoke produced empty translation remap.");

					if (!freeCamera->manipulate({ worldTranslationEvents.data(), remappedCount }))
						return fail("Camera manipulation utilities smoke failed to apply remapped translation.");

					const auto remappedPosition = freeCamera->getGimbal().getPosition();
					const auto positionDelta = remappedPosition - orientedPreset.goal.position;
					const float64_t3 expectedWorldDelta(1.25, 0.5, 2.0);
					if (!hlsl::nearlyEqualVec3(positionDelta, expectedWorldDelta, 1e-6))
						return fail("Camera manipulation utilities smoke changed world-space translation semantics.");

					CameraPreset pitchPreset = initialFreePreset;
					pitchPreset.goal.orientation = makeQuaternionFromEulerDegrees(float64_t3(60.0, 0.0, 0.0));
					const auto pitchResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, freeCamera, pitchPreset);
					if (!pitchResult.succeeded())
						return fail("Camera manipulation utilities smoke failed to prepare Free camera pitch clamp.");

					SCameraConstraintSettings freeConstraints;
					freeConstraints.enabled = true;
					freeConstraints.clampPitch = true;
					freeConstraints.pitchMinDeg = -15.f;
					freeConstraints.pitchMaxDeg = 15.f;
					if (!nbl::core::applyCameraConstraints(m_cameraGoalSolver, freeCamera, freeConstraints))
						return fail("Camera manipulation utilities smoke failed to clamp Free camera orientation.");

					const auto freeEulerDeg = getQuaternionEulerDegrees(freeCamera->getGimbal().getOrientation());
					if (std::abs(static_cast<double>(freeEulerDeg.x - 15.f)) > 0.1)
						return fail("Camera manipulation utilities smoke produced wrong clamped Free camera pitch.");

					const auto restoreFree = nbl::core::applyPresetDetailed(m_cameraGoalSolver, freeCamera, initialFreePreset);
					if (!restoreFree.succeeded() || !comparePresetToCameraStrict(freeCamera, initialFreePreset))
						return fail("Camera manipulation utilities smoke failed to restore Free camera baseline.");
				}

				if (hasOrbitPreset && orbitCamera && initialOrbitPreset.goal.hasDistance)
				{
					CameraPreset farOrbitPreset = initialOrbitPreset;
					farOrbitPreset.goal.distance = initialOrbitPreset.goal.distance + 10.f;
					const auto farOrbitResult = nbl::core::applyPresetDetailed(m_cameraGoalSolver, orbitCamera, farOrbitPreset);
					if (!farOrbitResult.succeeded())
						return fail("Camera manipulation utilities smoke failed to prepare Orbit distance clamp.");

					SCameraConstraintSettings orbitConstraints;
					orbitConstraints.enabled = true;
					orbitConstraints.clampDistance = true;
					orbitConstraints.minDistance = std::max(0.1f, initialOrbitPreset.goal.distance * 0.5f);
					orbitConstraints.maxDistance = initialOrbitPreset.goal.distance * 0.75f;
					if (!nbl::core::applyCameraConstraints(m_cameraGoalSolver, orbitCamera, orbitConstraints))
						return fail("Camera manipulation utilities smoke failed to clamp Orbit distance.");

					ICamera::SphericalTargetState clampedOrbitState;
					if (!orbitCamera->tryGetSphericalTargetState(clampedOrbitState) ||
						std::abs(static_cast<double>(clampedOrbitState.distance - orbitConstraints.maxDistance)) > 1e-6)
					{
						return fail("Camera manipulation utilities smoke produced wrong clamped Orbit distance.");
					}

					const auto restoreOrbit = nbl::core::applyPresetDetailed(m_cameraGoalSolver, orbitCamera, initialOrbitPreset);
					if (!restoreOrbit.succeeded() || !comparePresetToCameraStrict(orbitCamera, initialOrbitPreset))
						return fail("Camera manipulation utilities smoke failed to restore Orbit baseline.");
				}

				if (hasDollyZoomPreset && dollyZoomCamera)
				{
					float dynamicFov = 0.0f;
					if (!dollyZoomCamera->tryGetDynamicPerspectiveFov(dynamicFov))
						return fail("Camera projection utilities smoke failed to query DollyZoom dynamic FOV.");

					auto perspectiveProjection = IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Perspective>(0.1f, 100.f, 60.f);
					if (!nbl::core::syncDynamicPerspectiveProjection(dollyZoomCamera, perspectiveProjection))
						return fail("Camera projection utilities smoke failed to sync dynamic perspective projection.");
					if (std::abs(static_cast<double>(perspectiveProjection.getParameters().m_planar.perspective.fov - dynamicFov)) > 1e-6)
						return fail("Camera projection utilities smoke produced wrong dynamic perspective FOV.");

					auto orthographicProjection = IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Orthographic>(0.1f, 100.f, 10.f);
					if (nbl::core::syncDynamicPerspectiveProjection(dollyZoomCamera, orthographicProjection))
						return fail("Camera projection utilities smoke unexpectedly synced orthographic projection.");
				}

				{
					if (getCameraTypeLabel(ICamera::CameraKind::DollyZoom) != "Dolly Zoom")
						return fail("Camera text utilities smoke failed for Dolly Zoom label.");
					if (getCameraTypeDescription(ICamera::CameraKind::Path) != "Move along a target path")
						return fail("Camera text utilities smoke failed for Path description.");
					if (describeGoalStateMask(ICamera::GoalStateNone) != "Pose only")
						return fail("Camera text utilities smoke failed for empty goal-state description.");
					if (describeGoalStateMask(ICamera::GoalStateSphericalTarget | ICamera::GoalStateDynamicPerspective) != "Spherical target, Dynamic perspective")
						return fail("Camera text utilities smoke failed for combined goal-state description.");

					CCameraGoalSolver::SApplyResult defaultApplyResult;
					const auto applyResultText = describeApplyResult(defaultApplyResult);
					if (applyResultText.find("status=Unsupported") == std::string::npos || applyResultText.find("events=0") == std::string::npos)
						return fail("Camera text utilities smoke failed for apply-result description.");

					SCameraPresetApplySummary summary;
					summary.targetCount = 2u;
					summary.successCount = 2u;
					summary.approximateCount = 1u;
					const auto summaryText = nbl::ui::describePresetApplySummary(summary, "none");
					if (summaryText.find("targets=2") == std::string::npos || summaryText.find("approximate=1") == std::string::npos)
						return fail("Camera text utilities smoke failed for preset-apply summary description.");
				}

				m_headlessCameraSmokePassed = true;
				std::cout << "[headless-camera-smoke] PASS cameras=" << cameras.size() << std::endl;
				return true;
			}

			m_ciMode = program.get<bool>("--ci");
			if (m_ciMode)
			{
				m_ciScreenshotPath = localOutputCWD / "cameraz_ci.png";
				m_ciStartedAt = clock_t::now();
			}
			m_scriptedInput.log = program.get<bool>("--script-log");
			m_scriptVisualDebugCli = program.get<bool>("--script-visual-debug");
			m_disableScreenshotsCli = program.get<bool>("--no-screenshots");

			// Create imput system
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));
			m_logFormatter = core::make_smart_refctd_ptr<CUILogFormatter>();

			if (!base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			{
				const std::optional<std::filesystem::path> cameraJsonFile = program.is_used("--file") ? std::optional<std::filesystem::path>(program.get<std::string>("--file")) : std::optional<std::filesystem::path>(std::nullopt);

				camera_json_t j;
				auto loadDefaultConfig = [&]() -> bool
				{
					const auto configPath = std::filesystem::path(nbl::system::SCameraAppResourcePaths::DefaultCameraConfigRelativePath);
					std::string jsonError;
					if (!nbl::system::loadJsonFromPath(*m_system, configPath, j, &jsonError))
						return logFail("%s", jsonError.c_str());
					return true;
				};

				std::string jsonError;
				std::filesystem::path resolvedCameraJsonFile;
				const bool hasUserConfig = cameraJsonFile.has_value();
				if (hasUserConfig)
				{
					resolvedCameraJsonFile = nbl::system::resolveInputPath(localInputCWD, cameraJsonFile.value());
				}

				if (hasUserConfig && nbl::system::loadJsonFromPath(*m_system, resolvedCameraJsonFile, j, &jsonError))
				{
					// Loaded from explicit user path.
				}
				else
				{
					if (hasUserConfig)
						m_logger->log("Cannot open input \"%s\" json file (%s). Switching to default config.", ILogger::ELL_WARNING, resolvedCameraJsonFile.string().c_str(), jsonError.c_str());
					else
						m_logger->log("No input json file provided. Switching to default config.", ILogger::ELL_INFO);

					if (!loadDefaultConfig())
						return false;
				}

				std::optional<CCameraSequenceScript> pendingScriptedSequence;
				bool scriptedInputParseFailed = false;
				std::string scriptedInputParseError;

				auto resetScriptedInputRuntimeState = [&]() -> void
				{
					m_scriptedInput.nextEventIndex = 0;
					m_scriptedInput.checkRuntime = {};
					m_scriptedInput.nextCaptureIndex = 0;
					m_scriptedInput.failed = false;
					m_scriptedInput.summaryReported = false;
				};

				auto finalizeScriptedInput = [&]() -> void
				{
					nbl::system::finalizeScriptedTimeline(m_scriptedInput.timeline, m_disableScreenshotsCli);
				};

				auto applyParsedScriptedInput = [&](nbl::system::CCameraScriptedInputParseResult parsed) -> void
				{
					pendingScriptedSequence.reset();
					scriptedInputParseFailed = false;
					scriptedInputParseError.clear();
					m_scriptedInput.timeline.clear();
					resetScriptedInputRuntimeState();
					m_scriptedInput.exclusive = false;
					m_scriptedInput.hardFail = false;
					m_scriptedInput.visualDebug = false;
					m_scriptedInput.visualTargetFps = 0.f;
					m_scriptedInput.visualCameraHoldSeconds = 0.f;
					m_scriptedInput.visualActivePlanarValid = false;
					m_scriptedInput.visualActivePlanarIx = 0u;
					m_scriptedInput.visualActivePlanarStartFrame = 0u;
					m_scriptedInput.scriptedLeftMouseDown = false;
					m_scriptedInput.scriptedRightMouseDown = false;
					m_scriptedInput.framePacerInitialized = false;
					m_scriptedInput.capturePrefix = "script";
					m_scriptedInput.captureOutputDir = localOutputCWD;

					m_scriptedInput.enabled = parsed.enabled;
					if (parsed.hasLog)
						m_scriptedInput.log = parsed.log || m_scriptedInput.log;
					m_scriptedInput.hardFail = parsed.hardFail;
					m_scriptedInput.visualDebug = parsed.visualDebug;
					m_scriptedInput.visualTargetFps = parsed.visualTargetFps;
					m_scriptedInput.visualCameraHoldSeconds = parsed.visualCameraHoldSeconds;
					if (m_scriptVisualDebugCli)
						m_scriptedInput.visualDebug = true;
					if (m_scriptedInput.visualDebug)
					{
						if (m_scriptedInput.visualTargetFps <= 0.f)
							m_scriptedInput.visualTargetFps = 60.f;
						if (m_scriptedInput.visualCameraHoldSeconds <= 0.f)
							m_scriptedInput.visualCameraHoldSeconds = 3.f;
					}

					if (parsed.hasEnableActiveCameraMovement)
						enableActiveCameraMovement = parsed.enableActiveCameraMovement;
					else if (m_scriptedInput.enabled)
						enableActiveCameraMovement = true;

					m_scriptedInput.exclusive = parsed.exclusive;
					m_scriptedInput.capturePrefix = parsed.capturePrefix.empty() ? "script" : parsed.capturePrefix;

					if (parsed.cameraControls.hasKeyboardScale)
						m_cameraControls.keyboardScale = parsed.cameraControls.keyboardScale;
					if (parsed.cameraControls.hasMouseMoveScale)
						m_cameraControls.mouseMoveScale = parsed.cameraControls.mouseMoveScale;
					if (parsed.cameraControls.hasMouseScrollScale)
						m_cameraControls.mouseScrollScale = parsed.cameraControls.mouseScrollScale;
					if (parsed.cameraControls.hasTranslationScale)
						m_cameraControls.translationScale = parsed.cameraControls.translationScale;
					if (parsed.cameraControls.hasRotationScale)
						m_cameraControls.rotationScale = parsed.cameraControls.rotationScale;

					for (const auto& warning : parsed.warnings)
						m_logger->log("%s", ILogger::ELL_WARNING, warning.c_str());

					pendingScriptedSequence = std::move(parsed.sequence);
					m_scriptedInput.timeline = std::move(parsed.timeline);
					finalizeScriptedInput();
				};

				if (program.is_used("--script"))
				{
					nbl::system::path scriptPath = nbl::system::resolveInputPath(localInputCWD, program.get<std::string>("--script"));
					nbl::system::CCameraScriptedInputParseResult parsed;
					if (!nbl::system::loadCameraScriptedInputFromFile(scriptPath, parsed, &scriptedInputParseError))
					{
						logFail("Camera sequence script parse failed: %s", scriptedInputParseError.c_str());
						return false;
					}
					applyParsedScriptedInput(std::move(parsed));
				}
				else if (j.contains("scripted_input"))
				{
					std::stringstream scriptedInputStream;
					scriptedInputStream << j["scripted_input"].dump();
					nbl::system::CCameraScriptedInputParseResult parsed;
					if (!nbl::system::readCameraScriptedInput(scriptedInputStream, parsed, &scriptedInputParseError))
						scriptedInputParseFailed = true;
					else
						applyParsedScriptedInput(std::move(parsed));
					if (scriptedInputParseFailed)
					{
						logFail("Camera sequence script parse failed: %s", scriptedInputParseError.c_str());
						return false;
					}
				}

				std::vector<smart_refctd_ptr<ICamera>> cameras;
				std::string cameraConfigError;
				if (!nbl::system::tryLoadCameraCollectionFromJson(j, cameraConfigError, cameras))
				{
					logFail("%s", cameraConfigError.c_str());
					return false;
				}

				std::vector<IPlanarProjection::CProjection> projections;
				for (const auto& jProjection : j["projections"])
				{
					if (jProjection.contains("type"))
					{
						float zNear, zFar;

						if (!jProjection.contains("zNear"))
						{
							logFail("Expected \"zNear\" keyword for planar projection definition!");
							return false;
						}

						if (!jProjection.contains("zFar"))
						{
							logFail("Expected \"zFar\" keyword for planar projection definition!");
							return false;
						}

						zNear = jProjection["zNear"].get<float>();
						zFar = jProjection["zFar"].get<float>();

						if (jProjection["type"] == "perspective")
						{
							if (!jProjection.contains("fov"))
							{
								logFail("Expected \"fov\" keyword for planar perspective projection definition!");
								return false;
							}

							float fov = jProjection["fov"].get<float>();
							projections.emplace_back(IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Perspective>(zNear, zFar, fov));
						}
						else if (jProjection["type"] == "orthographic")
						{
							if (!jProjection.contains("orthoWidth"))
							{
								logFail("Expected \"orthoWidth\" keyword for planar orthographic projection definition!");
								return false;
							}

							float orthoWidth = jProjection["orthoWidth"].get<float>();
							projections.emplace_back(IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Orthographic>(zNear, zFar, orthoWidth));
						}
						else
						{
							logFail("Unsupported projection!");
							return false;
						}
					}
				}

				struct
				{
					std::vector<IGimbalBindingLayout::keyboard_to_virtual_events_t> keyboard;
					std::vector<IGimbalBindingLayout::mouse_to_virtual_events_t> mouse;
				} bindings;

				const char* bindingLayoutsKey = j.contains("bindings") ? "bindings" : nullptr;

				if (bindingLayoutsKey)
				{
					const auto& jBindings = j[bindingLayoutsKey];

					if (jBindings.contains("keyboard"))
					{
						for (const auto& jKeyboard : jBindings["keyboard"])
						{
							if (jKeyboard.contains("mappings"))
							{
								auto& binding = bindings.keyboard.emplace_back();
								for (const auto& [key, value] : jKeyboard["mappings"].items())
								{
									const auto nativeCode = stringToKeyCode(key.c_str());

									if (nativeCode == EKC_NONE)
									{
										logFail("Invalid native key \"%s\" code mapping for keyboard binding", key.c_str());
										return false;
									}

									binding[nativeCode] = CVirtualGimbalEvent::stringToVirtualEvent(value.get<std::string>());
								}
							}
							else
							{
								logFail("Expected \"mappings\" keyword for keyboard binding definition!");
								return false;
							}
						}
					}
					else
					{
						logFail("Expected \"keyboard\" keyword in bindings definition!");
						return false;
					}

					if (jBindings.contains("mouse"))
					{
						for (const auto& jMouse : jBindings["mouse"])
						{
							if (jMouse.contains("mappings"))
							{
								auto& binding = bindings.mouse.emplace_back();
								for (const auto& [key, value] : jMouse["mappings"].items())
								{
									const auto nativeCode = stringToMouseCode(key.c_str());

									if (nativeCode == EMC_NONE)
									{
										logFail("Invalid native key \"%s\" code mapping for mouse binding", key.c_str());
										return false;
									}

									binding[nativeCode] = CVirtualGimbalEvent::stringToVirtualEvent(value.get<std::string>());
								}
							}
							else
							{
								logFail("Expected \"mappings\" keyword for mouse binding definition!");
								return false;
							}
						}
					}
					else
					{
						logFail("Expected \"mouse\" keyword in bindings definition");
						return false;
					}
				}
				else
				{
					logFail("Expected \"bindings\" keyword in camera JSON");
					return false;
				}

				if (j.contains("viewports") && j.contains("planars"))
				{
					for (const auto& jPlanar : j["planars"])
					{
						if (!jPlanar.contains("camera"))
						{
							logFail("Expected \"camera\" value in planar object");
							return false;
						}

						if (!jPlanar.contains("viewports"))
						{
							logFail("Expected \"viewports\" list in planar object");
							return false;
						}

						const auto cameraIx = jPlanar["camera"].get<uint32_t>();
						auto boundViewports = jPlanar["viewports"].get<std::vector<uint32_t>>();

						auto& planar = m_planarProjections.emplace_back() = planar_projection_t::create(smart_refctd_ptr(cameras[cameraIx]));
						for (const auto viewportIx : boundViewports)
						{
							auto& viewport = j["viewports"][viewportIx];
							const char* viewportBindingsKey = viewport.contains("bindings") ? "bindings" : nullptr;
							if (!viewport.contains("projection") || !viewportBindingsKey)
							{
								logFail("\"projection\" or \"bindings\" missing in viewport object index %d", viewportIx);
								return false;
							}

							const auto projectionIx = viewport["projection"].get<uint32_t>();
							auto& projection = planar->getPlanarProjections().emplace_back(projections[projectionIx]);
							auto& projectionBinding = projection.getInputBinding();
							const auto& jViewportBindings = viewport[viewportBindingsKey];

							const bool hasKeyboardBound = jViewportBindings.contains("keyboard");
							const bool hasMouseBound = jViewportBindings.contains("mouse");

							if (hasKeyboardBound)
							{
								auto keyboardBindingIx = jViewportBindings["keyboard"].get<uint32_t>();
								projectionBinding.updateKeyboardMapping([&](auto& map) { map = bindings.keyboard[keyboardBindingIx]; });
							}
							else
								projectionBinding.updateKeyboardMapping([&](auto& map) { map = {}; }); // clean the map if not bound

							if (hasMouseBound)
							{
								auto mouseBindingIx = jViewportBindings["mouse"].get<uint32_t>();
								projectionBinding.updateMouseMapping([&](auto& map) { map = bindings.mouse[mouseBindingIx]; });
							}
							else
								projectionBinding.updateMouseMapping([&](auto& map) { map = {}; }); // clean the map if not bound
						}

					}
				}
				else
				{
					logFail("Expected \"viewports\" and \"planars\" lists in JSON");
					return false;
				}

				if (m_planarProjections.empty())
				{
					logFail("Expected at least 1 planar");
					return false;
				}

				// init render window planar references - we make all render windows start with focus on first
				// planar but in a way that first window has the planar's perspective preset bound & second orthographic
				for (uint32_t i = 0u; i < windowBindings.size(); ++i)
				{
					auto& binding = windowBindings[i];

					auto& planar = m_planarProjections[binding.activePlanarIx = 0];
					binding.pickDefaultProjections(planar->getPlanarProjections());

					if (i)
						binding.boundProjectionIx = binding.lastBoundOrthoPresetProjectionIx.value();
					else
						binding.boundProjectionIx = binding.lastBoundPerspectivePresetProjectionIx.value();
				}

				m_initialPlanarPresets.clear();
				m_initialPlanarPresets.reserve(m_planarProjections.size());
				for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
				{
					auto* camera = m_planarProjections[planarIx]->getCamera();
					const std::string presetName = "Planar " + std::to_string(planarIx);
					const auto captureAnalysis = nbl::core::analyzeCameraCapture(m_cameraGoalSolver, camera);
					if (!captureAnalysis.canCapture)
					{
						const auto kindLabel = camera ? std::string(getCameraTypeLabel(camera->getKind())) : std::string("Unknown");
						const auto reason = !captureAnalysis.hasCamera ? "missing camera" :
							(!captureAnalysis.capturedGoal ? "capture failed" :
								(!captureAnalysis.finiteGoal ? "non-finite goal" : "unknown"));
						return logFail("Failed to capture initial planar preset %u for camera kind \"%s\": %s",
							planarIx, kindLabel.c_str(), reason);
					}

					CameraPreset preset = {};
					if (!nbl::core::tryCapturePreset(captureAnalysis, camera, presetName, preset))
						return logFail("Failed to build initial planar preset %u for camera kind \"%s\".",
							planarIx,
							camera ? std::string(getCameraTypeLabel(camera->getKind())).c_str() : "Unknown");
					m_initialPlanarPresets.emplace_back(std::move(preset));
				}

				resetFollowTargetToDefault();
				m_planarFollowConfigs.clear();
				m_planarFollowConfigs.reserve(m_planarProjections.size());
				for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
				{
					auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
					auto config = makeDefaultFollowConfig(camera);
					m_planarFollowConfigs.emplace_back(config);
					if (config.enabled)
						captureFollowOffsetsForPlanar(planarIx);
				}
				bindManipulatedModel();

				if (pendingScriptedSequence.has_value())
				{
					auto expandCameraSequenceScript = [&](const CCameraSequenceScript& sequence) -> bool
					{
						CCameraScriptedTimeline timeline;
						resetScriptedInputRuntimeState();

						auto resolvePlanarIx = [&](const CCameraSequenceSegment& segment) -> std::optional<uint32_t>
						{
							std::optional<uint32_t> match;
							for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
							{
								auto* camera = m_planarProjections[planarIx]->getCamera();
								if (!camera)
									continue;

								const bool kindMatch = segment.cameraKind == ICamera::CameraKind::Unknown || camera->getKind() == segment.cameraKind;
								const bool identifierMatch = segment.cameraIdentifier.empty() || camera->getIdentifier() == segment.cameraIdentifier;
								if (!(kindMatch && identifierMatch))
									continue;

								if (match.has_value())
									return std::nullopt;
								match = planarIx;
							}
							return match;
						};

						const bool useWindow = nbl::core::sequenceScriptUsesMultiplePresentations(sequence);
						nbl::system::appendScriptedActionEvent(timeline, 0u, CCameraScriptedInputEvent::ActionData::Kind::SetUseWindow, useWindow ? 1 : 0);

						const CCameraSequenceTrackedTargetPose referenceTrackedTargetPose = {
							.position = getDefaultFollowTargetPosition(),
							.orientation = getDefaultFollowTargetOrientation()
						};
						uint64_t frameCursor = 0u;
						for (const auto& segment : sequence.segments)
						{
							auto planarIx = resolvePlanarIx(segment);
							if (!planarIx.has_value())
							{
								const auto kindLabel = segment.cameraKind != ICamera::CameraKind::Unknown ? std::string(getCameraTypeLabel(segment.cameraKind)) : std::string("Unknown");
								logFail("Sequence segment \"%s\" has ambiguous or missing camera match for kind \"%s\" identifier \"%s\".",
									segment.name.c_str(), kindLabel.c_str(), segment.cameraIdentifier.c_str());
								return false;
							}
							const bool useTrackedTargetFollow = nbl::core::sequenceSegmentUsesTrackedTargetTrack(segment) &&
								planarIx.value() < m_planarFollowConfigs.size() &&
								m_planarFollowConfigs[planarIx.value()].enabled &&
								m_planarFollowConfigs[planarIx.value()].mode != ECameraFollowMode::Disabled;

							nbl::core::CCameraSequenceCompiledSegment compiledSegment;
							std::string trackError;
							if (!nbl::core::compileSequenceSegmentFromReference(
								sequence,
								segment,
								m_initialPlanarPresets[planarIx.value()],
								referenceTrackedTargetPose,
								compiledSegment,
								&trackError))
							{
								logFail("Sequence segment \"%s\" failed to compile: %s", segment.name.c_str(), trackError.c_str());
								return false;
							}

							if (compiledSegment.presentations.size() > windowBindings.size())
							{
								m_logger->log("Sequence segment \"%s\" requests %zu presentations, only %zu windows are available. Extra presentations will be ignored.",
									ILogger::ELL_WARNING, segment.name.c_str(), compiledSegment.presentations.size(), windowBindings.size());
							}

							std::string buildError;
							if (!nbl::system::appendCompiledSequenceSegmentToScriptedTimeline(
								timeline,
								frameCursor,
								compiledSegment,
								{
									.planarIx = planarIx.value(),
									.availableWindowCount = windowBindings.size(),
									.useWindow = useWindow,
									.includeFollowTargetLock = useTrackedTargetFollow
								},
								&buildError))
							{
								logFail("Sequence segment \"%s\" failed to build scripted runtime data: %s",
									segment.name.c_str(), buildError.c_str());
								return false;
							}

							frameCursor += compiledSegment.durationFrames;
						}

						nbl::system::finalizeScriptedTimeline(timeline, m_disableScreenshotsCli);
						m_scriptedInput.timeline = std::move(timeline);
						return true;
					};

					if (!expandCameraSequenceScript(*pendingScriptedSequence))
						return false;
				}
			}

			// First create the resources that don't depend on a swapchain
			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			// The nice thing about having a triple buffer is that you don't need to do acrobatics to account for the formats available to the surface.
			// You can transcode to the swapchain's format while copying, and I actually recommend to do surface rotation, tonemapping and OETF application there.
			const auto format = asset::EF_R8G8B8A8_SRGB;
			// Could be more clever and use the copy Triple Buffer to Swapchain as an opportunity to do a MSAA resolve or something
			const auto samples = IGPUImage::ESCF_1_BIT;

			// Create the renderpass
			{
				const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
					{{
						{
							.format = format,
							.samples = samples,
							.mayAlias = false
						},
					/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
					/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
					/*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED, // because we clear we don't care about contents when we grab the triple buffer img again
					/*.finalLayout = */IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL // put it already in the correct layout for the blit operation
				}},
				IGPURenderpass::SCreationParams::ColorAttachmentsEnd
				};
				IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
					{},
					IGPURenderpass::SCreationParams::SubpassesEnd
				};
				subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} };
				// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
				IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
					// wipe-transition to ATTACHMENT_OPTIMAL
					{
						.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.dstSubpass = 0,
						.memoryBarrier = {
						// we can have NONE as Sources because the semaphore wait is ALL_COMMANDS
						// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
					// leave view offsets and flags default
				},
					// ATTACHMENT_OPTIMAL to PRESENT_SRC
					{
						.srcSubpass = 0,
						.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.memoryBarrier = {
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
							// we can have NONE as the Destinations because the semaphore signal is ALL_COMMANDS
							// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
						}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
				};

				IGPURenderpass::SCreationParams params = {};
				params.colorAttachments = colorAttachments;
				params.subpasses = subpasses;
				params.dependencies = dependencies;
				m_renderpass = m_device->createRenderpass(params);
				if (!m_renderpass)
					return logFail("Failed to Create a Renderpass!");
			}

			// We just live life in easy mode and have the Swapchain Creation Parameters get deduced from the surface.
			// We don't need any control over the format of the swapchain because we'll be only using Renderpasses this time!
			ISwapchain::SSharedCreationParams sharedParams = {};
			sharedParams.imageUsage |= IGPUImage::EUF_TRANSFER_SRC_BIT;
			auto swapchainResources = std::make_unique<CSwapchainResources>();
			if (!m_surface || !m_surface->init(m_surface->pickQueue(m_device.get()), std::move(swapchainResources), sharedParams))
				return logFail("Failed to Create a Swapchain!");

			// Normally you'd want to recreate these images whenever the swapchain is resized in some increment, like 64 pixels or something.
			// But I'm super lazy here and will just create "worst case sized images" and waste all the VRAM I can get.
			const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
			for (auto i = 0; i < MaxFramesInFlight; i++)
			{
				auto& image = m_tripleBuffers[i];
				{
					IGPUImage::SCreationParams params = {};
					params = asset::IImage::SCreationParams{
						.type = IGPUImage::ET_2D,
						.samples = samples,
						.format = format,
						.extent = {dpyInfo.resX,dpyInfo.resY,1},
						.mipLevels = 1,
						.arrayLayers = 1,
						.flags = IGPUImage::ECF_NONE,
						// in this example I'll be using a renderpass to clear the image, and then a blit to copy it to the swapchain
						.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_TRANSFER_SRC_BIT
					};
					image = m_device->createImage(std::move(params));
					if (!image)
						return logFail("Failed to Create Triple Buffer Image!");

					// use dedicated allocations, we have plenty of allocations left, even on Win32
					if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
						return logFail("Failed to allocate Device Memory for Image %d", i);
				}
				image->setObjectDebugName(("Triple Buffer Image " + std::to_string(i)).c_str());

				// create framebuffers for the images
				{
					auto imageView = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						// give it a Transfer SRC usage flag so we can transition to the Tranfer SRC layout with End Renderpass
						.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_TRANSFER_SRC_BIT,
						.image = core::smart_refctd_ptr(image),
						.viewType = IGPUImageView::ET_2D,
						.format = format
						});
					const auto& imageParams = image->getCreationParameters();
					IGPUFramebuffer::SCreationParams params = { {
						.renderpass = core::smart_refctd_ptr(m_renderpass),
						.depthStencilAttachments = nullptr,
						.colorAttachments = &imageView.get(),
						.width = imageParams.extent.width,
						.height = imageParams.extent.height,
						.layers = imageParams.arrayLayers
					} };
					m_framebuffers[i] = m_device->createFramebuffer(std::move(params));
					if (!m_framebuffers[i])
						return logFail("Failed to Create a Framebuffer for Image %d", i);
				}
			}

			// This time we'll create all CommandBuffers from one CommandPool, to keep life simple. However the Pool must support individually resettable CommandBuffers
			// because they cannot be pre-recorded because the fraembuffers/swapchain images they use will change when a swapchain recreates.
			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(),MaxFramesInFlight }, core::smart_refctd_ptr(m_logger)))
				return logFail("Failed to Create CommandBuffers!");

			// UI
			{
				{
					nbl::ext::imgui::UI::SCreationParameters params;
					params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
					params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
					params.assetManager = m_assetMgr;
					params.pipelineCache = nullptr;
					params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, TotalUISampleTexturesAmount);
					params.renderpass = smart_refctd_ptr<IGPURenderpass>(m_renderpass);
					params.subpassIx = 0u;
					params.transfer = getTransferUpQueue();
					params.utilities = m_utils;

					const auto vertexKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_vertex">(m_device.get());
					const auto fragmentKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_fragment">(m_device.get());
					auto vertexShader = nbl::system::loadPrecompiledShaderFromAppResources(*m_assetMgr, m_logger.get(), vertexKey);
					auto fragmentShader = nbl::system::loadPrecompiledShaderFromAppResources(*m_assetMgr, m_logger.get(), fragmentKey);
					if (!vertexShader || !fragmentShader)
						return logFail("Failed to load precompiled ImGui shaders.");

					params.spirv = nbl::ext::imgui::UI::SCreationParameters::PrecompiledShaders{
						.vertex = std::move(vertexShader),
						.fragment = std::move(fragmentShader)
					};

					m_ui.manager = nbl::ext::imgui::UI::create(std::move(params));
				}

				if (!m_ui.manager)
					return false;

				// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
				const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);

				IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = TotalUISampleTexturesAmount;
				descriptorPoolInfo.maxSets = 1u;
				descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

				m_descriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
				assert(m_descriptorSetPool);

				m_descriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
				assert(m_ui.descriptorSet);

				m_ui.manager->registerListener([this]() -> void { imguiListen(); });
				{
					const auto ds = float32_t2{ m_window->getWidth(), m_window->getHeight() };

					wInit.trsEditor.iPos = iPaddingOffset;
					wInit.trsEditor.iSize = { 0.0f, ds.y - wInit.trsEditor.iPos.y * 2 };

					const float panelWidth = std::clamp(ds.x * 0.33f, 380.0f, ds.x * 0.48f);
					wInit.planars.iSize = { panelWidth, ds.y - iPaddingOffset.y * 2 };
					wInit.planars.iPos = { ds.x - wInit.planars.iSize.x - iPaddingOffset.x, 0 + iPaddingOffset.y };

					{
						const float renderPaddingX = 0.0f;
						const float renderPaddingY = 0.0f;
						const float splitGap = 4.0f;
						float leftX = renderPaddingX;
						float eachXSize = std::max(0.0f, ds.x - 2.0f * renderPaddingX);
						float eachYSize = (ds.y - 2.0f * renderPaddingY - (wInit.renderWindows.size() - 1) * splitGap) / wInit.renderWindows.size();
						
						for (size_t i = 0; i < wInit.renderWindows.size(); ++i)
						{
							auto& rw = wInit.renderWindows[i];
							rw.iPos = { leftX, renderPaddingY + i * (eachYSize + splitGap) };
							rw.iSize = { eachXSize, eachYSize };
						}
					}
				}
			}

			// Geometry Creator Render Scene FBOs
			{
				const uint32_t addtionalBufferOwnershipFamilies[] = { getGraphicsQueue()->getFamilyIndex() };
				m_scene = CGeometryCreatorScene::create(
					{
						.transferQueue = getTransferUpQueue(),
						.utilities = m_utils.get(),
						.logger = m_logger.get(),
						.addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies
					},
					CSimpleDebugRenderer::DefaultPolygonGeometryPatch
				);

				if (!m_scene)
					return logFail("Could not create geometry creator scene!");

				{
					IGPURenderpass::SCreationParams params = {};
					const IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
						{{
							{
								.format = sceneRenderDepthFormat,
								.samples = IGPUImage::ESCF_1_BIT,
								.mayAlias = false
							},
							/*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
							/*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
							/*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED},
							/*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
						}},
						IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
					};
					params.depthStencilAttachments = depthAttachments;
					const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
						{{
							{
								.format = finalSceneRenderFormat,
								.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
								.mayAlias = false
							},
							/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
							/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
							/*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED,
							/*.finalLayout = */ IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
						}},
						IGPURenderpass::SCreationParams::ColorAttachmentsEnd
					};
					params.colorAttachments = colorAttachments;
					IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
						{},
						IGPURenderpass::SCreationParams::SubpassesEnd
					};
					subpasses[0].depthStencilAttachment = {{.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}};
					subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
					params.subpasses = subpasses;
					const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
						{
							.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
							.dstSubpass = 0,
							.memoryBarrier = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.srcAccessMask = ACCESS_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
							}
						},
						{
							.srcSubpass = 0,
							.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
							.memoryBarrier = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT|PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
								.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
							}
						},
						IGPURenderpass::SCreationParams::DependenciesEnd
					};
					params.dependencies = {};
					m_sceneRenderpass = m_device->createRenderpass(std::move(params));
					if (!m_sceneRenderpass)
						return logFail("Failed to create Scene Renderpass!");
				}

				{
					nbl::system::SSpaceEnvBlobHeader envBlobHeader = {};
					std::vector<uint8_t> envBlobPayload;
					const auto spaceEnvSearchRoots = nbl::system::makeSpaceEnvSearchRoots(localInputCWD);
					nbl::system::loadFirstSpaceEnvBlobFromRoots(*m_system, spaceEnvSearchRoots, envBlobHeader, envBlobPayload);
					if (envBlobPayload.empty())
						return logFail("Failed to load space environment blob from available assets.");

					const E_FORMAT envFormat = EF_R16G16B16A16_SFLOAT;
					const asset::VkExtent3D envExtent = { envBlobHeader.width, envBlobHeader.height, 1u };
					constexpr uint32_t envMipLevels = 1u;
					constexpr uint32_t envArrayLayers = 1u;
					const E_FORMAT envGpuFormat = envFormat;
					const std::array<IImage::SBufferCopy, 1u> envRegions = {{
						{
							.bufferOffset = 0ull,
							.bufferRowLength = 0u,
							.bufferImageHeight = 0u,
							.imageSubresource = {
								.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
								.mipLevel = 0u,
								.baseArrayLayer = 0u,
								.layerCount = envArrayLayers
							},
							.imageOffset = { 0, 0, 0 },
							.imageExtent = envExtent
						}
					}};

					IGPUImage::SCreationParams imageParams = {};
					imageParams = asset::IImage::SCreationParams{
						.type = IGPUImage::ET_2D,
						.samples = IGPUImage::ESCF_1_BIT,
						.format = envGpuFormat,
						.extent = envExtent,
						.mipLevels = envMipLevels,
						.arrayLayers = envArrayLayers,
						.flags = IGPUImage::ECF_NONE,
						.usage = IGPUImage::EUF_SAMPLED_BIT | IGPUImage::EUF_TRANSFER_DST_BIT
					};
					m_spaceEnvImage = m_device->createImage(std::move(imageParams));
					if (!m_spaceEnvImage)
						return logFail("Failed to create space environment image.");
					m_spaceEnvImage->setObjectDebugName("61_UI Space Environment");

					auto memReqs = m_spaceEnvImage->getMemoryReqs();
					memReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
					if (!m_device->allocate(memReqs, m_spaceEnvImage.get()).isValid())
						return logFail("Failed to allocate memory for space environment image.");

					auto uploadResult = m_utils->autoSubmit(
						SIntendedSubmitInfo{ .queue = getGraphicsQueue() },
						[&](SIntendedSubmitInfo& submitInfo) -> bool
						{
							auto* recordingInfo = submitInfo.getCommandBufferForRecording();
							if (!recordingInfo)
								return false;

							auto* cmdbuf = recordingInfo->cmdbuf;
							using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
							const image_barrier_t preBarrier[] = {
								{
									.barrier = {
										.dep = {
											.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
											.srcAccessMask = ACCESS_FLAGS::NONE,
											.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
											.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
										}
									},
									.image = m_spaceEnvImage.get(),
									.subresourceRange = {
										.aspectMask = IGPUImage::EAF_COLOR_BIT,
										.baseMipLevel = 0u,
										.levelCount = envMipLevels,
										.baseArrayLayer = 0u,
										.layerCount = envArrayLayers
									},
									.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
									.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
								}
							};
							const IGPUCommandBuffer::SPipelineBarrierDependencyInfo preDep = { .imgBarriers = preBarrier };
							bool success = cmdbuf->pipelineBarrier(asset::EDF_NONE, preDep);
							success = success && m_utils->updateImageViaStagingBuffer(
								submitInfo,
								envBlobPayload.data(),
								envFormat,
								m_spaceEnvImage.get(),
								IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
								std::span<const IImage::SBufferCopy>(envRegions));
							recordingInfo = submitInfo.getCommandBufferForRecording();
							if (!recordingInfo)
								return false;
							cmdbuf = recordingInfo->cmdbuf;

							const image_barrier_t postBarrier[] = {
								{
									.barrier = {
										.dep = {
											.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
											.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
											.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
											.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
										}
									},
									.image = m_spaceEnvImage.get(),
									.subresourceRange = {
										.aspectMask = IGPUImage::EAF_COLOR_BIT,
										.baseMipLevel = 0u,
										.levelCount = envMipLevels,
										.baseArrayLayer = 0u,
										.layerCount = envArrayLayers
									},
									.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
									.newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
								}
							};
							const IGPUCommandBuffer::SPipelineBarrierDependencyInfo postDep = { .imgBarriers = postBarrier };
							success = success && cmdbuf->pipelineBarrier(asset::EDF_NONE, postDep);
							return success;
						});
					if (uploadResult.copy() != IQueue::RESULT::SUCCESS)
						return logFail("Failed to upload space environment map.");

					IGPUImageView::SCreationParams viewParams = {};
					viewParams.subUsages = IGPUImage::EUF_SAMPLED_BIT;
					viewParams.image = core::smart_refctd_ptr(m_spaceEnvImage);
					viewParams.viewType = IGPUImageView::ET_2D;
					viewParams.format = envGpuFormat;
					viewParams.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
					viewParams.subresourceRange.baseMipLevel = 0u;
					viewParams.subresourceRange.levelCount = envMipLevels;
					viewParams.subresourceRange.baseArrayLayer = 0u;
					viewParams.subresourceRange.layerCount = envArrayLayers;
					m_spaceEnvImageView = m_device->createImageView(std::move(viewParams));
					if (!m_spaceEnvImageView)
						return logFail("Failed to create space environment image view.");

					IGPUSampler::SParams samplerParams = {};
					samplerParams.MinFilter = ISampler::ETF_LINEAR;
					samplerParams.MaxFilter = ISampler::ETF_LINEAR;
					samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
					samplerParams.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
					samplerParams.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
					samplerParams.TextureWrapW = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
					samplerParams.AnisotropicFilter = 0u;
					samplerParams.CompareEnable = false;
					samplerParams.CompareFunc = ISampler::ECO_ALWAYS;
					m_spaceEnvSampler = m_device->createSampler(samplerParams);
					if (!m_spaceEnvSampler)
						return logFail("Failed to create space environment sampler.");

					const IGPUDescriptorSetLayout::SBinding bindings[] = {
						{
							.binding = 0u,
							.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
							.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
							.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
							.count = 1u,
							.immutableSamplers = &m_spaceEnvSampler
						}
					};
					m_spaceEnvDescriptorSetLayout = m_device->createDescriptorSetLayout(std::span{ bindings });
					if (!m_spaceEnvDescriptorSetLayout)
						return logFail("Failed to create space environment descriptor set layout.");

					const asset::SPushConstantRange pushConstantRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.offset = 0u,
						.size = sizeof(SpaceEnvPushConstants)
					};
					auto pipelineLayout = m_device->createPipelineLayout(
						{ &pushConstantRange, 1u },
						core::smart_refctd_ptr(m_spaceEnvDescriptorSetLayout),
						nullptr,
						nullptr,
						nullptr);
					if (!pipelineLayout)
						return logFail("Failed to create space environment pipeline layout.");

					const auto spaceFragKey = nbl::this_example::builtin::build::get_spirv_key<"sky_env_fragment">(m_device.get());
					auto fragmentShader = nbl::system::loadPrecompiledShaderFromAppResources(*m_assetMgr, m_logger.get(), spaceFragKey);
					if (!fragmentShader)
						return logFail("Failed to load space environment fragment shader.");

					nbl::ext::FullScreenTriangle::ProtoPipeline fsTriProto(m_assetMgr.get(), m_device.get(), m_logger.get());
					if (!fsTriProto)
						return logFail("Failed to create FullScreenTriangle prototype pipeline.");

					const IGPUPipelineBase::SShaderSpecInfo fragmentSpec = {
						.shader = fragmentShader.get(),
						.entryPoint = "main"
					};
					m_spaceEnvPipeline = fsTriProto.createPipeline(fragmentSpec, pipelineLayout.get(), m_sceneRenderpass.get());
					if (!m_spaceEnvPipeline)
						return logFail("Failed to create space environment pipeline.");

					uint32_t setCount = 1u;
					const IGPUDescriptorSetLayout* setLayouts[] = { m_spaceEnvDescriptorSetLayout.get() };
					m_spaceEnvDescriptorPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, setLayouts, &setCount);
					if (!m_spaceEnvDescriptorPool)
						return logFail("Failed to create space environment descriptor pool.");
					m_spaceEnvDescriptorSet = m_spaceEnvDescriptorPool->createDescriptorSet(core::smart_refctd_ptr(m_spaceEnvDescriptorSetLayout));
					if (!m_spaceEnvDescriptorSet)
						return logFail("Failed to create space environment descriptor set.");

					IGPUDescriptorSet::SDescriptorInfo info = {};
					info.desc = m_spaceEnvImageView;
					info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

					IGPUDescriptorSet::SWriteDescriptorSet write = {};
					write.dstSet = m_spaceEnvDescriptorSet.get();
					write.binding = 0u;
					write.arrayElement = 0u;
					write.count = 1u;
					write.info = &info;
					if (!m_device->updateDescriptorSets({ &write, 1u }, {}))
						return logFail("Failed to update space environment descriptor set.");
				}

				const auto& geometries = m_scene->getInitParams().geometries;
				if (geometries.empty())
					return logFail("No geometries found for scene!");
				m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), m_sceneRenderpass.get(), 0, { &geometries.front().get(), geometries.size() });
				if (!m_renderer)
					return logFail("Failed to create debug renderer!");
				{
					const asset::SPushConstantRange singlePcRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX,
						.offset = offsetof(ext::frustum::PushConstants, spc),
						.size = sizeof(ext::frustum::SSinglePC)
					};

					ext::frustum::CDrawFrustum::SCreationParameters frustumParams = {};
					frustumParams.transfer = getTransferUpQueue();
					frustumParams.assetManager = m_assetMgr;
					frustumParams.drawMode = ext::frustum::CDrawFrustum::DrawMode::DM_SINGLE;
					frustumParams.singlePipelineLayout = ext::frustum::CDrawFrustum::createPipelineLayoutFromPCRange(m_device.get(), singlePcRange);
					frustumParams.renderpass = core::smart_refctd_ptr(m_sceneRenderpass);
					frustumParams.utilities = m_utils;
					m_drawFrustum = ext::frustum::CDrawFrustum::create(std::move(frustumParams));
					if (!m_drawFrustum)
						return logFail("Failed to create frustum drawer.");
				}

				{
					const auto& pipelines = m_renderer->getInitParams().pipelines;
					m_gridGeometryIx = std::nullopt;
					m_followTargetGeometryIx = std::nullopt;
					auto ix = 0u;
					for (const auto& name : m_scene->getInitParams().geometryNames)
					{
						if (name == "Cube")
						{
							if (!m_followTargetGeometryIx.has_value())
								m_followTargetGeometryIx = ix;
						}
						else if (name == "Cone")
							m_renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
						else if (name == "Grid")
							m_gridGeometryIx = ix;
						ix++;
					}
				}
				m_renderer->m_instances.resize(1u + (m_gridGeometryIx.has_value() ? 1u : 0u) + (m_followTargetGeometryIx.has_value() ? 1u : 0u));

				const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
				for (uint32_t i = 0u; i < windowBindings.size(); ++i)
				{
					auto& binding = windowBindings[i];
					binding.sceneColorView = createAttachmentView(m_device.get(), finalSceneRenderFormat, dpyInfo.resX, dpyInfo.resY, "UI Scene Color Attachment");
					binding.sceneDepthView = createAttachmentView(m_device.get(), sceneRenderDepthFormat, dpyInfo.resX, dpyInfo.resY, "UI Scene Depth Attachment");
					binding.sceneFramebuffer = createSceneFramebuffer(m_device.get(), m_sceneRenderpass.get(), binding.sceneColorView.get(), binding.sceneDepthView.get());
					if (!binding.sceneFramebuffer)
						return logFail("Could not create geometry creator scene[%d]!", i);
				}
			}

			oracle.reportBeginFrameRecord();

			if (base_t::argv.size() >= 3 && argv[1] == "-timeout_seconds")
				timeout = std::chrono::seconds(std::atoi(argv[2].c_str()));
			start = clock_t::now();
			return true;

}




