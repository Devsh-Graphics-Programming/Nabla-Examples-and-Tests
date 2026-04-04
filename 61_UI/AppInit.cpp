#include "app/App.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <span>
#include <vector>

namespace
{
	struct SpaceEnvBlobHeader final
	{
		uint32_t magic = 0u;
		uint32_t width = 0u;
		uint32_t height = 0u;
		uint32_t format = 0u;
		uint64_t payloadSize = 0ull;
	};

	constexpr uint32_t SpaceEnvBlobMagic = 0x31425645u; // "EVB1"
	constexpr uint32_t SpaceEnvBlobFormatRgba16Sfloat = 2u;

	bool loadSpaceEnvBlob(const std::filesystem::path& blobPath, SpaceEnvBlobHeader& outHeader, std::vector<uint8_t>& outPayload)
	{
		std::ifstream in(blobPath, std::ios::binary);
		if (!in.is_open())
			return false;

		in.read(reinterpret_cast<char*>(&outHeader), sizeof(outHeader));
		if (in.gcount() != sizeof(outHeader))
			return false;

		if (outHeader.magic != SpaceEnvBlobMagic || outHeader.format != SpaceEnvBlobFormatRgba16Sfloat)
			return false;
		if (outHeader.width == 0u || outHeader.height == 0u)
			return false;
		if (outHeader.payloadSize != static_cast<uint64_t>(outHeader.width) * outHeader.height * 8ull)
			return false;
		if (outHeader.payloadSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
			return false;

		outPayload.resize(static_cast<size_t>(outHeader.payloadSize));
		in.read(reinterpret_cast<char*>(outPayload.data()), static_cast<std::streamsize>(outPayload.size()));
		return in.gcount() == static_cast<std::streamsize>(outPayload.size());
	}

	constexpr float CameraDefaultMoveScale = 0.01f;
	constexpr float CameraDefaultRotateScale = 0.003f;
	constexpr float CameraOrbitMoveScale = 0.5f;

	void initializeCameraRigConfig(nbl::hlsl::ICamera& camera, const double moveScale, const double rotationScale)
	{
		camera.setMoveSpeedScale(moveScale);
		camera.setRotationSpeedScale(rotationScale);
		camera.updateKeyboardMapping([&](auto& map) { map = camera.getKeyboardMappingPreset(); });
		camera.updateMouseMapping([&](auto& map) { map = camera.getMouseMappingPreset(); });
		camera.updateImguizmoMapping([&](auto& map) { map = camera.getImguizmoMappingPreset(); });
	}

	bool createCameraFromJson(const nbl_json& jCamera, std::string& error, smart_refctd_ptr<nbl::hlsl::ICamera>& outCamera)
	{
		using namespace nbl::hlsl;

		if (!jCamera.contains("type"))
		{
			error = "Camera entry missing \"type\".";
			return false;
		}

		if (!jCamera.contains("position"))
		{
			error = "Camera entry missing \"position\".";
			return false;
		}

		const std::string type = jCamera["type"].get<std::string>();
		const bool withOrientation = jCamera.contains("orientation");
		const bool withTarget = jCamera.contains("target");

		auto position = [&]()
		{
			const auto jret = jCamera["position"].get<std::array<float, 3>>();
			return float32_t3(jret[0], jret[1], jret[2]);
		}();

		auto getOrientation = [&]()
		{
			const auto jret = jCamera["orientation"].get<std::array<float, 4>>();
			return glm::quat(jret[3], jret[0], jret[1], jret[2]);
		};

		auto getTarget = [&]()
		{
			const auto jret = jCamera["target"].get<std::array<float, 3>>();
			return float32_t3(jret[0], jret[1], jret[2]);
		};

		auto finalize = [&](auto&& camera, const double moveScale, const double rotationScale)
		{
			initializeCameraRigConfig(*camera, moveScale, rotationScale);
			outCamera = std::move(camera);
			return true;
		};

		if (type == "FPS")
		{
			if (!withOrientation)
			{
				error = "FPS camera requires \"orientation\".";
				return false;
			}
			return finalize(make_smart_refctd_ptr<CFPSCamera>(position, getOrientation()), CameraDefaultMoveScale, CameraDefaultRotateScale);
		}

		if (type == "Free")
		{
			if (!withOrientation)
			{
				error = "Free camera requires \"orientation\".";
				return false;
			}
			return finalize(make_smart_refctd_ptr<CFreeCamera>(position, getOrientation()), CameraDefaultMoveScale, CameraDefaultRotateScale);
		}

		if (!withTarget)
		{
			error = "Camera type \"" + type + "\" requires \"target\".";
			return false;
		}

		if (type == "Orbit")
			return finalize(make_smart_refctd_ptr<COrbitCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "Arcball")
			return finalize(make_smart_refctd_ptr<CArcballCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "Turntable")
			return finalize(make_smart_refctd_ptr<CTurntableCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "TopDown")
			return finalize(make_smart_refctd_ptr<CTopDownCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "Isometric")
			return finalize(make_smart_refctd_ptr<CIsometricCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "Chase")
			return finalize(make_smart_refctd_ptr<CChaseCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "Dolly")
			return finalize(make_smart_refctd_ptr<CDollyCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "Path")
			return finalize(make_smart_refctd_ptr<CPathCamera>(position, getTarget()), CameraOrbitMoveScale, CameraDefaultRotateScale);
		if (type == "DollyZoom")
		{
			float baseFov = 40.0f;
			if (jCamera.contains("baseFov"))
				baseFov = jCamera["baseFov"].get<float>();
			return finalize(make_smart_refctd_ptr<CDollyZoomCamera>(position, getTarget(), baseFov), CameraOrbitMoveScale, CameraDefaultRotateScale);
		}

		error = "Unsupported camera type \"" + type + "\".";
		return false;
	}
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

				auto configPath = [&]() -> std::filesystem::path
				{
					if (program.is_used("--file"))
					{
						std::filesystem::path path = program.get<std::string>("--file");
						if (path.is_relative())
							path = localInputCWD / path;
						return path.lexically_normal();
					}
					return (localInputCWD / "app_resources" / "cameras.json").lexically_normal();
				}();

				nbl_json j;
				{
					std::ifstream file(configPath);
					if (!file.is_open())
						return fail("Cannot open config \"" + configPath.string() + "\".");

					try
					{
						file >> j;
					}
					catch (const std::exception& e)
					{
						return fail("JSON parse error: " + std::string(e.what()));
					}
				}

				if (!j.contains("cameras") || !j["cameras"].is_array())
					return fail("Missing \"cameras\" array in config.");

				std::vector<smart_refctd_ptr<ICamera>> cameras;
				cameras.reserve(j["cameras"].size());
				for (const auto& jCamera : j["cameras"])
				{
					smart_refctd_ptr<ICamera> camera;
					std::string error;
					if (!createCameraFromJson(jCamera, error, camera))
						return fail(error);
					cameras.emplace_back(std::move(camera));
				}

				if (cameras.empty())
					return fail("No cameras defined.");

				auto angleDiffDeg = [](double a, double b) -> double
				{
					double d = std::fmod(a - b + 180.0, 360.0);
					if (d < 0.0)
						d += 360.0;
					return std::abs(d - 180.0);
				};

				auto isFinite3 = [](const auto& v) -> bool
				{
					return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
				};

				auto nearlyEqual3 = [](const auto& a, const auto& b, const double epsilon) -> bool
				{
					return std::abs(static_cast<double>(a.x - b.x)) <= epsilon &&
						std::abs(static_cast<double>(a.y - b.y)) <= epsilon &&
						std::abs(static_cast<double>(a.z - b.z)) <= epsilon;
				};

				auto computeDelta = [&](ICamera* camera, const float64_t3& beforePos, const float32_t3& beforeEulerDeg, double& outPosDelta, double& outRotDeltaDeg) -> bool
				{
					if (!camera)
						return false;
					const auto& gimbal = camera->getGimbal();
					const auto afterPos = gimbal.getPosition();
					const auto afterEuler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));
					if (!isFinite3(afterPos) || !isFinite3(afterEuler))
						return false;

					const double dx = static_cast<double>(afterPos.x - beforePos.x);
					const double dy = static_cast<double>(afterPos.y - beforePos.y);
					const double dz = static_cast<double>(afterPos.z - beforePos.z);
					outPosDelta = std::sqrt(dx * dx + dy * dy + dz * dz);
					outRotDeltaDeg = std::max({
						angleDiffDeg(afterEuler.x, beforeEulerDeg.x),
						angleDiffDeg(afterEuler.y, beforeEulerDeg.y),
						angleDiffDeg(afterEuler.z, beforeEulerDeg.z)
					});
					return true;
				};

				auto manipulateAndMeasure = [&](ICamera* camera, const std::vector<CVirtualGimbalEvent>& events, double& outPosDelta, double& outRotDeltaDeg) -> bool
				{
					outPosDelta = 0.0;
					outRotDeltaDeg = 0.0;
					if (!camera || events.empty())
						return false;

					const auto& beforeGimbal = camera->getGimbal();
					const float64_t3 beforePos = beforeGimbal.getPosition();
					const float32_t3 beforeEulerDeg = glm::degrees(glm::eulerAngles(beforeGimbal.getOrientation()));
					if (!isFinite3(beforePos) || !isFinite3(beforeEulerDeg))
						return false;

					if (!camera->manipulate({ events.data(), events.size() }))
						return false;

					if (!computeDelta(camera, beforePos, beforeEulerDeg, outPosDelta, outRotDeltaDeg))
						return false;

					return outPosDelta > 1e-9 || outRotDeltaDeg > 1e-9;
				};

				auto comparePresetToCamera = [&](ICamera* camera, const CameraPreset& preset, const double posEps, const double rotEpsDeg, const double scalarEps) -> bool
				{
					return comparePresetToCameraState(camera, preset, posEps, rotEpsDeg, scalarEps);
				};

				auto describePresetMismatch = [&](ICamera* camera, const CameraPreset& preset) -> std::string
				{
					return describePresetCameraMismatch(camera, preset);
				};

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
				CameraPreset initialChasePreset;
				CameraPreset initialDollyPreset;
				CameraPreset initialPathPreset;
				CameraPreset initialDollyZoomPreset;
				bool hasOrbitPreset = false;
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
					inputBinder.copyDefaultBindingsFromLayout(*camera);

					const auto initialPreset = capturePreset(camera, "smoke-initial");
					const auto initialCompatibility = analyzePresetCompatibility(camera, initialPreset);
					if (!initialCompatibility.exact || initialCompatibility.missingGoalStateMask != ICamera::GoalStateNone)
						return fail("Preset compatibility smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". missing=" + describeGoalStateMask(initialCompatibility.missingGoalStateMask));
					switch (camera->getKind())
					{
						case ICamera::CameraKind::Orbit:
							initialOrbitPreset = initialPreset;
							hasOrbitPreset = true;
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
					if (!applyPresetToCamera(camera, initialPreset))
						return fail("Preset no-op smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

					if (initialPreset.goal.hasTargetPosition)
					{
						CameraPreset shiftedPreset = initialPreset;
						shiftedPreset.goal.targetPosition += float64_t3(0.5, -0.25, 0.75);

						const auto shiftedResult = applyPresetToCameraDetailed(camera, shiftedPreset);
						if (!shiftedResult.succeeded() || !shiftedResult.changed() || !shiftedResult.exact)
							return fail("Preset target apply smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(shiftedResult));

						ICamera::SphericalTargetState shiftedState;
						if (!camera->tryGetSphericalTargetState(shiftedState) || !nearlyEqual3(shiftedState.target, shiftedPreset.goal.targetPosition, 1e-9))
							return fail("Preset target writeback smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

						const auto restoredResult = applyPresetToCameraDetailed(camera, initialPreset);
						if (!restoredResult.succeeded() || !restoredResult.exact)
							return fail("Preset restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(restoredResult));

						ICamera::SphericalTargetState restoredState;
						if (!camera->tryGetSphericalTargetState(restoredState) || !nearlyEqual3(restoredState.target, initialPreset.goal.targetPosition, 1e-9))
							return fail("Preset target restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

						if (!comparePresetToCamera(camera, initialPreset, 1e-6, 1e-4, 1e-9))
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

						const auto shiftedResult = applyPresetToCameraDetailed(camera, shiftedPreset);
						if (!shiftedResult.succeeded() || !shiftedResult.changed() || !comparePresetToCamera(camera, shiftedPreset, 1e-6, 0.1, 1e-6))
							return fail("Preset dynamic perspective apply smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describeApplyResult(shiftedResult) + " " + describePresetMismatch(camera, shiftedPreset));

						const auto restoredResult = applyPresetToCameraDetailed(camera, initialPreset);
						if (!restoredResult.succeeded() || !comparePresetToCamera(camera, initialPreset, 1e-6, 0.1, 1e-6))
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

					double directPosDelta = 0.0;
					double directRotDelta = 0.0;
					if (!manipulateAndMeasure(camera, directEvents, directPosDelta, directRotDelta))
						return fail("Direct manipulate smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");
					{
						const auto modifiedPreset = capturePreset(camera, "smoke-direct");
						const auto restoreInitial = applyPresetToCameraDetailed(camera, initialPreset);
						if (!restoreInitial.succeeded() || !comparePresetToCamera(camera, initialPreset, 1e-3, 0.1, 1e-4))
							return fail("Preset restore from direct smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, initialPreset));

						const auto applyModified = applyPresetToCameraDetailed(camera, modifiedPreset);
						if (!applyModified.succeeded() || !applyModified.changed() || !comparePresetToCamera(camera, modifiedPreset, 1e-3, 0.1, 1e-4))
							return fail("Preset apply from direct smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, modifiedPreset));

						const auto restoreAgain = applyPresetToCameraDetailed(camera, initialPreset);
						if (!restoreAgain.succeeded() || !comparePresetToCamera(camera, initialPreset, 1e-3, 0.1, 1e-4))
							return fail("Preset final restore smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\". " + describePresetMismatch(camera, initialPreset));
					}

					bool keyboardOk = false;
					double keyboardPosDelta = 0.0;
					double keyboardRotDelta = 0.0;
					for (const auto key : keyboardCandidates)
					{
						inputBinder.copyDefaultBindingsFromLayout(*camera);
						auto keyboardEvents = collectKeyboardVirtualEvents(inputBinder, key);
						if (keyboardEvents.empty())
							continue;
						if (manipulateAndMeasure(camera, keyboardEvents, keyboardPosDelta, keyboardRotDelta))
						{
							keyboardOk = true;
							break;
						}
					}
					if (!keyboardOk)
						return fail("Keyboard binding smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");

					const auto mousePreset = camera->getMouseMappingPreset();
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

					double mouseMovePosDelta = 0.0;
					double mouseMoveRotDelta = 0.0;
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

						inputBinder.copyDefaultBindingsFromLayout(*camera);
						auto mouseMoveEvents = collectMouseVirtualEvents(inputBinder, { filteredMoveLookDown.data(), filteredMoveLookDown.size() });
						if (mouseMoveEvents.empty())
							return fail("Mouse move virtual events missing for camera \"" + std::string(camera->getIdentifier()) + "\".");
						if (!manipulateAndMeasure(camera, mouseMoveEvents, mouseMovePosDelta, mouseMoveRotDelta))
							return fail("Mouse move binding smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");
					}

					double mouseScrollPosDelta = 0.0;
					double mouseScrollRotDelta = 0.0;
					if (hasScrollMapping)
					{
						SMouseEvent scrollEv(std::chrono::microseconds(16667));
						scrollEv.window = nullptr;
						scrollEv.type = ui::SMouseEvent::EET_SCROLL;
						scrollEv.scrollEvent.verticalScroll = 4;
						scrollEv.scrollEvent.horizontalScroll = 2;
						const std::array<SMouseEvent, 1u> rawScroll = { scrollEv };
						auto filteredScroll = filterOrbitMouseEvents(camera, rawScroll, false);

						inputBinder.copyDefaultBindingsFromLayout(*camera);
						auto mouseScrollEvents = collectMouseVirtualEvents(inputBinder, { filteredScroll.data(), filteredScroll.size() });
						if (mouseScrollEvents.empty())
							return fail("Mouse scroll virtual events missing for camera \"" + std::string(camera->getIdentifier()) + "\".");
						if (!manipulateAndMeasure(camera, mouseScrollEvents, mouseScrollPosDelta, mouseScrollRotDelta))
							return fail("Mouse scroll binding smoke failed for camera \"" + std::string(camera->getIdentifier()) + "\".");
					}

					std::cout << "[headless-camera-smoke][pass] " << camera->getIdentifier()
						<< " direct_pos_delta=" << directPosDelta
						<< " direct_rot_delta_deg=" << directRotDelta
						<< " kb_pos_delta=" << keyboardPosDelta
						<< " kb_rot_delta_deg=" << keyboardRotDelta
						<< " mouse_move_pos_delta=" << mouseMovePosDelta
						<< " mouse_move_rot_delta_deg=" << mouseMoveRotDelta
						<< " mouse_scroll_pos_delta=" << mouseScrollPosDelta
						<< " mouse_scroll_rot_delta_deg=" << mouseScrollRotDelta
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

					const auto baselinePreset = capturePreset(targetCamera, std::string(label) + "-baseline");
					const auto applyResult = applyPresetToCameraDetailed(targetCamera, sourcePreset);
					if (!applyResult.succeeded() || !applyResult.approximate() || !applyResult.hasIssue(expectedIssue))
						return fail(std::string("Cross-kind preset smoke failed for ") + label + ". " + describeApplyResult(applyResult));

					const auto restoreResult = applyPresetToCameraDetailed(targetCamera, baselinePreset);
					if (!restoreResult.succeeded() || !comparePresetToCamera(targetCamera, baselinePreset, 1e-6, 0.1, 1e-6))
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

					const auto baselinePreset = capturePreset(targetCamera, std::string(label) + "-baseline");
					const auto applyResult = applyPresetToCameraDetailed(targetCamera, sourcePreset);
					if (!applyResult.succeeded() || !applyResult.exact || !comparePresetToCamera(targetCamera, sourcePreset, 1e-6, 0.1, 1e-6))
					{
						return fail(std::string("Exact cross-kind preset smoke failed for ") + label + ". " +
							describeApplyResult(applyResult) + " " + describePresetMismatch(targetCamera, sourcePreset));
					}

					const auto restoreResult = applyPresetToCameraDetailed(targetCamera, baselinePreset);
					if (!restoreResult.succeeded() || !restoreResult.exact || !comparePresetToCamera(targetCamera, baselinePreset, 1e-6, 0.1, 1e-6))
					{
						return fail(std::string("Exact cross-kind preset restore smoke failed for ") + label + ". " +
							describeApplyResult(restoreResult) + " " + describePresetMismatch(targetCamera, baselinePreset));
					}

					return true;
				};

				ICamera* orbitCamera = findCameraByKind(ICamera::CameraKind::Orbit);
				ICamera* chaseCamera = findCameraByKind(ICamera::CameraKind::Chase);
				ICamera* dollyCamera = findCameraByKind(ICamera::CameraKind::Dolly);
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

			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			{
				smart_refctd_ptr<system::IFileArchive> examplesHeaderArch, examplesSourceArch, examplesBuildArch, thisExampleArch, thisExampleBuildArch;
#ifdef NBL_EMBED_BUILTIN_RESOURCES
				examplesHeaderArch = core::make_smart_refctd_ptr<nbl::builtin::examples::include::CArchive>(smart_refctd_ptr(m_logger));
				examplesSourceArch = core::make_smart_refctd_ptr<nbl::builtin::examples::src::CArchive>(smart_refctd_ptr(m_logger));
				examplesBuildArch = core::make_smart_refctd_ptr<nbl::builtin::examples::build::CArchive>(smart_refctd_ptr(m_logger));

	#ifdef _NBL_THIS_EXAMPLE_BUILTIN_C_ARCHIVE_H_
				thisExampleArch = make_smart_refctd_ptr<nbl::this_example::builtin::CArchive>(smart_refctd_ptr(m_logger));
	#endif

	#ifdef _NBL_THIS_EXAMPLE_BUILTIN_BUILD_C_ARCHIVE_H_
				thisExampleBuildArch = make_smart_refctd_ptr<nbl::this_example::builtin::build::CArchive>(smart_refctd_ptr(m_logger));
	#endif
#else
				examplesHeaderArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"../common/include/nbl/examples", smart_refctd_ptr(m_logger), m_system.get());
				examplesSourceArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"../common/src/nbl/examples", smart_refctd_ptr(m_logger), m_system.get());
				examplesBuildArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(NBL_EXAMPLES_BUILD_MOUNT_POINT, smart_refctd_ptr(m_logger), m_system.get());
				thisExampleArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"app_resources", smart_refctd_ptr(m_logger), m_system.get());
	#ifdef NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
				thisExampleBuildArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT, smart_refctd_ptr(m_logger), m_system.get());
	#endif
#endif
				m_system->mount(std::move(examplesHeaderArch),"nbl/examples");
				m_system->mount(std::move(examplesSourceArch),"nbl/examples");
				m_system->mount(std::move(examplesBuildArch),"nbl/examples");
				if (thisExampleArch)
					m_system->mount(std::move(thisExampleArch),"app_resources");
				if (thisExampleBuildArch)
					m_system->mount(std::move(thisExampleBuildArch),"app_resources");
			}

			{
				const std::optional<std::string> cameraJsonFile = program.is_used("--file") ? program.get<std::string>("--file") : std::optional<std::string>(std::nullopt);

				nbl_json j;
				auto loadDefaultConfig = [&]() -> bool
				{
#ifdef _NBL_THIS_EXAMPLE_BUILTIN_C_ARCHIVE_H_
					auto assets = make_smart_refctd_ptr<this_example::builtin::CArchive>(smart_refctd_ptr(m_logger));
					auto pFile = assets->getFile("cameras.json", IFile::ECF_READ, "");
					if (!pFile)
						return logFail("Could not open builtin cameras.json!");

					string config;
					IFile::success_t result;
					config.resize(pFile->getSize());
					pFile->read(result, config.data(), 0, pFile->getSize());
					j = nbl_json::parse(config);
					return true;
#else
					const auto fallbackPath = localInputCWD / "app_resources" / "cameras.json";
					std::ifstream fallbackFile(fallbackPath);
					if (!fallbackFile.is_open())
						return logFail("Cannot open default config \"%s\".", fallbackPath.string().c_str());
					fallbackFile >> j;
					return true;
#endif
				};

				auto file = cameraJsonFile.has_value() ? std::ifstream(cameraJsonFile.value()) : std::ifstream();
				if (!file.is_open())
				{
					if (cameraJsonFile.has_value())
						m_logger->log("Cannot open input \"%s\" json file. Switching to default config.", ILogger::ELL_WARNING, cameraJsonFile.value().c_str());
					else
						m_logger->log("No input json file provided. Switching to default config.", ILogger::ELL_INFO);

					if (!loadDefaultConfig())
						return false;
				}
				else
				{
					file >> j;
				}

				auto loadScriptJson = [&](const std::string& path, nbl_json& out) -> bool
				{
					std::ifstream sfile(path);
					if (!sfile.is_open())
					{
						m_logger->log("Cannot open scripted input file \"%s\".", ILogger::ELL_ERROR, path.c_str());
						return false;
					}
					sfile >> out;
					return true;
				};

				auto parseScriptedInput = [&](const nbl_json& script) -> void
				{
					m_scriptedInput.events.clear();
					m_scriptedInput.checks.clear();
					m_scriptedInput.captureFrames.clear();
					m_scriptedInput.nextEventIndex = 0;
					m_scriptedInput.nextCheckIndex = 0;
					m_scriptedInput.nextCaptureIndex = 0;
					m_scriptedInput.failed = false;
					m_scriptedInput.summaryReported = false;
					m_scriptedInput.baselineValid = false;
					m_scriptedInput.stepValid = false;
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

					if (script.contains("enabled"))
						m_scriptedInput.enabled = script["enabled"].get<bool>();
					else
						m_scriptedInput.enabled = true;

					if (script.contains("log"))
						m_scriptedInput.log = script["log"].get<bool>() || m_scriptedInput.log;

					if (script.contains("hard_fail"))
						m_scriptedInput.hardFail = script["hard_fail"].get<bool>();

					if (script.contains("visual_debug"))
						m_scriptedInput.visualDebug = script["visual_debug"].get<bool>();
					if (script.contains("visual_debug_target_fps"))
						m_scriptedInput.visualTargetFps = script["visual_debug_target_fps"].get<float>();
					if (script.contains("visual_debug_hold_seconds"))
						m_scriptedInput.visualCameraHoldSeconds = script["visual_debug_hold_seconds"].get<float>();
					if (m_scriptVisualDebugCli)
						m_scriptedInput.visualDebug = true;
					if (m_scriptedInput.visualDebug)
					{
						if (m_scriptedInput.visualTargetFps <= 0.f)
							m_scriptedInput.visualTargetFps = 60.f;
						if (m_scriptedInput.visualCameraHoldSeconds <= 0.f)
							m_scriptedInput.visualCameraHoldSeconds = 3.f;
					}

					if (script.contains("enableActiveCameraMovement"))
						enableActiveCameraMovement = script["enableActiveCameraMovement"].get<bool>();
					else if (m_scriptedInput.enabled)
						enableActiveCameraMovement = true;

					if (script.contains("exclusive_input"))
						m_scriptedInput.exclusive = script["exclusive_input"].get<bool>() || m_scriptedInput.exclusive;
					if (script.contains("exclusive"))
						m_scriptedInput.exclusive = script["exclusive"].get<bool>() || m_scriptedInput.exclusive;

					if (script.contains("capture_prefix"))
						m_scriptedInput.capturePrefix = script["capture_prefix"].get<std::string>();
					if (m_scriptedInput.capturePrefix.empty())
						m_scriptedInput.capturePrefix = "script";
					if (script.contains("capture_frames"))
						for (const auto& frame : script["capture_frames"])
							m_scriptedInput.captureFrames.emplace_back(frame.get<uint64_t>());

					if (script.contains("camera_controls"))
					{
						const auto& controls = script["camera_controls"];
						if (controls.contains("keyboard_scale"))
							m_cameraControls.keyboardScale = controls["keyboard_scale"].get<float>();
						if (controls.contains("mouse_move_scale"))
							m_cameraControls.mouseMoveScale = controls["mouse_move_scale"].get<float>();
						if (controls.contains("mouse_scroll_scale"))
							m_cameraControls.mouseScrollScale = controls["mouse_scroll_scale"].get<float>();
						if (controls.contains("translation_scale"))
							m_cameraControls.translationScale = controls["translation_scale"].get<float>();
						if (controls.contains("rotation_scale"))
							m_cameraControls.rotationScale = controls["rotation_scale"].get<float>();
					}

					if (script.contains("events"))
						for (const auto& ev : script["events"])
						{
						if (!ev.contains("frame") || !ev.contains("type"))
						{
							m_logger->log("Scripted input event missing \"frame\" or \"type\".", ILogger::ELL_WARNING);
							continue;
						}

						const auto frame = ev["frame"].get<uint64_t>();
						const auto type = ev["type"].get<std::string>();
						const bool captureFrame = ev.value("capture", false);

						if (type == "keyboard")
						{
							if (!ev.contains("key") || !ev.contains("action"))
							{
								m_logger->log("Scripted keyboard event missing \"key\" or \"action\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto keyStr = ev["key"].get<std::string>();
							const auto actionStr = ev["action"].get<std::string>();
							const auto key = ui::stringToKeyCode(keyStr);
							if (key == ui::EKC_NONE)
							{
								m_logger->log("Scripted keyboard event has invalid key \"%s\".", ILogger::ELL_WARNING, keyStr.c_str());
								continue;
							}

							ui::SKeyboardEvent::E_KEY_ACTION action = ui::SKeyboardEvent::ECA_UNITIALIZED;
							if (actionStr == "pressed" || actionStr == "press")
								action = ui::SKeyboardEvent::ECA_PRESSED;
							else if (actionStr == "released" || actionStr == "release")
								action = ui::SKeyboardEvent::ECA_RELEASED;

							if (action == ui::SKeyboardEvent::ECA_UNITIALIZED)
							{
								m_logger->log("Scripted keyboard event has invalid action \"%s\".", ILogger::ELL_WARNING, actionStr.c_str());
								continue;
							}

							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Keyboard;
							entry.keyboard.key = key;
							entry.keyboard.action = action;
							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else if (type == "mouse")
						{
							if (!ev.contains("kind"))
							{
								m_logger->log("Scripted mouse event missing \"kind\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto kind = ev["kind"].get<std::string>();
							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Mouse;

							if (kind == "move")
							{
								entry.mouse.type = ui::SMouseEvent::EET_MOVEMENT;
								entry.mouse.dx = ev.value("dx", 0);
								entry.mouse.dy = ev.value("dy", 0);
							}
							else if (kind == "scroll")
							{
								entry.mouse.type = ui::SMouseEvent::EET_SCROLL;
								entry.mouse.v = ev.value("v", 0);
								entry.mouse.h = ev.value("h", 0);
							}
							else if (kind == "click")
							{
								if (!ev.contains("button") || !ev.contains("action"))
								{
									m_logger->log("Scripted click event missing \"button\" or \"action\".", ILogger::ELL_WARNING);
									continue;
								}

								const auto buttonStr = ev["button"].get<std::string>();
								const auto actionStr = ev["action"].get<std::string>();

								ui::E_MOUSE_BUTTON button = ui::EMB_LEFT_BUTTON;
								if (buttonStr == "LEFT_BUTTON") button = ui::EMB_LEFT_BUTTON;
								else if (buttonStr == "RIGHT_BUTTON") button = ui::EMB_RIGHT_BUTTON;
								else if (buttonStr == "MIDDLE_BUTTON") button = ui::EMB_MIDDLE_BUTTON;
								else if (buttonStr == "BUTTON_4") button = ui::EMB_BUTTON_4;
								else if (buttonStr == "BUTTON_5") button = ui::EMB_BUTTON_5;
								else
								{
									m_logger->log("Scripted click event has invalid button \"%s\".", ILogger::ELL_WARNING, buttonStr.c_str());
									continue;
								}

								ui::SMouseEvent::SClickEvent::E_ACTION action = ui::SMouseEvent::SClickEvent::EA_UNITIALIZED;
								if (actionStr == "pressed" || actionStr == "press")
									action = ui::SMouseEvent::SClickEvent::EA_PRESSED;
								else if (actionStr == "released" || actionStr == "release")
									action = ui::SMouseEvent::SClickEvent::EA_RELEASED;

								if (action == ui::SMouseEvent::SClickEvent::EA_UNITIALIZED)
								{
									m_logger->log("Scripted click event has invalid action \"%s\".", ILogger::ELL_WARNING, actionStr.c_str());
									continue;
								}

								entry.mouse.type = ui::SMouseEvent::EET_CLICK;
								entry.mouse.button = button;
								entry.mouse.action = action;
								entry.mouse.x = ev.value("x", 0);
								entry.mouse.y = ev.value("y", 0);
							}
							else
							{
								m_logger->log("Scripted mouse event has invalid kind \"%s\".", ILogger::ELL_WARNING, kind.c_str());
								continue;
							}

							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else if (type == "imguizmo")
						{
							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Imguizmo;

							if (ev.contains("delta_trs"))
							{
								const auto arr = ev["delta_trs"].get<std::array<float, 16>>();
								float m16[16];
								for (size_t i = 0u; i < 16u; ++i)
									m16[i] = arr[i];
								entry.imguizmo = *reinterpret_cast<float32_t4x4*>(m16);
							}
							else
							{
								const auto t = ev.contains("translation") ? ev["translation"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
								const auto r = ev.contains("rotation_deg") ? ev["rotation_deg"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
								const auto s = ev.contains("scale") ? ev["scale"].get<std::array<float, 3>>() : std::array<float, 3>{1.f, 1.f, 1.f};

								float m16[16];
								float tr[3] = { t[0], t[1], t[2] };
								float rot[3] = { r[0], r[1], r[2] };
								float sc[3] = { s[0], s[1], s[2] };

								ImGuizmo::RecomposeMatrixFromComponents(tr, rot, sc, m16);
								entry.imguizmo = *reinterpret_cast<float32_t4x4*>(m16);
							}

							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else if (type == "action")
						{
							if (!ev.contains("action"))
							{
								m_logger->log("Scripted action event missing \"action\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto actionStr = ev["action"].get<std::string>();
							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Action;

							auto getValueInt = [&]() -> int32_t
							{
								if (ev.contains("value"))
									return ev["value"].get<int32_t>();
								if (ev.contains("index"))
									return ev["index"].get<int32_t>();
								return 0;
							};

							if (actionStr == "set_active_render_window")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow;
								entry.action.value = getValueInt();
							}
							else if (actionStr == "set_active_planar")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetActivePlanar;
								entry.action.value = getValueInt();
							}
							else if (actionStr == "set_projection_type")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetProjectionType;
								if (ev.contains("value") && ev["value"].is_string())
								{
									const auto valueStr = ev["value"].get<std::string>();
									if (valueStr == "perspective")
										entry.action.value = static_cast<int32_t>(IPlanarProjection::CProjection::Perspective);
									else if (valueStr == "orthographic")
										entry.action.value = static_cast<int32_t>(IPlanarProjection::CProjection::Orthographic);
									else
									{
										m_logger->log("Scripted action projection type has invalid value \"%s\".", ILogger::ELL_WARNING, valueStr.c_str());
										continue;
									}
								}
								else
								{
									entry.action.value = getValueInt();
								}
							}
							else if (actionStr == "set_projection_index")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetProjectionIndex;
								entry.action.value = getValueInt();
							}
							else if (actionStr == "set_use_window")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetUseWindow;
								entry.action.value = ev.value("value", false) ? 1 : 0;
							}
							else if (actionStr == "set_left_handed")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetLeftHanded;
								entry.action.value = ev.value("value", false) ? 1 : 0;
							}
							else if (actionStr == "reset_active_camera")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::ResetActiveCamera;
								entry.action.value = 1;
							}
							else
							{
								m_logger->log("Scripted action event has invalid action \"%s\".", ILogger::ELL_WARNING, actionStr.c_str());
								continue;
							}

							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else
						{
							m_logger->log("Scripted input event has invalid type \"%s\".", ILogger::ELL_WARNING, type.c_str());
						}
						}

					if (script.contains("checks"))
					{
						for (const auto& chk : script["checks"])
						{
							if (!chk.contains("frame") || !chk.contains("kind"))
							{
								m_logger->log("Scripted check missing \"frame\" or \"kind\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto frame = chk["frame"].get<uint64_t>();
							const auto kind = chk["kind"].get<std::string>();

							ScriptedInputCheck entry;
							entry.frame = frame;

							if (kind == "baseline")
							{
								entry.kind = ScriptedInputCheck::Kind::Baseline;
							}
							else if (kind == "imguizmo_virtual")
							{
								entry.kind = ScriptedInputCheck::Kind::ImguizmoVirtual;
								entry.tolerance = chk.value("tolerance", entry.tolerance);

								if (!chk.contains("events"))
								{
									m_logger->log("Imguizmo virtual check missing \"events\".", ILogger::ELL_WARNING);
									continue;
								}

								for (const auto& ev : chk["events"])
								{
									if (!ev.contains("type") || !ev.contains("magnitude"))
									{
										m_logger->log("Imguizmo virtual check event missing \"type\" or \"magnitude\".", ILogger::ELL_WARNING);
										continue;
									}

									const auto typeStr = ev["type"].get<std::string>();
									const auto type = CVirtualGimbalEvent::stringToVirtualEvent(typeStr);
									if (type == CVirtualGimbalEvent::None)
									{
										m_logger->log("Imguizmo virtual check event has invalid type \"%s\".", ILogger::ELL_WARNING, typeStr.c_str());
										continue;
									}

									ScriptedInputCheck::ExpectedVirtualEvent expected;
									expected.type = type;
									expected.magnitude = ev["magnitude"].get<double>();
									entry.expectedVirtualEvents.emplace_back(expected);
								}
							}
							else if (kind == "gimbal_near")
							{
								entry.kind = ScriptedInputCheck::Kind::GimbalNear;
								entry.posTolerance = chk.value("pos_tolerance", entry.posTolerance);
								entry.eulerToleranceDeg = chk.value("euler_tolerance_deg", entry.eulerToleranceDeg);

								if (chk.contains("position"))
								{
									const auto pos = chk["position"].get<std::array<float, 3>>();
									entry.expectedPos = float32_t3(pos[0], pos[1], pos[2]);
									entry.hasExpectedPos = true;
								}
								if (chk.contains("euler_deg"))
								{
									const auto euler = chk["euler_deg"].get<std::array<float, 3>>();
									entry.expectedEulerDeg = float32_t3(euler[0], euler[1], euler[2]);
									entry.hasExpectedEuler = true;
								}
							}
							else if (kind == "gimbal_delta")
							{
								entry.kind = ScriptedInputCheck::Kind::GimbalDelta;
								entry.posTolerance = chk.value("pos_tolerance", entry.posTolerance);
								entry.eulerToleranceDeg = chk.value("euler_tolerance_deg", entry.eulerToleranceDeg);
							}
							else if (kind == "gimbal_step")
							{
								entry.kind = ScriptedInputCheck::Kind::GimbalStep;

								if (chk.contains("min_pos_delta"))
								{
									entry.minPosDelta = chk["min_pos_delta"].get<float>();
									entry.hasPosDeltaConstraint = true;
								}
								if (chk.contains("max_pos_delta"))
								{
									entry.posTolerance = chk["max_pos_delta"].get<float>();
									entry.hasPosDeltaConstraint = true;
								}
								else if (chk.contains("pos_tolerance"))
								{
									entry.posTolerance = chk["pos_tolerance"].get<float>();
									entry.hasPosDeltaConstraint = true;
								}

								if (chk.contains("min_euler_delta_deg"))
								{
									entry.minEulerDeltaDeg = chk["min_euler_delta_deg"].get<float>();
									entry.hasEulerDeltaConstraint = true;
								}
								if (chk.contains("max_euler_delta_deg"))
								{
									entry.eulerToleranceDeg = chk["max_euler_delta_deg"].get<float>();
									entry.hasEulerDeltaConstraint = true;
								}
								else if (chk.contains("euler_tolerance_deg"))
								{
									entry.eulerToleranceDeg = chk["euler_tolerance_deg"].get<float>();
									entry.hasEulerDeltaConstraint = true;
								}

								if (!entry.hasPosDeltaConstraint && !entry.hasEulerDeltaConstraint)
								{
									m_logger->log("gimbal_step check requires at least one delta constraint.", ILogger::ELL_WARNING);
									continue;
								}
							}
							else
							{
								m_logger->log("Scripted check has invalid kind \"%s\".", ILogger::ELL_WARNING, kind.c_str());
								continue;
							}

							m_scriptedInput.checks.emplace_back(entry);
						}
					}

					std::sort(m_scriptedInput.events.begin(), m_scriptedInput.events.end(),
						[](const ScriptedInputEvent& a, const ScriptedInputEvent& b) { return a.frame < b.frame; });
					std::sort(m_scriptedInput.checks.begin(), m_scriptedInput.checks.end(),
						[](const ScriptedInputCheck& a, const ScriptedInputCheck& b) { return a.frame < b.frame; });
					if (!m_scriptedInput.captureFrames.empty())
					{
						std::sort(m_scriptedInput.captureFrames.begin(), m_scriptedInput.captureFrames.end());
						m_scriptedInput.captureFrames.erase(std::unique(m_scriptedInput.captureFrames.begin(), m_scriptedInput.captureFrames.end()), m_scriptedInput.captureFrames.end());
					}
					if (m_disableScreenshotsCli)
					{
						m_scriptedInput.captureFrames.clear();
						m_scriptedInput.nextCaptureIndex = 0;
					}
				};

				if (program.is_used("--script"))
				{
					system::path scriptPath = program.get<std::string>("--script");
					if (scriptPath.is_relative())
						scriptPath = localInputCWD / scriptPath;
					nbl_json scriptJson;
					if (!loadScriptJson(scriptPath.string(), scriptJson))
						return false;
					parseScriptedInput(scriptJson);
				}
				else if (j.contains("scripted_input"))
				{
					parseScriptedInput(j["scripted_input"]);
				}

				std::vector<smart_refctd_ptr<ICamera>> cameras;
				for (const auto& jCamera : j["cameras"])
				{
					if (jCamera.contains("type"))
					{
						if (!jCamera.contains("position"))
						{
							logFail("Expected \"position\" keyword for camera definition!");
							return false;
						}

						smart_refctd_ptr<ICamera> camera;
						std::string error;
						if (!createCameraFromJson(jCamera, error, camera))
						{
							logFail("%s", error.c_str());
							return false;
						}
						cameras.emplace_back(std::move(camera));
					}
					else
					{
						logFail("Expected \"type\" keyword for camera definition!");
						return false;
					}
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

				const char* bindingLayoutsKey = j.contains("bindings") ? "bindings" : (j.contains("controllers") ? "controllers" : nullptr);

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
							const char* viewportBindingsKey = viewport.contains("bindings") ? "bindings" : (viewport.contains("controllers") ? "controllers" : nullptr);
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
					m_initialPlanarPresets.emplace_back(capturePreset(camera, presetName));
				}
			}

			// Create asset manager
			m_assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));

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
			// TODO: improve the queue allocation/choice and allocate a dedicated presentation queue to improve responsiveness and race to present.
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
					constexpr std::array<size_t, 5> ImGuiStreamingBufferSizes = {
						32ull * 1024ull * 1024ull,
						16ull * 1024ull * 1024ull,
						8ull * 1024ull * 1024ull,
						4ull * 1024ull * 1024ull,
						2ull * 1024ull * 1024ull
					};
					auto createImGuiStreamingBuffer = [&](size_t size) -> smart_refctd_ptr<nbl::ext::imgui::UI::SCachedCreationParams::streaming_buffer_t>
					{
						constexpr uint32_t minStreamingBufferAllocationSize = 128u;
						constexpr uint32_t maxStreamingBufferAllocationAlignment = 4096u;

						auto getRequiredAccessFlags = [&](const bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>& properties)
						{
							bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> flags(IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);

							if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
								flags |= IDeviceMemoryAllocation::EMCAF_READ;
							if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
								flags |= IDeviceMemoryAllocation::EMCAF_WRITE;

							return flags;
						};

						IGPUBuffer::SCreationParams mdiCreationParams = {};
						mdiCreationParams.usage = nbl::ext::imgui::UI::SCachedCreationParams::RequiredUsageFlags;
						mdiCreationParams.size = size;

						auto buffer = m_utils->getLogicalDevice()->createBuffer(std::move(mdiCreationParams));
						if (!buffer)
						{
							m_logger->log("Failed to create ImGui streaming buffer object for size=%zu.", ILogger::ELL_WARNING, size);
							return nullptr;
						}

						buffer->setObjectDebugName("ImGui MDI Upstream Buffer");

						auto memoryReqs = buffer->getMemoryReqs();
						const auto upStreamingBits = m_utils->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
						const auto reqMemoryTypeBits = memoryReqs.memoryTypeBits;
						memoryReqs.memoryTypeBits &= upStreamingBits;
						if (!memoryReqs.memoryTypeBits)
						{
							m_logger->log("No compatible up-streaming memory type for ImGui buffer size=%zu reqBits=0x%08x upBits=0x%08x.", ILogger::ELL_WARNING, size, reqMemoryTypeBits, upStreamingBits);
							return nullptr;
						}

						auto allocation = m_utils->getLogicalDevice()->allocate(memoryReqs, buffer.get(), nbl::ext::imgui::UI::SCachedCreationParams::RequiredAllocateFlags);
						if (!allocation.isValid())
						{
							m_logger->log("Failed to allocate ImGui streaming buffer memory for size=%zu reqBits=0x%08x upBits=0x%08x filteredBits=0x%08x sizeReq=%llu.", ILogger::ELL_WARNING, size, reqMemoryTypeBits, upStreamingBits, memoryReqs.memoryTypeBits, memoryReqs.size);
							return nullptr;
						}

						auto memory = allocation.memory;

						if (!memory->map({ 0ull, memoryReqs.size }, getRequiredAccessFlags(memory->getMemoryPropertyFlags())))
						{
							m_logger->log("Could not map ImGui streaming buffer memory for size=%zu.", ILogger::ELL_WARNING, size);
							return nullptr;
						}

						return make_smart_refctd_ptr<nbl::ext::imgui::UI::SCachedCreationParams::streaming_buffer_t>(
							SBufferRange<IGPUBuffer>{0ull, mdiCreationParams.size, std::move(buffer)},
							maxStreamingBufferAllocationAlignment,
							minStreamingBufferAllocationSize);
					};

					smart_refctd_ptr<nbl::ext::imgui::UI::SCachedCreationParams::streaming_buffer_t> imguiStreamingBuffer = nullptr;
					for (const auto candidateSize : ImGuiStreamingBufferSizes)
					{
						imguiStreamingBuffer = createImGuiStreamingBuffer(candidateSize);
						if (imguiStreamingBuffer)
							break;
					}
					if (!imguiStreamingBuffer)
						return logFail("Failed to create ImGui streaming buffer.");

					nbl::ext::imgui::UI::SCreationParameters params;
					params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
					params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
					params.assetManager = m_assetManager;
					params.pipelineCache = nullptr;
					params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, TotalUISampleTexturesAmount);
					params.renderpass = smart_refctd_ptr<IGPURenderpass>(m_renderpass);
					params.streamingBuffer = std::move(imguiStreamingBuffer);
					params.subpassIx = 0u;
					params.transfer = getTransferUpQueue();
					params.utilities = m_utils;

					auto loadPrecompiledShader = [&](const std::string_view key) -> smart_refctd_ptr<IShader>
					{
						IAssetLoader::SAssetLoadParams loadParams = {};
						loadParams.logger = m_logger.get();
						loadParams.workingDirectory = "app_resources";
						auto bundle = m_assetManager->getAsset(key.data(), loadParams);
						const auto& contents = bundle.getContents();
						if (contents.empty())
							return nullptr;
						return IAsset::castDown<IShader>(contents[0]);
					};

					const auto vertexKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_vertex">(m_device.get());
					const auto fragmentKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_fragment">(m_device.get());
					auto vertexShader = loadPrecompiledShader(vertexKey.data());
					auto fragmentShader = loadPrecompiledShader(fragmentKey.data());
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
					constexpr std::string_view SpaceEnvBlobCandidates[] = {
						"rich_blue_nebulae_1_8k.rgba16f.envblob"
					};

					SpaceEnvBlobHeader envBlobHeader = {};
					std::vector<uint8_t> envBlobPayload;
					const std::array<path, 3u> SpaceEnvSearchRoots = {
						(localInputCWD / ".." / "media" / "envmap").lexically_normal(),
						(localInputCWD / ".." / "media").lexically_normal(),
						localInputCWD / "app_resources"
					};
					for (const auto candidate : SpaceEnvBlobCandidates)
					{
						for (const auto& root : SpaceEnvSearchRoots)
						{
							const auto candidatePath = root / candidate;
							if (loadSpaceEnvBlob(candidatePath, envBlobHeader, envBlobPayload))
							{
								break;
							}
						}
						if (!envBlobPayload.empty())
							break;
					}
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
					m_spaceEnvDescriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
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

					auto loadPrecompiledShader = [&](const std::string_view key) -> smart_refctd_ptr<IShader>
					{
						IAssetLoader::SAssetLoadParams loadParams = {};
						loadParams.logger = m_logger.get();
						loadParams.workingDirectory = "app_resources";
						auto bundle = m_assetManager->getAsset(key.data(), loadParams);
						const auto& contents = bundle.getContents();
						if (contents.empty())
							return nullptr;
						return IAsset::castDown<IShader>(contents[0]);
					};
					const auto spaceFragKey = nbl::this_example::builtin::build::get_spirv_key<"sky_env_fragment">(m_device.get());
					auto fragmentShader = loadPrecompiledShader(spaceFragKey.data());
					if (!fragmentShader)
						return logFail("Failed to load space environment fragment shader.");

					nbl::ext::FullScreenTriangle::ProtoPipeline fsTriProto(m_assetManager.get(), m_device.get(), m_logger.get());
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
				m_renderer = CSimpleDebugRenderer::create(m_assetManager.get(), m_sceneRenderpass.get(), 0, { &geometries.front().get(), geometries.size() });
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
					frustumParams.assetManager = m_assetManager;
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
					auto ix = 0u;
					for (const auto& name : m_scene->getInitParams().geometryNames)
					{
						if (name == "Cone")
							m_renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
						else if (name == "Grid")
							m_gridGeometryIx = ix;
						ix++;
					}
				}
				m_renderer->m_instances.resize(m_gridGeometryIx.has_value() ? 2u : 1u);

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


