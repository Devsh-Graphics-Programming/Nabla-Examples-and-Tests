#include "app/App.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <span>
#include <string_view>
#include <vector>
#include "app/AppCameraConfigUtilities.hpp"
#include "app/AppResourceUtilities.hpp"
#include "camera/CCameraPathUtilities.hpp"
#include "camera/CCameraPersistence.hpp"
#include "camera/CCameraScriptedRuntimePersistence.hpp"
#include "camera/CCameraSmokeRegressionUtilities.hpp"
#include "nlohmann/json.hpp"
#include "AppHeadlessCameraSmokeHelpers.inl"
#include "AppHeadlessCameraSmokeChecks.inl"

bool App::runHeadlessCameraSmoke(argparse::ArgumentParser& program, smart_refctd_ptr<ISystem>&& system)
{
	auto fail = [&](const std::string& msg) -> bool
	{
		m_cliRuntime.headlessCameraSmokePassed = false;
		return reportHeadlessCameraSmokeFailure(*this, msg);
	};
	const auto runSmokeStep = [&](auto&& fn) -> bool
	{
		std::string smokeError;
		if (fn(smokeError))
			return true;
		return fail(smokeError);
	};

	if (!initializeMountedCameraResources(std::move(system)))
		return fail("Failed to initialize mounted resources for headless camera smoke.");

	nbl::system::SCameraPlanarRuntimeBootstrap runtimeBootstrap = {};
	std::string jsonError;
	if (!nbl::system::tryBuildCameraPlanarRuntimeBootstrap(
			getCameraAppResourceContext(),
			{
				.requestedPath = program.is_used("--file") ? std::optional<nbl::system::path>(program.get<std::string>("--file")) : std::optional<nbl::system::path>(std::nullopt),
				.fallbackToDefault = false
			},
			runtimeBootstrap,
			&jsonError))
	{
		return fail(jsonError);
	}

	auto& cameraCollections = runtimeBootstrap.collections;
	auto& cameras = cameraCollections.cameras;
	auto& smokePlanars = runtimeBootstrap.planars;

	if (!runSmokeStep([&](std::string& smokeError) { return verifyScriptedRuntimeFrameBatch(&smokeError); }))
		return false;
	if (!runSmokeStep([&](std::string& smokeError) { return verifyScriptedRuntimeParser(&smokeError); }))
		return false;
	if (!runSmokeStep([&](std::string& smokeError) { return verifyScriptedCheckRunner(m_cameraGoalSolver, &smokeError); }))
		return false;

	SCameraSmokePresetInventory initialPresets = {};
	if (!runSmokeStep([&](std::string& smokeError)
	{
		return runPerCameraPresetAndBindingSmoke(m_cameraGoalSolver, { cameras.data(), cameras.size() }, initialPresets, smokeError);
	}))
	{
		return false;
	}

	const auto cameraInventory = collectSmokeCameras({ cameras.data(), cameras.size() });
	const SCameraSmokeResolvedState resolvedSmokeState = {
		.goalSolver = m_cameraGoalSolver,
		.system = getCameraAppResourceContext().system,
		.initialPresets = initialPresets,
		.orbitCamera = cameraInventory.orbit,
		.freeCamera = cameraInventory.free,
		.chaseCamera = cameraInventory.chase,
		.dollyCamera = cameraInventory.dolly,
		.dollyZoomCamera = cameraInventory.dollyZoom
	};

	if (!runSmokeStep([&](std::string& smokeError)
	{
		return verifyFollowSmoke(
			resolvedSmokeState,
			{ cameras.data(), cameras.size() },
			{ smokePlanars.data(), smokePlanars.size() },
			[](ICamera* camera) { return nbl::core::makeDefaultFollowConfig(camera); },
			[](const CTrackedTarget& trackedTarget, const std::string_view label, std::string& error)
			{
				return verifyFollowTargetMarkerAlignmentForSmoke(trackedTarget, label, error);
			},
			[this, smokePlanarsSpan = std::span<const smart_refctd_ptr<planar_projection_t>>(smokePlanars.data(), smokePlanars.size())](ICamera* camera, const CTrackedTarget& trackedTarget, const std::string_view label, std::string& error)
			{
				return verifyOffsetFollowRecaptureForSmoke(m_cameraGoalSolver, smokePlanarsSpan, camera, trackedTarget, label, error);
			},
			smokeError);
	}))
	{
		return false;
	}

	if (!runSmokeStep([&](std::string& smokeError) { return verifyCrossKindAndPresentationSmoke(resolvedSmokeState, smokeError); }))
		return false;
	if (!runSmokeStep([&](std::string& smokeError) { return verifyPersistenceAndPlaybackSmoke(resolvedSmokeState, smokeError); }))
		return false;
	if (!runSmokeStep([&](std::string& smokeError) { return verifySequenceCompileSmoke(resolvedSmokeState, smokeError); }))
		return false;
	if (!runSmokeStep([&](std::string& smokeError) { return verifyRangeAndUtilitySmoke(resolvedSmokeState, smokeError); }))
		return false;

	m_cliRuntime.headlessCameraSmokePassed = true;
	std::cout << "[headless-camera-smoke] PASS cameras=" << cameras.size() << std::endl;
	return true;
}
