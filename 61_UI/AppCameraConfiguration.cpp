#include "app/App.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/AppCameraConfigUtilities.hpp"
#include "app/AppResourceUtilities.hpp"
#include "app/AppViewportBindingUtilities.hpp"
#include "camera/CCameraPersistence.hpp"
#include "camera/CCameraScriptedRuntimePersistence.hpp"

bool App::initializeCameraConfiguration(const argparse::ArgumentParser& program)
{
	nbl::system::SCameraPlanarRuntimeBootstrap runtimeBootstrap = {};
	std::optional<CCameraSequenceScript> pendingScriptedSequence;
	if (!tryBuildCameraConfigurationBootstrap(program, runtimeBootstrap, pendingScriptedSequence))
		return false;

	return initializePlanarRuntimeState(runtimeBootstrap, pendingScriptedSequence);
}

bool App::tryBuildCameraConfigurationBootstrap(
	const argparse::ArgumentParser& program,
	nbl::system::SCameraPlanarRuntimeBootstrap& outRuntimeBootstrap,
	std::optional<CCameraSequenceScript>& outPendingScriptedSequence)
{
	const std::optional<nbl::system::path> cameraJsonFile =
		program.is_used("--file") ?
		std::optional<nbl::system::path>(program.get<std::string>("--file")) :
		std::optional<nbl::system::path>(std::nullopt);

	std::string jsonError;
	if (!nbl::system::tryBuildCameraPlanarRuntimeBootstrap(
			getCameraAppResourceContext(),
			{
				.requestedPath = cameraJsonFile,
				.fallbackToDefault = true
			},
			outRuntimeBootstrap,
			&jsonError))
		return logFail("%s", jsonError.c_str());
	auto& cameraConfig = outRuntimeBootstrap.loadResult;
	auto& cameraCollections = outRuntimeBootstrap.collections;

	const bool hasUserConfig = cameraJsonFile.has_value();
	if (cameraConfig.usedDefaultConfig())
	{
		if (hasUserConfig)
			m_logger->log("Cannot open input \"%s\" json file (%s). Switching to default config.", ILogger::ELL_WARNING, cameraJsonFile.value().string().c_str(), cameraConfig.requestedPathError.c_str());
		else
			m_logger->log("No input json file provided. Switching to default config.", ILogger::ELL_INFO);
	}

	outPendingScriptedSequence.reset();
	if (!tryLoadConfiguredScriptedInput(program, cameraCollections, outPendingScriptedSequence))
		return false;
	return true;
}

bool App::initializePlanarRuntimeState(
	const nbl::system::SCameraPlanarRuntimeBootstrap& runtimeBootstrap,
	const std::optional<CCameraSequenceScript>& pendingScriptedSequence)
{
	m_planarProjections = runtimeBootstrap.planars;

	if (!nbl::ui::initializeWindowBindingDefaults(
			getPlanarProjectionSpan(),
			std::span<SWindowControlBinding>(m_viewports.windowBindings.data(), m_viewports.windowBindings.size())))
	{
		return logFail("Failed to initialize default viewport bindings.");
	}

	std::string cameraConfigError;
	if (!nbl::system::tryCaptureInitialPlanarPresets(
			m_cameraGoalSolver,
			getPlanarProjectionSpan(),
			m_presetAuthoring.initialPlanarPresets,
			cameraConfigError))
	{
		return logFail("%s", cameraConfigError.c_str());
	}

	initializePlanarFollowConfigs();
	bindManipulatedModel();

	if (pendingScriptedSequence.has_value() && !expandPendingScriptedSequence(*pendingScriptedSequence))
		return false;

	return true;
}

SCameraFollowConfig App::makeExampleDefaultFollowConfig(const ICamera* const camera) const
{
	auto config = nbl::core::CCameraFollowUtilities::makeDefaultFollowConfig(camera);
	if (!camera)
		return config;

	switch (camera->getKind())
	{
		case ICamera::CameraKind::Free:
			config.enabled = true;
			config.mode = ECameraFollowMode::LookAtTarget;
			break;
		default:
			break;
	}

	return config;
}

void App::initializePlanarFollowConfigs()
{
	resetFollowTargetToDefault();
	m_sceneInteraction.planarFollowConfigs.clear();
	m_sceneInteraction.planarFollowConfigs.reserve(m_planarProjections.size());
	for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
	{
		auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
		auto config = makeExampleDefaultFollowConfig(camera);
		m_sceneInteraction.planarFollowConfigs.emplace_back(config);
		if (config.enabled)
			captureFollowOffsetsForPlanar(planarIx);
	}
}
