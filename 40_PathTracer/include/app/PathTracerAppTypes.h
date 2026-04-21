// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_PATH_TRACER_APP_TYPES_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATH_TRACER_APP_TYPES_H_INCLUDED_

#include "io/CSceneLoader.h"
#include "io/PathTracerReport.h"

#include "nbl/core/decl/Types.h"
#include "nbl/system/path.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nbl::this_example
{

using nbl::system::path;

enum class SensorWorkflow : uint8_t
{
	RenderAllThenInteractive,
	RenderAllThenTerminate,
	RenderSensorThenInteractive,
	InteractiveAtSensor
};

struct AppArguments
{
	path scenePath = {};
	std::optional<path> sceneEntry = std::nullopt;
	path sceneListPath = {};
	SensorWorkflow workflow = SensorWorkflow::RenderAllThenInteractive;
	uint32_t sensor = 0u;
	bool headless = false;
	bool deferPostProcess = false;
	path outputDir = {};
	path reportDir = {};
	path referenceDir = {};
	PathTracerReport::SCompareSettings compare = {};
};

struct SceneJob
{
	path scenePath = {};
	std::optional<path> sceneEntry = std::nullopt;
	uint32_t sensor = 0u;
	bool deferPostProcess = false;
	PathTracerReport::SCompareSettings compare = {};
};

struct ExportOutputs
{
	path tonemap = {};
	path rwmcCascades = {};
	path albedo = {};
	path normal = {};
	path denoised = {};

	std::string artifactName = {};
	std::string displayName = {};
	std::vector<std::string> referenceNames = {};
	std::string tonemapDisplay = {};
	std::string rwmcCascadesDisplay = {};
	std::string albedoDisplay = {};
	std::string normalDisplay = {};
	std::string denoisedDisplay = {};
};

struct PostProcessJob
{
	ExportOutputs outputs = {};
	CSceneLoader::SLoadResult::SSensor::SDynamic::SPostProcess postProcess = {};
};

std::string trim(std::string_view input);
std::string stripWrappingQuotes(std::string_view input);
std::string toLowerCopy(std::string_view input);
bool isOptionToken(const std::string& token);
bool hasZipExtension(const path& filePath);
void splitSceneArchiveEntry(std::string& sceneValue, std::optional<std::string>& sceneEntry);
std::vector<std::string> tokenizeSceneListLine(std::string_view line);
std::optional<SensorWorkflow> parseSensorWorkflow(std::string_view rawValue);
path appendBeforeExtension(path input, std::string_view suffix);
std::string sceneNameFromOutputPath(const path& outputPath);
std::string sceneReferenceNameFromPath(const path& scenePath, const std::optional<path>& sceneEntry);
void addUniqueName(std::vector<std::string>& names, const std::string& name);
std::string makeArtifactName(size_t jobIndex, const std::string& baseName);
void replaceDeducedSceneOutputName(path& outputFile, const std::string& sourceSceneName);
std::string makeGenericPathString(const path& input);
std::vector<std::string> normalizeCompatibleArguments(const nbl::core::vector<std::string>& rawArguments);

}

#endif
