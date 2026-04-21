// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "app/PathTracerAppTypes.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace nbl::this_example
{

std::string trim(std::string_view input)
{
	size_t begin = 0u;
	while (begin<input.size() && std::isspace(static_cast<unsigned char>(input[begin])))
		++begin;

	size_t end = input.size();
	while (end>begin && std::isspace(static_cast<unsigned char>(input[end-1u])))
		--end;

	return std::string(input.substr(begin,end-begin));
}

std::string stripWrappingQuotes(std::string_view input)
{
	auto stripped = trim(input);
	if (stripped.size()>=2u)
	{
		const char first = stripped.front();
		const char last = stripped.back();
		if ((first=='\"' && last=='\"') || (first=='\'' && last=='\''))
			stripped = stripped.substr(1u,stripped.size()-2u);
	}
	return stripped;
}

std::string toLowerCopy(std::string_view input)
{
	std::string lowered(input);
	for (char& c : lowered)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return lowered;
}

bool isOptionToken(const std::string& token)
{
	return !token.empty() && token.front()=='-';
}

bool hasZipExtension(const path& filePath)
{
	return toLowerCopy(filePath.extension().string())==".zip";
}

void splitSceneArchiveEntry(std::string& sceneValue, std::optional<std::string>& sceneEntry)
{
	if (const auto lowered = toLowerCopy(sceneValue); lowered.find(".zip")!=std::string::npos)
	{
		const auto zipPos = lowered.find(".zip");
		const size_t entryPos = sceneValue.find_first_not_of(" \t",zipPos+4u);
		if (entryPos!=std::string::npos)
		{
			sceneEntry = trim(sceneValue.substr(entryPos));
			sceneValue = trim(sceneValue.substr(0u,zipPos+4u));
		}
	}
}

std::vector<std::string> tokenizeSceneListLine(std::string_view line)
{
	std::vector<std::string> tokens;
	std::string token;
	bool quoted = false;
	char quote = 0;

	for (const char c : line)
	{
		if (quoted)
		{
			if (c==quote)
				quoted = false;
			else
				token += c;
			continue;
		}

		if (c=='\"' || c=='\'')
		{
			quoted = true;
			quote = c;
			continue;
		}

		if (std::isspace(static_cast<unsigned char>(c)))
		{
			if (!token.empty())
			{
				tokens.push_back(std::move(token));
				token.clear();
			}
			continue;
		}

		token += c;
	}

	if (!token.empty())
		tokens.push_back(std::move(token));
	return tokens;
}

std::optional<SensorWorkflow> parseSensorWorkflow(std::string_view rawValue)
{
	const auto lowered = toLowerCopy(rawValue);
	if (lowered=="renderalltheninteractive" || lowered=="render-all-then-interactive" || lowered=="all-interactive")
		return SensorWorkflow::RenderAllThenInteractive;
	if (lowered=="renderallthenterminate" || lowered=="render-all-then-terminate" || lowered=="all-terminate")
		return SensorWorkflow::RenderAllThenTerminate;
	if (lowered=="rendersensortheninteractive" || lowered=="render-sensor-then-interactive" || lowered=="sensor-interactive")
		return SensorWorkflow::RenderSensorThenInteractive;
	if (lowered=="interactiveatsensor" || lowered=="interactive-at-sensor" || lowered=="interactive")
		return SensorWorkflow::InteractiveAtSensor;
	return std::nullopt;
}

path appendBeforeExtension(path input, std::string_view suffix)
{
	auto stem = input.stem().string();
	stem += suffix;
	input.replace_filename(stem + input.extension().string());
	return input;
}

std::string sceneNameFromOutputPath(const path& outputPath)
{
	auto name = outputPath.stem().string();
	if (name.rfind("Render_",0u)==0u)
		name.erase(0u,std::char_traits<char>::length("Render_"));
	return name.empty() ? "scene":name;
}

std::string sceneReferenceNameFromPath(const path& scenePath, const std::optional<path>& sceneEntry)
{
	std::string name;
	if (hasZipExtension(scenePath))
	{
		name = scenePath.stem().string();
		const auto entry = sceneEntry.value_or(path("scene.xml"));
		if (!entry.empty())
			name += "_" + entry.stem().string();
	}
	else
	{
		name = scenePath.stem().string();
		if (name=="scene" && scenePath.has_parent_path())
			name = scenePath.parent_path().filename().string()+"_"+name;
	}
	return name.empty() ? "scene":name;
}

void addUniqueName(std::vector<std::string>& names, const std::string& name)
{
	if (!name.empty() && std::find(names.begin(),names.end(),name)==names.end())
		names.push_back(name);
}

std::string makeArtifactName(const size_t jobIndex, const std::string& baseName)
{
	std::ostringstream out;
	out << std::setw(2) << std::setfill('0') << jobIndex << "_" << baseName;
	const auto name = out.str();
	return name.empty() ? "scene":name;
}

void replaceDeducedSceneOutputName(path& outputFile, const std::string& sourceSceneName)
{
	if (sourceSceneName.empty())
		return;

	auto stem = outputFile.stem().string();
	const bool hasRenderPrefix = stem.rfind("Render_",0u)==0u;
	auto baseName = hasRenderPrefix ? stem.substr(std::char_traits<char>::length("Render_")):stem;
	if (baseName!="scene" && baseName.rfind("scene_Sensor_",0u)!=0u)
		return;

	const auto suffix = baseName=="scene" ? std::string{}:baseName.substr(std::char_traits<char>::length("scene"));
	const auto newStem = std::string(hasRenderPrefix ? "Render_":"")+sourceSceneName+suffix;
	outputFile.replace_filename(newStem+outputFile.extension().string());
}

std::string makeGenericPathString(const path& input)
{
	return input.generic_string();
}

std::vector<std::string> normalizeCompatibleArguments(const nbl::core::vector<std::string>& rawArguments)
{
	if (rawArguments.empty())
		return {};

	std::vector<std::string> normalized;
	normalized.reserve(rawArguments.size()*2u);
	normalized.push_back(rawArguments.front());

	for (size_t i = 1u; i < rawArguments.size(); ++i)
	{
		const auto& token = rawArguments[i];

		if (token.rfind("-SCENE=",0u)==0u)
		{
			auto sceneValue = stripWrappingQuotes(token.substr(7u));
			std::optional<std::string> sceneEntry = std::nullopt;
			splitSceneArchiveEntry(sceneValue,sceneEntry);
			if (sceneEntry==std::nullopt && hasZipExtension(path(sceneValue)) && i+1u<rawArguments.size() && !isOptionToken(rawArguments[i+1u]))
				sceneEntry = stripWrappingQuotes(rawArguments[++i]);

			normalized.push_back("--scene");
			normalized.push_back(sceneValue);
			if (sceneEntry.has_value())
			{
				normalized.push_back("--scene-entry");
				normalized.push_back(sceneEntry.value());
			}
			continue;
		}

		if (token=="-PROCESS_SENSORS" || token.rfind("-PROCESS_SENSORS=",0u)==0u)
		{
			std::string workflow;
			if (token=="-PROCESS_SENSORS")
			{
				if (i+1u>=rawArguments.size())
					continue;
				workflow = stripWrappingQuotes(rawArguments[++i]);
			}
			else
			{
				workflow = stripWrappingQuotes(token.substr(std::char_traits<char>::length("-PROCESS_SENSORS=")));
			}

			normalized.push_back("--process-sensors");
			normalized.push_back(workflow);
			if (i+1u<rawArguments.size() && !isOptionToken(rawArguments[i+1u]))
			{
				normalized.push_back("--sensor");
				normalized.push_back(stripWrappingQuotes(rawArguments[++i]));
			}
			continue;
		}

		if (token=="-DEFER_DENOISE")
		{
			normalized.push_back("--defer-denoise");
			continue;
		}

		if (token=="-TERMINATE")
		{
			normalized.push_back("--process-sensors");
			normalized.push_back("RenderAllThenTerminate");
			continue;
		}

		normalized.push_back(token);
	}

	return normalized;
}

}
