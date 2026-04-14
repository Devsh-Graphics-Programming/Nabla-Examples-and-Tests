// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "AppStub.hpp"

#include "AppBootstrap.hpp"
#include "AppInputParser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{

std::string jsonEscape(std::string_view input)
{
	std::string escaped;
	escaped.reserve(input.size());
	for (const char c : input)
	{
		switch (c)
		{
		case '\\': escaped += "\\\\"; break;
		case '"': escaped += "\\\""; break;
		case '\n': escaped += "\\n"; break;
		case '\r': escaped += "\\r"; break;
		case '\t': escaped += "\\t"; break;
		default: escaped += c; break;
		}
	}
	return escaped;
}

std::filesystem::path getManifestPath(const std::filesystem::path& outputPath)
{
	auto manifestPath = outputPath;
	manifestPath += ".stub.json";
	return manifestPath;
}

bool writeStubManifest(
	const std::filesystem::path& outputPath,
	const DenoiserTonemapperInvocation& invocation)
{
	const auto manifestPath = getManifestPath(outputPath);
	std::filesystem::create_directories(manifestPath.parent_path());

	std::ofstream manifest(manifestPath, std::ios::binary | std::ios::trunc);
	if (!manifest.is_open())
		return false;

	manifest << "{\n";
	manifest << "  \"mode\": \"stub\",\n";
	manifest << "  \"output\": \"" << jsonEscape(outputPath.string()) << "\",\n";
	manifest << "  \"color\": \"" << jsonEscape(invocation.colorFile) << "\",\n";
	manifest << "  \"albedo\": \"" << jsonEscape(invocation.albedoFile.value_or("")) << "\",\n";
	manifest << "  \"normal\": \"" << jsonEscape(invocation.normalFile.value_or("")) << "\",\n";
	manifest << "  \"color_channel\": \"" << jsonEscape(invocation.colorChannelName.value_or("")) << "\",\n";
	manifest << "  \"albedo_channel\": \"" << jsonEscape(invocation.albedoChannelName.value_or("")) << "\",\n";
	manifest << "  \"normal_channel\": \"" << jsonEscape(invocation.normalChannelName.value_or("")) << "\",\n";
	manifest << "  \"camera_transform\": \"" << jsonEscape(invocation.cameraTransform) << "\",\n";
	manifest << "  \"bloom_psf\": \"" << jsonEscape(invocation.bloomPsfFile) << "\",\n";
	manifest << "  \"denoiser_exposure_bias\": " << invocation.denoiserExposureBias << ",\n";
	manifest << "  \"denoiser_blend_factor\": " << invocation.denoiserBlendFactor << ",\n";
	manifest << "  \"bloom_relative_scale\": " << invocation.bloomRelativeScale << ",\n";
	manifest << "  \"bloom_intensity\": " << invocation.bloomIntensity << ",\n";
	manifest << "  \"tonemapper\": \"" << jsonEscape(invocation.tonemapper) << "\",\n";
	manifest << "  \"argv\": [";
	for (size_t i = 0u; i < invocation.rawArguments.size(); ++i)
	{
		if (i)
			manifest << ", ";
		manifest << "\"" << jsonEscape(invocation.rawArguments[i]) << "\"";
	}
	manifest << "]\n";
	manifest << "}\n";
	return true;
}

bool copyStubOutput(const std::filesystem::path& source, const std::filesystem::path& destination)
{
	std::filesystem::create_directories(destination.parent_path());
	std::error_code ec;
	std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing, ec);
	return !ec;
}

}

int runStubApp(int argc, char* argv[])
{
	const auto arguments = getInputArguments(argc, argv);
	std::string errorMessage;
	const auto invocations = parseDenoiserTonemapperInvocations(std::vector<std::string>(arguments.begin(), arguments.end()), errorMessage);
	if (invocations.empty())
	{
		std::cerr << "Stub denoiser could not parse invocation: " << errorMessage << std::endl;
		return 1;
	}

	for (const auto& invocation : invocations)
	{
		const auto colorPath = std::filesystem::path(invocation.colorFile);
		const auto outputPath = std::filesystem::path(invocation.output);
		if (!std::filesystem::exists(colorPath))
		{
			std::cerr << "Stub denoiser could not find COLOR_FILE at \"" << colorPath.string() << "\"" << std::endl;
			return 3;
		}

		if (!copyStubOutput(colorPath, outputPath))
		{
			std::cerr << "Stub denoiser could not create OUTPUT at \"" << outputPath.string() << "\"" << std::endl;
			return 4;
		}

		if (!writeStubManifest(outputPath, invocation))
		{
			std::cerr << "Stub denoiser could not create manifest for \"" << outputPath.string() << "\"" << std::endl;
			return 5;
		}

		std::cout << "Stub denoiser wrote \"" << outputPath.string() << "\" and \"" << getManifestPath(outputPath).string() << "\"" << std::endl;
	}

	return 0;
}
