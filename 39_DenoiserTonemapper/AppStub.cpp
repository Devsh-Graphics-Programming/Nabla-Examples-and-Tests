// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "AppStub.hpp"

#include "AppInputParser.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{

using nlohmann_json = nlohmann::json;

class DenoiserTonemapperStubRuntime final
{
	public:
		static std::filesystem::path getManifestPath(const std::filesystem::path& outputPath)
		{
			auto manifestPath = outputPath;
			manifestPath += ".stub.json";
			return manifestPath;
		}

		static bool writeManifest(const std::filesystem::path& outputPath, const DenoiserTonemapperInvocation& invocation)
		{
			const auto manifestPath = getManifestPath(outputPath);
			std::filesystem::create_directories(manifestPath.parent_path());

			std::ofstream manifestStream(manifestPath, std::ios::binary | std::ios::trunc);
			if (!manifestStream.is_open())
				return false;

			nlohmann_json manifest =
			{
				{"mode","stub"},
				{"output",outputPath.string()},
				{"color",invocation.colorFile},
				{"albedo",invocation.albedoFile.value_or("")},
				{"normal",invocation.normalFile.value_or("")},
				{"color_channel",invocation.colorChannelName.value_or("")},
				{"albedo_channel",invocation.albedoChannelName.value_or("")},
				{"normal_channel",invocation.normalChannelName.value_or("")},
				{"camera_transform",invocation.cameraTransform},
				{"bloom_psf",invocation.bloomPsfFile},
				{"denoiser_exposure_bias",invocation.denoiserExposureBias},
				{"denoiser_blend_factor",invocation.denoiserBlendFactor},
				{"bloom_relative_scale",invocation.bloomRelativeScale},
				{"bloom_intensity",invocation.bloomIntensity},
				{"tonemapper",invocation.tonemapper},
				{"argv",invocation.rawArguments}
			};

			manifestStream << std::setw(2) << manifest << '\n';
			return static_cast<bool>(manifestStream);
		}

		static bool copyOutput(const std::filesystem::path& source, const std::filesystem::path& destination)
		{
			std::filesystem::create_directories(destination.parent_path());
			std::error_code ec;
			std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing, ec);
			return !ec;
		}
};

}

int DenoiserTonemapperStubApp::run(int argc, char* argv[])
{
	const auto arguments = DenoiserTonemapperInputParser::collectInputArguments(argc, argv);
	std::string errorMessage;
	const auto invocations = DenoiserTonemapperInputParser::parseInvocations(
		std::vector<std::string>(arguments.begin(), arguments.end()),
		errorMessage);
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

		if (!DenoiserTonemapperStubRuntime::copyOutput(colorPath, outputPath))
		{
			std::cerr << "Stub denoiser could not create OUTPUT at \"" << outputPath.string() << "\"" << std::endl;
			return 4;
		}

		if (!DenoiserTonemapperStubRuntime::writeManifest(outputPath, invocation))
		{
			std::cerr << "Stub denoiser could not create manifest for \"" << outputPath.string() << "\"" << std::endl;
			return 5;
		}

		std::cout
			<< "Stub denoiser wrote \"" << outputPath.string() << "\" and \""
			<< DenoiserTonemapperStubRuntime::getManifestPath(outputPath).string()
			<< "\"" << std::endl;
	}

	return 0;
}
