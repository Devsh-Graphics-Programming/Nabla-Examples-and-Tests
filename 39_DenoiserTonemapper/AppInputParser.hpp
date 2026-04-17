// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _DENOISER_TONEMAPPER_APP_INPUT_PARSER_
#define _DENOISER_TONEMAPPER_APP_INPUT_PARSER_

#include "nabla.h"

#include <optional>
#include <string>
#include <vector>

struct DenoiserTonemapperInvocation
{
	std::vector<std::string> rawArguments = {};
	std::string colorFile = {};
	std::optional<std::string> albedoFile = {};
	std::optional<std::string> normalFile = {};
	std::optional<std::string> colorChannelName = {};
	std::optional<std::string> albedoChannelName = {};
	std::optional<std::string> normalChannelName = {};
	std::string cameraTransform = {};
	float denoiserExposureBias = 0.f;
	float denoiserBlendFactor = 0.f;
	std::string bloomPsfFile = {};
	float bloomRelativeScale = 0.f;
	float bloomIntensity = 0.f;
	std::string tonemapper = {};
	std::string output = {};
};

class DenoiserTonemapperInputParser final
{
	public:
		static constexpr const char* ProgramName = "denoisertonemapper";

		static nbl::core::vector<std::string> collectInputArguments(int argc, char* argv[]);
		static std::vector<DenoiserTonemapperInvocation> parseInvocations(
			const std::vector<std::string>& arguments,
			std::string& errorMessage);
};

#endif
