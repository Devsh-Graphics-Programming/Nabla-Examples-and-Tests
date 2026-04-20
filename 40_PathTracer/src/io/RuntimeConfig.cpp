// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "io/RuntimeConfig.h"

#include "nbl/system/path.h"

#include "nlohmann/json.hpp"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace nbl::this_example
{
namespace
{
using nbl::system::path;

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

bool endsWith(std::string_view input, std::string_view suffix)
{
	return input.size()>=suffix.size() && input.compare(input.size()-suffix.size(),suffix.size(),suffix)==0;
}

bool hasCliOption(const std::vector<std::string>& arguments, std::string_view option)
{
	const std::string optionWithValue = std::string(option)+"=";
	for (size_t i=1u; i<arguments.size(); ++i)
	{
		const auto& argument = arguments[i];
		if (argument==option || argument.rfind(optionWithValue,0u)==0u)
			return true;
	}
	return false;
}

std::string runtimeConfigNameFromExecutable(path executablePath)
{
	const auto stem = toLowerCopy(executablePath.stem().string());
	if (endsWith(stem,"_d"))
		return "debug";
	if (endsWith(stem,"_rwdi"))
		return "relwithdebinfo";
	return "release";
}

path getRuntimeConfigPath(const std::vector<std::string>& arguments)
{
	if (arguments.empty())
		return {};

	path executablePath = arguments.front();
	if (executablePath.is_relative())
		executablePath = (std::filesystem::current_path()/executablePath).lexically_normal();
	const auto configName = runtimeConfigNameFromExecutable(executablePath);
	return (executablePath.parent_path()/"config"/("pt."+configName+".json")).lexically_normal();
}

std::string normalizeConfigCliValue(const path& configPath, std::string_view option, std::string value)
{
	value = stripWrappingQuotes(value);
	if (option=="--output-dir" || option=="--report-dir" || option=="--reference-dir")
	{
		path outputPath = value;
		if (outputPath.is_relative())
			outputPath = (configPath.parent_path()/outputPath).lexically_normal();
		value = outputPath.generic_string();
	}
	return value;
}

}

bool applyRuntimeConfigDefaults(std::vector<std::string>& arguments)
{
	const auto configPath = getRuntimeConfigPath(arguments);
	if (configPath.empty() || !std::filesystem::exists(configPath))
		return true;

	std::ifstream configStream(configPath);
	if (!configStream)
	{
		std::fprintf(stderr,"Failed to open runtime config: %s\n",configPath.string().c_str());
		return false;
	}

	nlohmann::json config;
	try
	{
		config = nlohmann::json::parse(configStream);
	}
	catch (const std::exception& e)
	{
		std::fprintf(stderr,"Failed to parse runtime config %s: %s\n",configPath.string().c_str(),e.what());
		return false;
	}

	const auto cli = config.find("cli");
	if (cli==config.end())
		return true;
	if (!cli->is_object())
	{
		std::fprintf(stderr,"Runtime config %s has non-object cli section.\n",configPath.string().c_str());
		return false;
	}

	for (auto it=cli->begin(); it!=cli->end(); ++it)
	{
		const std::string option = it.key();
		if (option.empty() || option.front()!='-' || hasCliOption(arguments,option))
			continue;

		const auto& value = it.value();
		if (value.is_boolean())
		{
			if (value.get<bool>())
				arguments.push_back(option);
			continue;
		}

		std::string serializedValue;
		if (value.is_string())
			serializedValue = value.get<std::string>();
		else if (value.is_number_integer() || value.is_number_unsigned() || value.is_number_float())
			serializedValue = value.dump();
		else
		{
			std::fprintf(stderr,"Runtime config %s has unsupported value for %s.\n",configPath.string().c_str(),option.c_str());
			return false;
		}

		arguments.push_back(option);
		arguments.push_back(normalizeConfigCliValue(configPath,option,serializedValue));
	}

	return true;
}

}
