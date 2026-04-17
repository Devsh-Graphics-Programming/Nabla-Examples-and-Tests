// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "AppInputParser.hpp"

#include <argparse/argparse.hpp>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{

class DenoiserTonemapperInputParserUtils final
{
	public:
		static constexpr std::string_view BatchOptionShort = "-batch";
		static constexpr std::string_view BatchOptionLong = "--batch";

		static std::string trim(std::string_view input)
		{
			size_t begin = 0u;
			while (begin<input.size() && std::isspace(static_cast<unsigned char>(input[begin])))
				++begin;

			size_t end = input.size();
			while (end>begin && std::isspace(static_cast<unsigned char>(input[end-1u])))
				--end;

			return std::string(input.substr(begin,end-begin));
		}

		static std::string stripWrappingQuotes(std::string_view input)
		{
			auto stripped = trim(input);
			if (stripped.size()>=2u)
			{
				const char first = stripped.front();
				const char last = stripped.back();
				if ((first=='"' && last=='"') || (first=='\'' && last=='\''))
					stripped = stripped.substr(1u,stripped.size()-2u);
			}
			return stripped;
		}

		static bool isBatchOption(std::string_view token)
		{
			return token==BatchOptionShort || token==BatchOptionLong;
		}

		static std::optional<std::string_view> mapCompatibilityOption(std::string_view key)
		{
			if (key=="COLOR_FILE")
				return "--color-file";
			if (key=="ALBEDO_FILE")
				return "--albedo-file";
			if (key=="NORMAL_FILE")
				return "--normal-file";
			if (key=="COLOR_CHANNEL_NAME")
				return "--color-channel-name";
			if (key=="ALBEDO_CHANNEL_NAME")
				return "--albedo-channel-name";
			if (key=="NORMAL_CHANNEL_NAME")
				return "--normal-channel-name";
			if (key=="CAMERA_TRANSFORM")
				return "--camera-transform";
			if (key=="DENOISER_EXPOSURE_BIAS")
				return "--denoiser-exposure-bias";
			if (key=="DENOISER_BLEND_FACTOR")
				return "--denoiser-blend-factor";
			if (key=="BLOOM_PSF_FILE")
				return "--bloom-psf-file";
			if (key=="BLOOM_RELATIVE_SCALE")
				return "--bloom-relative-scale";
			if (key=="BLOOM_INTENSITY")
				return "--bloom-intensity";
			if (key=="TONEMAPPER")
				return "--tonemapper";
			if (key=="OUTPUT")
				return "--output";
			return std::nullopt;
		}

		static std::vector<std::string> splitBatchLine(const std::string& line, std::string& errorMessage)
		{
			std::vector<std::string> tokens;
			std::string current;
			bool insideQuotes = false;
			char quoteCharacter = '\0';

			for (size_t i = 0u; i < line.size(); ++i)
			{
				const char c = line[i];
				if (insideQuotes)
				{
					if (c==quoteCharacter)
					{
						insideQuotes = false;
						continue;
					}
					if (c=='\\' && i+1u<line.size() && line[i+1u]==quoteCharacter)
					{
						current.push_back(line[i+1u]);
						++i;
						continue;
					}
					current.push_back(c);
					continue;
				}

				if (std::isspace(static_cast<unsigned char>(c)))
				{
					if (!current.empty())
					{
						tokens.push_back(current);
						current.clear();
					}
					continue;
				}

				if (c=='"' || c=='\'')
				{
					insideQuotes = true;
					quoteCharacter = c;
					continue;
				}

				current.push_back(c);
			}

			if (insideQuotes)
			{
				errorMessage = "Batch input line contains an unmatched quote.";
				return {};
			}
			if (!current.empty())
				tokens.push_back(current);

			return tokens;
		}

		static std::vector<std::vector<std::string>> loadBatchInvocations(const std::filesystem::path& batchFile, std::string& errorMessage)
		{
			std::ifstream input(batchFile, std::ios::binary);
			if (!input.is_open())
			{
				errorMessage = "Could not open batch file \""+batchFile.string()+"\".";
				return {};
			}

			std::vector<std::vector<std::string>> invocations;
			std::string line;
			size_t lineNumber = 0u;
			while (std::getline(input,line))
			{
				++lineNumber;
				if (!line.empty() && line.back()=='\r')
					line.pop_back();

				if (trim(line).empty())
					continue;

				auto tokens = splitBatchLine(line,errorMessage);
				if (tokens.empty())
				{
					if (errorMessage.empty())
						continue;
					errorMessage = "Batch file \""+batchFile.string()+"\" line "+std::to_string(lineNumber)+": "+errorMessage;
					return {};
				}

				std::vector<std::string> invocation = {DenoiserTonemapperInputParser::ProgramName};
				invocation.insert(invocation.end(),tokens.begin(),tokens.end());
				invocations.push_back(std::move(invocation));
			}

			if (invocations.empty())
				errorMessage = "Batch file \""+batchFile.string()+"\" did not contain any invocations.";
			return invocations;
		}

		static std::vector<std::string> normalizeInvocationArguments(const std::vector<std::string>& rawArguments, std::string& errorMessage)
		{
			if (rawArguments.empty())
			{
				errorMessage = "Invocation argument list is empty.";
				return {};
			}

			std::vector<std::string> normalizedArguments = {rawArguments.front()};
			for (size_t i = 1u; i < rawArguments.size(); ++i)
			{
				const auto& token = rawArguments[i];
				if (token.empty())
					continue;

				if (token.rfind("--",0u)==0u)
				{
					const auto equalsPos = token.find('=');
					if (equalsPos!=std::string::npos)
					{
						normalizedArguments.push_back(token.substr(0u,equalsPos));
						normalizedArguments.push_back(stripWrappingQuotes(token.substr(equalsPos+1u)));
						continue;
					}

					if (i+1u>=rawArguments.size())
					{
						errorMessage = "Missing value for option \""+token+"\".";
						return {};
					}

					normalizedArguments.push_back(token);
					normalizedArguments.push_back(stripWrappingQuotes(rawArguments[++i]));
					continue;
				}

				if (token[0]=='-')
				{
					if (isBatchOption(token))
					{
						errorMessage = "\"-batch\" is only supported at the top level invocation.";
						return {};
					}

					const auto equalsPos = token.find('=');
					if (equalsPos==std::string::npos)
					{
						errorMessage = "Compatibility option \""+token+"\" must use the form -NAME=value.";
						return {};
					}

					const auto compatibilityName = token.substr(1u,equalsPos-1u);
					const auto normalizedOption = mapCompatibilityOption(compatibilityName);
					if (!normalizedOption.has_value())
					{
						errorMessage = "Unknown compatibility option \""+compatibilityName+"\".";
						return {};
					}

					normalizedArguments.emplace_back(normalizedOption.value());
					normalizedArguments.push_back(stripWrappingQuotes(token.substr(equalsPos+1u)));
					continue;
				}

				errorMessage = "Unexpected positional argument \""+token+"\".";
				return {};
			}

			return normalizedArguments;
		}

		static std::optional<DenoiserTonemapperInvocation> parseSingleInvocation(const std::vector<std::string>& rawArguments, std::string& errorMessage)
		{
			auto normalizedArguments = normalizeInvocationArguments(rawArguments,errorMessage);
			if (normalizedArguments.empty())
				return std::nullopt;

			argparse::ArgumentParser parser(DenoiserTonemapperInputParser::ProgramName);
			parser.add_description("Shared CLI parser for the denoiser tonemapper executable and stub mode.");
			parser.add_argument("--color-file")
				.required();
			parser.add_argument("--albedo-file");
			parser.add_argument("--normal-file");
			parser.add_argument("--color-channel-name");
			parser.add_argument("--albedo-channel-name");
			parser.add_argument("--normal-channel-name");
			parser.add_argument("--camera-transform")
				.required();
			parser.add_argument("--denoiser-exposure-bias")
				.required()
				.scan<'g',float>();
			parser.add_argument("--denoiser-blend-factor")
				.required()
				.scan<'g',float>();
			parser.add_argument("--bloom-psf-file")
				.required();
			parser.add_argument("--bloom-relative-scale")
				.required()
				.scan<'g',float>();
			parser.add_argument("--bloom-intensity")
				.required()
				.scan<'g',float>();
			parser.add_argument("--tonemapper")
				.required();
			parser.add_argument("--output")
				.required();

			try
			{
				parser.parse_args(normalizedArguments);
			}
			catch (const std::exception& e)
			{
				errorMessage = e.what();
				return std::nullopt;
			}

			DenoiserTonemapperInvocation invocation;
			invocation.rawArguments.assign(rawArguments.begin()+1u,rawArguments.end());
			invocation.colorFile = parser.get<std::string>("--color-file");
			if (const auto value = parser.present("--albedo-file"); value.has_value())
				invocation.albedoFile = value.value();
			if (const auto value = parser.present("--normal-file"); value.has_value())
				invocation.normalFile = value.value();
			if (const auto value = parser.present("--color-channel-name"); value.has_value())
				invocation.colorChannelName = value.value();
			if (const auto value = parser.present("--albedo-channel-name"); value.has_value())
				invocation.albedoChannelName = value.value();
			if (const auto value = parser.present("--normal-channel-name"); value.has_value())
				invocation.normalChannelName = value.value();
			invocation.cameraTransform = parser.get<std::string>("--camera-transform");
			invocation.denoiserExposureBias = parser.get<float>("--denoiser-exposure-bias");
			invocation.denoiserBlendFactor = parser.get<float>("--denoiser-blend-factor");
			invocation.bloomPsfFile = parser.get<std::string>("--bloom-psf-file");
			invocation.bloomRelativeScale = parser.get<float>("--bloom-relative-scale");
			invocation.bloomIntensity = parser.get<float>("--bloom-intensity");
			invocation.tonemapper = parser.get<std::string>("--tonemapper");
			invocation.output = parser.get<std::string>("--output");

			if (invocation.normalFile.has_value() && !invocation.albedoFile.has_value())
			{
				errorMessage = "NORMAL_FILE requires ALBEDO_FILE.";
				return std::nullopt;
			}
			if (invocation.albedoChannelName.has_value() && !invocation.albedoFile.has_value())
			{
				errorMessage = "ALBEDO_CHANNEL_NAME requires ALBEDO_FILE.";
				return std::nullopt;
			}
			if (invocation.normalChannelName.has_value() && !invocation.normalFile.has_value())
			{
				errorMessage = "NORMAL_CHANNEL_NAME requires NORMAL_FILE.";
				return std::nullopt;
			}

			return invocation;
		}

		static std::optional<std::filesystem::path> tryGetInlineBatchPath(std::string_view token)
		{
			for (const auto option : {BatchOptionShort,BatchOptionLong})
			{
				const std::string prefix = std::string(option)+"=";
				if (token.rfind(prefix,0u)==0u)
					return std::filesystem::path(stripWrappingQuotes(token.substr(prefix.size())));
			}
			return std::nullopt;
		}
};

}

nbl::core::vector<std::string> DenoiserTonemapperInputParser::collectInputArguments(int argc, char* argv[])
{
	nbl::core::vector<std::string> arguments;
	arguments.reserve(argc > 0 ? argc : 1);
	if (argc>0 && argv && argv[0])
		arguments.emplace_back(argv[0]);
	else
		arguments.emplace_back(ProgramName);

	if (argc>1)
	{
		std::cout << "Guess input from Commandline arguments" << std::endl;
		for (auto i = 1; i < argc; ++i)
			arguments.emplace_back(argv[i]);
	}
	else
	{
		std::cout << "No arguments provided, running demo mode from ../exampleInputArguments.txt" << std::endl;
		arguments.emplace_back("-batch");
		arguments.emplace_back("../exampleInputArguments.txt");
	}

	return arguments;
}

std::vector<DenoiserTonemapperInvocation> DenoiserTonemapperInputParser::parseInvocations(
	const std::vector<std::string>& arguments,
	std::string& errorMessage)
{
	errorMessage.clear();
	if (arguments.empty())
	{
		errorMessage = "No arguments were provided.";
		return {};
	}
	if (arguments.size()==1u)
	{
		errorMessage = "No denoiser invocation arguments were provided.";
		return {};
	}

	std::vector<std::vector<std::string>> rawInvocations;
	if (DenoiserTonemapperInputParserUtils::isBatchOption(arguments[1]))
	{
		if (arguments.size()!=3u)
		{
			errorMessage = "Batch mode expects exactly one path argument.";
			return {};
		}
		rawInvocations = DenoiserTonemapperInputParserUtils::loadBatchInvocations(
			DenoiserTonemapperInputParserUtils::stripWrappingQuotes(arguments[2]),
			errorMessage);
	}
	else if (const auto batchFile = DenoiserTonemapperInputParserUtils::tryGetInlineBatchPath(arguments[1]); batchFile.has_value())
	{
		if (arguments.size()!=2u)
		{
			errorMessage = "Inline batch mode expects exactly one batch file.";
			return {};
		}
		rawInvocations = DenoiserTonemapperInputParserUtils::loadBatchInvocations(batchFile.value(),errorMessage);
	}
	else
	{
		rawInvocations.push_back(arguments);
	}

	if (rawInvocations.empty())
		return {};

	std::vector<DenoiserTonemapperInvocation> invocations;
	invocations.reserve(rawInvocations.size());
	for (size_t i = 0u; i < rawInvocations.size(); ++i)
	{
		auto invocation = DenoiserTonemapperInputParserUtils::parseSingleInvocation(rawInvocations[i],errorMessage);
		if (!invocation.has_value())
		{
			if (rawInvocations.size()>1u)
				errorMessage = "Invocation "+std::to_string(i)+" failed: "+errorMessage;
			return {};
		}
		invocations.push_back(std::move(invocation.value()));
	}

	return invocations;
}
