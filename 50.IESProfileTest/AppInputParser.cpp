// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "AppInputParser.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace scene;
using namespace nbl::examples;
using namespace nlohmann;

bool AppInputParser::parse(std::vector<std::string>& out, const std::string input, const std::string cwd)
{
    const auto jInputFile = std::filesystem::absolute(input);
    const auto sjInputFile = jInputFile.string();

    std::ifstream file(sjInputFile.c_str());
    if (!file.is_open()) {

        logger.log("Could not open \"%s\" file.", system::ILogger::ELL_ERROR, sjInputFile.c_str());
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const auto jsonBuffer = buffer.str();

    if (jsonBuffer.empty()) 
    {
        logger.log("\"%s\" file is empty!", system::ILogger::ELL_ERROR, sjInputFile.c_str());
        return false;
    }

    const auto jsonMap = json::parse(jsonBuffer.c_str());

    if (!jsonMap["directories"].is_array())
    {
        logger.log("\"%s\" file is empty!", system::ILogger::ELL_ERROR, sjInputFile.c_str());
        return false;
    }

    if (!jsonMap["files"].is_array())
    {
        logger.log("\"%s\" file's field \"files\" is not an array!", system::ILogger::ELL_ERROR, sjInputFile.c_str());
        return false;
    }

    if (!jsonMap["writeAssets"].is_boolean())
    {
        logger.log("\"%s\" file's field \"writeAssets\" is not a boolean!", system::ILogger::ELL_ERROR, sjInputFile.c_str());
        return false;
    }

    auto addFile = [&](const std::string_view in) -> bool
    {
        auto path = std::filesystem::absolute(cwd / std::filesystem::path(in));

        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && path.extension() == ".ies")
            out.push_back(path.string());
        else
        {
            logger.log("Invalid \"%s\" input!", system::ILogger::ELL_ERROR, path.string().c_str());
            return false;
        }

        return true;
    };

    auto addFiles = [&](const std::string_view directoryPath) -> bool
    {
        auto directory(std::filesystem::absolute(cwd / std::filesystem::path(directoryPath)));
        if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) 
        {
            logger.log("Invalid \"%s\" directory!", system::ILogger::ELL_ERROR, directory.string().c_str());
            return false;
        }

        for (const auto& entry : std::filesystem::directory_iterator(directory))
            if (!addFile(entry.path().string().c_str()))
                return false;

        return true;
    };

    // parse json
    {
        std::vector<std::string_view> jDirectories;
        jsonMap["directories"].get_to(jDirectories);

        for (const auto& it : jDirectories)
            if (!addFiles(it))
                return false;

        std::vector<std::string_view> jFiles;
        jsonMap["files"].get_to(jFiles);

        for (const auto& it : jFiles)
            if (!addFile(it))
                return false;
    }

    return true;
}