// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_THIS_EXAMPLE_APP_RESOURCE_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_RESOURCE_UTILITIES_HPP_

#include <array>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"
#include "nlohmann/json.hpp"

namespace nbl::system
{

using camera_json_t = nlohmann::json;

struct SSpaceEnvBlobHeader final
{
    uint32_t magic = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t format = 0u;
    uint64_t payloadSize = 0ull;
};

struct SCameraAppResourcePaths final
{
    static constexpr uint32_t SpaceEnvBlobMagic = 0x31425645u; // "EVB1"
    static constexpr uint32_t SpaceEnvBlobFormatRgba16Sfloat = 2u;
    static constexpr std::string_view AppResourcesWorkingDirectory = "app_resources";
    static constexpr std::string_view DefaultCameraConfigRelativePath = "app_resources/cameras.json";
    static constexpr std::string_view SpaceEnvBlobCandidate = "rich_blue_nebulae_1_8k.rgba16f.envblob";
};

inline bool loadFileBytes(ISystem& system, const std::filesystem::path& path, std::vector<uint8_t>& outPayload)
{
    ISystem::future_t<core::smart_refctd_ptr<IFile>> future;
    system.createFile(future, path, IFile::ECF_READ | IFile::ECF_MAPPABLE);
    auto file = future.acquire();
    if (!file || !file->get())
        return false;

    auto& input = *file->get();
    const auto fileSize = input.getSize();
    outPayload.resize(fileSize);
    if (outPayload.empty())
        return true;

    IFile::success_t readResult;
    input.read(readResult, outPayload.data(), 0, fileSize);
    return static_cast<bool>(readResult);
}

inline bool loadFileText(ISystem& system, const std::filesystem::path& path, std::string& outText)
{
    std::vector<uint8_t> payload;
    if (!loadFileBytes(system, path, payload))
        return false;

    outText.assign(reinterpret_cast<const char*>(payload.data()), payload.size());
    return true;
}

inline bool parseJsonText(std::string_view text, camera_json_t& outJson, std::string* error = nullptr)
{
    try
    {
        outJson = camera_json_t::parse(text);
        return true;
    }
    catch (const std::exception& e)
    {
        if (error)
            *error = "JSON parse error: " + std::string(e.what());
        return false;
    }
}

inline bool loadJsonFromPath(
    ISystem& system,
    const std::filesystem::path& path,
    camera_json_t& outJson,
    std::string* error = nullptr)
{
    std::string jsonText;
    if (!loadFileText(system, path, jsonText))
    {
        if (error)
            *error = "Cannot open config \"" + path.string() + "\".";
        return false;
    }

    return parseJsonText(jsonText, outJson, error);
}

inline path resolveInputPath(const path& localInputCWD, path pathValue)
{
    if (pathValue.is_relative())
        pathValue = (localInputCWD / pathValue).lexically_normal();
    return pathValue;
}

inline bool parseSpaceEnvBlobBytes(
    std::span<const uint8_t> blobBytes,
    SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload)
{
    if (blobBytes.size() < sizeof(SSpaceEnvBlobHeader))
        return false;

    std::memcpy(&outHeader, blobBytes.data(), sizeof(outHeader));

    if (outHeader.magic != SCameraAppResourcePaths::SpaceEnvBlobMagic ||
        outHeader.format != SCameraAppResourcePaths::SpaceEnvBlobFormatRgba16Sfloat)
    {
        return false;
    }
    if (outHeader.width == 0u || outHeader.height == 0u)
        return false;
    if (outHeader.payloadSize != static_cast<uint64_t>(outHeader.width) * outHeader.height * 8ull)
        return false;
    if (outHeader.payloadSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return false;

    const size_t payloadOffset = sizeof(outHeader);
    if (blobBytes.size() != payloadOffset + static_cast<size_t>(outHeader.payloadSize))
        return false;

    outPayload.resize(static_cast<size_t>(outHeader.payloadSize));
    std::memcpy(outPayload.data(), blobBytes.data() + payloadOffset, outPayload.size());
    return true;
}

inline bool loadSpaceEnvBlob(
    ISystem& system,
    const std::filesystem::path& blobPath,
    SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload)
{
    std::vector<uint8_t> blobBytes;
    if (!loadFileBytes(system, blobPath, blobBytes))
        return false;
    return parseSpaceEnvBlobBytes(blobBytes, outHeader, outPayload);
}

inline std::array<path, 3u> makeSpaceEnvSearchRoots(const path& localInputCWD)
{
    return {
        (localInputCWD / ".." / "media" / "envmap").lexically_normal(),
        (localInputCWD / ".." / "media").lexically_normal(),
        (localInputCWD / SCameraAppResourcePaths::AppResourcesWorkingDirectory).lexically_normal()
    };
}

inline bool loadFirstSpaceEnvBlobFromRoots(
    ISystem& system,
    const std::array<path, 3u>& searchRoots,
    SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload)
{
    for (const auto& root : searchRoots)
    {
        if (loadSpaceEnvBlob(system, root / SCameraAppResourcePaths::SpaceEnvBlobCandidate, outHeader, outPayload))
            return true;
    }
    return false;
}

inline core::smart_refctd_ptr<asset::IShader> loadPrecompiledShaderFromAppResources(
    asset::IAssetManager& assetManager,
    ILogger* logger,
    const std::string_view key)
{
    asset::IAssetLoader::SAssetLoadParams loadParams = {};
    loadParams.logger = logger;
    loadParams.workingDirectory = SCameraAppResourcePaths::AppResourcesWorkingDirectory;
    auto bundle = assetManager.getAsset(key.data(), loadParams);
    const auto& contents = bundle.getContents();
    if (contents.empty())
        return nullptr;
    return asset::IAsset::castDown<asset::IShader>(contents[0]);
}

} // namespace nbl::system

#endif // _NBL_THIS_EXAMPLE_APP_RESOURCE_UTILITIES_HPP_
