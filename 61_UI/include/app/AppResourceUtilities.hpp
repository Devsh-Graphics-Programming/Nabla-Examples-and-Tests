#ifndef _NBL_THIS_EXAMPLE_APP_RESOURCE_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_RESOURCE_UTILITIES_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"

namespace nbl::system
{

struct SSpaceEnvBlobHeader final
{
    uint32_t magic = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t format = 0u;
    uint64_t payloadSize = 0ull;
};

struct SCameraMountedResourcePaths final
{
    static constexpr std::string_view AppResourcesWorkingDirectory = "app_resources";
    static constexpr std::string_view MountedSharedEnvmapWorkingDirectory = "app_resources/shared_envmap";
    static constexpr std::string_view SharedEnvmapRelativeDirectory = "../media/envmap";
};

struct SCameraConfigResourcePaths final
{
    static constexpr size_t CandidateCount = 2u;
    static constexpr std::string_view DefaultCameraConfigRelativePath = "cameras.json";
};

struct SCameraEnvmapResourcePaths final
{
    static constexpr uint32_t SpaceEnvBlobMagic = 0x31425645u;
    static constexpr uint32_t SpaceEnvBlobFormatRgba16Sfloat = 2u;
    static constexpr size_t CandidateCount = 2u;
    static constexpr std::string_view SpaceEnvBlobCandidate = "rich_blue_nebulae_1_8k.rgba16f.envblob";
};

struct SCameraAppResourceContext final
{
    ISystem* system = nullptr;
    path localInputCWD = {};

    inline explicit operator bool() const
    {
        return system != nullptr;
    }
};

inline SCameraAppResourceContext makeCameraAppResourceContext(ISystem& system, const path& localInputCWD)
{
    return {
        .system = &system,
        .localInputCWD = localInputCWD
    };
}

enum class ECameraConfigLoadSource : uint8_t
{
    RequestedPath,
    DefaultConfig
};

struct SCameraConfigLoadRequest final
{
    std::optional<path> requestedPath = std::nullopt;
    bool fallbackToDefault = false;
};

struct SCameraConfigLoadResult final
{
    std::string text = {};
    path loadedPath = {};
    ECameraConfigLoadSource source = ECameraConfigLoadSource::DefaultConfig;
    bool requestedPathLoadFailed = false;
    std::string requestedPathError = {};

    inline bool usedRequestedPath() const
    {
        return source == ECameraConfigLoadSource::RequestedPath;
    }

    inline bool usedDefaultConfig() const
    {
        return source == ECameraConfigLoadSource::DefaultConfig;
    }
};

struct SCameraScriptTextLoadResult final
{
    std::string text = {};
    path loadedPath = {};
};

bool tryLoadCameraConfigText(
    const SCameraAppResourceContext& context,
    const SCameraConfigLoadRequest& request,
    SCameraConfigLoadResult& outResult,
    std::string* error = nullptr);

bool tryLoadCameraScriptText(
    const SCameraAppResourceContext& context,
    const path& scriptPath,
    SCameraScriptTextLoadResult& outResult,
    std::string* error = nullptr);

bool mountOptionalSharedEnvmapResources(
    const SCameraAppResourceContext& context,
    ILogger* logger = nullptr);

bool loadPreferredSpaceEnvBlob(
    const SCameraAppResourceContext& context,
    SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload,
    path* outLoadedPath = nullptr);

core::smart_refctd_ptr<asset::IShader> loadPrecompiledShaderFromAppResources(
    asset::IAssetManager& assetManager,
    ILogger* logger,
    std::string_view key);

} // namespace nbl::system

#endif // _NBL_THIS_EXAMPLE_APP_RESOURCE_UTILITIES_HPP_
