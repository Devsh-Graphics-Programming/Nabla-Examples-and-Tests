#include "app/AppResourceUtilities.hpp"

#include <cstring>
#include <filesystem>
#include <limits>

#include "app/AppResourcePathUtilities.hpp"
#include "camera/CCameraFileUtilities.hpp"

inline bool parseSpaceEnvBlobBytes(
    std::span<const uint8_t> blobBytes,
    nbl::system::SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload)
{
    if (blobBytes.size() < sizeof(nbl::system::SSpaceEnvBlobHeader))
        return false;

    std::memcpy(&outHeader, blobBytes.data(), sizeof(outHeader));

    if (outHeader.magic != nbl::system::SCameraEnvmapResourcePaths::SpaceEnvBlobMagic ||
        outHeader.format != nbl::system::SCameraEnvmapResourcePaths::SpaceEnvBlobFormatRgba16Sfloat)
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
    nbl::system::ISystem& system,
    const nbl::system::path& blobPath,
    nbl::system::SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload)
{
    std::vector<uint8_t> blobBytes;
    if (!nbl::system::CCameraFileUtilities::readBinaryFile(system, blobPath, blobBytes))
        return false;
    return parseSpaceEnvBlobBytes(blobBytes, outHeader, outPayload);
}

namespace nbl::system
{

bool mountOptionalSharedEnvmapResources(
    const SCameraAppResourceContext& context,
    ILogger* logger)
{
    if (!context)
        return false;

    auto sharedEnvmapDirectory = getSharedEnvmapDirectory(context.localInputCWD);
    std::error_code ec;
    if (!std::filesystem::exists(sharedEnvmapDirectory, ec) || ec)
        return false;

    auto sharedEnvmapArchive = make_smart_refctd_ptr<CMountDirectoryArchive>(
        std::move(sharedEnvmapDirectory),
        core::smart_refctd_ptr<ILogger>(logger),
        context.system);
    context.system->mount(
        std::move(sharedEnvmapArchive),
        SCameraMountedResourcePaths::MountedSharedEnvmapWorkingDirectory.data());
    return true;
}

bool loadPreferredSpaceEnvBlob(
    const SCameraAppResourceContext& context,
    SSpaceEnvBlobHeader& outHeader,
    std::vector<uint8_t>& outPayload,
    path* outLoadedPath)
{
    if (!context)
        return false;

    const auto candidates = makeSpaceEnvBlobCandidates(context.localInputCWD);
    return loadFirstCandidatePath(
        candidates.asSpan(),
        [&](const path& candidate) -> bool
        {
            return loadSpaceEnvBlob(*context.system, candidate, outHeader, outPayload);
        },
        outLoadedPath);
}

core::smart_refctd_ptr<asset::IShader> loadPrecompiledShaderFromAppResources(
    asset::IAssetManager& assetManager,
    ILogger* logger,
    const std::string_view key)
{
    asset::IAssetLoader::SAssetLoadParams loadParams = {};
    loadParams.logger = logger;
    loadParams.workingDirectory = SCameraMountedResourcePaths::AppResourcesWorkingDirectory;
    auto bundle = assetManager.getAsset(key.data(), loadParams);
    const auto& contents = bundle.getContents();
    if (contents.empty())
        return nullptr;
    return asset::IAsset::castDown<asset::IShader>(contents[0]);
}

} // namespace nbl::system
