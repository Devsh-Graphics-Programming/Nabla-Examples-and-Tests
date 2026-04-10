#ifndef _NBL_THIS_EXAMPLE_APP_RESOURCE_PATH_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_RESOURCE_PATH_UTILITIES_HPP_

#include <array>
#include <span>

#include "app/AppResourceUtilities.hpp"
#include "nbl/ext/Cameras/CCameraFileUtilities.hpp"

namespace nbl::system
{

enum class EResourceLookupPolicy : uint8_t
{
    MountedOnly,
    RequestedPath
};

struct SCameraTextResourceErrorPrefixes final
{
    static constexpr std::string_view CameraConfig = "Cannot open config";
    static constexpr std::string_view ScriptedInput = "Cannot open scripted input file";
    static constexpr std::string_view MissingContext = "Camera app resource context is not initialized.";
};

struct STextResourceLoadRequest final
{
    path pathValue = {};
    EResourceLookupPolicy lookupPolicy = EResourceLookupPolicy::RequestedPath;
    std::string_view openErrorPrefix = {};
};

template<size_t CandidateCount>
struct SCameraAppResourcePathCandidates final
{
    std::array<path, CandidateCount> paths = {};
    size_t count = 0u;

    inline std::span<const path> asSpan() const
    {
        return { paths.data(), count };
    }

    inline bool appendUnique(const path& candidate)
    {
        for (size_t i = 0u; i < count; ++i)
        {
            if (paths[i] == candidate)
                return true;
        }

        if (count >= CandidateCount)
            return false;

        paths[count++] = candidate;
        return true;
    }
};

inline path resolveInputPath(const path& localInputCWD, path pathValue)
{
    if (pathValue.is_relative())
        pathValue = (localInputCWD / pathValue).lexically_normal();
    return pathValue;
}

inline bool isMountedAppResourcePath(const path& pathValue)
{
    if (pathValue.empty())
        return false;

    const auto begin = pathValue.begin();
    if (begin == pathValue.end())
        return false;

    return *begin == SCameraMountedResourcePaths::AppResourcesWorkingDirectory;
}

inline path makeMountedAppResourcePath(const path& relativePath)
{
    if (relativePath.empty() || relativePath.is_absolute() || isMountedAppResourcePath(relativePath))
        return relativePath;

    return path(SCameraMountedResourcePaths::AppResourcesWorkingDirectory) / relativePath;
}

template<size_t CandidateCount>
inline SCameraAppResourcePathCandidates<CandidateCount> makeResourcePathCandidates(
    const path& localInputCWD,
    const path& pathValue,
    const EResourceLookupPolicy lookupPolicy)
{
    SCameraAppResourcePathCandidates<CandidateCount> candidates = {};
    if (pathValue.empty())
        return candidates;

    if (pathValue.is_absolute())
    {
        candidates.appendUnique(pathValue);
        return candidates;
    }

    if (lookupPolicy == EResourceLookupPolicy::MountedOnly || isMountedAppResourcePath(pathValue))
    {
        candidates.appendUnique(makeMountedAppResourcePath(pathValue));
        return candidates;
    }

    candidates.appendUnique(resolveInputPath(localInputCWD, pathValue));
    return candidates;
}

template<typename Loader>
inline bool loadFirstCandidatePath(
    std::span<const path> candidates,
    Loader&& loader,
    path* outLoadedPath = nullptr)
{
    for (const auto& candidate : candidates)
    {
        if (!loader(candidate))
            continue;

        if (outLoadedPath)
            *outLoadedPath = candidate;
        return true;
    }
    return false;
}

inline bool loadTextResource(
    ISystem& system,
    const path& localInputCWD,
    const STextResourceLoadRequest& request,
    std::string& outText,
    path* outLoadedPath = nullptr,
    std::string* error = nullptr)
{
    const auto candidates = makeResourcePathCandidates<SCameraConfigResourcePaths::CandidateCount>(
        localInputCWD,
        request.pathValue,
        request.lookupPolicy);
    if (loadFirstCandidatePath(
            candidates.asSpan(),
            [&](const path& candidate) -> bool
            {
                return CCameraFileUtilities::readTextFile(system, candidate, outText);
            },
            outLoadedPath))
    {
        return true;
    }

    if (error)
        *error = std::string(request.openErrorPrefix) + " \"" + request.pathValue.string() + "\".";
    return false;
}

inline bool loadDefaultCameraConfigText(
    ISystem& system,
    const path& localInputCWD,
    std::string& outText,
    path* outLoadedPath = nullptr,
    std::string* error = nullptr)
{
    return loadTextResource(
        system,
        localInputCWD,
        {
            .pathValue = path(SCameraConfigResourcePaths::DefaultCameraConfigRelativePath),
            .lookupPolicy = EResourceLookupPolicy::MountedOnly,
            .openErrorPrefix = SCameraTextResourceErrorPrefixes::CameraConfig
        },
        outText,
        outLoadedPath,
        error);
}

inline bool loadRequestedCameraConfigText(
    ISystem& system,
    const path& localInputCWD,
    const path& requestedPath,
    std::string& outText,
    path* outLoadedPath = nullptr,
    std::string* error = nullptr)
{
    return loadTextResource(
        system,
        localInputCWD,
        {
            .pathValue = requestedPath,
            .lookupPolicy = EResourceLookupPolicy::RequestedPath,
            .openErrorPrefix = SCameraTextResourceErrorPrefixes::CameraConfig
        },
        outText,
        outLoadedPath,
        error);
}

inline path getSharedEnvmapDirectory(const path& localInputCWD)
{
    return resolveInputPath(
        localInputCWD,
        path(SCameraMountedResourcePaths::SharedEnvmapRelativeDirectory));
}

inline SCameraAppResourcePathCandidates<SCameraEnvmapResourcePaths::CandidateCount> makeSpaceEnvBlobCandidates(const path& localInputCWD)
{
    return makeResourcePathCandidates<SCameraEnvmapResourcePaths::CandidateCount>(
        localInputCWD,
        path(SCameraMountedResourcePaths::MountedSharedEnvmapWorkingDirectory) /
            path(SCameraEnvmapResourcePaths::SpaceEnvBlobCandidate),
        EResourceLookupPolicy::MountedOnly);
}

} // namespace nbl::system

#endif // _NBL_THIS_EXAMPLE_APP_RESOURCE_PATH_UTILITIES_HPP_
