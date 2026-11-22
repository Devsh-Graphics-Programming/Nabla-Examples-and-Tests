#include "app/AppCameraConfigUtilities.hpp"

namespace nbl::system
{

bool tryLoadCameraConfigCollections(
    const SCameraAppResourceContext& context,
    const SCameraConfigLoadRequest& request,
    SCameraConfigLoadResult& outLoadResult,
    SCameraConfigCollections& outCollections,
    std::string* const error)
{
    outLoadResult = {};
    outCollections = {};

    std::string loadOrParseError;
    if (!tryLoadCameraConfigText(context, request, outLoadResult, &loadOrParseError))
    {
        if (error)
            *error = loadOrParseError;
        return false;
    }

    if (!tryBuildCameraConfigCollections(std::string_view(outLoadResult.text), outCollections, loadOrParseError))
    {
        if (error)
            *error = loadOrParseError;
        return false;
    }

    return true;
}

bool tryBuildCameraPlanarRuntimeBootstrap(
    const SCameraAppResourceContext& context,
    const SCameraConfigLoadRequest& request,
    SCameraPlanarRuntimeBootstrap& outBootstrap,
    std::string* const error)
{
    outBootstrap = {};

    std::string runtimeError;
    if (!tryLoadCameraConfigCollections(
            context,
            request,
            outBootstrap.loadResult,
            outBootstrap.collections,
            &runtimeError))
    {
        if (error)
            *error = runtimeError;
        return false;
    }

    if (!tryBuildCameraPlanarRuntime(outBootstrap.collections, outBootstrap.planars, runtimeError))
    {
        if (error)
            *error = runtimeError;
        return false;
    }

    return true;
}

} // namespace nbl::system
