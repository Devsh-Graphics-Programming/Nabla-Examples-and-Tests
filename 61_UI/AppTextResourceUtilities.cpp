#include "app/AppResourceUtilities.hpp"

#include "app/AppResourcePathUtilities.hpp"

namespace nbl::system
{

bool tryLoadCameraConfigText(
    const SCameraAppResourceContext& context,
    const SCameraConfigLoadRequest& request,
    SCameraConfigLoadResult& outResult,
    std::string* error)
{
    outResult = {};
    if (!context)
    {
        if (error)
            *error = SCameraTextResourceErrorPrefixes::MissingContext;
        return false;
    }

    if (request.requestedPath)
    {
        std::string requestedError;
        if (loadRequestedCameraConfigText(
                *context.system,
                context.localInputCWD,
                request.requestedPath.value(),
                outResult.text,
                &outResult.loadedPath,
                &requestedError))
        {
            outResult.source = ECameraConfigLoadSource::RequestedPath;
            return true;
        }

        outResult.requestedPathLoadFailed = true;
        outResult.requestedPathError = requestedError;
        if (!request.fallbackToDefault)
        {
            if (error)
                *error = outResult.requestedPathError;
            return false;
        }
    }

    if (!loadDefaultCameraConfigText(*context.system, context.localInputCWD, outResult.text, &outResult.loadedPath, error))
        return false;

    outResult.source = ECameraConfigLoadSource::DefaultConfig;
    return true;
}

bool tryLoadCameraScriptText(
    const SCameraAppResourceContext& context,
    const path& scriptPath,
    SCameraScriptTextLoadResult& outResult,
    std::string* error)
{
    outResult = {};
    if (!context)
    {
        if (error)
            *error = SCameraTextResourceErrorPrefixes::MissingContext;
        return false;
    }

    return loadTextResource(
        *context.system,
        context.localInputCWD,
        {
            .pathValue = scriptPath,
            .lookupPolicy = EResourceLookupPolicy::RequestedPath,
            .openErrorPrefix = SCameraTextResourceErrorPrefixes::ScriptedInput
        },
        outResult.text,
        &outResult.loadedPath,
        error);
}

} // namespace nbl::system
