#ifndef _NBL_THIS_EXAMPLE_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_INCLUDED_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "camera/CCameraScriptedActionUtilities.hpp"
#include "nbl/ext/Cameras/CCameraScriptedRuntime.hpp"
#include "nbl/ext/Cameras/CCameraSequenceScriptPersistence.hpp"
#include "nbl/system/path.h"

namespace nbl::this_example
{

struct CCameraScriptedControlOverrides
{
    bool hasKeyboardScale = false;
    float keyboardScale = 1.f;
    bool hasMouseMoveScale = false;
    float mouseMoveScale = 1.f;
    bool hasMouseScrollScale = false;
    float mouseScrollScale = 1.f;
    bool hasTranslationScale = false;
    float translationScale = 1.f;
    bool hasRotationScale = false;
    float rotationScale = 1.f;
};

struct CCameraScriptedInputParseResult
{
    bool enabled = true;
    bool hasLog = false;
    bool log = false;
    bool hardFail = false;
    bool visualDebug = false;
    float visualTargetFps = 0.f;
    float visualCameraHoldSeconds = 0.f;
    bool hasEnableActiveCameraMovement = false;
    bool enableActiveCameraMovement = true;
    bool exclusive = false;
    std::string capturePrefix = "script";
    CCameraScriptedControlOverrides cameraControls = {};
    nbl::system::CCameraScriptedTimeline timeline = {};
    std::vector<CCameraScriptedActionEvent> actionEvents = {};
    std::optional<nbl::core::CCameraSequenceScript> sequence;
    std::vector<std::string> warnings;
};

struct CCameraScriptedRuntimePersistenceUtilities final
{
    static inline void appendScriptedInputParseWarning(CCameraScriptedInputParseResult& out, std::string warning)
    {
        out.warnings.emplace_back(std::move(warning));
    }

    static bool readCameraScriptedInput(std::string_view text, CCameraScriptedInputParseResult& out, std::string* error = nullptr);
    static bool loadCameraScriptedInputFromFile(nbl::system::ISystem& system, const nbl::system::path& path, CCameraScriptedInputParseResult& out, std::string* error = nullptr);
};

} // namespace nbl::this_example

#endif
