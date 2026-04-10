#include "app/AppCameraConfigUtilities.hpp"

#include <array>
#include <functional>
#include <initializer_list>
#include <type_traits>

#include "keysmapping.hpp"
#include "nlohmann/json.hpp"
#include "nbl/ext/Cameras/CArcballCamera.hpp"
#include "nbl/ext/Cameras/CChaseCamera.hpp"
#include "nbl/ext/Cameras/CDollyCamera.hpp"
#include "nbl/ext/Cameras/CDollyZoomCamera.hpp"
#include "nbl/ext/Cameras/CFPSCamera.hpp"
#include "nbl/ext/Cameras/CFreeLockCamera.hpp"
#include "nbl/ext/Cameras/CIsometricCamera.hpp"
#include "nbl/ext/Cameras/COrbitCamera.hpp"
#include "nbl/ext/Cameras/CPathCamera.hpp"
#include "nbl/ext/Cameras/CTopDownCamera.hpp"
#include "nbl/ext/Cameras/CTurntableCamera.hpp"

namespace nbl::system
{

using camera_json_t = nlohmann::json;

struct SCameraConfigJsonKeys final
{
    static constexpr std::string_view Type = "type";
    static constexpr std::string_view Position = "position";
    static constexpr std::string_view Orientation = "orientation";
    static constexpr std::string_view Target = "target";
    static constexpr std::string_view BaseFov = "baseFov";
    static constexpr std::string_view Cameras = "cameras";
    static constexpr std::string_view Projections = "projections";
    static constexpr std::string_view Bindings = "bindings";
    static constexpr std::string_view Keyboard = "keyboard";
    static constexpr std::string_view Mouse = "mouse";
    static constexpr std::string_view Mappings = "mappings";
    static constexpr std::string_view ScriptedInput = "scripted_input";
    static constexpr std::string_view Viewports = "viewports";
    static constexpr std::string_view Planars = "planars";
    static constexpr std::string_view Projection = "projection";
    static constexpr std::string_view Camera = "camera";
    static constexpr std::string_view ZNear = "zNear";
    static constexpr std::string_view ZFar = "zFar";
    static constexpr std::string_view Fov = "fov";
    static constexpr std::string_view OrthoWidth = "orthoWidth";
};

struct SCameraConfigTypeNames final
{
    static constexpr std::string_view Fps = "FPS";
    static constexpr std::string_view Free = "Free";
    static constexpr std::string_view Orbit = "Orbit";
    static constexpr std::string_view Arcball = "Arcball";
    static constexpr std::string_view Turntable = "Turntable";
    static constexpr std::string_view TopDown = "TopDown";
    static constexpr std::string_view Isometric = "Isometric";
    static constexpr std::string_view Chase = "Chase";
    static constexpr std::string_view Dolly = "Dolly";
    static constexpr std::string_view PathRig = "PathRig";
    static constexpr std::string_view DollyZoom = "DollyZoom";
    static constexpr std::string_view PerspectiveProjection = "perspective";
    static constexpr std::string_view OrthographicProjection = "orthographic";
};

template<typename Camera>
bool tryCreateOrientationCameraFromSpec(
    const camera_json_t& jCamera,
    const double moveScale,
    const double rotationScale,
    std::string_view typeName,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera);

template<typename Camera>
bool tryCreateTargetCameraFromSpec(
    const camera_json_t& jCamera,
    const double moveScale,
    const double rotationScale,
    std::string_view typeName,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera);

bool tryCreateDollyZoomCameraFromSpec(
    const camera_json_t& jCamera,
    const double moveScale,
    const double rotationScale,
    std::string_view typeName,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera);

struct SCameraConfigCameraFactorySpec final
{
    using create_t = bool (*)(
        const camera_json_t&,
        double,
        double,
        std::string_view,
        std::string&,
        core::smart_refctd_ptr<core::ICamera>&);

    std::string_view typeName = {};
    create_t create = nullptr;
    double moveScale = SCameraAppCameraFactoryDefaults::DefaultMoveScale;
    double rotationScale = SCameraAppCameraFactoryDefaults::DefaultRotateScale;
};

inline constexpr std::array<SCameraConfigCameraFactorySpec, 11u> CameraFactorySpecs = {{
    { SCameraConfigTypeNames::Fps, &tryCreateOrientationCameraFromSpec<core::CFPSCamera>, SCameraAppCameraFactoryDefaults::DefaultMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Free, &tryCreateOrientationCameraFromSpec<core::CFreeCamera>, SCameraAppCameraFactoryDefaults::DefaultMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Orbit, &tryCreateTargetCameraFromSpec<core::COrbitCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Arcball, &tryCreateTargetCameraFromSpec<core::CArcballCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Turntable, &tryCreateTargetCameraFromSpec<core::CTurntableCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::TopDown, &tryCreateTargetCameraFromSpec<core::CTopDownCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Isometric, &tryCreateTargetCameraFromSpec<core::CIsometricCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Chase, &tryCreateTargetCameraFromSpec<core::CChaseCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::Dolly, &tryCreateTargetCameraFromSpec<core::CDollyCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::PathRig, &tryCreateTargetCameraFromSpec<core::CPathCamera>, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale },
    { SCameraConfigTypeNames::DollyZoom, &tryCreateDollyZoomCameraFromSpec, SCameraAppCameraFactoryDefaults::TargetRigMoveScale, SCameraAppCameraFactoryDefaults::DefaultRotateScale }
}};

inline bool jsonContainsAll(const camera_json_t& json, std::initializer_list<std::string_view> keys)
{
    for (const auto key : keys)
    {
        if (!json.contains(key))
            return false;
    }
    return true;
}

template<typename T, size_t Count>
inline std::array<T, Count> readJsonArray(const camera_json_t& json, const std::string_view key)
{
    return json[key].get<std::array<T, Count>>();
}

inline hlsl::float64_t3 readJsonFloat64Vec3(const camera_json_t& json, const std::string_view key)
{
    const auto value = readJsonArray<float, 3u>(json, key);
    return hlsl::float64_t3(value[0], value[1], value[2]);
}

inline hlsl::camera_quaternion_t<hlsl::float64_t> readJsonQuaternion(const camera_json_t& json, const std::string_view key)
{
    const auto value = readJsonArray<float, 4u>(json, key);
    return hlsl::CCameraMathUtilities::makeQuaternionFromComponents<hlsl::float64_t>(value[0], value[1], value[2], value[3]);
}

template<typename Camera, typename... Args>
inline core::smart_refctd_ptr<core::ICamera> makeCameraAsBase(Args&&... args)
{
    return core::make_smart_refctd_ptr<Camera>(std::forward<Args>(args)...);
}

template<typename TMap, typename TResolveCode>
bool tryLoadBindingMapFromJson(
    const camera_json_t& jBinding,
    const char* bindingTypeLabel,
    const char* bindingCodeLabel,
    TResolveCode&& resolveCode,
    TMap& outBinding,
    std::string& error)
{
    using code_type = std::remove_cvref_t<std::invoke_result_t<TResolveCode, const char*>>;
    outBinding.clear();
    if (!jBinding.contains(SCameraConfigJsonKeys::Mappings))
    {
        error = std::string("Expected \"mappings\" keyword for ") + bindingTypeLabel + " binding definition.";
        return false;
    }

    for (const auto& [key, value] : jBinding[SCameraConfigJsonKeys::Mappings].items())
    {
        const auto nativeCode = resolveCode(key.c_str());
        if (nativeCode == code_type{})
        {
            error = std::string("Invalid native ") + bindingCodeLabel + " \"" + key + "\" code mapping for " + bindingTypeLabel + " binding.";
            return false;
        }

        outBinding[nativeCode] = core::CVirtualGimbalEvent::stringToVirtualEvent(value.get<std::string>());
    }

    return true;
}

inline void initializeCameraMotionConfig(core::ICamera& camera, const double moveScale, const double rotationScale)
{
    camera.setMotionScales(moveScale, rotationScale);
}

template<typename Camera>
inline bool tryCreateOrientationCameraFromSpec(
    const camera_json_t& jCamera,
    const double moveScale,
    const double rotationScale,
    std::string_view typeName,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera)
{
    if (!jsonContainsAll(jCamera, { SCameraConfigJsonKeys::Position, SCameraConfigJsonKeys::Orientation }))
    {
        error = std::string(typeName) + " camera requires \"position\" and \"orientation\".";
        return false;
    }

    auto camera = makeCameraAsBase<Camera>(
        readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Position),
        readJsonQuaternion(jCamera, SCameraConfigJsonKeys::Orientation));
    initializeCameraMotionConfig(*camera, moveScale, rotationScale);
    outCamera = std::move(camera);
    return true;
}

template<typename Camera>
inline bool tryCreateTargetCameraFromSpec(
    const camera_json_t& jCamera,
    const double moveScale,
    const double rotationScale,
    std::string_view typeName,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera)
{
    if (!jsonContainsAll(jCamera, { SCameraConfigJsonKeys::Position, SCameraConfigJsonKeys::Target }))
    {
        error = std::string(typeName) + " camera requires \"position\" and \"target\".";
        return false;
    }

    auto camera = makeCameraAsBase<Camera>(
        readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Position),
        readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Target));
    initializeCameraMotionConfig(*camera, moveScale, rotationScale);
    outCamera = std::move(camera);
    return true;
}

inline bool tryCreateDollyZoomCameraFromSpec(
    const camera_json_t& jCamera,
    const double moveScale,
    const double rotationScale,
    std::string_view typeName,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera)
{
    if (!jsonContainsAll(jCamera, { SCameraConfigJsonKeys::Position, SCameraConfigJsonKeys::Target }))
    {
        error = std::string(typeName) + " camera requires \"position\" and \"target\".";
        return false;
    }

    auto camera =
        jCamera.contains(SCameraConfigJsonKeys::BaseFov) ?
        makeCameraAsBase<core::CDollyZoomCamera>(
            readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Position),
            readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Target),
            jCamera[SCameraConfigJsonKeys::BaseFov].get<float>()) :
        makeCameraAsBase<core::CDollyZoomCamera>(
            readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Position),
            readJsonFloat64Vec3(jCamera, SCameraConfigJsonKeys::Target));

    initializeCameraMotionConfig(*camera, moveScale, rotationScale);
    outCamera = std::move(camera);
    return true;
}

inline const SCameraConfigCameraFactorySpec* findCameraFactorySpec(const std::string_view typeName)
{
    for (const auto& spec : CameraFactorySpecs)
    {
        if (spec.typeName == typeName)
            return &spec;
    }
    return nullptr;
}

inline bool tryParseViewportBindingSelectionFromJson(
    const camera_json_t& json,
    const char* label,
    SCameraViewportBindingSelection& outSelection,
    std::string& error)
{
    outSelection = {};
    if (!json.is_object())
    {
        error = std::string("Expected object for ") + label + ".";
        return false;
    }

    const auto tryParseIx = [&](const std::string_view key, std::optional<uint32_t>& outIx) -> bool
    {
        if (!json.contains(key))
            return true;
        if (!json[key].is_number_unsigned())
        {
            error = std::string("Expected unsigned integer for \"") + std::string(key) + "\" in " + label + ".";
            return false;
        }
        outIx = json[key].get<uint32_t>();
        return true;
    };

    return tryParseIx(SCameraConfigJsonKeys::Keyboard, outSelection.keyboard) &&
        tryParseIx(SCameraConfigJsonKeys::Mouse, outSelection.mouse);
}

inline bool tryParseViewportConfigFromJson(
    const camera_json_t& json,
    SCameraViewportConfig& outConfig,
    std::string& error)
{
    outConfig = {};
    if (!jsonContainsAll(json, { SCameraConfigJsonKeys::Projection, SCameraConfigJsonKeys::Bindings }))
    {
        error = "\"projection\" or \"bindings\" missing in viewport object.";
        return false;
    }
    if (!json[SCameraConfigJsonKeys::Projection].is_number_unsigned())
    {
        error = "Expected unsigned integer for viewport projection index.";
        return false;
    }

    outConfig.projectionIx = json[SCameraConfigJsonKeys::Projection].get<uint32_t>();
    return tryParseViewportBindingSelectionFromJson(
        json[SCameraConfigJsonKeys::Bindings],
        "viewport bindings",
        outConfig.bindings,
        error);
}

inline bool tryParsePlanarConfigFromJson(
    const camera_json_t& json,
    SCameraPlanarConfig& outConfig,
    std::string& error)
{
    outConfig = {};
    if (!jsonContainsAll(json, { SCameraConfigJsonKeys::Camera, SCameraConfigJsonKeys::Viewports }))
    {
        error = "Expected \"camera\" value and \"viewports\" list in planar object.";
        return false;
    }
    if (!json[SCameraConfigJsonKeys::Camera].is_number_unsigned())
    {
        error = "Expected unsigned integer camera index in planar object.";
        return false;
    }
    if (!json[SCameraConfigJsonKeys::Viewports].is_array())
    {
        error = "Expected array for planar viewport indices.";
        return false;
    }

    outConfig.cameraIx = json[SCameraConfigJsonKeys::Camera].get<uint32_t>();
    outConfig.viewportIxs = json[SCameraConfigJsonKeys::Viewports].get<std::vector<uint32_t>>();
    return true;
}

template<typename Collection, typename ResolveCode>
bool tryLoadBindingCollectionFromJson(
    const camera_json_t& root,
    const std::string_view key,
    const char* bindingTypeLabel,
    const char* bindingCodeLabel,
    ResolveCode&& resolveCode,
    Collection& outCollection,
    std::string& error)
{
    outCollection.clear();
    if (!root.contains(key))
    {
        error = std::string("Expected \"") + std::string(key) + "\" keyword in bindings definition.";
        return false;
    }
    if (!root[key].is_array())
    {
        error = std::string("\"") + std::string(key) + "\" bindings must be an array.";
        return false;
    }

    outCollection.reserve(root[key].size());
    for (const auto& bindingJson : root[key])
    {
        auto& binding = outCollection.emplace_back();
        if (!tryLoadBindingMapFromJson(
                bindingJson,
                bindingTypeLabel,
                bindingCodeLabel,
                std::forward<ResolveCode>(resolveCode),
                binding,
                error))
        {
            return false;
        }
    }

    return true;
}

inline bool tryAppendProjectionFromJson(
    const camera_json_t& jProjection,
    std::vector<core::IPlanarProjection::CProjection>& outProjections,
    std::string& error)
{
    if (!jsonContainsAll(jProjection, {
            SCameraConfigJsonKeys::Type,
            SCameraConfigJsonKeys::ZNear,
            SCameraConfigJsonKeys::ZFar }))
    {
        error = "Projection entry requires \"type\", \"zNear\", and \"zFar\".";
        return false;
    }

    const float zNear = jProjection[SCameraConfigJsonKeys::ZNear].get<float>();
    const float zFar = jProjection[SCameraConfigJsonKeys::ZFar].get<float>();
    const auto type = jProjection[SCameraConfigJsonKeys::Type].get<std::string>();

    if (type == SCameraConfigTypeNames::PerspectiveProjection)
    {
        if (!jProjection.contains(SCameraConfigJsonKeys::Fov))
        {
            error = "Perspective projection requires \"fov\".";
            return false;
        }

        outProjections.emplace_back(
            core::IPlanarProjection::CProjection::create<core::IPlanarProjection::CProjection::Perspective>(
                zNear,
                zFar,
                jProjection[SCameraConfigJsonKeys::Fov].get<float>()));
        return true;
    }

    if (type == SCameraConfigTypeNames::OrthographicProjection)
    {
        if (!jProjection.contains(SCameraConfigJsonKeys::OrthoWidth))
        {
            error = "Orthographic projection requires \"orthoWidth\".";
            return false;
        }

        outProjections.emplace_back(
            core::IPlanarProjection::CProjection::create<core::IPlanarProjection::CProjection::Orthographic>(
                zNear,
                zFar,
                jProjection[SCameraConfigJsonKeys::OrthoWidth].get<float>()));
        return true;
    }

    error = "Unsupported projection type \"" + type + "\".";
    return false;
}

inline bool tryCreateCameraFromJson(
    const camera_json_t& jCamera,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera)
{
    if (!jCamera.contains(SCameraConfigJsonKeys::Type))
    {
        error = "Camera entry missing \"type\".";
        return false;
    }

    if (!jCamera.contains(SCameraConfigJsonKeys::Position))
    {
        error = "Camera entry missing \"position\".";
        return false;
    }

    const auto type = jCamera[SCameraConfigJsonKeys::Type].get<std::string>();
    const auto* spec = findCameraFactorySpec(type);
    if (!spec || !spec->create)
    {
        error = "Unsupported camera type \"" + type + "\".";
        return false;
    }

    return spec->create(
        jCamera,
        spec->moveScale,
        spec->rotationScale,
        spec->typeName,
        error,
        outCamera);
}

inline bool tryParseCameraConfigJsonText(
    const std::string_view text,
    camera_json_t& outJson,
    std::string* const error)
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

bool tryLoadCameraCollectionFromJson(
    const camera_json_t& json,
    std::string& error,
    std::vector<core::smart_refctd_ptr<core::ICamera>>& outCameras)
{
    outCameras.clear();
    if (!json.contains(SCameraConfigJsonKeys::Cameras) || !json[SCameraConfigJsonKeys::Cameras].is_array())
    {
        error = "Missing \"cameras\" array in config.";
        return false;
    }

    outCameras.reserve(json[SCameraConfigJsonKeys::Cameras].size());
    for (const auto& jCamera : json[SCameraConfigJsonKeys::Cameras])
    {
        core::smart_refctd_ptr<core::ICamera> camera;
        if (!tryCreateCameraFromJson(jCamera, error, camera))
            return false;
        outCameras.emplace_back(std::move(camera));
    }

    if (outCameras.empty())
    {
        error = "No cameras defined.";
        return false;
    }

    return true;
}

bool tryLoadProjectionCollectionFromJson(
    const camera_json_t& json,
    std::string& error,
    std::vector<core::IPlanarProjection::CProjection>& outProjections)
{
    outProjections.clear();
    if (!json.contains(SCameraConfigJsonKeys::Projections) || !json[SCameraConfigJsonKeys::Projections].is_array())
    {
        error = "Missing \"projections\" array in config.";
        return false;
    }

    outProjections.reserve(json[SCameraConfigJsonKeys::Projections].size());
    for (const auto& jProjection : json[SCameraConfigJsonKeys::Projections])
    {
        if (!tryAppendProjectionFromJson(jProjection, outProjections, error))
            return false;
    }

    return true;
}

bool tryLoadInputBindingCollectionsFromJson(
    const camera_json_t& json,
    std::string& error,
    SCameraInputBindingCollections& outBindings)
{
    outBindings = {};
    if (!json.contains(SCameraConfigJsonKeys::Bindings))
    {
        error = "Expected \"bindings\" keyword in camera JSON.";
        return false;
    }

    const auto& jBindings = json[SCameraConfigJsonKeys::Bindings];
    if (!tryLoadBindingCollectionFromJson(
            jBindings,
            SCameraConfigJsonKeys::Keyboard,
            "keyboard",
            "key",
            [](const char* key) { return stringToKeyCode(key); },
            outBindings.keyboard,
            error))
    {
        return false;
    }

    if (!tryLoadBindingCollectionFromJson(
            jBindings,
            SCameraConfigJsonKeys::Mouse,
            "mouse",
            "key",
            [](const char* key) { return stringToMouseCode(key); },
            outBindings.mouse,
            error))
    {
        return false;
    }

    return true;
}

bool tryLoadPlanarConfigCollectionsFromJson(
    const camera_json_t& json,
    std::string& error,
    SCameraPlanarConfigCollections& outPlanarConfig)
{
    outPlanarConfig = {};
    if (!(json.contains(SCameraConfigJsonKeys::Viewports) && json.contains(SCameraConfigJsonKeys::Planars)))
    {
        error = "Expected \"viewports\" and \"planars\" lists in JSON.";
        return false;
    }
    if (!json[SCameraConfigJsonKeys::Viewports].is_array() || !json[SCameraConfigJsonKeys::Planars].is_array())
    {
        error = "\"viewports\" and \"planars\" must be arrays.";
        return false;
    }

    outPlanarConfig.viewports.reserve(json[SCameraConfigJsonKeys::Viewports].size());
    for (const auto& jViewport : json[SCameraConfigJsonKeys::Viewports])
    {
        auto& viewportConfig = outPlanarConfig.viewports.emplace_back();
        if (!tryParseViewportConfigFromJson(jViewport, viewportConfig, error))
            return false;
    }

    outPlanarConfig.planars.reserve(json[SCameraConfigJsonKeys::Planars].size());
    for (const auto& jPlanar : json[SCameraConfigJsonKeys::Planars])
    {
        auto& planarConfig = outPlanarConfig.planars.emplace_back();
        if (!tryParsePlanarConfigFromJson(jPlanar, planarConfig, error))
            return false;
    }

    if (!outPlanarConfig.valid())
    {
        error = "No planars defined.";
        return false;
    }
    return true;
}

bool tryBuildCameraConfigCollections(
    const camera_json_t& json,
    SCameraConfigCollections& outCollections,
    std::string& error)
{
    outCollections = {};
    if (json.contains(SCameraConfigJsonKeys::ScriptedInput))
        outCollections.embeddedScriptedInputText = json[SCameraConfigJsonKeys::ScriptedInput].dump();

    if (!tryLoadCameraCollectionFromJson(json, error, outCollections.cameras))
        return false;

    if (!tryLoadProjectionCollectionFromJson(json, error, outCollections.projections))
        return false;

    if (!tryLoadInputBindingCollectionsFromJson(json, error, outCollections.bindings))
        return false;

    if (!tryLoadPlanarConfigCollectionsFromJson(json, error, outCollections.planarConfig))
        return false;

    return true;
}

bool tryBuildCameraConfigCollections(
    const std::string_view text,
    SCameraConfigCollections& outCollections,
    std::string& error)
{
    camera_json_t json = {};
    if (!tryParseCameraConfigJsonText(text, json, &error))
        return false;

    return tryBuildCameraConfigCollections(json, outCollections, error);
}

} // namespace nbl::system
