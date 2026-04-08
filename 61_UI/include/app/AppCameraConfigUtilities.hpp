#ifndef _APP_CAMERA_CONFIG_UTILITIES_HPP_
#define _APP_CAMERA_CONFIG_UTILITIES_HPP_

#include <array>
#include <string>
#include <vector>

#include "app/AppResourceUtilities.hpp"
#include "camera/CArcballCamera.hpp"
#include "camera/CChaseCamera.hpp"
#include "camera/CDollyCamera.hpp"
#include "camera/CDollyZoomCamera.hpp"
#include "camera/CFPSCamera.hpp"
#include "camera/CFreeLockCamera.hpp"
#include "camera/CIsometricCamera.hpp"
#include "camera/COrbitCamera.hpp"
#include "camera/CPathCamera.hpp"
#include "camera/CTopDownCamera.hpp"
#include "camera/CTurntableCamera.hpp"

namespace nbl::system
{

struct SCameraConfigFactoryMotionScales final
{
    static constexpr double DefaultMove = core::ICamera::DefaultMoveSpeedScale;
    static constexpr double DefaultRotate = core::ICamera::DefaultRotationSpeedScale;
    static constexpr double TargetRigMove = 0.5;
};

inline void initializeCameraMotionConfig(core::ICamera& camera, const double moveScale, const double rotationScale)
{
    camera.setMotionScales(moveScale, rotationScale);
}

inline bool tryCreateCameraFromJson(
    const camera_json_t& jCamera,
    std::string& error,
    core::smart_refctd_ptr<core::ICamera>& outCamera)
{
    if (!jCamera.contains("type"))
    {
        error = "Camera entry missing \"type\".";
        return false;
    }

    if (!jCamera.contains("position"))
    {
        error = "Camera entry missing \"position\".";
        return false;
    }

    const std::string type = jCamera["type"].get<std::string>();
    const bool withOrientation = jCamera.contains("orientation");
    const bool withTarget = jCamera.contains("target");

    const auto position = [&]()
    {
        const auto value = jCamera["position"].get<std::array<float, 3>>();
        return hlsl::float64_t3(value[0], value[1], value[2]);
    }();

    const auto getOrientation = [&]()
    {
        const auto value = jCamera["orientation"].get<std::array<float, 4>>();
        return hlsl::makeQuaternionFromComponents<hlsl::float64_t>(value[0], value[1], value[2], value[3]);
    };

    const auto getTarget = [&]()
    {
        const auto value = jCamera["target"].get<std::array<float, 3>>();
        return hlsl::float64_t3(value[0], value[1], value[2]);
    };

    const auto finalize = [&](auto&& camera, const double moveScale, const double rotationScale)
    {
        initializeCameraMotionConfig(*camera, moveScale, rotationScale);
        outCamera = std::move(camera);
        return true;
    };

    if (type == "FPS")
    {
        if (!withOrientation)
        {
            error = "FPS camera requires \"orientation\".";
            return false;
        }
        return finalize(core::make_smart_refctd_ptr<core::CFPSCamera>(position, getOrientation()), SCameraConfigFactoryMotionScales::DefaultMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    }

    if (type == "Free")
    {
        if (!withOrientation)
        {
            error = "Free camera requires \"orientation\".";
            return false;
        }
        return finalize(core::make_smart_refctd_ptr<core::CFreeCamera>(position, getOrientation()), SCameraConfigFactoryMotionScales::DefaultMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    }

    if (!withTarget)
    {
        error = "Camera type \"" + type + "\" requires \"target\".";
        return false;
    }

    if (type == "Orbit")
        return finalize(core::make_smart_refctd_ptr<core::COrbitCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "Arcball")
        return finalize(core::make_smart_refctd_ptr<core::CArcballCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "Turntable")
        return finalize(core::make_smart_refctd_ptr<core::CTurntableCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "TopDown")
        return finalize(core::make_smart_refctd_ptr<core::CTopDownCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "Isometric")
        return finalize(core::make_smart_refctd_ptr<core::CIsometricCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "Chase")
        return finalize(core::make_smart_refctd_ptr<core::CChaseCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "Dolly")
        return finalize(core::make_smart_refctd_ptr<core::CDollyCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "Path")
        return finalize(core::make_smart_refctd_ptr<core::CPathCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    if (type == "DollyZoom")
    {
        if (jCamera.contains("baseFov"))
            return finalize(core::make_smart_refctd_ptr<core::CDollyZoomCamera>(position, getTarget(), jCamera["baseFov"].get<float>()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
        return finalize(core::make_smart_refctd_ptr<core::CDollyZoomCamera>(position, getTarget()), SCameraConfigFactoryMotionScales::TargetRigMove, SCameraConfigFactoryMotionScales::DefaultRotate);
    }

    error = "Unsupported camera type \"" + type + "\".";
    return false;
}

inline bool tryLoadCameraCollectionFromJson(
    const camera_json_t& json,
    std::string& error,
    std::vector<core::smart_refctd_ptr<core::ICamera>>& outCameras)
{
    outCameras.clear();
    if (!json.contains("cameras") || !json["cameras"].is_array())
    {
        error = "Missing \"cameras\" array in config.";
        return false;
    }

    outCameras.reserve(json["cameras"].size());
    for (const auto& jCamera : json["cameras"])
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

} // namespace nbl::system

#endif // _APP_CAMERA_CONFIG_UTILITIES_HPP_
