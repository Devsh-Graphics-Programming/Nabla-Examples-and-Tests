#ifndef _NBL_THIS_EXAMPLE_CAMERA_CONSTRAINT_UTILITIES_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_CAMERA_CONSTRAINT_UTILITIES_HPP_INCLUDED_

#include <algorithm>

#include "CCameraGoalSolver.hpp"
#include "CCameraPresetFlow.hpp"

namespace nbl::this_example
{

struct SCameraConstraintDefaults final
{
    static constexpr float PitchMinDeg = -80.0f;
    static constexpr float PitchMaxDeg = 80.0f;
    static constexpr float YawMinDeg = -180.0f;
    static constexpr float YawMaxDeg = 180.0f;
    static constexpr float RollMinDeg = -180.0f;
    static constexpr float RollMaxDeg = 180.0f;
    static constexpr float MinDistance = nbl::ext::cameras::ICamera::DefaultMinTargetDistance;
    static constexpr float MaxDistance = nbl::ext::cameras::ICamera::DefaultMaxTargetDistance;
};

struct SCameraConstraintSettings
{
    bool enabled = false;
    bool clampPitch = false;
    bool clampYaw = false;
    bool clampRoll = false;
    bool clampDistance = false;
    float pitchMinDeg = SCameraConstraintDefaults::PitchMinDeg;
    float pitchMaxDeg = SCameraConstraintDefaults::PitchMaxDeg;
    float yawMinDeg = SCameraConstraintDefaults::YawMinDeg;
    float yawMaxDeg = SCameraConstraintDefaults::YawMaxDeg;
    float rollMinDeg = SCameraConstraintDefaults::RollMinDeg;
    float rollMaxDeg = SCameraConstraintDefaults::RollMaxDeg;
    float minDistance = SCameraConstraintDefaults::MinDistance;
    float maxDistance = SCameraConstraintDefaults::MaxDistance;
};

struct CCameraConstraintUtilities final
{
    static inline bool applyCameraConstraints(
        const CCameraGoalSolver& solver,
        nbl::ext::cameras::ICamera* camera,
        const SCameraConstraintSettings& constraints)
    {
        if (!constraints.enabled || !camera)
            return false;

        if (camera->hasCapability(nbl::ext::cameras::ICamera::SphericalTarget))
        {
            if (!constraints.clampDistance)
                return false;

            nbl::ext::cameras::ICamera::SphericalTargetState sphericalState;
            if (!camera->tryGetSphericalTargetState(sphericalState))
                return false;

            const float clamped = std::clamp<float>(sphericalState.distance, constraints.minDistance, constraints.maxDistance);
            if (clamped == sphericalState.distance)
                return false;

            return camera->trySetSphericalDistance(clamped);
        }

        if (!(constraints.clampPitch || constraints.clampYaw || constraints.clampRoll))
            return false;

        const auto& gimbal = camera->getGimbal();
        const auto pos = gimbal.getPosition();
        const auto eulerDeg = nbl::ext::cameras::CCameraMathUtilities::getCameraOrientationEulerDegrees(gimbal.getOrientation());

        auto clamped = eulerDeg;
        if (constraints.clampPitch)
            clamped.x = std::clamp(clamped.x, static_cast<decltype(clamped.x)>(constraints.pitchMinDeg), static_cast<decltype(clamped.x)>(constraints.pitchMaxDeg));
        if (constraints.clampYaw)
            clamped.y = std::clamp(clamped.y, static_cast<decltype(clamped.y)>(constraints.yawMinDeg), static_cast<decltype(clamped.y)>(constraints.yawMaxDeg));
        if (constraints.clampRoll)
            clamped.z = std::clamp(clamped.z, static_cast<decltype(clamped.z)>(constraints.rollMinDeg), static_cast<decltype(clamped.z)>(constraints.rollMaxDeg));

        if (clamped.x == eulerDeg.x && clamped.y == eulerDeg.y && clamped.z == eulerDeg.z)
            return false;

        CCameraPreset preset;
        preset.goal.position = pos;
        preset.goal.orientation = nbl::ext::cameras::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(clamped);
        return CCameraPresetFlowUtilities::applyPreset(solver, camera, preset);
    }
};

} // namespace nbl::this_example

#endif
