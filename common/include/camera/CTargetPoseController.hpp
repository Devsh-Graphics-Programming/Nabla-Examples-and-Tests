#ifndef _C_TARGET_POSE_CONTROLLER_HPP_
#define _C_TARGET_POSE_CONTROLLER_HPP_

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "ICamera.hpp"
#include "CFPSCamera.hpp"
#include "CFreeLockCamera.hpp"
#include "CSphericalTargetCamera.hpp"
#include "glm/glm/gtc/quaternion.hpp"

namespace nbl::hlsl
{

struct CTargetPose
{
    float64_t3 position = float64_t3(0.0);
    glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    bool hasDistance = false;
    float distance = 0.f;
    bool hasOrbitState = false;
    double orbitU = 0.0;
    double orbitV = 0.0;
    float orbitDistance = 0.f;
};

class CTargetPoseController
{
public:
    bool buildEvents(ICamera* camera, const CTargetPose& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        out.clear();
        if (!camera)
            return false;

        if (auto* orbit = dynamic_cast<CSphericalTargetCamera*>(camera))
            return buildOrbitEvents(orbit, target, out);

        return buildFreeEvents(camera, target, out);
    }

    bool apply(ICamera* camera, const CTargetPose& target) const
    {
        std::vector<CVirtualGimbalEvent> events;
        if (!buildEvents(camera, target, events))
            return false;
        return camera->manipulate({ events.data(), events.size() });
    }

private:
    static constexpr double Pi = 3.14159265358979323846;
    static constexpr double HalfPi = 0.5 * Pi;

    inline double wrapAngleRad(double angle) const
    {
        while (angle > Pi)
            angle -= 2.0 * Pi;
        while (angle < -Pi)
            angle += 2.0 * Pi;
        return angle;
    }

    inline void appendSignedEvent(std::vector<CVirtualGimbalEvent>& events, double value,
        CVirtualGimbalEvent::VirtualEventType positive, CVirtualGimbalEvent::VirtualEventType negative) const
    {
        if (value == 0.0)
            return;
        auto& ev = events.emplace_back();
        ev.type = (value > 0.0) ? positive : negative;
        ev.magnitude = std::abs(value);
    }

    inline std::pair<double, double> computePitchYawFromOrientation(const glm::quat& orientation) const
    {
        const auto mat = glm::mat3_cast(orientation);
        const auto forward = float64_t3(mat[2][0], mat[2][1], mat[2][2]);
        const double pitch = std::atan2(std::sqrt(forward.x * forward.x + forward.z * forward.z), forward.y) - HalfPi;
        const double yaw = std::atan2(forward.x, forward.z);
        return { pitch, yaw };
    }

    inline float64_t3 extractYawPitchRollYXZ(const glm::quat& delta) const
    {
        const auto m = glm::mat3_cast(delta);
        const double sp = std::clamp(-static_cast<double>(m[1][2]), -1.0, 1.0);
        const double pitch = std::asin(sp);
        const double cp = std::cos(pitch);

        double yaw = 0.0;
        double roll = 0.0;
        if (std::abs(cp) > 1e-6)
        {
            yaw = std::atan2(static_cast<double>(m[0][2]), static_cast<double>(m[2][2]));
            roll = std::atan2(static_cast<double>(m[1][0]), static_cast<double>(m[1][1]));
        }
        else
        {
            yaw = std::atan2(-static_cast<double>(m[2][0]), static_cast<double>(m[0][0]));
            roll = 0.0;
        }

        return float64_t3(pitch, yaw, roll);
    }

    inline bool computeOrbitStateFromPositionTarget(const float64_t3& position, const float64_t3& target,
        double& outU, double& outV, float& outDistance, float minDistance, float maxDistance) const
    {
        const auto localSpherePosition = position - target;
        const double dist = length(localSpherePosition);
        if (!std::isfinite(dist))
            return false;

        const double clamped = std::clamp(dist, static_cast<double>(minDistance), static_cast<double>(maxDistance));
        outDistance = static_cast<float>(clamped);

        if (clamped > 0.0)
        {
            const auto localUnit = localSpherePosition / clamped;
            outU = std::atan2(localUnit.y, localUnit.x);
            outV = std::asin(localUnit.z);
        }

        return true;
    }

    template<typename T>
    inline bool buildOrbitEvents(T* orbit, const CTargetPose& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        double targetU = orbit->getU();
        double targetV = orbit->getV();
        float targetDistance = orbit->getDistance();

        if (target.hasOrbitState)
        {
            targetU = target.orbitU;
            targetV = target.orbitV;
            targetDistance = target.orbitDistance;
        }
        else
        {
            const auto orbitTarget = orbit->getTarget();
            if (!computeOrbitStateFromPositionTarget(target.position, orbitTarget, targetU, targetV, targetDistance, T::MinDistance, T::MaxDistance))
                return false;
        }

        targetDistance = std::clamp(targetDistance, T::MinDistance, T::MaxDistance);

        const double deltaU = targetU - orbit->getU();
        const double deltaV = targetV - orbit->getV();
        const double deltaDistance = static_cast<double>(targetDistance - orbit->getDistance());

        const double moveScale = orbit->getMoveSpeedScale();
        const double moveDenom = 0.01 * (moveScale == 0.0 ? 1.0 : moveScale);

        appendSignedEvent(out, deltaV / moveDenom, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendSignedEvent(out, deltaU / moveDenom, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendSignedEvent(out, deltaDistance / 0.01, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

        return !out.empty();
    }

    inline bool buildFreeEvents(ICamera* camera, const CTargetPose& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        const auto& gimbal = camera->getGimbal();
        const auto currentPos = gimbal.getPosition();
        const auto right = gimbal.getXAxis();
        const auto up = gimbal.getYAxis();
        const auto forward = gimbal.getZAxis();

        const auto deltaWorld = target.position - currentPos;
        const float64_t3 localDelta(
            hlsl::dot(deltaWorld, right),
            hlsl::dot(deltaWorld, up),
            hlsl::dot(deltaWorld, forward));

        appendSignedEvent(out, localDelta.x, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendSignedEvent(out, localDelta.y, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendSignedEvent(out, localDelta.z, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

        if (auto* fps = dynamic_cast<CFPSCamera*>(camera))
        {
            const auto [curPitch, curYaw] = computePitchYawFromOrientation(gimbal.getOrientation());
            const auto [tgtPitch, tgtYaw] = computePitchYawFromOrientation(target.orientation);

            const double rotScale = fps->getRotationSpeedScale();
            const double invScale = rotScale == 0.0 ? 1.0 : (1.0 / rotScale);

            const double deltaPitch = wrapAngleRad(tgtPitch - curPitch) * invScale;
            const double deltaYaw = wrapAngleRad(tgtYaw - curYaw) * invScale;

            appendSignedEvent(out, deltaPitch, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
            appendSignedEvent(out, deltaYaw, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
        }
        else if (auto* freeCam = dynamic_cast<CFreeCamera*>(camera))
        {
            const auto deltaQuat = glm::normalize(target.orientation) * glm::inverse(gimbal.getOrientation());
            const auto angles = extractYawPitchRollYXZ(deltaQuat);

            appendSignedEvent(out, angles.x, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
            appendSignedEvent(out, angles.y, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
            appendSignedEvent(out, angles.z, CVirtualGimbalEvent::RollRight, CVirtualGimbalEvent::RollLeft);
        }

        return !out.empty();
    }
};

} // namespace nbl::hlsl

#endif // _C_TARGET_POSE_CONTROLLER_HPP_
