#ifndef _C_CAMERA_GOAL_SOLVER_HPP_
#define _C_CAMERA_GOAL_SOLVER_HPP_

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

struct CCameraGoal
{
    float64_t3 position = float64_t3(0.0);
    glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    ICamera::CameraKind sourceKind = ICamera::CameraKind::Unknown;
    uint32_t sourceCapabilities = ICamera::None;
    uint32_t sourceGoalStateMask = ICamera::GoalStateNone;
    bool hasTargetPosition = false;
    float64_t3 targetPosition = float64_t3(0.0);
    bool hasDistance = false;
    float distance = 0.f;
    bool hasOrbitState = false;
    double orbitU = 0.0;
    double orbitV = 0.0;
    float orbitDistance = 0.f;
    bool hasPathState = false;
    ICamera::PathState pathState = {};
    bool hasDynamicPerspectiveState = false;
    ICamera::DynamicPerspectiveState dynamicPerspectiveState = {};
};

class CCameraGoalSolver
{
public:
    struct SCompatibilityResult
    {
        bool sameKind = false;
        bool exact = false;
        uint32_t requiredGoalStateMask = ICamera::GoalStateNone;
        uint32_t supportedGoalStateMask = ICamera::GoalStateNone;
        uint32_t missingGoalStateMask = ICamera::GoalStateNone;
    };

    struct SApplyResult
    {
        enum class EStatus : uint8_t
        {
            Unsupported,
            Failed,
            AlreadySatisfied,
            AppliedAbsoluteOnly,
            AppliedVirtualEvents,
            AppliedAbsoluteAndVirtualEvents
        };

        enum EIssue : uint32_t
        {
            NoIssue = 0u,
            UsedAbsolutePoseFallback = 1u << 0,
            MissingSphericalTargetState = 1u << 1,
            MissingPathState = 1u << 2,
            MissingDynamicPerspectiveState = 1u << 3,
            VirtualEventReplayFailed = 1u << 4
        };

        EStatus status = EStatus::Unsupported;
        bool exact = false;
        uint32_t eventCount = 0u;
        uint32_t issues = NoIssue;

        inline bool succeeded() const
        {
            return status != EStatus::Unsupported && status != EStatus::Failed;
        }

        inline bool changed() const
        {
            return status == EStatus::AppliedAbsoluteOnly ||
                status == EStatus::AppliedVirtualEvents ||
                status == EStatus::AppliedAbsoluteAndVirtualEvents;
        }

        inline bool approximate() const
        {
            return succeeded() && !exact;
        }

        inline bool hasIssue(EIssue issue) const
        {
            return (issues & issue) == issue;
        }
    };

    bool buildEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        out.clear();
        if (!camera)
            return false;

        if (camera->hasCapability(ICamera::SphericalTarget))
            return buildSphericalEvents(camera, target, out);

        return buildFreeEvents(camera, target, out);
    }

    bool capture(ICamera* camera, CCameraGoal& out) const
    {
        out = {};
        if (!camera)
            return false;

        const auto& gimbal = camera->getGimbal();
        out.position = gimbal.getPosition();
        out.orientation = gimbal.getOrientation();
        out.sourceKind = camera->getKind();
        out.sourceCapabilities = camera->getCapabilities();
        out.sourceGoalStateMask = camera->getGoalStateMask();

        ICamera::SphericalTargetState sphericalState;
        if (camera->tryGetSphericalTargetState(sphericalState))
        {
            out.targetPosition = sphericalState.target;
            out.hasTargetPosition = true;
            out.distance = sphericalState.distance;
            out.hasDistance = true;
            out.orbitDistance = sphericalState.distance;
            out.orbitU = sphericalState.u;
            out.orbitV = sphericalState.v;
            out.hasOrbitState = true;
        }

        ICamera::DynamicPerspectiveState dynamicState;
        if (camera->tryGetDynamicPerspectiveState(dynamicState))
        {
            out.hasDynamicPerspectiveState = true;
            out.dynamicPerspectiveState = dynamicState;
        }

        ICamera::PathState pathState;
        if (camera->tryGetPathState(pathState))
        {
            out.hasPathState = true;
            out.pathState = pathState;
        }

        return true;
    }

    SCompatibilityResult analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const
    {
        SCompatibilityResult result;
        if (!camera)
            return result;

        result.sameKind = target.sourceKind == ICamera::CameraKind::Unknown || target.sourceKind == camera->getKind();
        result.supportedGoalStateMask = camera->getGoalStateMask();
        result.requiredGoalStateMask = getRequiredGoalStateMask(target);
        result.missingGoalStateMask = result.requiredGoalStateMask & ~result.supportedGoalStateMask;
        result.exact = result.missingGoalStateMask == ICamera::GoalStateNone;
        return result;
    }

    SApplyResult applyDetailed(ICamera* camera, const CCameraGoal& target) const
    {
        SApplyResult result;
        if (!camera)
            return result;

        bool exact = true;
        bool absoluteChanged = false;

        if (!camera->hasCapability(ICamera::SphericalTarget))
        {
            bool poseChanged = false;
            bool poseExact = false;
            if (tryApplyAbsoluteReferencePose(camera, target, poseChanged, poseExact))
            {
                result.issues |= SApplyResult::UsedAbsolutePoseFallback;
                absoluteChanged = absoluteChanged || poseChanged;
                if (poseExact && !target.hasDynamicPerspectiveState)
                {
                    result.status = poseChanged ?
                        SApplyResult::EStatus::AppliedAbsoluteOnly :
                        SApplyResult::EStatus::AlreadySatisfied;
                    result.exact = true;
                    return result;
                }
            }
        }

        if (target.hasTargetPosition)
        {
            ICamera::SphericalTargetState beforeState;
            if (!camera->tryGetSphericalTargetState(beforeState))
            {
                result.issues |= SApplyResult::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                const auto beforeTarget = beforeState.target;
                if (!camera->trySetSphericalTarget(target.targetPosition))
                {
                    result.issues |= SApplyResult::MissingSphericalTargetState;
                    exact = false;
                }
                else
                {
                    ICamera::SphericalTargetState afterState;
                    if (!camera->tryGetSphericalTargetState(afterState))
                    {
                        result.issues |= SApplyResult::MissingSphericalTargetState;
                        exact = false;
                    }
                    else
                    {
                        absoluteChanged = afterState.target != beforeTarget;
                        exact = exact && afterState.target == target.targetPosition;
                    }
                }
            }
        }

        if (target.hasDistance || target.hasOrbitState)
        {
            ICamera::SphericalTargetState beforeState;
            if (!camera->tryGetSphericalTargetState(beforeState))
            {
                result.issues |= SApplyResult::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                const float desiredDistance = target.hasOrbitState ? target.orbitDistance : target.distance;
                const float beforeDistance = beforeState.distance;
                if (!camera->trySetSphericalDistance(desiredDistance))
                {
                    result.issues |= SApplyResult::MissingSphericalTargetState;
                    exact = false;
                }
                else
                {
                    ICamera::SphericalTargetState afterState;
                    if (!camera->tryGetSphericalTargetState(afterState))
                    {
                        result.issues |= SApplyResult::MissingSphericalTargetState;
                        exact = false;
                    }
                    else
                    {
                        absoluteChanged = absoluteChanged || afterState.distance != beforeDistance;
                        exact = exact && std::abs(static_cast<double>(afterState.distance - desiredDistance)) <= 1e-6;
                    }
                }
            }
        }

        if (target.hasPathState)
        {
            ICamera::PathState beforeState;
            if (!camera->tryGetPathState(beforeState))
            {
                result.issues |= SApplyResult::MissingPathState;
                exact = false;
            }
            else if (!camera->trySetPathState(target.pathState))
            {
                result.issues |= SApplyResult::MissingPathState;
                exact = false;
            }
            else
            {
                ICamera::PathState afterState;
                if (!camera->tryGetPathState(afterState))
                {
                    result.issues |= SApplyResult::MissingPathState;
                    exact = false;
                }
                else
                {
                    const bool pathChanged = !nearlyEqual(beforeState.angle, afterState.angle) ||
                        !nearlyEqual(beforeState.radius, afterState.radius) ||
                        !nearlyEqual(beforeState.height, afterState.height);
                    const bool pathExact = nearlyEqual(afterState.angle, target.pathState.angle) &&
                        nearlyEqual(afterState.radius, target.pathState.radius) &&
                        nearlyEqual(afterState.height, target.pathState.height);

                    absoluteChanged = absoluteChanged || pathChanged;
                    exact = exact && pathExact;
                }
            }
        }

        if (target.hasDynamicPerspectiveState)
        {
            ICamera::DynamicPerspectiveState beforeState;
            if (!camera->tryGetDynamicPerspectiveState(beforeState))
            {
                result.issues |= SApplyResult::MissingDynamicPerspectiveState;
                exact = false;
            }
            else if (!camera->trySetDynamicPerspectiveState(target.dynamicPerspectiveState))
            {
                result.issues |= SApplyResult::MissingDynamicPerspectiveState;
                exact = false;
            }
            else
            {
                ICamera::DynamicPerspectiveState afterState;
                if (!camera->tryGetDynamicPerspectiveState(afterState))
                {
                    result.issues |= SApplyResult::MissingDynamicPerspectiveState;
                    exact = false;
                }
                else
                {
                    const bool dynamicChanged = !nearlyEqual(beforeState.baseFov, afterState.baseFov) ||
                        !nearlyEqual(beforeState.referenceDistance, afterState.referenceDistance);
                    const bool dynamicExact = nearlyEqual(afterState.baseFov, target.dynamicPerspectiveState.baseFov) &&
                        nearlyEqual(afterState.referenceDistance, target.dynamicPerspectiveState.referenceDistance);

                    absoluteChanged = absoluteChanged || dynamicChanged;
                    exact = exact && dynamicExact;
                }
            }
        }

        std::vector<CVirtualGimbalEvent> events;
        buildEvents(camera, target, events);
        result.eventCount = static_cast<uint32_t>(events.size());
        result.exact = exact;

        if (events.empty())
        {
            if (absoluteChanged)
                result.status = SApplyResult::EStatus::AppliedAbsoluteOnly;
            else if (exact)
                result.status = SApplyResult::EStatus::AlreadySatisfied;
            return result;
        }

        if (camera->manipulate({ events.data(), events.size() }))
        {
            result.status = absoluteChanged ?
                SApplyResult::EStatus::AppliedAbsoluteAndVirtualEvents :
                SApplyResult::EStatus::AppliedVirtualEvents;
            return result;
        }

        if (absoluteChanged)
        {
            result.status = SApplyResult::EStatus::AppliedAbsoluteOnly;
            result.exact = false;
            return result;
        }

        result.issues |= SApplyResult::VirtualEventReplayFailed;
        result.status = SApplyResult::EStatus::Failed;
        result.exact = false;
        return result;
    }

    bool apply(ICamera* camera, const CCameraGoal& target) const
    {
        return applyDetailed(camera, target).succeeded();
    }

private:
    static constexpr double Pi = 3.14159265358979323846;
    static constexpr double HalfPi = 0.5 * Pi;

    struct SSphericalGoal
    {
        float64_t3 target = float64_t3(0.0);
        double u = 0.0;
        double v = 0.0;
        float distance = 0.f;
    };

    inline double wrapAngleRad(double angle) const
    {
        while (angle > Pi)
            angle -= 2.0 * Pi;
        while (angle < -Pi)
            angle += 2.0 * Pi;
        return angle;
    }

    inline bool nearlyEqual(double a, double b, double eps = 1e-6) const
    {
        return std::abs(a - b) <= eps;
    }

    inline uint32_t getRequiredGoalStateMask(const CCameraGoal& target) const
    {
        uint32_t mask = ICamera::GoalStateNone;
        if (target.hasTargetPosition || target.hasDistance || target.hasOrbitState)
            mask |= ICamera::GoalStateSphericalTarget;
        if (target.hasDynamicPerspectiveState)
            mask |= ICamera::GoalStateDynamicPerspective;
        if (target.hasPathState)
            mask |= ICamera::GoalStatePath;
        return mask;
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

    inline double getMoveMagnitudeDenominator(const ICamera* camera) const
    {
        const double moveScale = camera->getMoveSpeedScale();
        return 0.01 * (moveScale == 0.0 ? 1.0 : moveScale);
    }

    inline double getRotationMagnitudeDenominator(const ICamera* camera) const
    {
        const double rotationScale = camera->getRotationSpeedScale();
        return rotationScale == 0.0 ? 1.0 : rotationScale;
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
        const auto m = getMatrix3x3As4x4(matrix<float64_t, 3, 3>(glm::mat3_cast(delta)));
        const double yaw = std::atan2(static_cast<double>(m[2][0]), static_cast<double>(m[2][2]));
        const double c2 = std::sqrt(static_cast<double>(m[0][1] * m[0][1] + m[1][1] * m[1][1]));
        const double pitch = std::atan2(-static_cast<double>(m[2][1]), c2);
        const double s1 = std::sin(yaw);
        const double c1 = std::cos(yaw);
        const double roll = std::atan2(
            s1 * static_cast<double>(m[1][2]) - c1 * static_cast<double>(m[1][0]),
            c1 * static_cast<double>(m[0][0]) - s1 * static_cast<double>(m[0][2]));
        return float64_t3(pitch, yaw, roll);
    }

    inline bool computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const
    {
        outPositionDelta = 0.0;
        outRotationDeltaDeg = 0.0;
        if (!camera)
            return false;

        const auto& gimbal = camera->getGimbal();
        const auto currentPos = gimbal.getPosition();
        const auto currentOrientation = glm::normalize(gimbal.getOrientation());
        const auto targetOrientation = glm::normalize(target.orientation);

        const double dx = static_cast<double>(currentPos.x - target.position.x);
        const double dy = static_cast<double>(currentPos.y - target.position.y);
        const double dz = static_cast<double>(currentPos.z - target.position.z);
        outPositionDelta = std::sqrt(dx * dx + dy * dy + dz * dz);

        const double orientationDot = std::clamp(static_cast<double>(std::abs(glm::dot(currentOrientation, targetOrientation))), 0.0, 1.0);
        outRotationDeltaDeg = glm::degrees(2.0 * std::acos(orientationDot));
        return std::isfinite(outPositionDelta) && std::isfinite(outRotationDeltaDeg);
    }

    inline bool tryApplyAbsoluteReferencePose(ICamera* camera, const CCameraGoal& target, bool& outChanged, bool& outExact) const
    {
        outChanged = false;
        outExact = false;
        if (!camera)
            return false;

        switch (camera->getKind())
        {
            case ICamera::CameraKind::Free:
            case ICamera::CameraKind::FPS:
                break;
            default:
                return false;
        }

        double beforePosDelta = 0.0;
        double beforeRotDeltaDeg = 0.0;
        if (!computePoseMismatch(camera, target, beforePosDelta, beforeRotDeltaDeg))
            return false;

        if (beforePosDelta <= 1e-6 && beforeRotDeltaDeg <= 0.1)
        {
            outExact = true;
            return true;
        }

        auto targetFrame = getMatrix3x3As4x4(matrix<float64_t, 3, 3>(glm::mat3_cast(glm::normalize(target.orientation))));
        targetFrame[3] = float64_t4(target.position, 1.0);

        camera->manipulate({}, &targetFrame);

        double afterPosDelta = 0.0;
        double afterRotDeltaDeg = 0.0;
        if (!computePoseMismatch(camera, target, afterPosDelta, afterRotDeltaDeg))
            return false;

        outChanged = (std::abs(afterPosDelta - beforePosDelta) > 1e-9) || (std::abs(afterRotDeltaDeg - beforeRotDeltaDeg) > 1e-9);
        outExact = afterPosDelta <= 1e-6 && afterRotDeltaDeg <= 0.1;
        return true;
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

    inline bool resolveSphericalGoal(ICamera* camera, const CCameraGoal& target, const ICamera::SphericalTargetState& sphericalState, SSphericalGoal& outGoal) const
    {
        outGoal.target = target.hasTargetPosition ? target.targetPosition : sphericalState.target;
        outGoal.u = sphericalState.u;
        outGoal.v = sphericalState.v;
        outGoal.distance = sphericalState.distance;

        if (target.hasOrbitState)
        {
            outGoal.u = target.orbitU;
            outGoal.v = target.orbitV;
            outGoal.distance = target.orbitDistance;
        }
        else
        {
            if (!computeOrbitStateFromPositionTarget(target.position, outGoal.target, outGoal.u, outGoal.v, outGoal.distance, sphericalState.minDistance, sphericalState.maxDistance))
                return false;
        }

        if (target.hasDistance && !target.hasOrbitState)
            outGoal.distance = target.distance;

        outGoal.distance = std::clamp(outGoal.distance, sphericalState.minDistance, sphericalState.maxDistance);
        return true;
    }

    inline bool buildOrbitTranslateEvents(ICamera* camera, const ICamera::SphericalTargetState& sphericalState, const SSphericalGoal& goal, std::vector<CVirtualGimbalEvent>& out) const
    {
        const double moveDenom = getMoveMagnitudeDenominator(camera);
        appendSignedEvent(out, wrapAngleRad(goal.v - sphericalState.v) / moveDenom, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendSignedEvent(out, wrapAngleRad(goal.u - sphericalState.u) / moveDenom, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendSignedEvent(out, static_cast<double>(goal.distance - sphericalState.distance) / 0.01, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);
        return !out.empty();
    }

    inline bool buildRotateDistanceEvents(ICamera* camera, const ICamera::SphericalTargetState& sphericalState, const SSphericalGoal& goal,
        std::vector<CVirtualGimbalEvent>& out, bool allowYaw, bool allowPitch,
        CVirtualGimbalEvent::VirtualEventType distancePositive, CVirtualGimbalEvent::VirtualEventType distanceNegative) const
    {
        const double rotationDenom = getRotationMagnitudeDenominator(camera);
        if (allowYaw)
            appendSignedEvent(out, wrapAngleRad(goal.u - sphericalState.u) / rotationDenom, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
        if (allowPitch)
            appendSignedEvent(out, wrapAngleRad(goal.v - sphericalState.v) / rotationDenom, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
        if (distancePositive != CVirtualGimbalEvent::None && distanceNegative != CVirtualGimbalEvent::None)
            appendSignedEvent(out, static_cast<double>(goal.distance - sphericalState.distance) / 0.01, distancePositive, distanceNegative);
        return !out.empty();
    }

    inline bool buildPathEvents(ICamera* camera, const CCameraGoal& target, const ICamera::SphericalTargetState& sphericalState, std::vector<CVirtualGimbalEvent>& out) const
    {
        if (!camera)
            return false;

        const auto effectiveTarget = target.hasTargetPosition ? target.targetPosition : sphericalState.target;
        const auto currentOffset = camera->getGimbal().getPosition() - effectiveTarget;
        const auto desiredOffset = target.position - effectiveTarget;

        const double currentAngle = std::atan2(currentOffset.z, currentOffset.x);
        const double desiredAngle = std::atan2(desiredOffset.z, desiredOffset.x);
        const double currentRadius = std::sqrt(currentOffset.x * currentOffset.x + currentOffset.z * currentOffset.z);
        const double desiredRadius = std::sqrt(desiredOffset.x * desiredOffset.x + desiredOffset.z * desiredOffset.z);
        const double currentHeight = currentOffset.y;
        const double desiredHeight = desiredOffset.y;

        const double moveDenom = getMoveMagnitudeDenominator(camera);
        appendSignedEvent(out, (desiredRadius - currentRadius) / moveDenom, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendSignedEvent(out, (desiredHeight - currentHeight) / moveDenom, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendSignedEvent(out, wrapAngleRad(desiredAngle - currentAngle) / moveDenom, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);
        return !out.empty();
    }

    inline bool buildSphericalEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        ICamera::SphericalTargetState sphericalState;
        if (!camera || !camera->tryGetSphericalTargetState(sphericalState))
            return false;

        if (camera->getKind() == ICamera::CameraKind::Path)
            return buildPathEvents(camera, target, sphericalState, out);

        SSphericalGoal goal;
        if (!resolveSphericalGoal(camera, target, sphericalState, goal))
            return false;

        switch (camera->getKind())
        {
            case ICamera::CameraKind::Orbit:
            case ICamera::CameraKind::DollyZoom:
                return buildOrbitTranslateEvents(camera, sphericalState, goal, out);

            case ICamera::CameraKind::Turntable:
            case ICamera::CameraKind::Arcball:
                return buildRotateDistanceEvents(camera, sphericalState, goal, out, true, true, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

            case ICamera::CameraKind::TopDown:
                return buildRotateDistanceEvents(camera, sphericalState, goal, out, true, false, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

            case ICamera::CameraKind::Isometric:
                return buildRotateDistanceEvents(camera, sphericalState, goal, out, false, false, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

            case ICamera::CameraKind::Dolly:
                return buildRotateDistanceEvents(camera, sphericalState, goal, out, true, true, CVirtualGimbalEvent::None, CVirtualGimbalEvent::None);

            case ICamera::CameraKind::Chase:
                return buildRotateDistanceEvents(camera, sphericalState, goal, out, true, true, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);

            default:
                return buildOrbitTranslateEvents(camera, sphericalState, goal, out);
        }
    }

    inline bool buildFreeEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
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

        switch (camera->getKind())
        {
            case ICamera::CameraKind::FPS:
            {
                const auto [curPitch, curYaw] = computePitchYawFromOrientation(gimbal.getOrientation());
                const auto [tgtPitch, tgtYaw] = computePitchYawFromOrientation(target.orientation);

                const double rotScale = camera->getRotationSpeedScale();
                const double invScale = rotScale == 0.0 ? 1.0 : (1.0 / rotScale);

                const double deltaPitch = wrapAngleRad(tgtPitch - curPitch) * invScale;
                const double deltaYaw = wrapAngleRad(tgtYaw - curYaw) * invScale;

                appendSignedEvent(out, deltaPitch, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
                appendSignedEvent(out, deltaYaw, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
            } break;

            case ICamera::CameraKind::Free:
            {
                const auto deltaQuat = glm::inverse(gimbal.getOrientation()) * glm::normalize(target.orientation);
                const auto angles = extractYawPitchRollYXZ(deltaQuat);

                appendSignedEvent(out, angles.x, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
                appendSignedEvent(out, angles.y, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
                appendSignedEvent(out, angles.z, CVirtualGimbalEvent::RollRight, CVirtualGimbalEvent::RollLeft);
            } break;

            default:
                break;
        }

        return !out.empty();
    }
};

} // namespace nbl::hlsl

#endif // _C_CAMERA_GOAL_SOLVER_HPP_
