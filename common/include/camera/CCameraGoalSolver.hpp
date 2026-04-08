#ifndef _C_CAMERA_GOAL_SOLVER_HPP_
#define _C_CAMERA_GOAL_SOLVER_HPP_

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "CCameraGoal.hpp"

namespace nbl::core
{

/**
* Best-effort absolute layer built on top of the event-only camera core.
*
* It captures typed camera state into `CCameraGoal`, analyzes compatibility,
* and tries to apply goals back to cameras using typed hooks and virtual-event replay.
*/
class CCameraGoalSolver
{
public:
    //! Detailed capture result for tooling code.
    struct SCaptureResult
    {
        bool hasCamera = false;
        bool captured = false;
        bool finiteGoal = false;
        CCameraGoal goal = {};

        inline bool canUseGoal() const
        {
            return hasCamera && captured && finiteGoal;
        }
    };

    //! Compatibility of a goal with a target camera kind and state mask.
    struct SCompatibilityResult
    {
        bool sameKind = false;
        bool exact = false;
        uint32_t requiredGoalStateMask = ICamera::GoalStateNone;
        uint32_t supportedGoalStateMask = ICamera::GoalStateNone;
        uint32_t missingGoalStateMask = ICamera::GoalStateNone;
    };

    //! Outcome of a best-effort goal apply attempt.
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

        const auto canonicalTarget = canonicalizeGoal(target);

        if (camera->hasCapability(ICamera::SphericalTarget))
            return buildSphericalEvents(camera, canonicalTarget, out);

        return buildFreeEvents(camera, canonicalTarget, out);
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

        out = canonicalizeGoal(out);
        return true;
    }

    SCaptureResult captureDetailed(ICamera* camera) const
    {
        SCaptureResult result;
        result.hasCamera = camera != nullptr;
        if (!result.hasCamera)
            return result;

        result.captured = capture(camera, result.goal);
        result.finiteGoal = result.captured && isGoalFinite(result.goal);
        return result;
    }

    SCompatibilityResult analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const
    {
        SCompatibilityResult result;
        if (!camera)
            return result;

        const auto canonicalTarget = canonicalizeGoal(target);
        result.sameKind = canonicalTarget.sourceKind == ICamera::CameraKind::Unknown || canonicalTarget.sourceKind == camera->getKind();
        result.supportedGoalStateMask = camera->getGoalStateMask();
        result.requiredGoalStateMask = getRequiredGoalStateMask(canonicalTarget);
        result.missingGoalStateMask = result.requiredGoalStateMask & ~result.supportedGoalStateMask;
        result.exact = result.missingGoalStateMask == ICamera::GoalStateNone;
        return result;
    }

    SApplyResult applyDetailed(ICamera* camera, const CCameraGoal& target) const
    {
        SApplyResult result;
        if (!camera)
            return result;

        const auto canonicalTarget = canonicalizeGoal(target);

        bool exact = true;
        bool absoluteChanged = false;

        if (!camera->hasCapability(ICamera::SphericalTarget))
        {
            bool poseChanged = false;
            bool poseExact = false;
            if (tryApplyAbsoluteReferencePose(camera, canonicalTarget, poseChanged, poseExact))
            {
                result.issues |= SApplyResult::UsedAbsolutePoseFallback;
                absoluteChanged = absoluteChanged || poseChanged;
                if (poseExact && !canonicalTarget.hasDynamicPerspectiveState)
                {
                    result.status = poseChanged ?
                        SApplyResult::EStatus::AppliedAbsoluteOnly :
                        SApplyResult::EStatus::AlreadySatisfied;
                    result.exact = true;
                    return result;
                }
            }
        }

        if (canonicalTarget.hasTargetPosition)
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
                if (!camera->trySetSphericalTarget(canonicalTarget.targetPosition))
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
                        exact = exact && afterState.target == canonicalTarget.targetPosition;
                    }
                }
            }
        }

        if (canonicalTarget.hasDistance || canonicalTarget.hasOrbitState)
        {
            ICamera::SphericalTargetState beforeState;
            if (!camera->tryGetSphericalTargetState(beforeState))
            {
                result.issues |= SApplyResult::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                const float desiredDistance = canonicalTarget.hasOrbitState ? canonicalTarget.orbitDistance : canonicalTarget.distance;
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
                        exact = exact && std::abs(static_cast<double>(afterState.distance - desiredDistance)) <= ICamera::ScalarTolerance;
                    }
                }
            }
        }

        if (canonicalTarget.hasPathState)
        {
            ICamera::PathState beforeState;
            if (!camera->tryGetPathState(beforeState))
            {
                result.issues |= SApplyResult::MissingPathState;
                exact = false;
            }
            else if (!camera->trySetPathState(canonicalTarget.pathState))
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
                    const bool pathChanged = !hlsl::nearlyEqualScalar(beforeState.angle, afterState.angle, static_cast<double>(ICamera::ScalarTolerance)) ||
                        !hlsl::nearlyEqualScalar(beforeState.radius, afterState.radius, static_cast<double>(ICamera::ScalarTolerance)) ||
                        !hlsl::nearlyEqualScalar(beforeState.height, afterState.height, static_cast<double>(ICamera::ScalarTolerance));
                    const bool pathExact = hlsl::nearlyEqualScalar(afterState.angle, canonicalTarget.pathState.angle, static_cast<double>(ICamera::ScalarTolerance)) &&
                        hlsl::nearlyEqualScalar(afterState.radius, canonicalTarget.pathState.radius, static_cast<double>(ICamera::ScalarTolerance)) &&
                        hlsl::nearlyEqualScalar(afterState.height, canonicalTarget.pathState.height, static_cast<double>(ICamera::ScalarTolerance));

                    absoluteChanged = absoluteChanged || pathChanged;
                    exact = exact && pathExact;
                }
            }
        }

        if (canonicalTarget.hasDynamicPerspectiveState)
        {
            ICamera::DynamicPerspectiveState beforeState;
            if (!camera->tryGetDynamicPerspectiveState(beforeState))
            {
                result.issues |= SApplyResult::MissingDynamicPerspectiveState;
                exact = false;
            }
            else if (!camera->trySetDynamicPerspectiveState(canonicalTarget.dynamicPerspectiveState))
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
                    const bool dynamicChanged = !hlsl::nearlyEqualScalar(beforeState.baseFov, afterState.baseFov, static_cast<float>(ICamera::ScalarTolerance)) ||
                        !hlsl::nearlyEqualScalar(beforeState.referenceDistance, afterState.referenceDistance, static_cast<float>(ICamera::ScalarTolerance));
                    const bool dynamicExact = hlsl::nearlyEqualScalar(afterState.baseFov, canonicalTarget.dynamicPerspectiveState.baseFov, static_cast<float>(ICamera::ScalarTolerance)) &&
                        hlsl::nearlyEqualScalar(afterState.referenceDistance, canonicalTarget.dynamicPerspectiveState.referenceDistance, static_cast<float>(ICamera::ScalarTolerance));

                    absoluteChanged = absoluteChanged || dynamicChanged;
                    exact = exact && dynamicExact;
                }
            }
        }

        std::vector<CVirtualGimbalEvent> events;
        buildEvents(camera, canonicalTarget, events);
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
    struct SSphericalGoal
    {
        hlsl::float64_t3 target = hlsl::float64_t3(0.0);
        double u = 0.0;
        double v = 0.0;
        float distance = 0.f;
    };

    inline void appendSignedEvent(std::vector<CVirtualGimbalEvent>& events, double value,
        CVirtualGimbalEvent::VirtualEventType positive, CVirtualGimbalEvent::VirtualEventType negative) const
    {
        if (!std::isfinite(value) || std::abs(value) <= ICamera::TinyScalarEpsilon)
            return;
        auto& ev = events.emplace_back();
        ev.type = (value > 0.0) ? positive : negative;
        ev.magnitude = std::abs(value);
    }

    inline void appendScalarDeltaEvent(std::vector<CVirtualGimbalEvent>& events, const double delta, const double denominator,
        const double tolerance, CVirtualGimbalEvent::VirtualEventType positive, CVirtualGimbalEvent::VirtualEventType negative) const
    {
        if (!std::isfinite(delta) || std::abs(delta) <= tolerance)
            return;
        appendSignedEvent(events, delta / denominator, positive, negative);
    }

    inline void appendAngularDeltaEvent(std::vector<CVirtualGimbalEvent>& events, const double deltaRadians, const double denominator,
        const double toleranceDeg, CVirtualGimbalEvent::VirtualEventType positive, CVirtualGimbalEvent::VirtualEventType negative) const
    {
        if (!std::isfinite(deltaRadians) || std::abs(hlsl::degrees(deltaRadians)) <= toleranceDeg)
            return;
        appendSignedEvent(events, deltaRadians / denominator, positive, negative);
    }

    inline double getMoveMagnitudeDenominator(const ICamera* camera) const
    {
        const double moveScale = camera->getMoveSpeedScale();
        return ICamera::VirtualTranslationStep * (moveScale == 0.0 ? 1.0 : moveScale);
    }

    inline double getRotationMagnitudeDenominator(const ICamera* camera) const
    {
        const double rotationScale = camera->getRotationSpeedScale();
        return rotationScale == 0.0 ? 1.0 : rotationScale;
    }

    inline std::pair<double, double> computePitchYawFromOrientation(const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation) const
    {
        const auto mat = hlsl::getQuaternionBasisMatrix(orientation);
        const auto pitchYaw = hlsl::getPitchYawFromForwardVector(hlsl::float64_t3(mat[2][0], mat[2][1], mat[2][2]));
        return { pitchYaw.x, pitchYaw.y };
    }

    inline hlsl::float64_t3 extractYawPitchRollYXZ(const hlsl::camera_quaternion_t<hlsl::float64_t>& delta) const
    {
        return hlsl::getQuaternionEulerRadiansYXZ(delta);
    }

    inline bool computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const
    {
        outPositionDelta = 0.0;
        outRotationDeltaDeg = 0.0;
        if (!camera)
            return false;

        const auto& gimbal = camera->getGimbal();
        const auto currentPos = gimbal.getPosition();
        const auto currentOrientation = hlsl::normalizeQuaternion(gimbal.getOrientation());
        const auto targetOrientation = hlsl::normalizeQuaternion(target.orientation);

        outPositionDelta = hlsl::length(currentPos - target.position);

        outRotationDeltaDeg = hlsl::getQuaternionAngularDistanceDegrees(currentOrientation, targetOrientation);
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

        if (beforePosDelta <= ICamera::DefaultPositionTolerance && beforeRotDeltaDeg <= ICamera::DefaultAngularToleranceDeg)
        {
            outExact = true;
            return true;
        }

        auto targetFrame = hlsl::getMatrix3x3As4x4(hlsl::getQuaternionBasisMatrix(target.orientation));
        targetFrame[3] = hlsl::float64_t4(target.position, 1.0);

        camera->manipulate({}, &targetFrame);

        double afterPosDelta = 0.0;
        double afterRotDeltaDeg = 0.0;
        if (!computePoseMismatch(camera, target, afterPosDelta, afterRotDeltaDeg))
            return false;

        outChanged = (std::abs(afterPosDelta - beforePosDelta) > ICamera::TinyScalarEpsilon) ||
            (std::abs(afterRotDeltaDeg - beforeRotDeltaDeg) > ICamera::TinyScalarEpsilon);
        outExact = afterPosDelta <= ICamera::DefaultPositionTolerance && afterRotDeltaDeg <= ICamera::DefaultAngularToleranceDeg;
        return true;
    }

    inline bool computeOrbitStateFromPositionTarget(const hlsl::float64_t3& position, const hlsl::float64_t3& target,
        double& outU, double& outV, float& outDistance, float minDistance, float maxDistance) const
    {
        hlsl::float64_t clampedDistance = static_cast<hlsl::float64_t>(outDistance);
        if (!hlsl::tryBuildOrbitFromPosition(
                target,
                position,
                static_cast<hlsl::float64_t>(minDistance),
                static_cast<hlsl::float64_t>(maxDistance),
                outU,
                outV,
                clampedDistance))
        {
            return false;
        }

        outDistance = static_cast<float>(clampedDistance);
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
        appendAngularDeltaEvent(out, hlsl::wrapAngleRad(goal.v - sphericalState.v), moveDenom, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendAngularDeltaEvent(out, hlsl::wrapAngleRad(goal.u - sphericalState.u), moveDenom, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendScalarDeltaEvent(out, static_cast<double>(goal.distance - sphericalState.distance), ICamera::VirtualTranslationStep, ICamera::ScalarTolerance, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);
        return !out.empty();
    }

    inline bool buildRotateDistanceEvents(ICamera* camera, const ICamera::SphericalTargetState& sphericalState, const SSphericalGoal& goal,
        std::vector<CVirtualGimbalEvent>& out, bool allowYaw, bool allowPitch,
        CVirtualGimbalEvent::VirtualEventType distancePositive, CVirtualGimbalEvent::VirtualEventType distanceNegative) const
    {
        const double rotationDenom = getRotationMagnitudeDenominator(camera);
        if (allowYaw)
            appendAngularDeltaEvent(out, hlsl::wrapAngleRad(goal.u - sphericalState.u), rotationDenom, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
        if (allowPitch)
            appendAngularDeltaEvent(out, hlsl::wrapAngleRad(goal.v - sphericalState.v), rotationDenom, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
        if (distancePositive != CVirtualGimbalEvent::None && distanceNegative != CVirtualGimbalEvent::None)
            appendScalarDeltaEvent(out, static_cast<double>(goal.distance - sphericalState.distance), ICamera::VirtualTranslationStep, ICamera::ScalarTolerance, distancePositive, distanceNegative);
        return !out.empty();
    }

    inline bool buildPathEvents(ICamera* camera, const CCameraGoal& target, const ICamera::SphericalTargetState& sphericalState, std::vector<CVirtualGimbalEvent>& out) const
    {
        if (!camera)
            return false;

        const auto effectiveTarget = target.hasTargetPosition ? target.targetPosition : sphericalState.target;
        double currentAngle = 0.0;
        double desiredAngle = 0.0;
        double currentRadius = 0.0;
        double desiredRadius = 0.0;
        double currentHeight = 0.0;
        double desiredHeight = 0.0;
        constexpr double MinPathRadius = static_cast<double>(ICamera::SphericalMinDistance);

        if (!hlsl::tryBuildPathStateFromPosition(effectiveTarget, camera->getGimbal().getPosition(), MinPathRadius, currentAngle, currentRadius, currentHeight))
            return false;
        if (!hlsl::tryBuildPathStateFromPosition(effectiveTarget, target.position, MinPathRadius, desiredAngle, desiredRadius, desiredHeight))
            return false;

        const double moveDenom = getMoveMagnitudeDenominator(camera);
        appendScalarDeltaEvent(out, desiredRadius - currentRadius, moveDenom, ICamera::ScalarTolerance, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendScalarDeltaEvent(out, desiredHeight - currentHeight, moveDenom, ICamera::ScalarTolerance, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendAngularDeltaEvent(out, hlsl::wrapAngleRad(desiredAngle - currentAngle), moveDenom, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);
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
        const hlsl::float64_t3 localDelta(
            hlsl::dot(deltaWorld, right),
            hlsl::dot(deltaWorld, up),
            hlsl::dot(deltaWorld, forward));

        appendScalarDeltaEvent(out, localDelta.x, 1.0, ICamera::ScalarTolerance, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
        appendScalarDeltaEvent(out, localDelta.y, 1.0, ICamera::ScalarTolerance, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
        appendScalarDeltaEvent(out, localDelta.z, 1.0, ICamera::ScalarTolerance, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

        switch (camera->getKind())
        {
            case ICamera::CameraKind::FPS:
            {
                const auto [curPitch, curYaw] = computePitchYawFromOrientation(gimbal.getOrientation());
                const auto [tgtPitch, tgtYaw] = computePitchYawFromOrientation(target.orientation);

                const double rotScale = camera->getRotationSpeedScale();
                const double invScale = rotScale == 0.0 ? 1.0 : (1.0 / rotScale);

                const double deltaPitch = hlsl::wrapAngleRad(tgtPitch - curPitch) * invScale;
                const double deltaYaw = hlsl::wrapAngleRad(tgtYaw - curYaw) * invScale;

                appendAngularDeltaEvent(out, deltaPitch, 1.0, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
                appendAngularDeltaEvent(out, deltaYaw, 1.0, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
            } break;

            case ICamera::CameraKind::Free:
            {
                const auto deltaQuat = hlsl::inverseQuaternion(gimbal.getOrientation()) * hlsl::normalizeQuaternion(target.orientation);
                const auto angles = extractYawPitchRollYXZ(deltaQuat);

                appendAngularDeltaEvent(out, angles.x, 1.0, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown);
                appendAngularDeltaEvent(out, angles.y, 1.0, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft);
                appendAngularDeltaEvent(out, angles.z, 1.0, ICamera::DefaultAngularToleranceDeg, CVirtualGimbalEvent::RollRight, CVirtualGimbalEvent::RollLeft);
            } break;

            default:
                break;
        }

        return !out.empty();
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_GOAL_SOLVER_HPP_
