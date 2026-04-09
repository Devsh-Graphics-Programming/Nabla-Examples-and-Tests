#ifndef _C_CAMERA_GOAL_SOLVER_HPP_
#define _C_CAMERA_GOAL_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "CCameraGoal.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraVirtualEventUtilities.hpp"

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

        const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);

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
            out.orbitUv = sphericalState.orbitUv;
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

        out = CCameraGoalUtilities::canonicalizeGoal(out);
        return true;
    }

    SCaptureResult captureDetailed(ICamera* camera) const
    {
        SCaptureResult result;
        result.hasCamera = camera != nullptr;
        if (!result.hasCamera)
            return result;

        result.captured = capture(camera, result.goal);
        result.finiteGoal = result.captured && CCameraGoalUtilities::isGoalFinite(result.goal);
        return result;
    }

    SCompatibilityResult analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const
    {
        SCompatibilityResult result;
        if (!camera)
            return result;

        const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);
        result.sameKind = canonicalTarget.sourceKind == ICamera::CameraKind::Unknown || canonicalTarget.sourceKind == camera->getKind();
        result.supportedGoalStateMask = camera->getGoalStateMask();
        result.requiredGoalStateMask = CCameraGoalUtilities::getRequiredGoalStateMask(canonicalTarget);
        result.missingGoalStateMask = result.requiredGoalStateMask & ~result.supportedGoalStateMask;
        result.exact = result.missingGoalStateMask == ICamera::GoalStateNone;
        return result;
    }

    SApplyResult applyDetailed(ICamera* camera, const CCameraGoal& target) const
    {
        SApplyResult result;
        if (!camera)
            return result;

        const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);

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
                        exact = exact && hlsl::abs(static_cast<double>(afterState.distance - desiredDistance)) <= ICamera::ScalarTolerance;
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
                    const auto thresholds = SCameraPathDefaults::ComparisonThresholds;
                    const bool pathChanged = CCameraPathUtilities::pathStatesChanged(beforeState, afterState, thresholds);
                    const bool pathExact = CCameraPathUtilities::pathStatesNearlyEqual(afterState, canonicalTarget.pathState, thresholds);

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
    struct SGoalSolverDefaults final
    {
        static constexpr double UnitScale = 1.0;
        static inline const hlsl::float64_t3 UnitAxisDenominator = hlsl::float64_t3(UnitScale);
        static inline const hlsl::float64_t3 ScalarToleranceVec = hlsl::float64_t3(ICamera::ScalarTolerance);
        static inline const hlsl::float64_t3 AngularToleranceDegVec = hlsl::float64_t3(ICamera::DefaultAngularToleranceDeg);
    };

    inline void appendYawPitchRollEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const hlsl::float64_t3& eulerRadians,
        const double denominator,
        const bool includeRoll = true) const
    {
        static constexpr std::array<SCameraVirtualEventAxisBinding, 3u> RotationBindings = {{
            { CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown },
            { CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft },
            { CVirtualGimbalEvent::RollRight, CVirtualGimbalEvent::RollLeft }
        }};

        auto tolerances = SGoalSolverDefaults::AngularToleranceDegVec;
        if (!includeRoll)
            tolerances.z = std::numeric_limits<hlsl::float64_t>::infinity();

        appendAngularAxisEvents(
            events,
            eulerRadians,
            hlsl::float64_t3(denominator),
            tolerances,
            RotationBindings);
    }

    inline void appendPathDeltaEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraPathDelta& delta,
        const double moveDenominator) const
    {
        CCameraPathUtilities::appendPathAdvanceEvents(
            events,
            delta,
            moveDenominator,
            SCameraPathDefaults::ComparisonThresholds.angleToleranceDeg,
            SCameraPathDefaults::ComparisonThresholds.scalarTolerance);
    }

    inline double getMoveMagnitudeDenominator(const ICamera* camera) const
    {
        const double moveScale = camera->getMoveSpeedScale();
        return ICamera::VirtualTranslationStep * (moveScale == 0.0 ? SGoalSolverDefaults::UnitScale : moveScale);
    }

    inline double getRotationMagnitudeDenominator(const ICamera* camera) const
    {
        const double rotationScale = camera->getRotationSpeedScale();
        return rotationScale == 0.0 ? SGoalSolverDefaults::UnitScale : rotationScale;
    }

    inline bool computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const
    {
        outPositionDelta = 0.0;
        outRotationDeltaDeg = 0.0;
        if (!camera)
            return false;

        const auto& gimbal = camera->getGimbal();
        hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
        if (!hlsl::tryComputePoseDelta(gimbal.getPosition(), gimbal.getOrientation(), target.position, target.orientation, poseDelta))
            return false;

        outPositionDelta = poseDelta.position;
        outRotationDeltaDeg = poseDelta.rotationDeg;
        return true;
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

        const auto targetFrame = hlsl::composeTransformMatrix(target.position, target.orientation);

        camera->manipulate({}, &targetFrame);

        double afterPosDelta = 0.0;
        double afterRotDeltaDeg = 0.0;
        if (!computePoseMismatch(camera, target, afterPosDelta, afterRotDeltaDeg))
            return false;

        outChanged = !hlsl::isNearlyZeroScalar(afterPosDelta - beforePosDelta, static_cast<double>(ICamera::TinyScalarEpsilon)) ||
            !hlsl::isNearlyZeroScalar(afterRotDeltaDeg - beforeRotDeltaDeg, static_cast<double>(ICamera::TinyScalarEpsilon));
        outExact = afterPosDelta <= ICamera::DefaultPositionTolerance && afterRotDeltaDeg <= ICamera::DefaultAngularToleranceDeg;
        return true;
    }

    inline bool buildTargetRelativeEvents(
        ICamera* camera,
        const ICamera::SphericalTargetState& sphericalState,
        const SCameraTargetRelativeState& goal,
        std::vector<CVirtualGimbalEvent>& out,
        const SCameraTargetRelativeEventPolicy& policy) const
    {
        const auto delta = CCameraTargetRelativeUtilities::buildTargetRelativeDelta(sphericalState, goal);
        CCameraTargetRelativeUtilities::appendTargetRelativeDeltaEvents(
            out,
            delta,
            policy.translateOrbit ? getMoveMagnitudeDenominator(camera) : getRotationMagnitudeDenominator(camera),
            ICamera::DefaultAngularToleranceDeg,
            ICamera::VirtualTranslationStep,
            ICamera::ScalarTolerance,
            policy);
        return !out.empty();
    }

    inline bool buildPathEvents(ICamera* camera, const CCameraGoal& target, const ICamera::SphericalTargetState& sphericalState, std::vector<CVirtualGimbalEvent>& out) const
    {
        if (!camera)
            return false;

        const auto effectiveTarget = target.hasTargetPosition ? target.targetPosition : sphericalState.target;
        ICamera::PathState currentState = {};
        const ICamera::PathState* currentStateOverride = camera->tryGetPathState(currentState) ? &currentState : nullptr;
        SCameraPathStateTransition transition = {};
        if (!CCameraPathUtilities::tryBuildPathStateTransition(
                effectiveTarget,
                camera->getGimbal().getPosition(),
                target.position,
                SCameraPathDefaults::Limits,
                currentStateOverride,
                target.hasPathState ? &target.pathState : nullptr,
                transition))
        {
            return false;
        }

        const auto moveDenom = getMoveMagnitudeDenominator(camera);
        appendPathDeltaEvents(out, transition.delta, moveDenom);
        return !out.empty();
    }

    inline bool buildSphericalEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        ICamera::SphericalTargetState sphericalState;
        if (!camera || !camera->tryGetSphericalTargetState(sphericalState))
            return false;

        if (camera->getKind() == ICamera::CameraKind::Path)
            return buildPathEvents(camera, target, sphericalState, out);

        SCameraTargetRelativeState goal;
        if (!CCameraGoalUtilities::tryResolveCanonicalTargetRelativeState(target, sphericalState, goal))
            return false;

        switch (camera->getKind())
        {
            case ICamera::CameraKind::Orbit:
            case ICamera::CameraKind::DollyZoom:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::OrbitTranslatePolicy);

            case ICamera::CameraKind::Turntable:
            case ICamera::CameraKind::Arcball:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::RotateDistancePolicy);

            case ICamera::CameraKind::TopDown:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::TopDownPolicy);

            case ICamera::CameraKind::Isometric:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::IsometricPolicy);

            case ICamera::CameraKind::Dolly:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::DollyPolicy);

            case ICamera::CameraKind::Chase:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::ChasePolicy);

            default:
                return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::OrbitTranslatePolicy);
        }
    }

    inline bool buildFreeEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
    {
        const auto& gimbal = camera->getGimbal();
        const auto currentPos = gimbal.getPosition();
        const auto deltaWorld = target.position - currentPos;
        appendWorldTranslationAsLocalEvents(
            out,
            gimbal.getOrientation(),
            deltaWorld,
            SGoalSolverDefaults::UnitAxisDenominator,
            SGoalSolverDefaults::ScalarToleranceVec);

        switch (camera->getKind())
        {
            case ICamera::CameraKind::FPS:
            {
                const auto currentPitchYaw = hlsl::getPitchYawFromOrientation(gimbal.getOrientation());
                const auto targetPitchYaw = hlsl::getPitchYawFromOrientation(target.orientation);

                const double rotScale = camera->getRotationSpeedScale();
                const double invScale = rotScale == 0.0 ? SGoalSolverDefaults::UnitScale : (SGoalSolverDefaults::UnitScale / rotScale);

                appendYawPitchRollEvents(
                    out,
                    hlsl::float64_t3(
                        hlsl::wrapAngleRad(targetPitchYaw.x - currentPitchYaw.x) * invScale,
                        hlsl::wrapAngleRad(targetPitchYaw.y - currentPitchYaw.y) * invScale,
                        0.0),
                    SGoalSolverDefaults::UnitScale,
                    false);
            } break;

            case ICamera::CameraKind::Free:
            {
                appendYawPitchRollEvents(
                    out,
                    hlsl::getOrientationDeltaEulerRadiansYXZ(gimbal.getOrientation(), target.orientation),
                    SGoalSolverDefaults::UnitScale);
            } break;

            default:
                break;
        }

        return !out.empty();
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_GOAL_SOLVER_HPP_
