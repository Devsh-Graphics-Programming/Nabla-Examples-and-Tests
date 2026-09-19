// TODO: moved out of nbl::ext::Cameras into this example pending a rework of the camera tooling layer
// (goal / preset / keyframe / playback / persistence / follow / scripted runtime). It sits at global scope
// like the example's other headers. See README.md in this folder.

#ifndef _C_CAMERA_GOAL_SOLVER_HPP_
#define _C_CAMERA_GOAL_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "CCameraGoal.hpp"
#include "SCameraToolingThresholds.hpp"
#include "nbl/ext/Cameras/SCameraControls.hpp"
#include "nbl/core/util/bitflag.h"
#include <limits>

using namespace nbl;
using namespace nbl::ext::cameras;

/// @brief Goal capture, compatibility analysis, and goal application helper.
///
/// The solver captures canonical state into `CCameraGoal`, compares a goal
/// against one target camera, applies typed fragments directly when the camera
/// exposes them, and builds one control frame for whatever a typed setter could
/// not reach, which it applies through `manipulate(...)`.
class CCameraGoalSolver
{
public:
    /// @brief Detailed result returned by one goal-capture attempt.
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

    /// @brief Compatibility of a goal with a target camera kind and state mask.
    struct SCompatibilityResult
    {
        bool sameKind = false;
        bool exact = false;
        ICamera::goal_state_flags_t requiredGoalStateMask = ICamera::GoalStateNone;
        ICamera::goal_state_flags_t supportedGoalStateMask = ICamera::GoalStateNone;
        ICamera::goal_state_flags_t missingGoalStateMask = ICamera::GoalStateNone;
    };

    /// @brief Outcome of one goal-application attempt.
    struct SApplyResult
    {
        enum class EStatus : uint8_t
        {
            Unsupported,
            Failed,
            AlreadySatisfied,
            AppliedAbsoluteOnly,
            AppliedControls,
            AppliedAbsoluteAndControls
        };

        enum class EIssue : uint32_t
        {
            NoIssue = 0u,
            UsedAbsolutePoseFallback = core::createBitmask({ 0 }),
            MissingSphericalTargetState = core::createBitmask({ 1 }),
            MissingPathState = core::createBitmask({ 2 }),
            MissingDynamicPerspectiveState = core::createBitmask({ 3 }),
            ControlFrameFailed = core::createBitmask({ 4 })
        };

        EStatus status = EStatus::Unsupported;
        bool exact = false;
        /// @brief `ECameraControlAxis` mask the control frame carried into `manipulate`.
        uint32_t appliedAxes = 0u;
        core::bitflag<EIssue> issues = EIssue::NoIssue;

        inline bool succeeded() const
        {
            return status != EStatus::Unsupported && status != EStatus::Failed;
        }

        inline bool changed() const
        {
            return status == EStatus::AppliedAbsoluteOnly ||
                status == EStatus::AppliedControls ||
                status == EStatus::AppliedAbsoluteAndControls;
        }

        inline bool approximate() const
        {
            return succeeded() && !exact;
        }

        inline bool hasIssue(EIssue issue) const
        {
            return issues.hasFlags(issue);
        }
    };

    /// @brief Fill one control frame with whatever the typed setters cannot reach, masked to what the rig accepts.
    bool buildControls(ICamera* camera, const CCameraGoal& target, SCameraControls& out) const;
    bool capture(ICamera* camera, CCameraGoal& out) const;
    SCaptureResult captureDetailed(ICamera* camera) const;
    SCompatibilityResult analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const;
    SApplyResult applyDetailed(ICamera* camera, const CCameraGoal& target) const;
    bool apply(ICamera* camera, const CCameraGoal& target) const;

private:
    /// @brief Zero one value inside its deadband, so a goal already met yields an empty frame.
    static hlsl::float64_t deadbandScalar(hlsl::float64_t value, hlsl::float64_t tolerance);
    /// @brief The same for an angle in radians, against a tolerance in degrees.
    static hlsl::float64_t deadbandAngle(hlsl::float64_t radians, hlsl::float64_t toleranceDeg);
    bool computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const;
    bool tryApplyAbsoluteReferencePose(ICamera* camera, const CCameraGoal& target, bool& outChanged, bool& outExact) const;
    bool buildTargetRelativeControls(
        const ICamera::SphericalTargetState& sphericalState,
        const STargetOrbit& goal,
        SCameraControls& out) const;
    bool buildPathControls(
        ICamera* camera,
        const CCameraGoal& target,
        const ICamera::SphericalTargetState& sphericalState,
        SCameraControls& out) const;
    bool buildSphericalControls(ICamera* camera, const CCameraGoal& target, SCameraControls& out) const;
    bool buildFreeControls(ICamera* camera, const CCameraGoal& target, SCameraControls& out) const;
};


inline hlsl::float64_t CCameraGoalSolver::deadbandScalar(const hlsl::float64_t value, const hlsl::float64_t tolerance)
{
    if (!std::isfinite(value) || hlsl::abs(value) <= tolerance)
        return 0.0;
    return value;
}

inline hlsl::float64_t CCameraGoalSolver::deadbandAngle(const hlsl::float64_t radians, const hlsl::float64_t toleranceDeg)
{
    return deadbandScalar(radians, hlsl::radians(toleranceDeg));
}

inline bool CCameraGoalSolver::buildControls(ICamera* camera, const CCameraGoal& target, SCameraControls& out) const
{
    out = {};
    if (!camera)
        return false;

    const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);
    const bool built = camera->hasCapability(ICamera::SphericalTarget) ?
        buildSphericalControls(camera, canonicalTarget, out) :
        buildFreeControls(camera, canonicalTarget, out);

    // a rig refuses a whole frame carrying an axis it does not accept, so drop the residue it cannot take
    out = out.masked(camera->getAcceptedControls());
    return built && out.nonZeroAxes() != 0u;
}

inline bool CCameraGoalSolver::capture(ICamera* camera, CCameraGoal& out) const
{
    out = {};
    if (!camera)
        return false;

    const CCameraGimbal& gimbal = camera->getGimbal();
    out.position = hlsl::float64_t3(gimbal.getPosition());
    out.orientation = gimbal.getOrientation();
    out.sourceKind = camera->getKind();
    out.sourceCapabilities = ICamera::capability_flags_t(camera->getCapabilities());
    out.sourceGoalStateMask = ICamera::goal_state_flags_t(camera->getGoalStateMask());

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

inline CCameraGoalSolver::SCaptureResult CCameraGoalSolver::captureDetailed(ICamera* camera) const
{
    SCaptureResult result;
    result.hasCamera = camera != nullptr;
    if (!result.hasCamera)
        return result;

    result.captured = capture(camera, result.goal);
    result.finiteGoal = result.captured && CCameraGoalUtilities::isGoalFinite(result.goal);
    return result;
}

inline CCameraGoalSolver::SCompatibilityResult CCameraGoalSolver::analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const
{
    SCompatibilityResult result;
    if (!camera)
        return result;

    const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);
    result.sameKind = canonicalTarget.sourceKind == ICamera::CameraKind::Unknown || canonicalTarget.sourceKind == camera->getKind();
    result.supportedGoalStateMask = ICamera::goal_state_flags_t(camera->getGoalStateMask());
    result.requiredGoalStateMask = CCameraGoalUtilities::getRequiredGoalStateMask(canonicalTarget);
    result.missingGoalStateMask = result.requiredGoalStateMask & ~result.supportedGoalStateMask;
    result.exact = result.missingGoalStateMask == ICamera::GoalStateNone;
    return result;
}

inline CCameraGoalSolver::SApplyResult CCameraGoalSolver::applyDetailed(ICamera* camera, const CCameraGoal& target) const
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
            result.issues |= SApplyResult::EIssue::UsedAbsolutePoseFallback;
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
            result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
            exact = false;
        }
        else
        {
            const auto beforeTarget = beforeState.target;
            if (!camera->trySetSphericalTarget(canonicalTarget.targetPosition))
            {
                result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                ICamera::SphericalTargetState afterState;
                if (!camera->tryGetSphericalTargetState(afterState))
                {
                    result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
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
            result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
            exact = false;
        }
        else
        {
            const float desiredDistance = canonicalTarget.hasOrbitState ? canonicalTarget.orbitDistance : canonicalTarget.distance;
            const float beforeDistance = beforeState.distance;
            if (!camera->trySetSphericalDistance(desiredDistance))
            {
                result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                ICamera::SphericalTargetState afterState;
                if (!camera->tryGetSphericalTargetState(afterState))
                {
                    result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                    exact = false;
                }
                else
                {
                    absoluteChanged = absoluteChanged || afterState.distance != beforeDistance;
                    exact = exact && hlsl::abs(static_cast<double>(afterState.distance - desiredDistance)) <= SCameraToolingThresholds::ScalarTolerance;
                }
            }
        }
    }

    if (canonicalTarget.hasPathState)
    {
        ICamera::PathState beforeState;
        if (!camera->tryGetPathState(beforeState))
        {
            result.issues |= SApplyResult::EIssue::MissingPathState;
            exact = false;
        }
        else if (!camera->trySetPathState(canonicalTarget.pathState))
        {
            result.issues |= SApplyResult::EIssue::MissingPathState;
            exact = false;
        }
        else
        {
            ICamera::PathState afterState;
            if (!camera->tryGetPathState(afterState))
            {
                result.issues |= SApplyResult::EIssue::MissingPathState;
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
            result.issues |= SApplyResult::EIssue::MissingDynamicPerspectiveState;
            exact = false;
        }
        else if (!camera->trySetDynamicPerspectiveState(canonicalTarget.dynamicPerspectiveState))
        {
            result.issues |= SApplyResult::EIssue::MissingDynamicPerspectiveState;
            exact = false;
        }
        else
        {
            ICamera::DynamicPerspectiveState afterState;
            if (!camera->tryGetDynamicPerspectiveState(afterState))
            {
                result.issues |= SApplyResult::EIssue::MissingDynamicPerspectiveState;
                exact = false;
            }
            else
            {
                const bool dynamicChanged = !CCameraMathUtilities::nearlyEqualScalar(beforeState.baseFov, afterState.baseFov, static_cast<float>(SCameraToolingThresholds::ScalarTolerance)) ||
                    !CCameraMathUtilities::nearlyEqualScalar(beforeState.referenceDistance, afterState.referenceDistance, static_cast<float>(SCameraToolingThresholds::ScalarTolerance));
                const bool dynamicExact = CCameraMathUtilities::nearlyEqualScalar(afterState.baseFov, canonicalTarget.dynamicPerspectiveState.baseFov, static_cast<float>(SCameraToolingThresholds::ScalarTolerance)) &&
                    CCameraMathUtilities::nearlyEqualScalar(afterState.referenceDistance, canonicalTarget.dynamicPerspectiveState.referenceDistance, static_cast<float>(SCameraToolingThresholds::ScalarTolerance));

                absoluteChanged = absoluteChanged || dynamicChanged;
                exact = exact && dynamicExact;
            }
        }
    }

    SCameraControls controls = {};
    buildControls(camera, canonicalTarget, controls);
    result.appliedAxes = controls.nonZeroAxes();
    result.exact = exact;

    // an empty frame means every axis landed inside its deadband, which is the goal already being met
    if (result.appliedAxes == 0u)
    {
        if (absoluteChanged)
            result.status = SApplyResult::EStatus::AppliedAbsoluteOnly;
        else if (exact)
            result.status = SApplyResult::EStatus::AlreadySatisfied;
        return result;
    }

    if (camera->manipulate(controls))
    {
        result.status = absoluteChanged ?
            SApplyResult::EStatus::AppliedAbsoluteAndControls :
            SApplyResult::EStatus::AppliedControls;
        return result;
    }

    if (absoluteChanged)
    {
        result.status = SApplyResult::EStatus::AppliedAbsoluteOnly;
        result.exact = false;
        return result;
    }

    result.issues |= SApplyResult::EIssue::ControlFrameFailed;
    result.status = SApplyResult::EStatus::Failed;
    result.exact = false;
    return result;
}

inline bool CCameraGoalSolver::apply(ICamera* camera, const CCameraGoal& target) const
{
    return applyDetailed(camera, target).succeeded();
}

inline bool CCameraGoalSolver::computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const
{
    outPositionDelta = 0.0;
    outRotationDeltaDeg = 0.0;
    if (!camera)
        return false;

    const CCameraGimbal& gimbal = camera->getGimbal();
    SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
    if (!tryComputePoseDelta<hlsl::float64_t>(gimbal.getPosition(), gimbal.getOrientation(), target.position, target.orientation, poseDelta))
        return false;

    outPositionDelta = poseDelta.position;
    outRotationDeltaDeg = poseDelta.rotationDeg;
    return true;
}

inline bool CCameraGoalSolver::tryApplyAbsoluteReferencePose(ICamera* camera, const CCameraGoal& target, bool& outChanged, bool& outExact) const
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

    if (beforePosDelta <= SCameraToolingThresholds::DefaultPositionTolerance && beforeRotDeltaDeg <= SCameraToolingThresholds::DefaultAngularToleranceDeg)
    {
        outExact = true;
        return true;
    }

    const auto targetFrame = CCameraMathUtilities::composeTransformMatrix(target.position, target.orientation);

    camera->setPose(targetFrame);

    double afterPosDelta = 0.0;
    double afterRotDeltaDeg = 0.0;
    if (!computePoseMismatch(camera, target, afterPosDelta, afterRotDeltaDeg))
        return false;

    outChanged = !CCameraMathUtilities::isNearlyZeroScalar(afterPosDelta - beforePosDelta, static_cast<double>(SCameraToolingThresholds::TinyScalarEpsilon)) ||
        !CCameraMathUtilities::isNearlyZeroScalar(afterRotDeltaDeg - beforeRotDeltaDeg, static_cast<double>(SCameraToolingThresholds::TinyScalarEpsilon));
    outExact = afterPosDelta <= SCameraToolingThresholds::DefaultPositionTolerance && afterRotDeltaDeg <= SCameraToolingThresholds::DefaultAngularToleranceDeg;
    return true;
}

inline bool CCameraGoalSolver::buildTargetRelativeControls(
    const ICamera::SphericalTargetState& sphericalState,
    const STargetOrbit& goal,
    SCameraControls& out) const
{
    // orbit angles are (yaw, pitch) while `rotate` is laid out as pitch in x, yaw in y
    const auto yawDelta = CCameraMathUtilities::wrapAngleRad(goal.angles.x - sphericalState.orbitUv.x);
    const auto pitchDelta = CCameraMathUtilities::wrapAngleRad(goal.angles.y - sphericalState.orbitUv.y);
    out.rotate.x = deadbandAngle(pitchDelta, SCameraToolingThresholds::DefaultAngularToleranceDeg);
    out.rotate.y = deadbandAngle(yawDelta, SCameraToolingThresholds::DefaultAngularToleranceDeg);
    out.distance = deadbandScalar(goal.distance - static_cast<hlsl::float64_t>(sphericalState.distance), SCameraToolingThresholds::ScalarTolerance);

    return out.nonZeroAxes() != 0u;
}

inline bool CCameraGoalSolver::buildPathControls(
    ICamera* camera,
    const CCameraGoal& target,
    const ICamera::SphericalTargetState& sphericalState,
    SCameraControls& out) const
{
    if (!camera)
        return false;

    const auto effectiveTarget = target.hasTargetPosition ? target.targetPosition : sphericalState.target;
    ICamera::PathState currentState = {};
    const ICamera::PathState* currentStateOverride = camera->tryGetPathState(currentState) ? &currentState : nullptr;
    ICamera::PathStateLimits pathLimits = CCameraPathUtilities::makeDefaultPathLimits();
    camera->tryGetPathStateLimits(pathLimits);
    SCameraPathStateTransition transition = {};
    if (!CCameraPathUtilities::tryBuildPathStateTransition(
            effectiveTarget,
            camera->getGimbal().getPosition(),
            target.position,
            pathLimits,
            currentStateOverride,
            target.hasPathState ? &target.pathState : nullptr,
            transition))
    {
        return false;
    }

    // the path model reads `s`, `u` and `v` as its own coordinates and `roll` as radians, so the delta goes
    // straight into the path axes
    const auto& delta = transition.delta;
    const auto scalarTolerance = SCameraPathDefaults::ExactComparisonThresholds.scalarTolerance;
    out.path.s = deadbandScalar(delta.s, scalarTolerance);
    out.path.u = deadbandScalar(delta.u, scalarTolerance);
    out.path.v = deadbandScalar(delta.v, scalarTolerance);
    out.path.roll = deadbandAngle(delta.roll, SCameraPathDefaults::ExactComparisonThresholds.rollToleranceDeg);

    return out.nonZeroAxes() != 0u;
}

inline bool CCameraGoalSolver::buildSphericalControls(ICamera* camera, const CCameraGoal& target, SCameraControls& out) const
{
    ICamera::SphericalTargetState sphericalState;
    if (!camera || !camera->tryGetSphericalTargetState(sphericalState))
        return false;

    if (camera->getKind() == ICamera::CameraKind::Path)
        return buildPathControls(camera, target, sphericalState, out);

    STargetOrbit goal;
    if (!CCameraGoalUtilities::tryResolveCanonicalTargetRelativeState(target, sphericalState, goal))
        return false;

    // every target-relative rig wants the same orbit delta; which parts of it survive is the rig's own business
    return buildTargetRelativeControls(sphericalState, goal, out);
}

inline bool CCameraGoalSolver::buildFreeControls(ICamera* camera, const CCameraGoal& target, SCameraControls& out) const
{
    const CCameraGimbal& gimbal = camera->getGimbal();

    // both rigs apply `translate` in their own frame, so the world-space position error is rotated into it
    const auto deltaWorld = target.position - gimbal.getPosition();
    const auto deltaLocal = CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame<hlsl::float64_t>(gimbal.getOrientation(), deltaWorld);
    out.translate.x = deadbandScalar(deltaLocal.x, SCameraToolingThresholds::ScalarTolerance);
    out.translate.y = deadbandScalar(deltaLocal.y, SCameraToolingThresholds::ScalarTolerance);
    out.translate.z = deadbandScalar(deltaLocal.z, SCameraToolingThresholds::ScalarTolerance);

    constexpr auto AngularToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    switch (camera->getKind())
    {
        case ICamera::CameraKind::FPS:
        {
            // an FPS rig holds roll at zero, so only the pitch and yaw difference is asked for
            const auto current = CCameraMathUtilities::getPitchYawRollRadians(gimbal.getOrientation());
            const auto wanted = CCameraMathUtilities::getPitchYawRollRadians(target.orientation);
            out.rotate.x = deadbandAngle(CCameraMathUtilities::wrapAngleRad<hlsl::float64_t>(wanted.x - current.x), AngularToleranceDeg);
            out.rotate.y = deadbandAngle(CCameraMathUtilities::wrapAngleRad<hlsl::float64_t>(wanted.y - current.y), AngularToleranceDeg);
        } break;

        case ICamera::CameraKind::Free:
        {
            const auto euler = CCameraMathUtilities::getPitchYawRollDeltaRadians<hlsl::float64_t>(gimbal.getOrientation(), target.orientation);
            out.rotate.x = deadbandAngle(euler.x, AngularToleranceDeg);
            out.rotate.y = deadbandAngle(euler.y, AngularToleranceDeg);
            out.rotate.z = deadbandAngle(euler.z, AngularToleranceDeg);
        } break;

        default:
            break;
    }

    return out.nonZeroAxes() != 0u;
}


#endif // _C_CAMERA_GOAL_SOLVER_HPP_

