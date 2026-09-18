// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: moved out of nbl::ext::Cameras into this example pending a rework of the camera tooling layer
// (goal / preset / keyframe / playback / persistence / follow / scripted runtime). It sits at global scope
// like the example's other headers. See README.md in this folder.

#ifndef _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_
#define _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "CCameraFollowRegressionUtilities.hpp"
#include "CCameraScriptedRuntime.hpp"
#include "nbl/ext/Cameras/SCameraTypes.hpp"

using namespace nbl;
using namespace nbl::ext::cameras;

/// @brief Runtime state for authored scripted checks.
///
/// This state stores:
/// - the index of the next authored check to evaluate
/// - one baseline pose reference
/// - one step pose reference
struct CCameraScriptedCheckRuntimeState
{
    struct SPoseReference final : SCameraRigPose
    {
        bool valid = false;
    };

    size_t nextCheckIndex = 0u;
    SPoseReference baseline = {};
    SPoseReference step = {};
};

/// @brief Shared per-frame evaluation context for authored scripted checks.
struct CCameraScriptedCheckContext
{
    uint64_t frame = 0ull;
    ICamera* camera = nullptr;
    /// @brief What the scripted gizmo deltas of this frame amounted to, or null when the frame had none.
    const SCameraControls* imguizmoControls = nullptr;
    const CTrackedTarget* trackedTarget = nullptr;
    const SCameraFollowConfig* followConfig = nullptr;
    const SCameraProjectionContext* followProjectionContext = nullptr;
    const CCameraGoalSolver* goalSolver = nullptr;
};

/// @brief Reusable log entry produced by scripted check evaluation.
struct CCameraScriptedCheckLogEntry
{
    bool failure = false;
    std::string text;
};

/// @brief Result for one frame worth of scripted checks.
struct CCameraScriptedCheckFrameResult
{
    std::vector<CCameraScriptedCheckLogEntry> logs;
    bool hadFailures = false;
};

struct CCameraScriptedCheckRunnerUtilities final
{
    static void scriptedCheckSetStepReference(
        CCameraScriptedCheckRuntimeState& state,
        const hlsl::float64_t3& position,
        const hlsl::math::quaternion<hlsl::float64_t>& orientation);
    static void scriptedCheckSetBaselineReference(
        CCameraScriptedCheckRuntimeState& state,
        const hlsl::float64_t3& position,
        const hlsl::math::quaternion<hlsl::float64_t>& orientation);
    static bool scriptedCheckComputePoseDelta(
        const hlsl::float64_t3& currentPosition,
        const hlsl::math::quaternion<hlsl::float64_t>& currentOrientation,
        const hlsl::float64_t3& referencePosition,
        const hlsl::math::quaternion<hlsl::float64_t>& referenceOrientation,
        SCameraPoseDelta<hlsl::float64_t>& outDelta);

    template<typename Fn>
    static inline std::string buildScriptedCheckMessage(Fn&& formatter)
    {
        std::ostringstream oss;
        formatter(oss);
        return oss.str();
    }

    static void appendScriptedCheckLog(
        CCameraScriptedCheckFrameResult& result,
        bool failure,
        std::string&& text);

    /// @brief Evaluate all authored scripted checks scheduled for the current frame.
    static CCameraScriptedCheckFrameResult evaluateScriptedChecksForFrame(
        const std::vector<CCameraScriptedInputCheck>& checks,
        CCameraScriptedCheckRuntimeState& state,
        const CCameraScriptedCheckContext& context);
};


inline void CCameraScriptedCheckRunnerUtilities::scriptedCheckSetStepReference(
    CCameraScriptedCheckRuntimeState& state,
    const hlsl::float64_t3& position,
    const hlsl::math::quaternion<hlsl::float64_t>& orientation)
{
    state.step.valid = true;
    state.step.position = position;
    state.step.orientation = hlsl::normalize(orientation);
}

inline void CCameraScriptedCheckRunnerUtilities::scriptedCheckSetBaselineReference(
    CCameraScriptedCheckRuntimeState& state,
    const hlsl::float64_t3& position,
    const hlsl::math::quaternion<hlsl::float64_t>& orientation)
{
    state.baseline.valid = true;
    state.baseline.position = position;
    state.baseline.orientation = hlsl::normalize(orientation);
    scriptedCheckSetStepReference(state, position, orientation);
}

inline bool CCameraScriptedCheckRunnerUtilities::scriptedCheckComputePoseDelta(
    const hlsl::float64_t3& currentPosition,
    const hlsl::math::quaternion<hlsl::float64_t>& currentOrientation,
    const hlsl::float64_t3& referencePosition,
    const hlsl::math::quaternion<hlsl::float64_t>& referenceOrientation,
    SCameraPoseDelta<hlsl::float64_t>& outDelta)
{
    return CCameraMathUtilities::tryComputePoseDelta(
        currentPosition,
        currentOrientation,
        referencePosition,
        referenceOrientation,
        outDelta);
}

inline void CCameraScriptedCheckRunnerUtilities::appendScriptedCheckLog(
    CCameraScriptedCheckFrameResult& result,
    const bool failure,
    std::string&& text)
{
    result.logs.push_back({
        .failure = failure,
        .text = std::move(text)
    });
    result.hadFailures = result.hadFailures || failure;
}

inline CCameraScriptedCheckFrameResult CCameraScriptedCheckRunnerUtilities::evaluateScriptedChecksForFrame(
    const std::vector<CCameraScriptedInputCheck>& checks,
    CCameraScriptedCheckRuntimeState& state,
    const CCameraScriptedCheckContext& context)
{
    CCameraScriptedCheckFrameResult result = {};

    while (state.nextCheckIndex < checks.size() && checks[state.nextCheckIndex].frame == context.frame)
    {
        const auto& check = checks[state.nextCheckIndex];

        if (!context.camera)
        {
            appendScriptedCheckLog(
                result,
                true,
                buildScriptedCheckMessage([&](std::ostringstream& oss)
                {
                    oss << "[script][fail] check frame=" << context.frame << " no active camera";
                }));
            ++state.nextCheckIndex;
            continue;
        }

        const auto& gimbal = context.camera->getGimbal();
        const auto pos = gimbal.getPosition();
        const auto orientation = hlsl::normalize(gimbal.getOrientation());
        const auto eulerDeg = hlsl::_static_cast<hlsl::float32_t3>(CCameraMathUtilities::getCameraOrientationEulerDegrees(orientation));

        if (!CCameraMathUtilities::isFiniteVec3(pos) || !CCameraMathUtilities::isFiniteQuaternion(orientation) || !CCameraMathUtilities::isFiniteVec3(eulerDeg))
        {
            appendScriptedCheckLog(
                result,
                true,
                buildScriptedCheckMessage([&](std::ostringstream& oss)
                {
                    oss << "[script][fail] check frame=" << context.frame << " non-finite gimbal state";
                }));
            ++state.nextCheckIndex;
            continue;
        }

        switch (check.kind)
        {
            case CCameraScriptedInputCheck::Kind::Baseline:
            {
                scriptedCheckSetBaselineReference(state, pos, orientation);
                appendScriptedCheckLog(
                    result,
                    false,
                    buildScriptedCheckMessage([&](std::ostringstream& oss)
                    {
                        oss << std::fixed << std::setprecision(3);
                        oss << "[script][pass] baseline frame=" << context.frame
                            << " pos=(" << pos.x << ", " << pos.y << ", " << pos.z << ")"
                            << " euler_deg=(" << eulerDeg.x << ", " << eulerDeg.y << ", " << eulerDeg.z << ")";
                    }));
                break;
            }
            case CCameraScriptedInputCheck::Kind::ImguizmoVirtual:
            {
                bool ok = true;
                if (!context.imguizmoControls)
                {
                    ok = false;
                }
                else
                {
                    for (const auto& expected : check.expectedVirtualEvents)
                    {
                        const double actual = context.imguizmoControls->axis(expected.axis);
                        if (hlsl::abs(actual - expected.value) > check.tolerance)
                        {
                            ok = false;
                            appendScriptedCheckLog(
                                result,
                                true,
                                buildScriptedCheckMessage([&](std::ostringstream& oss)
                                {
                                    oss << std::fixed << std::setprecision(6);
                                    oss << "[script][fail] imguizmo_virtual frame=" << context.frame
                                        << " axis=" << cameraControlAxisName(expected.axis)
                                        << " expected=" << expected.value
                                        << " actual=" << actual
                                        << " tol=" << check.tolerance;
                                }));
                        }
                    }
                }

                if (ok)
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][pass] imguizmo_virtual frame=" << context.frame
                                << " axes=" << check.expectedVirtualEvents.size();
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalNear:
            {
                bool ok = true;
                if (check.hasExpectedPos)
                {
                    const double distance = hlsl::length(pos - hlsl::_static_cast<hlsl::float64_t3>(check.expectedPos));
                    if (distance > check.posTolerance)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_near frame=" << context.frame
                                    << " pos_diff=" << distance
                                    << " tol=" << check.posTolerance;
                            }));
                    }
                }
                if (check.hasExpectedEuler)
                {
                    const auto expectedOrientation = CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(
                        hlsl::_static_cast<hlsl::float64_t3>(check.expectedEulerDeg));
                    SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
                    if (!scriptedCheckComputePoseDelta(pos, orientation, pos, expectedOrientation, poseDelta))
                        poseDelta.rotationDeg = std::numeric_limits<hlsl::float64_t>::infinity();
                    const auto rotationDeltaDeg = poseDelta.rotationDeg;
                    if (rotationDeltaDeg > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_near frame=" << context.frame
                                    << " rot_delta_deg=" << rotationDeltaDeg
                                    << " tol=" << check.eulerToleranceDeg;
                            }));
                    }
                }

                if (ok)
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][pass] gimbal_near frame=" << context.frame;
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalDelta:
            {
                if (!state.baseline.valid)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] gimbal_delta frame=" << context.frame << " missing baseline";
                        }));
                    break;
                }

                SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
                if (!scriptedCheckComputePoseDelta(pos, orientation, state.baseline.position, state.baseline.orientation, poseDelta))
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] gimbal_delta frame=" << context.frame << " non-finite pose delta";
                        }));
                    break;
                }

                if (poseDelta.position > check.posTolerance || poseDelta.rotationDeg > check.eulerToleranceDeg)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][fail] gimbal_delta frame=" << context.frame
                                << " pos_diff=" << poseDelta.position
                                << " tol=" << check.posTolerance
                                << " rot_delta_deg=" << poseDelta.rotationDeg
                                << " tol=" << check.eulerToleranceDeg;
                        }));
                }
                else
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][pass] gimbal_delta frame=" << context.frame
                                << " pos_diff=" << poseDelta.position
                                << " rot_delta_deg=" << poseDelta.rotationDeg;
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalStep:
            {
                if (!state.step.valid)
                {
                    if (state.baseline.valid)
                    {
                        scriptedCheckSetStepReference(state, state.baseline.position, state.baseline.orientation);
                    }
                    else
                    {
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << "[script][fail] gimbal_step frame=" << context.frame << " missing step reference";
                            }));
                        scriptedCheckSetStepReference(state, pos, orientation);
                        ++state.nextCheckIndex;
                        continue;
                    }
                }

                SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
                if (!scriptedCheckComputePoseDelta(pos, orientation, state.step.position, state.step.orientation, poseDelta))
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] gimbal_step frame=" << context.frame << " non-finite pose delta";
                        }));
                    scriptedCheckSetStepReference(state, pos, orientation);
                    break;
                }

                bool ok = true;
                bool requiresProgress = false;
                bool hasProgress = false;
                if (check.hasPosDeltaConstraint)
                {
                    if (poseDelta.position > check.posTolerance)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " pos_delta=" << poseDelta.position
                                    << " max=" << check.posTolerance;
                            }));
                    }
                    if (check.minPosDelta > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || poseDelta.position >= check.minPosDelta;
                    }
                }
                if (check.hasEulerDeltaConstraint)
                {
                    if (poseDelta.rotationDeg > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " rot_delta_deg=" << poseDelta.rotationDeg
                                    << " max=" << check.eulerToleranceDeg;
                            }));
                    }
                    if (check.minEulerDeltaDeg > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || poseDelta.rotationDeg >= check.minEulerDeltaDeg;
                    }
                }
                if (requiresProgress && !hasProgress)
                {
                    ok = false;
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][fail] gimbal_step frame=" << context.frame
                                << " missing progress pos_delta=" << poseDelta.position
                                << " rot_delta_deg=" << poseDelta.rotationDeg;
                        }));
                }

                if (ok)
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][pass] gimbal_step frame=" << context.frame
                                << " pos_delta=" << poseDelta.position
                                << " rot_delta_deg=" << poseDelta.rotationDeg;
                        }));
                }
                scriptedCheckSetStepReference(state, pos, orientation);
                break;
            }
            case CCameraScriptedInputCheck::Kind::FollowTargetLock:
            {
                if (!context.followConfig)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << " missing follow config";
                        }));
                    break;
                }
                if (!context.trackedTarget)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << " missing tracked target";
                        }));
                    break;
                }
                if (!context.goalSolver)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << " missing goal solver";
                        }));
                    break;
                }

                SCameraFollowRegressionResult regression = {};
                std::string regressionError;
                CCameraGoal expectedFollowGoal = {};
                const auto thresholds = CCameraFollowRegressionUtilities::makeFollowRegressionThresholds(check.posTolerance, check.eulerToleranceDeg);
                const bool ok = CCameraFollowUtilities::tryBuildFollowGoal(
                        *context.goalSolver,
                        context.camera,
                        *context.trackedTarget,
                        *context.followConfig,
                        expectedFollowGoal) &&
                    CCameraFollowRegressionUtilities::validateFollowTargetContract(
                        context.camera,
                        *context.trackedTarget,
                        *context.followConfig,
                        expectedFollowGoal,
                        regression,
                        &regressionError,
                        context.followProjectionContext,
                        thresholds);

                if (!ok)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << ' '
                                << (regressionError.empty() ? "follow validation mismatch" : regressionError);
                        }));
                }
                else
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][pass] follow_lock frame=" << context.frame
                                << " angle_deg=" << regression.lockAngleDeg
                                << " target_distance=" << regression.targetDistance
                                << " screen_ndc=" << regression.projectedTarget.radius;
                        }));
                }
                break;
            }
        }

        ++state.nextCheckIndex;
    }

    return result;
}


#endif // _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_
