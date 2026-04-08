// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_
#define _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "CCameraFollowRegressionUtilities.hpp"
#include "CCameraScriptedRuntime.hpp"

namespace nbl::system
{

/**
* Runtime state for authored scripted checks.
*
* The state is intentionally small:
*
* - authored check data stays in `CCameraScriptedInputCheck`
* - the runner only remembers where the next check starts
* - baseline and step references are maintained here so consumers do not have to
*   keep duplicating the same bookkeeping
*/
struct CCameraScriptedCheckRuntimeState
{
    struct SPoseReference
    {
        bool valid = false;
        hlsl::float64_t3 position = hlsl::float64_t3(0.0);
        hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
    };

    size_t nextCheckIndex = 0u;
    SPoseReference baseline = {};
    SPoseReference step = {};
};

//! Shared per-frame evaluation context for authored scripted checks.
struct CCameraScriptedCheckContext
{
    uint64_t frame = 0ull;
    core::ICamera* camera = nullptr;
    const core::CVirtualGimbalEvent* imguizmoVirtual = nullptr;
    uint32_t imguizmoVirtualCount = 0u;
    const core::CTrackedTarget* trackedTarget = nullptr;
    const core::SCameraFollowConfig* followConfig = nullptr;
    const hlsl::float32_t4x4* followViewProjMatrix = nullptr;
    const core::CCameraGoalSolver* goalSolver = nullptr;
};

//! Reusable log entry produced by scripted check evaluation.
struct CCameraScriptedCheckLogEntry
{
    bool failure = false;
    std::string text;
};

//! Result for one frame worth of scripted checks.
struct CCameraScriptedCheckFrameResult
{
    std::vector<CCameraScriptedCheckLogEntry> logs;
    bool hadFailures = false;
};

inline void scriptedCheckSetStepReference(
    CCameraScriptedCheckRuntimeState& state,
    const hlsl::float64_t3& position,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
{
    state.step.valid = true;
    state.step.position = position;
    state.step.orientation = hlsl::normalizeQuaternion(orientation);
}

inline void scriptedCheckSetBaselineReference(
    CCameraScriptedCheckRuntimeState& state,
    const hlsl::float64_t3& position,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
{
    state.baseline.valid = true;
    state.baseline.position = position;
    state.baseline.orientation = hlsl::normalizeQuaternion(orientation);
    scriptedCheckSetStepReference(state, position, orientation);
}

inline float scriptedCheckComputeRotationDeltaDegrees(
    const hlsl::camera_quaternion_t<hlsl::float64_t>& currentOrientation,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& referenceOrientation)
{
    return static_cast<float>(hlsl::getQuaternionAngularDistanceDegrees(
        hlsl::normalizeQuaternion(currentOrientation),
        hlsl::normalizeQuaternion(referenceOrientation)));
}

template<typename Fn>
inline std::string buildScriptedCheckMessage(Fn&& formatter)
{
    std::ostringstream oss;
    formatter(oss);
    return oss.str();
}

inline void appendScriptedCheckLog(
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

//! Evaluate all authored scripted checks scheduled for the current frame.
inline CCameraScriptedCheckFrameResult evaluateScriptedChecksForFrame(
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
        const auto orientation = hlsl::normalizeQuaternion(gimbal.getOrientation());
        const auto eulerDeg = hlsl::getCastedVector<hlsl::float32_t>(hlsl::getQuaternionEulerDegrees(orientation));

        if (!hlsl::isFiniteVec3(pos) || !hlsl::isFiniteQuaternion(orientation) || !hlsl::isFiniteVec3(eulerDeg))
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
                if (!context.imguizmoVirtual || context.imguizmoVirtualCount == 0u)
                {
                    ok = false;
                }
                else
                {
                    for (const auto& expected : check.expectedVirtualEvents)
                    {
                        bool found = false;
                        double actual = 0.0;
                        for (uint32_t i = 0u; i < context.imguizmoVirtualCount; ++i)
                        {
                            if (context.imguizmoVirtual[i].type == expected.type)
                            {
                                found = true;
                                actual = context.imguizmoVirtual[i].magnitude;
                                break;
                            }
                        }

                        if (!found || std::abs(actual - expected.magnitude) > check.tolerance)
                        {
                            ok = false;
                            appendScriptedCheckLog(
                                result,
                                true,
                                buildScriptedCheckMessage([&](std::ostringstream& oss)
                                {
                                    oss << std::fixed << std::setprecision(6);
                                    oss << "[script][fail] imguizmo_virtual frame=" << context.frame
                                        << " type=" << core::CVirtualGimbalEvent::virtualEventToString(expected.type).data()
                                        << " expected=" << expected.magnitude
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
                                << " events=" << check.expectedVirtualEvents.size();
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalNear:
            {
                bool ok = true;
                if (check.hasExpectedPos)
                {
                    const hlsl::float64_t3 diff = pos - hlsl::getCastedVector<hlsl::float64_t>(check.expectedPos);
                    const double distance = hlsl::length(diff);
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
                    const auto expectedOrientation = hlsl::makeQuaternionFromEulerDegrees(
                        hlsl::getCastedVector<hlsl::float64_t>(check.expectedEulerDeg));
                    const auto rotationDeltaDeg = scriptedCheckComputeRotationDeltaDegrees(orientation, expectedOrientation);
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

                const hlsl::float64_t3 diff = pos - state.baseline.position;
                const double dpos = hlsl::length(diff);
                const auto rotationDeltaDeg = scriptedCheckComputeRotationDeltaDegrees(orientation, state.baseline.orientation);

                if (dpos > check.posTolerance || rotationDeltaDeg > check.eulerToleranceDeg)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][fail] gimbal_delta frame=" << context.frame
                                << " pos_diff=" << dpos
                                << " tol=" << check.posTolerance
                                << " rot_delta_deg=" << rotationDeltaDeg
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
                                << " pos_diff=" << dpos
                                << " rot_delta_deg=" << rotationDeltaDeg;
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

                const hlsl::float64_t3 diff = pos - state.step.position;
                const double dpos = hlsl::length(diff);
                const auto rotationDeltaDeg = scriptedCheckComputeRotationDeltaDegrees(orientation, state.step.orientation);

                bool ok = true;
                bool requiresProgress = false;
                bool hasProgress = false;
                if (check.hasPosDeltaConstraint)
                {
                    if (dpos > check.posTolerance)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " pos_delta=" << dpos
                                    << " max=" << check.posTolerance;
                            }));
                    }
                    if (check.minPosDelta > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || dpos >= check.minPosDelta;
                    }
                }
                if (check.hasEulerDeltaConstraint)
                {
                    if (rotationDeltaDeg > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " rot_delta_deg=" << rotationDeltaDeg
                                    << " max=" << check.eulerToleranceDeg;
                            }));
                    }
                    if (check.minEulerDeltaDeg > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || rotationDeltaDeg >= check.minEulerDeltaDeg;
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
                                << " missing progress pos_delta=" << dpos
                                << " rot_delta_deg=" << rotationDeltaDeg;
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
                                << " pos_delta=" << dpos
                                << " rot_delta_deg=" << rotationDeltaDeg;
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
                core::CCameraGoal expectedFollowGoal = {};
                SCameraFollowRegressionThresholds thresholds = {};
                thresholds.lockAngleToleranceDeg = check.eulerToleranceDeg;
                thresholds.projectedNdcTolerance = check.posTolerance;
                const bool ok = core::tryBuildFollowGoal(
                        *context.goalSolver,
                        context.camera,
                        *context.trackedTarget,
                        *context.followConfig,
                        expectedFollowGoal) &&
                    validateFollowTargetContract(
                        context.camera,
                        *context.trackedTarget,
                        *context.followConfig,
                        expectedFollowGoal,
                        regression,
                        &regressionError,
                        context.followViewProjMatrix,
                        thresholds);

                if (!ok)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << ' '
                                << (regressionError.empty() ? "follow contract mismatch" : regressionError);
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
                                << " screen_ndc=" << regression.projectedNdcRadius;
                        }));
                }
                break;
            }
        }

        ++state.nextCheckIndex;
    }

    return result;
}

} // namespace nbl::system

#endif // _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_
