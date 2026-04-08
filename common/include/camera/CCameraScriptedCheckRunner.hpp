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
    size_t nextCheckIndex = 0u;
    bool baselineValid = false;
    hlsl::float32_t3 baselinePos = hlsl::float32_t3(0.f);
    hlsl::float32_t3 baselineEulerDeg = hlsl::float32_t3(0.f);
    bool stepValid = false;
    hlsl::float32_t3 stepPos = hlsl::float32_t3(0.f);
    hlsl::float32_t3 stepEulerDeg = hlsl::float32_t3(0.f);
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
    const hlsl::float32_t3& pos,
    const hlsl::float32_t3& eulerDeg)
{
    state.stepValid = true;
    state.stepPos = pos;
    state.stepEulerDeg = eulerDeg;
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
        const auto eulerDeg = hlsl::getCastedVector<hlsl::float32_t>(hlsl::getQuaternionEulerDegrees(gimbal.getOrientation()));

        if (!hlsl::isFiniteVec3(pos) || !hlsl::isFiniteVec3(eulerDeg))
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
                state.baselineValid = true;
                state.baselinePos = pos;
                state.baselineEulerDeg = eulerDeg;
                scriptedCheckSetStepReference(state, pos, eulerDeg);
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
                    const auto deltaEulerDeg = hlsl::getWrappedEulerDistanceDegrees(eulerDeg, check.expectedEulerDeg);
                    const auto dx = deltaEulerDeg.x;
                    const auto dy = deltaEulerDeg.y;
                    const auto dz = deltaEulerDeg.z;
                    const auto maxAngle = std::max(dx, std::max(dy, dz));
                    if (maxAngle > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_near frame=" << context.frame
                                    << " euler_diff=" << maxAngle
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
                if (!state.baselineValid)
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

                const hlsl::float64_t3 diff = pos - hlsl::getCastedVector<hlsl::float64_t>(state.baselinePos);
                const double dpos = hlsl::length(diff);
                const auto deltaEulerDeg = hlsl::getWrappedEulerDistanceDegrees(eulerDeg, state.baselineEulerDeg);
                const auto dx = deltaEulerDeg.x;
                const auto dy = deltaEulerDeg.y;
                const auto dz = deltaEulerDeg.z;
                const auto dmax = std::max(dx, std::max(dy, dz));

                if (dpos > check.posTolerance || dmax > check.eulerToleranceDeg)
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
                                << " euler_diff=" << dmax
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
                                << " euler_diff=" << dmax;
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalStep:
            {
                if (!state.stepValid)
                {
                    if (state.baselineValid)
                    {
                        scriptedCheckSetStepReference(state, state.baselinePos, state.baselineEulerDeg);
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
                        scriptedCheckSetStepReference(state, pos, eulerDeg);
                        ++state.nextCheckIndex;
                        continue;
                    }
                }

                const hlsl::float64_t3 diff = pos - hlsl::getCastedVector<hlsl::float64_t>(state.stepPos);
                const double dpos = hlsl::length(diff);
                const auto deltaEulerDeg = hlsl::getWrappedEulerDistanceDegrees(eulerDeg, state.stepEulerDeg);
                const auto dx = deltaEulerDeg.x;
                const auto dy = deltaEulerDeg.y;
                const auto dz = deltaEulerDeg.z;
                const auto dmax = std::max(dx, std::max(dy, dz));

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
                    if (dmax > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " euler_delta=" << dmax
                                    << " max=" << check.eulerToleranceDeg;
                            }));
                    }
                    if (check.minEulerDeltaDeg > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || dmax >= check.minEulerDeltaDeg;
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
                                << " euler_delta=" << dmax;
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
                                << " euler_delta=" << dmax;
                        }));
                }
                scriptedCheckSetStepReference(state, pos, eulerDeg);
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
