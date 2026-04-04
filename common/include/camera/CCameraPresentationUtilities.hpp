// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESENTATION_UTILITIES_HPP_
#define _C_CAMERA_PRESENTATION_UTILITIES_HPP_

#include <string>

#include "CCameraTextUtilities.hpp"

namespace nbl::hlsl
{

//! Shared exactness-oriented filter used by preset presentation surfaces.
enum class EPresetApplyPresentationFilter : uint8_t
{
    All,
    Exact,
    BestEffort
};

//! Presentation-ready wrapper around analyzed goal apply compatibility.
struct SCameraGoalApplyPresentation final : SCameraGoalApplyAnalysis
{
    std::string compatibilityLabel;
    std::string policyLabel;

    inline bool matchesFilter(const EPresetApplyPresentationFilter mode) const
    {
        switch (mode)
        {
            case EPresetApplyPresentationFilter::All:
                return true;
            case EPresetApplyPresentationFilter::Exact:
                return hasCamera && exact();
            case EPresetApplyPresentationFilter::BestEffort:
                return hasCamera && !exact();
            default:
                return true;
        }
    }
};

//! Presentation-ready wrapper around analyzed camera capture viability.
struct SCameraCapturePresentation final : SCameraCaptureAnalysis
{
    std::string policyLabel;
};

//! Build presentation text for one analyzed goal-apply result.
inline SCameraGoalApplyPresentation makeGoalApplyPresentation(const SCameraGoalApplyAnalysis& analysis, const ICamera* targetCamera)
{
    SCameraGoalApplyPresentation presentation;
    static_cast<SCameraGoalApplyAnalysis&>(presentation) = analysis;
    presentation.compatibilityLabel = describeGoalApplyCompatibility(analysis, targetCamera);
    presentation.policyLabel = describeGoalApplyPolicy(analysis);
    return presentation;
}

//! Analyze one preset against one camera and return reusable presentation data.
inline SCameraGoalApplyPresentation analyzePresetPresentation(const CCameraGoalSolver& solver, const ICamera* camera, const CCameraPreset& preset)
{
    return makeGoalApplyPresentation(analyzePresetApply(solver, camera, preset), camera);
}

//! Analyze one camera capture path and return reusable presentation data.
inline SCameraCapturePresentation analyzeCapturePresentation(const CCameraGoalSolver& solver, ICamera* camera)
{
    SCameraCapturePresentation presentation;
    static_cast<SCameraCaptureAnalysis&>(presentation) = analyzeCameraCapture(solver, camera);
    presentation.policyLabel = describeCameraCapturePolicy(presentation, camera);
    return presentation;
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_PRESENTATION_UTILITIES_HPP_
