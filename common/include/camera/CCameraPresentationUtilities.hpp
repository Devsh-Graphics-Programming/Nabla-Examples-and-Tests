// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESENTATION_UTILITIES_HPP_
#define _C_CAMERA_PRESENTATION_UTILITIES_HPP_

#include <string>

#include "CCameraTextUtilities.hpp"

namespace nbl::core
{

//! Shared exactness-oriented filter used by preset presentation surfaces.
enum class EPresetApplyPresentationFilter : uint8_t
{
    All,
    Exact,
    BestEffort
};

//! Shared badge/pill policy derived from one analyzed presentation answer.
struct SCameraGoalApplyPresentationBadges final
{
    bool exact = false;
    bool bestEffort = false;
    bool dropsState = false;
    bool sharedStateOnly = false;
    bool blocked = false;
};

//! Presentation-ready wrapper around analyzed goal apply compatibility.
struct SCameraGoalApplyPresentation final : SCameraGoalApplyAnalysis
{
    SCameraGoalApplyPresentationBadges badges;
    std::string sourceKindLabel;
    std::string goalStateLabel;
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

inline SCameraGoalApplyPresentationBadges collectGoalApplyPresentationBadges(const SCameraGoalApplyPresentation& presentation);

//! Shared user-facing label for the exactness filter selector.
inline const char* getPresetApplyPresentationFilterLabel(const EPresetApplyPresentationFilter mode)
{
    switch (mode)
    {
        case EPresetApplyPresentationFilter::All:
            return "All";
        case EPresetApplyPresentationFilter::Exact:
            return "Exact";
        case EPresetApplyPresentationFilter::BestEffort:
            return "Best-effort";
        default:
            return "All";
    }
}

//! Build presentation text for one analyzed goal-apply result.
inline SCameraGoalApplyPresentation makeGoalApplyPresentation(const SCameraGoalApplyAnalysis& analysis, const ICamera* targetCamera)
{
    SCameraGoalApplyPresentation presentation;
    static_cast<SCameraGoalApplyAnalysis&>(presentation) = analysis;
    presentation.badges = collectGoalApplyPresentationBadges(presentation);
    presentation.sourceKindLabel = std::string(getCameraTypeLabel(presentation.goal.sourceKind));
    presentation.goalStateLabel = describeGoalStateMask(presentation.goal.sourceGoalStateMask);
    presentation.compatibilityLabel = describeGoalApplyCompatibility(analysis, targetCamera);
    presentation.policyLabel = describeGoalApplyPolicy(analysis);
    return presentation;
}

//! Analyze one preset against one camera and return reusable presentation data.
inline SCameraGoalApplyPresentation analyzePresetPresentation(const CCameraGoalSolver& solver, const ICamera* camera, const CCameraPreset& preset)
{
    return makeGoalApplyPresentation(analyzePresetApply(solver, camera, preset), camera);
}

//! Build reusable badge flags for one preset/keyframe compatibility answer.
inline SCameraGoalApplyPresentationBadges collectGoalApplyPresentationBadges(const SCameraGoalApplyPresentation& presentation)
{
    SCameraGoalApplyPresentationBadges badges;
    badges.exact = presentation.exact();
    badges.bestEffort = presentation.hasCamera && !presentation.exact();
    badges.dropsState = presentation.dropsGoalState();
    badges.sharedStateOnly = presentation.usesSharedStateOnly();
    badges.blocked = !presentation.canApply;
    return badges;
}

//! Analyze one camera capture path and return reusable presentation data.
inline SCameraCapturePresentation analyzeCapturePresentation(const CCameraGoalSolver& solver, ICamera* camera)
{
    SCameraCapturePresentation presentation;
    static_cast<SCameraCaptureAnalysis&>(presentation) = analyzeCameraCapture(solver, camera);
    presentation.policyLabel = describeCameraCapturePolicy(presentation, camera);
    return presentation;
}

} // namespace nbl::core

#endif // _C_CAMERA_PRESENTATION_UTILITIES_HPP_
