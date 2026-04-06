#ifndef _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_

#include <bitset>

#include "nbl/examples/examples.hpp"

// common api
#include "camera/CFPSCamera.hpp"
#include "camera/CFreeLockCamera.hpp"
#include "camera/CSphericalTargetCamera.hpp"
#include "camera/COrbitCamera.hpp"
#include "camera/CArcballCamera.hpp"
#include "camera/CTurntableCamera.hpp"
#include "camera/CTopDownCamera.hpp"
#include "camera/CIsometricCamera.hpp"
#include "camera/CChaseCamera.hpp"
#include "camera/CDollyCamera.hpp"
#include "camera/CDollyZoomCamera.hpp"
#include "camera/CPathCamera.hpp"
#include "camera/CCameraPreset.hpp"
#include "camera/CCameraPresetFlow.hpp"
#include "camera/CCameraKeyframeTrack.hpp"
#include "camera/CCameraPlaybackTimeline.hpp"
#include "camera/CCameraPersistence.hpp"
#include "camera/CCameraSequenceScript.hpp"
#include "camera/CCameraGoalAnalysis.hpp"
#include "camera/CCameraGoalSolver.hpp"
#include "camera/CCameraManipulationUtilities.hpp"
#include "camera/CCameraPresentationUtilities.hpp"
#include "camera/CCameraProjectionUtilities.hpp"
#include "camera/CCameraFollowUtilities.hpp"
#include "camera/CCameraFollowRegressionUtilities.hpp"
#include "camera/CCameraTextUtilities.hpp"
#include "camera/CGimbalInputBinder.hpp"

#include "camera/CCubeProjection.hpp"
#include "camera/CLinearProjection.hpp"
#include "camera/CPlanarProjection.hpp"
// extensions
#include "nbl/ext/Frustum/CDrawFrustum.h"

// the example's headers
#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

namespace hlsl = nbl::hlsl;
using nbl::hlsl::ICamera;
using nbl::hlsl::CFPSCamera;
using nbl::hlsl::CFreeCamera;
using nbl::hlsl::CSphericalTargetCamera;
using nbl::hlsl::COrbitCamera;
using nbl::hlsl::CArcballCamera;
using nbl::hlsl::CTurntableCamera;
using nbl::hlsl::CTopDownCamera;
using nbl::hlsl::CIsometricCamera;
using nbl::hlsl::CChaseCamera;
using nbl::hlsl::CDollyCamera;
using nbl::hlsl::CDollyZoomCamera;
using nbl::hlsl::CPathCamera;
using nbl::hlsl::CCameraGoal;
using nbl::hlsl::CTrackedTarget;
using nbl::hlsl::CCameraPreset;
using nbl::hlsl::CCameraKeyframe;
using nbl::hlsl::CCameraKeyframeTrack;
using nbl::hlsl::CCameraPlaybackCursor;
using nbl::hlsl::CCameraSequenceScript;
using nbl::hlsl::CCameraSequenceSegment;
using nbl::hlsl::CCameraSequenceKeyframe;
using nbl::hlsl::CCameraSequenceTrackedTargetPose;
using nbl::hlsl::CCameraSequenceTrackedTargetTrack;
using nbl::hlsl::CCameraSequencePresentation;
using nbl::hlsl::CCameraSequenceContinuitySettings;
using nbl::hlsl::SCameraPlaybackAdvanceResult;
using nbl::hlsl::SCameraPresetApplySummary;
using nbl::hlsl::SCameraGoalApplyAnalysis;
using nbl::hlsl::SCameraCaptureAnalysis;
using nbl::hlsl::SCameraFollowConfig;
using nbl::hlsl::SCameraGoalApplyPresentation;
using nbl::hlsl::SCameraGoalApplyPresentationBadges;
using nbl::hlsl::SCameraCapturePresentation;
using nbl::hlsl::SCameraConstraintSettings;
using nbl::hlsl::CCameraGoalSolver;
using nbl::hlsl::CGimbalInputBinder;
using nbl::hlsl::ECameraFollowMode;
using nbl::hlsl::EPresetApplyPresentationFilter;
using nbl::hlsl::IGimbalBindingLayout;
using nbl::hlsl::IPlanarProjection;
using nbl::hlsl::CPlanarProjection;
using nbl::hlsl::CVirtualGimbalEvent;
using nbl::hlsl::float32_t;
using nbl::hlsl::float32_t2;
using nbl::hlsl::float32_t3;
using nbl::hlsl::float32_t4;
using nbl::hlsl::float32_t3x3;
using nbl::hlsl::float32_t3x4;
using nbl::hlsl::float32_t4x4;
using nbl::hlsl::float64_t;
using nbl::hlsl::float64_t3;
using nbl::hlsl::float64_t4x4;
using nbl::hlsl::uint16_t2;
using nbl::hlsl::describeApplyResult;
using nbl::hlsl::describeGoalStateMask;
using nbl::hlsl::getCameraFollowModeLabel;
using nbl::hlsl::getCameraTypeDescription;
using nbl::hlsl::getCameraTypeLabel;
using nbl::hlsl::getCastedMatrix;
using nbl::hlsl::getCastedVector;
using nbl::hlsl::getMatrix3x4As4x4;
using nbl::hlsl::cameraFollowModeUsesLocalOffset;
using nbl::hlsl::cameraFollowModeUsesWorldOffset;
using nbl::hlsl::concatenateBFollowedByA;
using nbl::hlsl::mul;
using nbl::hlsl::syncDynamicPerspectiveProjection;

#endif // _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_
