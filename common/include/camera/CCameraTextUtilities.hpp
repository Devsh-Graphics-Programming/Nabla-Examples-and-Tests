// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_TEXT_UTILITIES_HPP_
#define _C_CAMERA_TEXT_UTILITIES_HPP_

#include <sstream>
#include <string>
#include <string_view>

#include "CCameraGoalSolver.hpp"

namespace nbl::hlsl
{

//! Return a short human-readable label for a camera kind.
inline std::string_view getCameraTypeLabel(const ICamera::CameraKind kind)
{
	switch (kind)
	{
		case ICamera::CameraKind::FPS: return "FPS";
		case ICamera::CameraKind::Free: return "Free";
		case ICamera::CameraKind::Orbit: return "Orbit";
		case ICamera::CameraKind::Arcball: return "Arcball";
		case ICamera::CameraKind::Turntable: return "Turntable";
		case ICamera::CameraKind::TopDown: return "TopDown";
		case ICamera::CameraKind::Isometric: return "Isometric";
		case ICamera::CameraKind::Chase: return "Chase";
		case ICamera::CameraKind::Dolly: return "Dolly";
		case ICamera::CameraKind::DollyZoom: return "Dolly Zoom";
		case ICamera::CameraKind::Path: return "Path";
		default: return "Unknown";
	}
}

//! Return a short human-readable label for a concrete camera instance.
inline std::string_view getCameraTypeLabel(const ICamera* camera)
{
	return camera ? getCameraTypeLabel(camera->getKind()) : "Unknown";
}

//! Return a short human-readable description for a camera kind.
inline std::string_view getCameraTypeDescription(const ICamera::CameraKind kind)
{
	switch (kind)
	{
		case ICamera::CameraKind::FPS: return "First-person WASD + mouse look";
		case ICamera::CameraKind::Free: return "Free-fly 6DOF with full rotation";
		case ICamera::CameraKind::Orbit: return "Orbit around target with dolly";
		case ICamera::CameraKind::Arcball: return "Arcball trackball around target";
		case ICamera::CameraKind::Turntable: return "Turntable yaw/pitch around target";
		case ICamera::CameraKind::TopDown: return "Fixed pitch top-down pan";
		case ICamera::CameraKind::Isometric: return "Fixed isometric view with pan";
		case ICamera::CameraKind::Chase: return "Target follow with chase controls";
		case ICamera::CameraKind::Dolly: return "Rig truck/dolly with look-at";
		case ICamera::CameraKind::DollyZoom: return "Orbit with dolly-zoom FOV";
		case ICamera::CameraKind::Path: return "Move along a target path";
		default: return "Unspecified camera behavior";
	}
}

//! Return a short human-readable description for a concrete camera instance.
inline std::string_view getCameraTypeDescription(const ICamera* camera)
{
	return camera ? getCameraTypeDescription(camera->getKind()) : "Unspecified camera behavior";
}

//! Describe the typed goal-state mask in a stable human-readable format.
inline std::string describeGoalStateMask(const uint32_t mask)
{
	if (mask == ICamera::GoalStateNone)
		return "Pose only";

	std::string out;
	auto append = [&](const char* label, const uint32_t bit) -> void
	{
		if ((mask & bit) != bit)
			return;
		if (!out.empty())
			out += ", ";
		out += label;
	};

	append("Spherical target", ICamera::GoalStateSphericalTarget);
	append("Dynamic perspective", ICamera::GoalStateDynamicPerspective);
	append("Path", ICamera::GoalStatePath);
	return out;
}

//! Describe a detailed goal-apply result for logs, smoke tests, and UI summaries.
inline std::string describeApplyResult(const CCameraGoalSolver::SApplyResult& result)
{
	std::ostringstream oss;
	oss << "status=";
	switch (result.status)
	{
		case CCameraGoalSolver::SApplyResult::EStatus::Unsupported: oss << "Unsupported"; break;
		case CCameraGoalSolver::SApplyResult::EStatus::Failed: oss << "Failed"; break;
		case CCameraGoalSolver::SApplyResult::EStatus::AlreadySatisfied: oss << "AlreadySatisfied"; break;
		case CCameraGoalSolver::SApplyResult::EStatus::AppliedAbsoluteOnly: oss << "AppliedAbsoluteOnly"; break;
		case CCameraGoalSolver::SApplyResult::EStatus::AppliedVirtualEvents: oss << "AppliedVirtualEvents"; break;
		case CCameraGoalSolver::SApplyResult::EStatus::AppliedAbsoluteAndVirtualEvents: oss << "AppliedAbsoluteAndVirtualEvents"; break;
	}
	oss << " exact=" << (result.exact ? "true" : "false")
		<< " events=" << result.eventCount;

	if (result.issues != CCameraGoalSolver::SApplyResult::NoIssue)
	{
		oss << " issues=";
		bool first = true;
		auto appendIssue = [&](const char* label, const CCameraGoalSolver::SApplyResult::EIssue issue) -> void
		{
			if (!result.hasIssue(issue))
				return;
			if (!first)
				oss << ",";
			oss << label;
			first = false;
		};

		appendIssue("absolute_pose_fallback", CCameraGoalSolver::SApplyResult::UsedAbsolutePoseFallback);
		appendIssue("missing_spherical_state", CCameraGoalSolver::SApplyResult::MissingSphericalTargetState);
		appendIssue("missing_path_state", CCameraGoalSolver::SApplyResult::MissingPathState);
		appendIssue("missing_dynamic_perspective_state", CCameraGoalSolver::SApplyResult::MissingDynamicPerspectiveState);
		appendIssue("virtual_event_replay_failed", CCameraGoalSolver::SApplyResult::VirtualEventReplayFailed);
	}

	return oss.str();
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_TEXT_UTILITIES_HPP_
