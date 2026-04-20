// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_CAMERAS_C_CAMERA_SIMPLE_FPS_UTILITIES_HPP_INCLUDED_
#define _NBL_EXAMPLES_CAMERAS_C_CAMERA_SIMPLE_FPS_UTILITIES_HPP_INCLUDED_

#include <cstdint>
#include <vector>

#include "nbl/ext/Cameras/CCameraInputBindingUtilities.hpp"
#include "nbl/ext/Cameras/CCameraMathUtilities.hpp"
#include "nbl/ext/Cameras/CFPSCamera.hpp"
#include "nbl/ext/Cameras/CGimbalInputBinder.hpp"

namespace nbl::examples
{

/// @brief Small example-side helpers for the most basic mouse+keyboard FPS usage.
///
/// This helper exists only to reduce repeated boilerplate in simple examples.
/// It does not wrap or replace the `ext/Cameras` stack.
/// Advanced paths such as ImGuizmo-driven edits, world-space gizmo translation,
/// goal solving, scripted playback, follow runtime, or non-FPS camera families
/// should keep using the full camera API directly.
struct CCameraSimpleFPSUtilities final
{
	enum class EMouseLookMode : uint8_t
	{
		HoldButton,
		AlwaysActive
	};

	/// @brief Mutable runtime state for the basic mouse+keyboard FPS input path.
	///
	/// This groups the FPS input binder and the small amount of state needed for
	/// button-gated mouse look.
	/// The binder is referenced through a non-owning pointer so examples keep full
	/// ownership of binding setup and rebinding policy.
	struct SBasicInputRuntime final
	{
		ui::CGimbalInputBinder* binder = nullptr;
		bool lookActive = false;
	};

	/// @brief Read-only configuration for the basic mouse+keyboard FPS input path.
	struct SBasicInputConfig final
	{
		EMouseLookMode lookMode = EMouseLookMode::HoldButton;
		ui::E_MOUSE_BUTTON lookButton = ui::EMB_LEFT_BUTTON;
	};

	/// @brief Example-facing FPS speed knobs matching the removed legacy wrapper.
	///
	/// `moveSpeed` and `rotationSpeed` are the same user-level values that the
	/// old example camera wrapper exposed. They are mapped to camera-local scales
	/// with the exact same numerical factors as before, so existing example
	/// values preserve the same motion semantics.
	struct SSpeedSettings final
	{
		double moveSpeed = 1.0;
		double rotationSpeed = 1.0;
	};

	/// @brief Create a normal `CFPSCamera` from a look-at pair and simple motion scales.
	///
	/// Returns `nullptr` when the look-at orientation cannot be resolved from the
	/// provided position, target, and preferred up-vector.
	static inline core::smart_refctd_ptr<core::CFPSCamera> createFromLookAt(
		const hlsl::float64_t3& position,
		const hlsl::float64_t3& target,
		const double moveSpeedScale,
		const double rotationSpeedScale,
		const hlsl::float64_t3& preferredUp = hlsl::float64_t3(0.0, 1.0, 0.0))
	{
		hlsl::camera_quaternion_t<double> orientation;
		if (!hlsl::CCameraMathUtilities::tryBuildLookAtOrientation(position, target, preferredUp, orientation))
			return nullptr;

		auto camera = core::make_smart_refctd_ptr<core::CFPSCamera>(position, orientation);
		if (!camera)
			return nullptr;
		camera->setMoveSpeedScale(moveSpeedScale);
		camera->setRotationSpeedScale(rotationSpeedScale);
		return camera;
	}

	/// @brief Apply example-facing speed settings using the exact old wrapper mapping.
	static inline void applySpeedSettings(
		core::CFPSCamera& camera,
		const SSpeedSettings& speedSettings)
	{
		camera.setMoveSpeedScale(toMoveSpeedScale(speedSettings.moveSpeed));
		camera.setRotationSpeedScale(toRotationSpeedScale(speedSettings.rotationSpeed));
	}

	/// @brief Create a normal `CFPSCamera` from a look-at pair using the same speed values as the old wrapper.
	static inline core::smart_refctd_ptr<core::CFPSCamera> createFromLookAt(
		const hlsl::float64_t3& position,
		const hlsl::float64_t3& target,
		const SSpeedSettings& speedSettings,
		const hlsl::float64_t3& preferredUp = hlsl::float64_t3(0.0, 1.0, 0.0))
	{
		return createFromLookAt(
			position,
			target,
			toMoveSpeedScale(speedSettings.moveSpeed),
			toRotationSpeedScale(speedSettings.rotationSpeed),
			preferredUp);
	}

	/// @brief Collect virtual FPS events from already-consumed mouse and keyboard events.
	///
	/// This path covers only the common `mouse + keyboard + button-gated look`
	/// setup used by simple examples.
	/// The caller keeps ownership of channel consumption and may reuse the same
	/// raw event spans for other example-local logic before or after this helper.
	static inline std::vector<core::CVirtualGimbalEvent> collectBasicVirtualEvents(
		const std::span<const ui::SMouseEvent> mouseEvents,
		const std::span<const ui::SKeyboardEvent> keyboardEvents,
		const std::chrono::microseconds nextPresentationTimestamp,
		SBasicInputRuntime& runtime,
		const SBasicInputConfig& config = {})
	{
		if (!runtime.binder)
			return {};

		std::vector<ui::SMouseEvent> cameraMouseEvents;

		for (const auto& event : mouseEvents)
		{
			if (config.lookMode == EMouseLookMode::HoldButton &&
				event.type == ui::SMouseEvent::EET_CLICK &&
				event.clickEvent.mouseButton == config.lookButton)
			{
				if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
					runtime.lookActive = true;
				else if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
					runtime.lookActive = false;
			}

			const bool allowLook = (config.lookMode == EMouseLookMode::AlwaysActive) || runtime.lookActive;
			if (event.type == ui::SMouseEvent::EET_MOVEMENT && allowLook)
				cameraMouseEvents.push_back(event);
		}

		auto collected = runtime.binder->collectVirtualEvents(
			nextPresentationTimestamp,
			{
				.keyboardEvents = keyboardEvents,
				.mouseEvents = { cameraMouseEvents.data(), cameraMouseEvents.size() }
			});

		return std::move(collected.events);
	}

private:
	// These are the exact same factors that the removed legacy example camera wrapper used.
	static inline constexpr double MoveSpeedScaleMultiplier = 2.0;
	static inline constexpr double RotationSpeedScaleMultiplier = 0.003;

	static inline double toMoveSpeedScale(const double moveSpeed)
	{
		return moveSpeed * MoveSpeedScaleMultiplier;
	}

	static inline double toRotationSpeedScale(const double rotationSpeed)
	{
		return rotationSpeed * RotationSpeedScaleMultiplier;
	}
};

} // namespace nbl::examples

#endif // _NBL_EXAMPLES_CAMERAS_C_CAMERA_SIMPLE_FPS_UTILITIES_HPP_INCLUDED_
