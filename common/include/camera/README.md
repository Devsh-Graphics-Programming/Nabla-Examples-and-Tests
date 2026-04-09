# Shared Camera API

This directory contains the reusable camera stack used by [`61_UI`](../../../61_UI/README.md).
It is the source of truth for camera semantics, typed camera state, scripted playback, follow behavior, and camera-focused validation.

## Scope

This stack is:

- a reusable camera runtime based on semantic virtual events
- a typed tooling layer for capture, restore, presets, playback, follow, and scripted validation
- a shared authoring and CI surface reused by `61_UI`

This stack is not:

- the engine-wide Nabla scene graph API
- a generic animation system for arbitrary scene objects
- a setter-heavy runtime camera API
- a `61_UI`-local convenience layer

## Design pillars

The current design is built around five rules.

### 1. Runtime is event-driven

The hot path stays on:

```text
input -> virtual events -> ICamera::manipulate(...)
```

Camera models do not expose arbitrary absolute runtime setters for position, target, yaw/pitch/roll, or similar mutable bags of state.

### 2. Tooling is typed-state driven

Capture, presets, restore, tracks, and validation use typed sidecar state:

- [`CCameraGoal.hpp`](CCameraGoal.hpp)
- [`CCameraPreset.hpp`](CCameraPreset.hpp)
- [`CCameraKeyframeTrack.hpp`](CCameraKeyframeTrack.hpp)
- [`CCameraPlaybackTimeline.hpp`](CCameraPlaybackTimeline.hpp)
- [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp)

That layer is allowed to reason in terms of canonical camera state because it is not the hot input path.

### 3. Input mapping is separate from camera semantics

The reusable stack keeps these responsibilities separate:

- physical input mapping
- virtual event processing
- camera semantics
- absolute state capture / restore
- scripted playback and validation

This keeps camera models reusable across keyboard, mouse, ImGuizmo, CI, and headless tools.

### 4. Projection-local state is allowed, runtime input processing is not

Projection types may own viewport-local binding layouts, but raw input processing stays in the input layer.

### 5. Follow and scripts stay above camera models

Follow and scripted playback are shared layers built on top of camera semantics.
They are not hardwired into `ICamera`.

## Namespace split

The stack is intentionally spread across existing Nabla namespaces.

- `nbl::hlsl`
  camera math, transform math, pose deltas, interpolation helpers, and reusable vector/quaternion types
- `nbl::core`
  runtime camera model, typed goal state, presets, tracks, follow, playback, path-rig helpers, and authored sequence data
- `nbl::ui`
  input binding layouts, input processors, runtime binders, default input presets, and presentation-facing UI helpers
- `nbl::system`
  persistence, scripted runtime parsing, scripted timeline building, scripted check execution, and follow validation

The shared camera math is written against `nbl::hlsl`.
Consumers are expected to use those `nbl::hlsl` types and helpers directly rather than duplicating local math wrappers.

## Runtime pipeline

The hot runtime path is:

```text
raw input
  -> IGimbalBindingLayout
  -> IGimbalInputProcessor / CGimbalInputBinder
  -> CVirtualGimbalEvent[]
  -> ICamera::manipulate(...)
  -> updated camera gimbal and cached view matrix
```

The main building blocks are:

- [`CVirtualGimbalEvent.hpp`](CVirtualGimbalEvent.hpp)
  shared semantic command language
- [`IGimbal.hpp`](IGimbal.hpp)
  gimbal math, event accumulation, and world-space pose handling
- [`ICamera.hpp`](ICamera.hpp)
  camera runtime interface and typed tooling hooks
- [`SCameraRigPose.hpp`](SCameraRigPose.hpp)
  shared typed pose transport used outside the hot runtime path

## Input layer

The input layer is split into three parts.

### Binding layout

- [`IGimbalBindingLayout.hpp`](IGimbalBindingLayout.hpp)

This layer stores static mappings from physical inputs to semantic virtual events.
It does not process runtime input.

### Input processing

- [`IGimbalInputProcessor.hpp`](IGimbalInputProcessor.hpp)
- [`CGimbalInputBinder.hpp`](CGimbalInputBinder.hpp)

`IGimbalInputProcessor` converts keyboard, mouse, and ImGuizmo input into virtual events.
`CGimbalInputBinder` is the convenience runtime wrapper that collects one frame of virtual events together with per-domain counts.

### Shared presets

- [`CCameraInputBindingUtilities.hpp`](CCameraInputBindingUtilities.hpp)
- [`CCameraScriptedUiInputUtilities.hpp`](CCameraScriptedUiInputUtilities.hpp)

These helpers provide shared default input presets for camera kinds and the scripted/UI-facing glue that reuses the same semantic event language.

## Core camera interface

The main runtime interface is [`ICamera.hpp`](ICamera.hpp).

Important properties:

- runtime entry point is `manipulate(std::span<const CVirtualGimbalEvent>, const hlsl::float64_t4x4*)`
- cameras own their own `CGimbal`
- motion scaling stays camera-local through `SMotionConfig`
- typed state hooks are optional and exist for tooling, not for direct runtime driving

`ICamera` currently exposes these shared typed states:

- `SphericalTargetState`
- `DynamicPerspectiveState`
- `PathState`
- `PathStateLimits`

The capability and state-mask surface is:

- `CameraKind`
- `CameraCapability`
- `GoalStateMask`

## Camera families

### Free cameras

- [`CFPSCamera.hpp`](CFPSCamera.hpp)
- [`CFreeLockCamera.hpp`](CFreeLockCamera.hpp)

These are pose-driven cameras without spherical target semantics.

### Spherical-target family

- [`CSphericalTargetCamera.hpp`](CSphericalTargetCamera.hpp)
- [`COrbitCamera.hpp`](COrbitCamera.hpp)
- [`CArcballCamera.hpp`](CArcballCamera.hpp)
- [`CTurntableCamera.hpp`](CTurntableCamera.hpp)
- [`CTopDownCamera.hpp`](CTopDownCamera.hpp)
- [`CIsometricCamera.hpp`](CIsometricCamera.hpp)
- [`CChaseCamera.hpp`](CChaseCamera.hpp)
- [`CDollyCamera.hpp`](CDollyCamera.hpp)

These cameras share:

- target position
- distance
- orbit angles in `orbitUv`

### Extended-state cameras

- [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp)
- [`CPathCamera.hpp`](CPathCamera.hpp)

`DollyZoom` adds dynamic perspective state.
`Path Rig` adds typed path-rig state and a pluggable path model.

## Path Rig design

`Path Rig` is described explicitly by typed state, typed limits, and a path model.

### Typed path state

`ICamera::PathState` stores:

- `s`
  path progress / angular progress in the default model
- `u`
  lateral / radial component in the default model
- `v`
  vertical component
- `roll`
  roll around the view axis / path tangent

The default shared target-relative model interprets `s/u/v/roll` as a target-relative path rig, but custom models may reuse the same coordinates differently.

### Typed path limits

`ICamera::PathStateLimits` stores per-instance limits:

- `minU`
- `minDistance`
- `maxDistance`

Those limits are part of the active camera state surface.
They are queried through `tryGetPathStateLimits(...)` and used by the solver, validation, and path-state sanitization.

### Shared path helpers

- [`CCameraPathMetadata.hpp`](CCameraPathMetadata.hpp)
- [`CCameraPathUtilities.hpp`](CCameraPathUtilities.hpp)

`CCameraPathUtilities` provides:

- default path metadata and identifiers
- path-state sanitization and comparison
- position-to-state and state-to-pose conversion
- distance updates
- delta and transition helpers
- the default `Path Rig` model

### Path model seam

The path seam is [`SCameraPathModel`](CCameraPathUtilities.hpp).
It is a named typed model, not an ad-hoc runtime hack.

It contains five callbacks:

- `resolveState`
  normalize requested state or derive initial state from target + position
- `controlLaw`
  turn accumulated runtime motion into a `SCameraPathDelta`
- `integrate`
  apply one delta to the current state under typed limits
- `evaluate`
  convert path state into canonical pose plus target-relative data
- `updateDistance`
  retarget state to a requested spherical distance

The default model is created by `CCameraPathUtilities::makeDefaultPathModel()`.

### Runtime behavior of `CPathCamera`

[`CPathCamera.hpp`](CPathCamera.hpp) keeps the public runtime entry point unchanged:

```text
virtual events -> gimbal accumulation -> path controlLaw -> integrate -> evaluate -> gimbal pose
```

`CPathCamera` owns:

- one active `SCameraPathModel`
- one active `PathState`
- one active `PathStateLimits`

and exposes:

- `getPathModel()`
- `getPathStateLimits()`
- `setPathModel(...)`
- `setPathStateLimits(...)`
- `tryGetPathState(...)`
- `trySetPathState(...)`

Construction is resilient:

- a complete custom model is accepted directly
- an incomplete model falls back to the shared default model
- invalid custom limits are sanitized or replaced with shared defaults

That means `Path Rig` is first-class and pluggable without changing `ICamera::manipulate(...)`.

## Goals, presets, tracks, and playback

The tooling side of the stack is built around a shared canonical camera state.

### Goal layer

- [`CCameraGoal.hpp`](CCameraGoal.hpp)
- [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp)
- [`CCameraGoalAnalysis.hpp`](CCameraGoalAnalysis.hpp)

`CCameraGoal` is the canonical typed transport object.
It extends [`SCameraRigPose.hpp`](SCameraRigPose.hpp) with optional state such as:

- target position
- orbit state
- path state
- dynamic perspective state

`CCameraGoalSolver` is the bridge between typed state and the event-driven runtime:

- capture camera state into a goal
- analyze compatibility
- apply what can be applied through typed hooks
- replay virtual events when needed

This is the main best-effort absolute layer.

### Presets and preset flow

- [`CCameraPreset.hpp`](CCameraPreset.hpp)
- [`CCameraPresetFlow.hpp`](CCameraPresetFlow.hpp)
- [`CCameraPresetPersistence.hpp`](CCameraPresetPersistence.hpp)

Presets are named `CCameraGoal` wrappers.
`CCameraPresetFlowUtilities` provides the high-level capture/apply helpers that most consumers should call.

### Keyframe tracks and playback

- [`CCameraKeyframeTrack.hpp`](CCameraKeyframeTrack.hpp)
- [`CCameraKeyframeTrackPersistence.hpp`](CCameraKeyframeTrackPersistence.hpp)
- [`CCameraPlaybackTimeline.hpp`](CCameraPlaybackTimeline.hpp)

Keyframe tracks are preset-based reusable playback data.
`CCameraPlaybackTimeline` owns only transport-like playback cursor logic.
Higher-level playback policy remains on the consumer side.

### Persistence and file helpers

- [`CCameraPersistence.hpp`](CCameraPersistence.hpp)
- [`CCameraFileUtilities.hpp`](CCameraFileUtilities.hpp)

These helpers cover camera presets, tracks, and related shared persistence tasks.

## Follow layer

Follow is deliberately not baked into `ICamera`.

### Tracked target

- [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp)

The tracked subject is `CTrackedTarget`.
It owns its own `ICamera::CGimbal`.

The follow source of truth is:

- tracked-target pose
- follow mode
- follow config

not a scene-node id or mesh handle.

### Follow modes

Current reusable modes are:

- `OrbitTarget`
- `LookAtTarget`
- `KeepWorldOffset`
- `KeepLocalOffset`

### Follow application and validation

- [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp)
- [`CCameraFollowRegressionUtilities.hpp`](CCameraFollowRegressionUtilities.hpp)

`CCameraFollowUtilities` builds goal state from tracked target + policy and applies it through `CCameraGoalSolver`.
`CCameraFollowRegressionUtilities` validates lock angle, projected target placement, distance consistency, and spherical writeback behavior.

## Scripted authoring and validation

The scripting side is split into compact authored data and expanded runtime payloads.

### Compact authored sequence

- [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp)
- [`CCameraSequenceScriptPersistence.hpp`](CCameraSequenceScriptPersistence.hpp)

This is the human-maintainable authored representation.
It stores:

- camera kind or identifier
- projection presentation requests
- compact goal keyframes
- compact tracked-target keyframes
- continuity thresholds
- capture fractions

It intentionally does not store frame-by-frame low-level event dumps.

### Expanded runtime payload

- [`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp)
- [`CCameraScriptedRuntimePersistence.hpp`](CCameraScriptedRuntimePersistence.hpp)

This layer stores:

- low-level scripted input events
- per-frame checks
- capture frame scheduling
- optional compact sequence payload parsed alongside the low-level runtime payload

### Sequence expansion

- [`CCameraSequenceScriptedBuilder.hpp`](CCameraSequenceScriptedBuilder.hpp)

This converts one compiled sequence segment into frame-by-frame scripted runtime payloads.

### Frame check execution

- [`CCameraScriptedCheckRunner.hpp`](CCameraScriptedCheckRunner.hpp)
- [`CCameraSmokeRegressionUtilities.hpp`](CCameraSmokeRegressionUtilities.hpp)

This layer evaluates authored per-frame checks against the active runtime state.
It is used by shared smoke and continuity coverage in `61_UI`.

## Projection layer

Projection state is a separate reusable layer.

- [`IProjection.hpp`](IProjection.hpp)
- [`ILinearProjection.hpp`](ILinearProjection.hpp)
- [`IPerspectiveProjection.hpp`](IPerspectiveProjection.hpp)
- [`IPlanarProjection.hpp`](IPlanarProjection.hpp)
- [`CLinearProjection.hpp`](CLinearProjection.hpp)
- [`CPlanarProjection.hpp`](CPlanarProjection.hpp)
- [`CCubeProjection.hpp`](CCubeProjection.hpp)
- [`IRange.hpp`](IRange.hpp)

Important rule:

- projections may own viewport-local binding layouts
- projections do not process raw runtime input

That separation keeps viewport glue out of reusable camera semantics.

## Presentation and UI-facing helpers

The shared layer also exposes small reusable helpers used by consumers such as `61_UI`:

- [`CCameraPresentationUtilities.hpp`](CCameraPresentationUtilities.hpp)
- [`CCameraProjectionUtilities.hpp`](CCameraProjectionUtilities.hpp)
- [`CCameraTextUtilities.hpp`](CCameraTextUtilities.hpp)
- [`CCameraViewportOverlayUtilities.hpp`](CCameraViewportOverlayUtilities.hpp)
- [`CCameraControlPanelUiUtilities.hpp`](CCameraControlPanelUiUtilities.hpp)
- [`CCameraScriptVisualDebugOverlayUtilities.hpp`](CCameraScriptVisualDebugOverlayUtilities.hpp)

These are still shared camera helpers.
They are presentation-facing, but they remain camera-domain rather than `61_UI`-local glue.

## 61_UI integration

`61_UI` is the full runnable integration for this stack:

- [`61_UI/README.md`](../../../61_UI/README.md)

`61_UI` provides:

- scene setup
- active planar / window routing
- ImGui control panel
- tracked-target visualization
- scripted smoke and continuity assets
- screenshot capture
- local logging

The shared camera layer remains the source of truth for camera semantics.
`61_UI` is the concrete harness that exercises them.

## Local configure, build, and test

Current local setup uses the Visual Studio 2022 dynamic preset.

Configure:

```powershell
cmake --preset user-configure-dynamic-msvc
```

Build `61_UI`:

```powershell
cmake --build build/dynamic/examples_tests/61_UI --config Debug --target 61_ui -- /m:1
```

Run the camera-focused tests:

```powershell
ctest --test-dir build/dynamic/examples_tests/61_UI -C Debug --output-on-failure -R NBL_61_UI_CAMERA_
```

Run the example manually:

```powershell
examples_tests/61_UI/bin/61_ui_d.exe
```

Run CI-style screenshot capture:

```powershell
examples_tests/61_UI/bin/61_ui_d.exe --ci
```

## Minimal integration examples

### Runtime input to camera

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

ui::CGimbalInputBinder binder;
ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(binder, *camera);

auto collected = binder.collectVirtualEvents(timestamp, {
    .mouseEvents = { mouseEvents.data(), mouseEvents.size() },
    .keyboardEvents = { keyEvents.data(), keyEvents.size() }
});

camera->manipulate(collected.events);
```

### Capture and apply a preset

```cpp
core::CCameraGoalSolver solver;

auto capture = solver.captureDetailed(camera.get());
if (capture.canUseGoal())
{
    core::CCameraPreset preset;
    core::CCameraPresetUtilities::assignGoalToPreset(preset, capture.goal);

    auto apply = core::CCameraPresetFlowUtilities::applyPresetDetailed(solver, camera.get(), preset);
    if (!apply.succeeded())
    {
        // report unsupported or approximate apply
    }
}
```

### Apply follow

```cpp
core::CTrackedTarget trackedTarget(position, orientation);

core::SCameraFollowConfig follow = {};
follow.enabled = true;
follow.mode = core::ECameraFollowMode::KeepLocalOffset;
follow.localOffset = hlsl::float64_t3(-4.0, 0.0, 1.0);

auto result = core::CCameraFollowUtilities::applyFollowToCamera(solver, camera.get(), trackedTarget, follow);
```

### Expand a compact sequence into runtime payloads

```cpp
system::CCameraScriptedTimeline timeline;

system::CCameraSequenceScriptedBuilderUtilities::appendCompiledSequenceSegmentToScriptedTimeline(
    timeline,
    baseFrame,
    compiledSegment,
    buildInfo);

system::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(timeline);
```

## Summary

The current stack is organized as:

```text
input layer
  -> semantic virtual events
  -> runtime camera models
  -> typed goal / preset / track tooling
  -> compact sequence authoring
  -> expanded scripted runtime
  -> shared validation
```

That split is intentional.
It is what keeps the runtime path small, the tooling path expressive, and the `61_UI` integration reusable instead of example-local.
