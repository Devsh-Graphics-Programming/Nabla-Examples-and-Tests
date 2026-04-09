# Shared Camera API

This directory contains the shared camera stack.
A complete runnable integration is [`61_UI`](../../../61_UI/README.md).

## What this stack is

This stack is a reusable camera framework with two clearly separated halves:

1. Runtime camera control
   The hot path that reacts to input and updates camera pose.
2. Tooling and validation
   The sidecar layer used for capture, restore, presets, tracks, playback, scripted validation, and follow.

The runtime half is intentionally event-driven.
The tooling half is intentionally state-driven.

That split is the core design decision.

## What this stack is not

This stack is not:

- engine-core Nabla camera API
- a generic scene animation system
- a direct scene-object-follow system
- a setter-heavy camera API with arbitrary absolute pose mutation in the hot runtime path

## Design goals

The design goals are:

- one semantic command language for keyboard, mouse, gizmo, scripts, and CI
- no input-device assumptions inside camera models
- no viewport glue inside camera models
- no direct dependence on application-specific UI concepts in the reusable camera layer
- best-effort absolute restore for tooling without turning cameras into mutable state bags
- reusable persistence, analysis, playback, follow, and validation helpers

## Namespace split

The stack is split across existing Nabla namespaces:

- `nbl::hlsl`
  math types and camera math helpers
- `nbl::core`
  camera runtime model, goals, presets, tracks, playback, follow, and authored sequence data
- `nbl::ui`
  binding layouts, input processors, binders, default input mappings, and user-facing presentation/text helpers
- `nbl::system`
  persistence, scripted runtime payloads, scripted parsing, scripted check execution, and follow-contract validation helpers

The shared camera math is written against `nbl::hlsl`.
Consumers of this stack are expected to talk to camera math through `nbl::hlsl` types and helpers rather than through direct `glm::...` calls.

## Why virtual events and not absolute setters

The runtime contract is intentionally built around virtual events such as:

- `MoveForward`
- `PanLeft`
- `TiltUp`
- `RollRight`

instead of runtime methods like:

- `setPosition(...)`
- `setTarget(...)`
- `setYawPitchRoll(...)`

The reason is architectural, not cosmetic.

### What the event-driven contract buys us

It gives us one shared runtime path for:

- live keyboard and mouse input
- ImGuizmo manipulation
- scripted playback
- CI validation
- future headless or tool-driven input sources

It also keeps camera semantics inside the camera type:

- `Orbit` means orbit
- `FPS` means FPS
- `Path Rig` means target-relative cylindrical path rig

instead of allowing every caller to overwrite camera internals arbitrarily.

### Why the absolute layer still exists

Tooling still needs absolute-ish operations:

- capture current state
- restore preset
- scrub playback
- compare two camera states
- run validation against expected state

That is why the absolute layer exists, but it is kept outside `ICamera`.

The intended pattern is:

`camera <-> goal/preset/track/solver`

not:

`camera exposes public setters for everything`

This is why the design uses:

- [`CCameraGoal.hpp`](CCameraGoal.hpp)
- [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp)
- [`CCameraPreset.hpp`](CCameraPreset.hpp)
- [`CCameraKeyframeTrack.hpp`](CCameraKeyframeTrack.hpp)

instead of making `ICamera` setter-heavy.

## High-level architecture

There are two main paths.

### Runtime path

```text
raw input
  -> binding layout
  -> input processor / binder
  -> virtual gimbal events
  -> ICamera::manipulate(...)
  -> updated camera gimbal / view
```

### Tooling path

```text
ICamera
  <-> CCameraGoal
  <-> CCameraPreset
  <-> CCameraKeyframeTrack
  <-> CCameraPlaybackTimeline
  <-> CCameraSequenceScript
```

### Follow path

```text
CTrackedTarget + SCameraFollowConfig
  -> CCameraGoal
  -> CCameraGoalSolver
  -> camera state
```

### Scripted sequence path

```text
CCameraSequenceScript
  -> compiled sequence segment
  -> scripted runtime timeline
  -> scripted check runner
  -> runtime logging / CI / screenshots
```

## Stack breakdown

### 1. Gimbal and semantic commands

- [`CVirtualGimbalEvent.hpp`](CVirtualGimbalEvent.hpp)
- [`IGimbal.hpp`](IGimbal.hpp)
- [`CCameraMathUtilities.hpp`](CCameraMathUtilities.hpp)

This is the mathematical foundation.

It defines:

- the shared semantic event language in `CVirtualGimbalEvent`
- low-level gimbal math
- reusable camera-oriented math helpers in `nbl::hlsl`
- accumulation of multiple semantic commands into one camera impulse

This is the shared command language used by all camera types and all runtime input sources.

### 2. Binding layout

- [`IGimbalBindingLayout.hpp`](IGimbalBindingLayout.hpp)

This layer stores static mappings such as:

- keyboard key -> virtual event
- mouse input -> virtual event
- ImGuizmo delta -> virtual event

This layer does not process runtime input.
It only stores how input should map to the semantic command language.

### 3. Runtime input processing

- [`IGimbalInputProcessor.hpp`](IGimbalInputProcessor.hpp)
- [`CGimbalInputBinder.hpp`](CGimbalInputBinder.hpp)
- [`CCameraInputBindingUtilities.hpp`](CCameraInputBindingUtilities.hpp)

The main runtime type is `IGimbalInputProcessor`.

This layer:

- receives actual keyboard and mouse streams
- receives ImGuizmo transforms
- emits virtual events for the current frame
- stores reusable default keyboard, mouse, and ImGuizmo binding presets for camera kinds

`CGimbalInputBinder` is the convenience runtime binder that a consumer should usually use.

### 4. Camera core

- [`ICamera.hpp`](ICamera.hpp)

This is the core contract that camera models implement.

Important properties:

- runtime entry point is `manipulate(...)`
- the runtime contract consumes virtual events only
- cameras own their own gimbal and view state
- cameras expose typed optional state hooks only for tooling

`ICamera` also exposes:

- `CameraKind`
- `CameraCapability`
- `GoalStateMask`
- motion config
- typed state hooks used by tooling

### 5. Projection layer

- [`ILinearProjection.hpp`](ILinearProjection.hpp)
- [`IPlanarProjection.hpp`](IPlanarProjection.hpp)
- [`CPlanarProjection.hpp`](CPlanarProjection.hpp)
- [`CLinearProjection.hpp`](CLinearProjection.hpp)
- [`CCubeProjection.hpp`](CCubeProjection.hpp)

This layer handles projection state.

Important rule:

- projection may own viewport-local binding layout state
- projection does not own raw input processing

That separation was one of the major cleanup goals of the refactor.

### 6. Goal / preset / track tooling

- [`CCameraGoal.hpp`](CCameraGoal.hpp)
- [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp)
- [`CCameraGoalAnalysis.hpp`](CCameraGoalAnalysis.hpp)
- [`CCameraPreset.hpp`](CCameraPreset.hpp)
- [`CCameraPresetFlow.hpp`](CCameraPresetFlow.hpp)
- [`CCameraKeyframeTrack.hpp`](CCameraKeyframeTrack.hpp)
- [`CCameraPlaybackTimeline.hpp`](CCameraPlaybackTimeline.hpp)
- [`CCameraPersistence.hpp`](CCameraPersistence.hpp)

This is the tooling half of the stack.

It covers:

- state capture
- compatibility analysis
- best-effort restore
- preset storage
- keyframe playback
- persistence
- UI-facing diagnostics and presentation helpers

### 7. Follow

- [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp)

Follow is deliberately not part of `ICamera`.

The tracked subject owns its own gimbal through `CTrackedTarget`.
Follow stays as a policy layer above the camera.

That means the camera API does not know about meshes, scene nodes, or any particular UI harness.
It only knows about:

- camera
- tracked target gimbal
- follow config
- best-effort goal application

### 8. Scripted sequence and validation

- [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp)
- [`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp)
- [`CCameraScriptedRuntimePersistence.hpp`](CCameraScriptedRuntimePersistence.hpp)
- [`CCameraSequenceScriptedBuilder.hpp`](CCameraSequenceScriptedBuilder.hpp)
- [`CCameraScriptedCheckRunner.hpp`](CCameraScriptedCheckRunner.hpp)
- [`CCameraFollowRegressionUtilities.hpp`](CCameraFollowRegressionUtilities.hpp)

This is the reusable scripting and CI half.

It supports two levels of representation:

1. Compact authored camera-domain script
2. Expanded frame-by-frame scripted runtime payload

That separation is important.
Authored assets stay short and meaningful.
Expanded runtime payloads stay normalized and reusable.

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
- orbit angles

They participate in the shared spherical goal flow.

### Extended-state cameras

- [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp)
- [`CPathCamera.hpp`](CPathCamera.hpp)

These extend the shared base with typed extra state:

- `DynamicPerspectiveState`
- `PathState`

## Capabilities and typed state

The core camera contract exposes:

- `CameraCapability`
- `GoalStateMask`

The currently relevant typed state is:

- `SphericalTargetState`
- `DynamicPerspectiveState`
- `PathState`

The rule is:

- if a camera can round-trip through shared spherical state, do not add fake extra state
- if a camera has real additional semantics that would be lost, add typed state explicitly

That is why:

- `Chase` and `Dolly` currently stay on shared spherical state
- `DollyZoom` has dynamic perspective state
- `Path Rig` has target-relative cylindrical path state

## Follow model

Follow is modeled around a tracked target gimbal, not around a scene object id.

### Source of truth

The source of truth is:

- `CTrackedTarget`

which literally owns a gimbal.

### Follow modes

Current modes:

- `LookAtTarget`
- `OrbitTarget`
- `KeepWorldOffset`
- `KeepLocalOffset`

### Follow invariants

For enabled modes, the camera must stay logically locked to the tracked target:

- camera-to-target direction must match the expected view direction
- projected target center error must stay small when projection is available
- spherical cameras must write target state back consistently
- camera-target distance must remain internally consistent

Those invariants are reusable and validated through:

- [`CCameraFollowRegressionUtilities.hpp`](CCameraFollowRegressionUtilities.hpp)

## Compact sequence design

The compact sequence format is deliberately camera-domain.

It describes:

- camera kind or identifier
- projection presentation requests
- goal keyframes
- tracked-target keyframes
- continuity thresholds
- capture fractions

It deliberately does not describe:

- runtime-specific window actions as authored source data
- frame-by-frame event dumps
- ImGuizmo matrices as authored motion primitives

This is why the new continuity asset became small and maintainable instead of being a giant generated dump.

## Scripted runtime design

The expanded scripted runtime exists so that a consumer can execute frame-by-frame logic without redefining runtime types locally.

It is split into:

- authored parsing and normalization
- timeline finalization
- segment-to-runtime expansion
- per-frame dequeue
- per-frame check evaluation

This keeps one runtime from owning a private scripting subsystem.

## Current validation story

The current camera-focused validation is exercised through scripted smoke and continuity tests.

### Smoke

Purpose:

- prove that camera selection and basic scripted manipulation still work
- validate preset, sequence, runtime, and follow helper contracts with small regression checks

### Continuity

Purpose:

- prove that camera motion remains smooth frame-to-frame
- prove that follow target lock remains valid during scripted target motion

This test now runs on the compact authored sequence format rather than a large expanded frame dump.

## Recommended integration patterns

### Minimal runtime integration

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

CGimbalInputBinder binder;
CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(binder, *camera);

auto collected = binder.collectVirtualEvents(timestamp, {
    .mouseEvents = { mouseEvents.data(), mouseEvents.size() },
    .keyboardEvents = { keyEvents.data(), keyEvents.size() }
});

camera->manipulate(collected.events);
```

### Preset / tooling integration

```cpp
CCameraGoalSolver solver;

auto capture = solver.captureDetailed(camera.get());
if (capture.canUseGoal())
{
    CCameraPreset preset;
    CCameraPresetUtilities::assignGoalToPreset(preset, capture.goal);

    auto apply = CCameraPresetFlowUtilities::applyPresetDetailed(solver, camera.get(), preset);
    if (!apply.succeeded())
    {
        // report exact vs best-effort or unsupported state
    }
}
```

### Follow integration

```cpp
CTrackedTarget trackedTarget(position, orientation);

SCameraFollowConfig follow = {};
follow.enabled = true;
follow.mode = ECameraFollowMode::KeepLocalOffset;
follow.localOffset = float64_t3(-4.0, 0.0, 1.0);

CCameraGoalSolver solver;
auto result = CCameraFollowUtilities::applyFollowToCamera(solver, camera.get(), trackedTarget, follow);
```

### Sequence / CI integration

```cpp
CCameraSequenceScript script = ...;
CCameraSequenceCompiledSegment segment = ...;

CCameraScriptedTimeline timeline;
CCameraSequenceScriptedBuilderUtilities::appendCompiledSequenceSegmentToScriptedTimeline(
    timeline,
    baseFrame,
    segment,
    buildInfo);

CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(timeline);

CCameraScriptedCheckRuntimeState state = {};
auto frameResult = CCameraScriptedCheckRunnerUtilities::evaluateScriptedChecksForFrame(
    timeline.checks,
    state,
    context);
```

## Why this split matters

The design deliberately keeps these concerns separate:

- input binding
- camera semantics
- absolute/tooling state
- follow policy
- scripted playback and validation

It also keeps the math side explicit:

- camera-space vectors, matrices, and quaternions come from `nbl::hlsl`
- runtime camera semantics stay in `nbl::core`
- input-device mappings stay in `nbl::ui`
- scripting, persistence, and validation helpers stay in `nbl::system`

That separation is what keeps the stack reusable.

If any one of those concerns leaks into the others:

- cameras become setter-heavy
- projections become input processors
- consumers own private copies of state math
- scripts become runtime-specific
- follow becomes scene-object-specific

The current refactor was mostly about removing exactly those leaks.
