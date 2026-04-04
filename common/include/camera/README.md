# Shared Camera API

This directory contains the reusable camera stack currently used by `61_UI`.
It lives in `examples_tests/common` on purpose: the API is shared across examples, but it is not engine-core API yet.

The current design goal is:

- camera core consumes virtual events only
- raw input binding stays outside the camera
- absolute goals and preset restore are best-effort helpers layered on top
- reusable math, preset, analysis, and keyframe-track utilities live in shared headers
- reusable playback cursor and timeline helpers live in shared headers
- `61_UI` is the current integration and validation surface

## Mental model

The stack is split into 5 layers:

1. `CVirtualGimbalEvent`
   A semantic camera command such as `MoveForward`, `PanLeft`, or `RollRight`.
2. Binding layout
   Static mapping from keyboard, mouse, or ImGuizmo inputs to virtual events.
3. Input processor / binder
   Runtime translation from external events into virtual events.
4. Camera model
   A concrete camera type that consumes virtual events and updates its pose.
5. Goal / preset / track utilities
   Best-effort capture, compatibility analysis, restore, blending, and playback helpers.

The intended flow is:

`raw input -> binding layout -> input binder -> virtual events -> ICamera::manipulate(...)`

The best-effort absolute layer sits beside that flow:

`camera state <-> CCameraGoal <-> CCameraPreset <-> CCameraKeyframeTrack`

and the playback cursor sits beside that state:

`CCameraPlaybackCursor <-> CCameraKeyframeTrack`

## Core contracts

### `IGimbal.hpp`

Defines `CVirtualGimbalEvent` and the low-level gimbal math.

Important points:

- `CVirtualGimbalEvent` is the shared semantic command language.
- Translation, rotation, and scale commands are represented explicitly.
- `IGimbal<T>::accumulate(...)` converts a stream of virtual events into a single impulse.

This is the base abstraction that makes scripted tests, headless CI, and manual input share one command path.

### `IGimbalBindingLayout.hpp`

Defines static binding-layout storage and mutation.

Use it when you need to describe:

- which keyboard keys trigger which virtual events
- which mouse channels trigger which virtual events
- which ImGuizmo transforms map to which virtual events

This header does not process input by itself. It only stores mapping layout.

### `IGimbalController.hpp`

The file name is legacy. The main type is `IGimbalInputProcessor`.

Use it when you need runtime input processing:

- keyboard event streams
- mouse event streams
- ImGuizmo delta transforms

`IGimbalInputProcessor` owns active runtime mappings and emits virtual events for the current frame.

### `CGimbalInputBinder.hpp`

High-level runtime binder built on top of `IGimbalInputProcessor`.

This is the easiest entry point for examples:

- copy active bindings from a layout
- copy preset/default bindings from a layout
- collect virtual events for one frame

Use `CGimbalInputBinder` in viewport or example glue.
Do not embed external input processing inside camera types.

### `ICamera.hpp`

Main camera contract.

Important rules:

- `ICamera::manipulate(...)` is the core runtime entry point
- camera core consumes virtual events only
- `ICamera` exposes inspection and capability queries
- camera core may expose typed optional state through `tryGet...State` / `trySet...State`

The typed state hooks are intentionally optional:

- `SphericalTargetState`
- `DynamicPerspectiveState`
- `PathState`

This allows best-effort goal solving without turning the core runtime contract into a setter-heavy API.

`ICamera` also exposes:

- `CameraKind`
- `CameraCapability`
- `GoalStateMask`
- motion config
- default input binding config

## Camera families

### Free cameras

- `CFPSCamera.hpp`
- `CFreeLockCamera.hpp`

These are pose-driven cameras without spherical target semantics.
Best-effort absolute apply may fall back to direct reference-pose restoration when event replay alone is insufficient.

### Spherical-target family

- `CSphericalTargetCamera.hpp`
- `COrbitCamera.hpp`
- `CArcballCamera.hpp`
- `CTurntableCamera.hpp`
- `CTopDownCamera.hpp`
- `CIsometricCamera.hpp`
- `CChaseCamera.hpp`
- `CDollyCamera.hpp`

These cameras share:

- target position
- distance
- orbit angles `u/v`

`CSphericalTargetCamera` is the common reusable base for that family.

Current contract:

- `Orbit`, `Arcball`, `Turntable`, `TopDown`, `Isometric`, `Chase`, and `Dolly` all participate in the shared spherical goal flow
- `Chase` and `Dolly` currently do not carry extra typed state beyond shared spherical state

### Extended state cameras

- `CDollyZoomCamera.hpp`
- `CPathCamera.hpp`

These extend the shared spherical model with extra typed state:

- `CDollyZoomCamera` uses `DynamicPerspectiveState`
- `CPathCamera` uses `PathState`

If a camera needs extra state that cannot be faithfully round-tripped through pose plus spherical target data, this is the pattern to follow.

## Projection layer

### `IProjection.hpp`, `ILinearProjection.hpp`

Base projection contracts and linear-projection math.

### `IPlanarProjection.hpp`

Planar camera projection wrapper used by `61_UI`.

`IPlanarProjection::CProjection` carries:

- perspective or orthographic parameters
- projection matrix update logic
- its own input binding storage for viewport-local bindings

Important design point:

- projection owns binding layout storage
- projection does not process raw input by itself
- runtime input processing should happen through `CGimbalInputBinder`

### `CPlanarProjection.hpp`, `CLinearProjection.hpp`, `CCubeProjection.hpp`

Concrete projection helpers on top of the above contracts.

## Goal, preset, and playback utilities

### `CCameraGoal.hpp`

Typed camera-state transport object plus reusable math helpers.

Use it for:

- canonicalizing captured state
- checking whether a goal is finite
- comparing actual state to expected state
- blending two camera states for playback
- describing mismatches
- determining required typed goal state

`CCameraGoal` is the shared language between capture, analysis, preset persistence, and playback interpolation.

### `CCameraGoalSolver.hpp`

Best-effort absolute layer for cameras.

Use it for:

- capture of typed camera state into `CCameraGoal`
- compatibility analysis against a target camera
- best-effort apply of a goal to a camera
- reconstruction of virtual events for replay-driven application

Important result types:

- `SCaptureResult`
- `SCompatibilityResult`
- `SApplyResult`

`SApplyResult` explicitly distinguishes:

- unsupported
- failed
- already satisfied
- applied by absolute fallback
- applied by virtual events
- mixed absolute + virtual event application

This is the main contract behind preset restore and cross-camera best-effort behavior.

### `CCameraGoalAnalysis.hpp`

Thin reusable analysis layer built on top of `CCameraGoalSolver`.

Use it when UI or higher-level tools need typed answers for:

- can this camera state be captured
- can this preset/goal be applied
- is the apply exact or best-effort
- does the target camera drop some goal state
- is the result only using shared state across different camera kinds

This keeps policy analysis out of example-local UI code.

### `CCameraPreset.hpp`

Reusable preset and keyframe state plus JSON IO.

Provides:

- `CCameraPreset`
- `CCameraKeyframe`
- goal-to-preset conversion helpers
- preset JSON serialization and deserialization

This is the storage format used by `61_UI` for preset authoring and playback persistence.

### `CCameraKeyframeTrack.hpp`

Reusable keyframe-track helpers on top of presets.

Provides:

- `CCameraKeyframeTrack`
- preset-at-time evaluation
- keyframe sorting
- playback-time clamping
- nearest-keyframe selection
- selected-keyframe access
- selected-keyframe preset replacement
- track JSON serialization and deserialization

This keeps playback-authoring logic reusable without forcing examples to reimplement keyframe math and storage flow.

### `CCameraPlaybackTimeline.hpp`

Reusable playback cursor and timeline helpers on top of keyframe tracks.

Provides:

- `CCameraPlaybackCursor`
- `SCameraPlaybackAdvanceResult`
- track duration helpers
- cursor reset and clamping
- per-frame time advance with loop and stop semantics

This keeps playback-time progression reusable without forcing examples to reimplement cursor stepping rules.

## Current integration status

The shared headers above are designed to be reusable by additional examples.
Right now the active migrated call-site is only:

- `61_UI`

That is intentional.
The current work is focused on stabilizing the reusable API surface first, then using `61_UI` as the validation and UX harness.

## Recommended integration pattern for a new example

If another example wants to adopt this stack later, the intended path is:

1. Instantiate a concrete `ICamera`.
2. Store default binding layouts on the camera and/or planar projection.
3. Use `CGimbalInputBinder` at runtime to translate external input into virtual events.
4. Feed the resulting event stream into `ICamera::manipulate(...)`.
5. Use `CCameraGoalSolver`, `CCameraGoalAnalysis`, `CCameraPreset`, `CCameraKeyframeTrack`, and `CCameraPlaybackTimeline` only for tooling features such as:
   - preset capture
   - preset restore
   - compatibility preview
   - playback interpolation
   - playback cursor stepping
   - scripted validation

That keeps the hot runtime path event-driven while still allowing higher-level tools to work with absolute camera goals in a controlled way.

## Legacy compatibility notes

- `CTargetPoseController.hpp` is currently only a compatibility include for `CCameraGoalSolver.hpp`.
- `IGimbalController.hpp` still has the old file name, but the main runtime-processing type is `IGimbalInputProcessor`.

## Non-goals

This layer is not yet:

- engine-core Nabla camera API
- a promise that every example already uses the stack
- a fully generic animation system beyond camera preset and keyframe utilities

It is a reusable example-shared camera framework currently validated through `61_UI`.
