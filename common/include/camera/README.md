# Shared Camera API

This directory contains the reusable camera stack currently used by `61_UI`.
It lives in `examples_tests/common` on purpose: the API is shared across examples, but it is not engine-core API yet.

The current design goal is:

- camera core consumes virtual events only
- raw input binding stays outside the camera
- absolute goals and preset restore are best-effort helpers layered on top
- reusable math, preset, analysis, and keyframe-track utilities live in shared headers
- reusable preset capture and apply flow helpers live in shared headers
- reusable playback cursor and timeline helpers live in shared headers
- reusable preset and keyframe persistence helpers live in shared headers
- reusable compact camera-sequence scripting helpers live in shared headers
- reusable tracked-target and follow helpers live in shared headers
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

and compact authored sequence scripts sit above the same shared camera-domain state:

`CCameraSequenceScript -> CCameraKeyframeTrack -> CCameraPreset -> CCameraGoal`

and tracked-target follow sits beside the same goal layer:

`CTrackedTarget + SCameraFollowConfig -> CCameraGoal -> CCameraGoalSolver`

and sequence-authored tracked-target motion can feed that same follow layer:

`CCameraSequenceScript.target_keyframes -> CCameraSequenceTrackedTargetTrack -> CTrackedTarget`

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

### `CCameraTextUtilities.hpp`

Reusable human-readable metadata and diagnostic text helpers for cameras.

### `CCameraFollowUtilities.hpp`

Reusable tracked-target and follow helpers layered on top of the shared camera API.

Important design points:

- follow stays outside `ICamera`
- the tracked subject owns its own gimbal through `CTrackedTarget`
- follow modes map target motion into `CCameraGoal`
- goal application still goes through `CCameraGoalSolver`
- follow semantics are defined against the tracked-target gimbal, not against any scene model or mesh

Current follow modes:

- `LookAtTarget`
- `OrbitTarget`
- `KeepWorldOffset`
- `KeepLocalOffset`

Current follow invariants:

- all enabled follow modes lock the final camera view onto the tracked-target position
- `LookAtTarget` keeps the camera world position and only rotates it toward the target
- `OrbitTarget` keeps the camera on a target-relative spherical/path rig and recenters the tracked target
- `KeepWorldOffset` keeps a world-space offset from the tracked target and recenters it
- `KeepLocalOffset` keeps an offset in the tracked-target local frame and recenters it
- the tracked target remains the source of truth; the camera does not own the subject

This is why the regression layer validates:

- camera forward axis vs. camera-to-target direction
- projected target center error in NDC
- spherical target writeback for spherical cameras
- target distance consistency after apply

This keeps the camera runtime contract event-driven while still allowing higher-level
tracking behavior to be reused by tools and examples.

Provides:

- camera-kind labels
- camera-kind descriptions
- goal-state mask descriptions
- detailed goal-apply result descriptions
- analyzed goal-apply compatibility and policy descriptions
- analyzed camera-capture policy descriptions
- aggregate preset-apply summary descriptions

This keeps camera-specific presentation and diagnostic text reusable without leaving it in example-local glue.

### `CCameraPresentationUtilities.hpp`

Reusable presentation-oriented wrappers built on top of shared camera analysis and text helpers.

Provides:

- exact-vs-best-effort preset presentation filtering
- reusable labels for presentation filters
- presentation-ready apply-analysis structs
- presentation-ready capture-analysis structs
- reusable badge flags for apply/result presentation
- presentation-ready source-kind and goal-state labels
- ready-to-render compatibility and policy labels

This keeps higher-level preset and capture presentation flow reusable without leaving it in example-local glue.

### `CCameraPreset.hpp`

Reusable preset and keyframe state plus JSON IO.

Provides:

- `CCameraPreset`
- `CCameraKeyframe`
- preset comparison helpers
- preset collection comparison helpers
- goal-to-preset conversion helpers
- preset JSON serialization and deserialization

This is the storage format used by `61_UI` for preset authoring and playback persistence.

### `CCameraPresetFlow.hpp`

Reusable preset capture, comparison, mismatch, and best-effort apply helpers.

Provides:

- preset capture from a camera and solver
- preset apply through the shared goal solver
- preset apply summaries across camera ranges
- preset-to-camera comparison helpers
- preset mismatch descriptions for diagnostics

This keeps solver-backed preset flow reusable without leaving the flow rules in example-local glue.

### `CCameraManipulationUtilities.hpp`

Reusable manipulation helpers that sit between raw collected virtual events and final camera state.

Provides:

- `SCameraConstraintSettings`
- virtual-event translation and rotation scaling
- world-translation remapping into local camera movement
- post-manipulation constraint clamping through the shared goal solver

This keeps example runtime manipulation policy reusable without leaving event-scaling and constraint logic in example-local glue.

### `CCameraProjectionUtilities.hpp`

Reusable helpers that synchronize camera-driven projection state with planar projections.

Provides:

- dynamic perspective FOV sync from camera state into `IPlanarProjection::CProjection`

This keeps camera-specific projection updates reusable without leaving them in example-local glue.

### `CCameraKeyframeTrack.hpp`

Reusable keyframe-track helpers on top of presets.

Provides:

- `CCameraKeyframeTrack`
- keyframe and track comparison helpers with optional selection-state checks
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

### `CCameraSequenceScript.hpp`

Reusable compact authored camera-sequence format for playback and validation.

Provides:

- `CCameraSequencePresentation`
- `CCameraSequenceContinuitySettings`
- `CCameraSequenceGoalDelta`
- `CCameraSequenceKeyframe`
- `CCameraSequenceSegment`
- `CCameraSequenceScript`
- `CCameraSequenceCompiledSegment`
- `CCameraSequenceCompiledFramePolicy`
- parsing and normalization helpers
- reusable sequence-to-track construction from captured reference presets
- reusable segment compilation from authored data into sampled times, capture offsets, and normalized tracks
- reusable frame-policy scheduling for baseline, continuity-step, follow-lock, and capture milestones

The important design rule is that the authored format stays camera-domain:

- segment and keyframe based
- keyed by camera kind or identifier
- keyed by projection presentation requests
- keyed by compact continuity thresholds and capture fractions

and deliberately does not store:

- expanded per-frame event dumps
- `61_UI`-specific window-routing commands as authored source data
- ImGuizmo matrices as the authored motion primitive

Tracked-target motion follows the same rule:

- authored `target_keyframes` describe only tracked-target pose over time
- they do not refer to a scene object id, mesh, or `61_UI` model
- consumers may map those poses to their own runtime objects, but the authored source of truth stays a tracked-target gimbal track

Track normalization rules:

- negative keyframe times clamp to `0`
- tracked-target keyframes are sorted by time before sampling
- duplicate tracked-target keyframe times collapse to the last authored pose

That makes the same authored sequence usable by any future consumer that understands the shared camera API, even if its runtime expansion path differs from `61_UI`.

### `CCameraScriptedRuntime.hpp`

Shared expanded scripted-runtime payloads for consumers that want a frame-by-frame execution path.

Provides:

- `CCameraScriptedInputEvent`
- `CCameraScriptedInputCheck`
- `CCameraScriptedTimeline`
- `CCameraScriptedFrameEvents`
- reusable scripted-timeline finalization helpers
- reusable event/check append helpers
- reusable per-frame event dequeue/bucketing

This sits one layer below the compact authored sequence:

- `CCameraSequenceScript` remains the authored source of truth
- `CCameraScriptedRuntime` is only the normalized expanded runtime contract
- consumers such as `61_UI` can adapt that contract to their own concrete event loop without redefining event/check structs locally

### `CCameraScriptedRuntimePersistence.hpp`

Reusable JSON parser and compatibility layer for low-level scripted runtime payloads.

Provides:

- `CCameraScriptedControlOverrides`
- `CCameraScriptedInputParseResult`
- shared parsing of legacy low-level runtime JSON:
  - `events`
  - `checks`
  - `capture_frames`
  - top-level scripted debug and control overrides
- compatibility parsing for legacy key names such as `KEY_KEY_W`
- optional handoff into `CCameraSequenceScript` when the same file contains compact `segments`

This keeps old scripted-runtime assets reusable without leaving parser logic inside `61_UI`.
It also means other future consumers can accept the same low-level payloads or progressively migrate them
to compact camera-sequence scripts without changing the shared parsing contract.

### `CCameraSequenceScriptedBuilder.hpp`

Reusable authored-sequence to scripted-runtime builder helpers.

Provides:

- `CCameraSequenceScriptedSegmentBuildInfo`
- reusable conversion from one compiled sequence segment into:
  - scripted `Action` events
  - scripted `Goal` events
  - scripted tracked-target transforms
  - baseline/continuity/follow-lock checks
  - capture frame milestones

This sits between `CCameraSequenceScript.hpp` and `CCameraScriptedRuntime.hpp`:

- `CCameraSequenceScript` owns authored compact segment semantics
- `CCameraSequenceScriptedBuilder` expands compiled segments into shared runtime payloads
- `61_UI` only resolves camera/planar targets and feeds those shared payloads into its local loop

### `CCameraScriptedCheckRunner.hpp`

Reusable scripted-check runtime state and per-frame evaluation helpers.

Provides:

- `CCameraScriptedCheckRuntimeState`
- `CCameraScriptedCheckContext`
- `CCameraScriptedCheckLogEntry`
- `CCameraScriptedCheckFrameResult`
- reusable baseline, near, delta, step, and follow-lock evaluation for one frame

This sits above `CCameraScriptedRuntime.hpp`:

- `CCameraScriptedRuntime` only normalizes authored expanded payloads
- `CCameraScriptedCheckRunner` evaluates those payloads against live camera state
- consumers can keep only their logging and runtime-object lookup glue locally

### `CCameraPersistence.hpp`

Reusable JSON and file persistence helpers for preset collections and keyframe tracks.

Provides:

- preset collection JSON serialization and deserialization
- preset collection file save/load helpers
- keyframe-track file save/load helpers

This keeps example-level save/load glue out of `61_UI` while reusing the same preset and track formats.

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
5. Use `CCameraGoalSolver`, `CCameraGoalAnalysis`, `CCameraPreset`, `CCameraPresetFlow`, `CCameraKeyframeTrack`, `CCameraPlaybackTimeline`, and `CCameraPersistence` only for tooling features such as:
   - preset capture
   - preset restore
   - compatibility preview
   - preset comparison and mismatch diagnostics
   - playback interpolation
   - playback cursor stepping
   - preset and keyframe persistence
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
