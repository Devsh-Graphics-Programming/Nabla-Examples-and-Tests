# Camera tooling (moved out of the Cameras extension)

Everything in this folder used to live in `include/nbl/ext/Cameras`. It moved here because this example is its
only user, and because it is under review rather than settled.

## What this layer does

It captures the state of one camera into a single value, then applies that value to a camera, possibly of a
different kind.

`CCameraGoal` is that value. It is a pose plus a set of optional fragments: an orbited target, a distance, orbit
angles, a path state, a dynamic-perspective state. Each fragment has a `has` flag, because no single camera kind
owns all of them. An arcball has orbit angles and no path state; an FPS camera has neither.

`CCameraGoalSolver` reads a camera into a goal, reports whether a given camera can represent a given goal, and
applies a goal back onto a camera. Application runs in tiers: absolute pose first for rigs without spherical
state, then the typed setters on `ICamera`, then synthesized virtual events replayed through `manipulate` for
anything that has no direct setter. The result says which tier ran and whether it was exact.

Everything else is built on that:

- **presets, keyframes, playback, persistence** store and replay goals
- **follow** authors a goal each frame from a tracked subject and a mode
- **follow regression, smoke regression, scripted runtime and the check runner** assert that what came out
  matches what was asked for
- **sequence scripts** author the whole thing compactly
- **text and presentation helpers** turn the results into UI strings

## Why it left the extension

Three reasons, in order of weight.

It forces every camera kind to answer questions about state it does not have. A goal is the union of all rigs'
internal state, so adding a camera kind means deciding what that kind reports for every fragment, what happens
when a goal carrying a foreign fragment is applied to it, and how the solver should approximate the difference.
That cost is paid on every new rig, and it is paid in this layer rather than in the rig.

It dominated review. It is about 5400 lines, roughly 43% of the extension, and none of it is needed to make a
camera work.

It had exactly one consumer. Extension code should earn its place by being shared.

## What is true right now

The cut is clean in one direction. Nothing in the extension includes anything here; every dependency points from
this folder outward into `nbl/ext/Cameras/`, never back. That is what made the move possible, and it is worth
preserving.

There is no namespace. These types sit at global scope, like `app/AppTypes.hpp` and the rest of the example's
headers. Each file opens with `using namespace nbl::ext::cameras;` so the extension's types stay usable
unqualified, which is what keeps the diff against the extension version small. The later arrivals
(`CCameraProjectionUtilities`, `CFileUtilities`, `CInputCodeNames`, `SCameraToolingThresholds`) are small enough
that they spell out `nbl::` instead, so they do not depend on another header having issued those directives first.

The layer is header-only. The eight `.cpp` files were merged into their headers, so each unit is one file.

One loose end lives outside this folder: the typed hooks on `ICamera` (`tryGetSphericalTargetState`,
`tryGetPathState`, `tryGetDynamicPerspectiveState`, `trySetSphericalTarget`, `trySetSphericalDistance`) now have
no consumer inside the extension. They exist for the solver here. There is a TODO on them in `ICamera.hpp`.

## Before anything goes back

A file returns when it has a second consumer, or when the design stops pushing one rig's state onto another and
starts letting each rig own its state with explicit conversions between kinds. Until then, treat this folder as
a staging area, not a library.

Things worth a second look while reviewing, none of them confirmed:

- `CCameraFollowRegressionUtilities::tryComputeProjectedFollowTargetMetrics` transposes the projection matrix but
  not the view matrix, while `app/AppViewportBindingUtilities.hpp` composes the same two with a plain `mul`. That
  looks inconsistent, but the conventions here have not been traced end to end, so it may well be deliberate.
- `validateFollowTargetContract` appears to compare the measured camera-to-target distance against a
  recomputation of the same quantity, which would leave only non-finite input able to fail it. Likely an issue,
  not verified. There is a matching TODO at the site.
- "Target" is used for three different things across this layer: the point an orbit rig revolves around, the
  tracked subject being followed, and the goal being aimed at. That is a readability observation, not a defect.

## Files

**Goal and solving** — `CCameraGoal`, `CCameraGoalSolver`, `CCameraGoalAnalysis`, `CCameraPresentationUtilities`

**Presets, keyframes, playback** — `CCameraPreset`, `CCameraPresetFlow`, `CCameraKeyframeTrack`,
`CCameraPlaybackTimeline`

**Persistence** — `CCameraPersistence`, `CCameraPresetPersistence`, `CCameraKeyframeTrackPersistence`,
`CCameraSequenceScriptPersistence`, `CCameraJsonPersistenceUtilities`

**Follow** — `CCameraFollowUtilities`, `CCameraFollowRegressionUtilities`

**Scripting and validation** — `CCameraSequenceScript`, `CCameraScriptedRuntime`, `CCameraScriptedCheckRunner`,
`CCameraScriptedUiInputUtilities`, `CCameraSmokeRegressionUtilities`

**Text and UI** — `CCameraTextUtilities`, `CCameraControlPanelUiUtilities`,
`CCameraScriptVisualDebugOverlayUtilities`, `CCameraViewportOverlayUtilities`

The last three came from `examples_tests/common/include/camera/`, where they also had only this example as a user.

**Small helpers, moved later for the same reason** — `CCameraProjectionUtilities` (pushes a dynamic-perspective FOV
into a projection entry), `CFileUtilities` (whole-file read/write through `ISystem`), `CInputCodeNames` (stable
string names for key codes and mouse buttons, used by the binding editor and the scripted input files),
`SCameraToolingThresholds` (the comparison tolerances of the solver, follow, presets and scripted checks) and
`SCameraPoseDelta` with `tryComputePoseDelta` (in `CCameraGoal`)

Files already local to this example before the move, and not part of it: `CCameraConstraintUtilities`,
`CCameraScriptedActionUtilities`, `CCameraScriptedRuntimePersistence`, `CCameraSequenceScriptedBuilder`.
