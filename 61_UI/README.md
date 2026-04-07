# 61_UI Cameraz

`61_UI` is the current integration, UX, and validation harness for the shared camera stack in
[`../common/include/camera`](../common/include/camera/README.md).

If you want the architecture, design rationale, and reusable API breakdown, start there first.
This README focuses on what `61_UI` adds on top of the shared layer.

## Role of this example

`61_UI` is the only actively migrated consumer right now.

It is used to:

- exercise all current camera models in one scene
- validate the shared input, goal, preset, playback, follow, and scripting layers
- provide an interactive manual playground
- provide CI-oriented smoke and continuity coverage

The example is intentionally not the source of truth for camera semantics.
Its job is to consume the shared camera APIs and expose them through a visible, testable UI.

## What `61_UI` contributes locally

The reusable camera layer stops at shared camera-domain contracts.
`61_UI` adds the local glue needed to turn that into an example application:

- scene setup and demo geometry
- planar/window routing
- ImGui control panel and transform editor
- screenshot capture
- scripted visual-debug HUD
- local logging and failure reporting

That means the shared camera layer owns:

- camera semantics
- follow semantics
- compact sequence semantics
- scripted runtime payloads
- scripted check semantics

and `61_UI` owns:

- how those things are presented and driven in one concrete sample

## Cameras in the scene

`app_resources/cameras.json` configures the currently showcased camera set:

- FPS
- Orbit
- Free
- Arcball
- Turntable
- TopDown
- Isometric
- Chase
- Dolly
- DollyZoom
- Path

These are exposed through the active planar/view configuration in the example UI.

## Follow target in `61_UI`

`61_UI` exposes one tracked target in the default scene.

Important rule:

- the tracked target is the reusable `CTrackedTarget` gimbal
- it is not the large cone
- it is not any scene object id
- the rendered marker is only a visualization of that gimbal

This is important because the shared follow layer is intentionally modeled around:

- tracked target pose
- follow mode
- follow config

and not around a mesh reference.

### Default follow usage in the scene

Current default setup:

- `Orbit`, `Arcball`, `Turntable`, `TopDown`, `Isometric`, `DollyZoom`, `Path`
  use `OrbitTarget`
- `Chase`, `Dolly`
  use `KeepLocalOffset`

Manual runtime and scripted continuity both drive the same follow layer.

## Scripted assets

`61_UI` currently uses two camera-focused scripted assets:

- `app_resources/cameraz_smoke_all.json`
- `app_resources/cameraz_continuity.json`

### Smoke

Purpose:

- validate basic camera selection and movement
- validate helper contracts in a small, cheap run

### Continuity

Purpose:

- validate smooth frame-to-frame motion
- validate tracked-target follow lock during scripted target motion
- provide a readable visual-debug showcase

The continuity asset is now a compact authored camera-sequence script.
It is no longer a giant committed frame dump.

## Shared pieces consumed by `61_UI`

The example now consumes these shared scripting and follow pieces directly:

- [`CCameraSequenceScript.hpp`](../common/include/camera/CCameraSequenceScript.hpp)
- [`CCameraScriptedRuntime.hpp`](../common/include/camera/CCameraScriptedRuntime.hpp)
- [`CCameraScriptedRuntimePersistence.hpp`](../common/include/camera/CCameraScriptedRuntimePersistence.hpp)
- [`CCameraSequenceScriptedBuilder.hpp`](../common/include/camera/CCameraSequenceScriptedBuilder.hpp)
- [`CCameraScriptedCheckRunner.hpp`](../common/include/camera/CCameraScriptedCheckRunner.hpp)
- [`CCameraFollowUtilities.hpp`](../common/include/camera/CCameraFollowUtilities.hpp)
- [`CCameraFollowRegressionUtilities.hpp`](../common/include/camera/CCameraFollowRegressionUtilities.hpp)

That means `61_UI` no longer owns a private scripting model or private follow math.

## Manual usage

Typical manual workflow:

1. Pick a camera/planar in the UI.
2. Manipulate the camera through mouse, keyboard, or ImGuizmo-backed controls.
3. Use presets and playback tools if needed.
4. Move the tracked target marker.
5. Observe how follow-enabled cameras react.

## CI and validation

`CMakeLists.txt` registers two dedicated tests:

- `NBL_61_UI_CAMERA_SMOKE`
- `NBL_61_UI_CAMERA_CONTINUITY`

Run from `build_vs2026/examples_tests/61_UI`:

```powershell
ctest -C Debug --output-on-failure -R NBL_61_UI_CAMERA_
```

## Build and run

Build:

```powershell
cmake --build build_vs2026/examples_tests/61_UI --config Debug --target 61_ui -- /m:1
```

Run manual smoke-style playback:

```powershell
./61_ui_d.exe --script app_resources/cameraz_smoke_all.json --script-log
```

Run continuity in CI-style mode:

```powershell
./61_ui_d.exe --ci --script app_resources/cameraz_continuity.json --script-log --script-visual-debug
```

Notes:

- continuity visual run is about `47 s`
- `visual_debug` can also be authored in JSON
- the compact continuity asset stays camera-domain and reusable instead of storing example-specific frame dumps
