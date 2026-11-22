# 61_UI Cameraz

`61_UI` is the full runnable integration and validation target for the shared camera stack documented in [`../../include/nbl/ext/Cameras/README.md`](../../include/nbl/ext/Cameras/README.md).

If you want the reusable API design, start there first.
This README focuses on what `61_UI` adds on top of the shared layer.

## Role of this example

`61_UI` is used to:

- exercise all current camera kinds in one visible scene
- exercise the extension's input layer, and the goal, preset, playback, follow and scripted layers kept in [`include/camera/`](include/camera/)
- validate `referenceFrame` behavior across all camera kinds
- provide a manual playground for camera behavior
- provide CI-oriented smoke and continuity coverage

It does not define camera semantics.
It consumes the camera extension and exposes it through one concrete, testable app.

## What `61_UI` owns locally

The camera extension owns camera semantics: the rigs, `manipulate(...)` and `setPose(...)`, control frames, the
mouse/keyboard controller and the projections.

Everything else lives in this example.

[`include/camera/`](include/camera/) is a staging area for code that used to be in the extension and has this example
as its only user. Its [README](include/camera/README.md) explains why each part moved and what has to be true before
it goes back:

- goal capture and apply, presets, keyframes, playback and their persistence
- follow and follow regression
- compact sequence scripts, the scripted runtime and the scripted check runner
- small helpers: dynamic-perspective FOV sync (`CCameraProjectionUtilities`), whole-file I/O (`CFileUtilities`) and
  key/mouse-button names (`CInputCodeNames`)

The rest is the glue needed to turn that into an application:

- scene setup and demo geometry
- planar / window routing
- explicit render-window selection for planar editing
- ImGui control panel
- transform editor and gizmo glue
- screenshot capture
- runtime logging and failure reporting
- local visual-debug presentation

## Camera set

`app_resources/cameras.json` configures the showcased cameras.
The current set is:

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
- Path Rig

These are exposed through the active planar / viewport configuration in the UI.

## Follow target

`61_UI` exposes one tracked target in the default scene.

Tracked-target rule:

- the tracked subject is `CTrackedTarget` from [`CCameraFollowUtilities.hpp`](include/camera/CCameraFollowUtilities.hpp)
- it owns its own gimbal
- it is not the large cone mesh
- the rendered marker is only a visualization of the tracked-target gimbal

This matters because the follow layer is modeled around:

- tracked-target pose
- follow mode
- follow config

and not around a scene-node or mesh id.

### Default follow usage

The default scene uses:

- `Free`
  with `LookAtTarget`
- `Orbit`, `Arcball`, `Turntable`, `TopDown`, `Isometric`, `DollyZoom`, `Path Rig`
  with `OrbitTarget`
- `Chase`, `Dolly`
  with `KeepLocalOffset`

Manual runtime and scripted continuity both drive the same follow layer.

## Scripted assets

`61_UI` currently ships two camera-focused scripted assets:

- `app_resources/cameraz_smoke_all.json`
- `app_resources/cameraz_continuity.json`

### Smoke

Purpose:

- validate basic camera selection and movement
- validate `referenceFrame` application for every runtime camera kind
- validate shared helpers in a short, cheap run

### Continuity

Purpose:

- validate smooth frame-to-frame motion
- validate follow lock while the tracked target moves
- validate typed restore and replay paths against the same camera semantics
- provide a readable visual-debug showcase

The continuity asset is a compact authored camera-sequence script.
It is no longer a giant committed frame dump.

## Extension pieces consumed by `61_UI`

From the camera extension:

- every camera kind, through [`ICamera.hpp`](../../include/nbl/ext/Cameras/ICamera.hpp)
- [`SCameraControls.hpp`](../../include/nbl/ext/Cameras/SCameraControls.hpp)
- [`CCameraMouseKeyboardController.hpp`](../../include/nbl/ext/Cameras/CCameraMouseKeyboardController.hpp)
- [`CCameraMouseKeyboardPresets.hpp`](../../include/nbl/ext/Cameras/CCameraMouseKeyboardPresets.hpp)
- [`CCameraWithProjections.hpp`](../../include/nbl/ext/Cameras/CCameraWithProjections.hpp) and [`CPlanarProjection.hpp`](../../include/nbl/ext/Cameras/CPlanarProjection.hpp)
- [`CCameraPathUtilities.hpp`](../../include/nbl/ext/Cameras/CCameraPathUtilities.hpp) and [`CCameraKindUtilities.hpp`](../../include/nbl/ext/Cameras/CCameraKindUtilities.hpp)

The scripting model, follow math and camera restore logic are local, in [`include/camera/`](include/camera/).

## Reference-frame and gizmo validation

`61_UI` is also the concrete harness for the shared `referenceFrame` seam used by ImGuizmo and other pose-driven tools.

The current smoke coverage checks:

- rigid reference application for `FPS` and `Free`
- legal-state projection from `referenceFrame` for all target-relative cameras
- typed `Path Rig` projection through the active path model
- restore back to baseline after reference-frame application

`61_UI` is also the app used to exercise world-space and local-space gizmo semantics end-to-end against the shared camera API.

## Local build and run

Current local setup uses the Visual Studio 2022 dynamic preset.

Configure:

```powershell
cmake --preset user-configure-dynamic-msvc
```

Build:

```powershell
cmake --build build/dynamic/examples_tests/61_UI --config Debug --target 61_ui -- /m
```

Run tests:

```powershell
ctest --test-dir build/dynamic/examples_tests/61_UI -C Debug --output-on-failure -R NBL_61_UI_CAMERA_
```

Run the example:

```powershell
examples_tests/61_UI/bin/61_ui_d.exe
```

Run CI-style screenshot capture:

```powershell
examples_tests/61_UI/bin/61_ui_d.exe --ci
```

Run smoke-style scripted playback:

```powershell
examples_tests/61_UI/bin/61_ui_d.exe --script app_resources/cameraz_smoke_all.json --script-log
```

Run continuity with visual debug:

```powershell
examples_tests/61_UI/bin/61_ui_d.exe --ci --script app_resources/cameraz_continuity.json --script-log --script-visual-debug
```

## Typical manual workflow

1. Pick a camera and planar in the UI.
2. Drive the camera with keyboard, mouse, or ImGuizmo-backed controls.
3. Capture or restore presets if needed.
4. Move the tracked target marker.
5. Observe follow-enabled cameras and scripted overlays.

## Summary

`61_UI` is the app-layer harness around the camera extension.
It proves that the extension works end-to-end in a visible scene, with follow, presets, scripted playback, and CI validation all going through the same underlying camera semantics.
