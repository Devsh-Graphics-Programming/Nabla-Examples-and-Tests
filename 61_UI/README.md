# 61_UI Cameraz

This example demonstrates interactive camera control in the ImGui-based UI sample.
It contains a scripted-input harness that can drive camera actions in CI and validate behavior with frame-based checks.

## Shared camera API

`61_UI` is the active integration and validation surface for the reusable camera stack in `../common/include/camera`.

See:

- [`../common/include/camera/README.md`](../common/include/camera/README.md)

That shared layer covers:

- virtual gimbal events
- binding layouts and runtime input binders
- reusable camera kinds and typed state hooks
- best-effort goal capture and apply utilities
- preset and keyframe-track storage helpers
- tracked-target and follow helpers built on top of shared goals

At the moment other examples are not being migrated yet.
The reusable API is growing in `common/include/camera`, while `61_UI` stays the only active call-site.

## Cameras in this scene

`app_resources/cameras.json` defines 11 camera types:

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

Each planar uses one of the configured input binding layouts and can be switched at runtime by scripted `action` events.

## Follow target integration

`61_UI` also exposes one tracked target in the default scene.

That target owns its own gimbal and is integrated through the shared follow layer rather than
through direct camera hacks. The example can:

- manipulate the tracked target with the scene gizmo
- show a marker for the tracked target
- let selected cameras follow it through reusable follow modes

The current default follow setup is:

- `Orbit`, `Arcball`, `Turntable`, `TopDown`, `Isometric`, `DollyZoom`, `Path`
  use `OrbitTarget`
- `Chase`, `Dolly`
  use `KeepLocalOffset`

The follow layer is live-scene behavior only for now.
Scripted continuity/CI runs intentionally do not re-apply follow every frame yet, because the
current sequence asset does not author target animation separately from camera motion.

## Short math context

Each camera is represented by a gimbal pose `(R, p)` and produces a view matrix from camera basis vectors and position.

For a world-space point `x_w`, clip-space projection is:

`x_c = P * V * x_w`

where:

- `V` is camera view transform
- `P` is selected planar projection (perspective or orthographic)

The scripted smoothness checks use per-step deltas:

- position delta: `d_pos = ||p_t - p_{t-1}||`
- rotation delta: `d_rot = max(angleDiff(euler_t, euler_{t-1}))`

and validate them against configured `[min, max]` ranges.

## Scripted test assets

- `app_resources/cameraz_smoke_all.json`
- `app_resources/cameraz_continuity.json`

### Smoke script

Goal: verify that every camera can be selected and responds to scripted input.

Per camera sequence:

1. select planar
2. store `baseline`
3. apply one `imguizmo` movement step
4. run `gimbal_step` check

PASS means each camera produced a finite and expected movement delta.
FAIL means missing movement, out-of-range movement, invalid state, or missing reference.

### Continuity script

Goal: verify smooth frame-to-frame behavior (no visible teleport-like jumps).

The continuity asset is now a compact authored camera-sequence spec, not a committed frame dump.
`61_UI` expands that shared camera-domain description into its own runtime scripted checks.

Per authored segment:

1. select planar
2. store `baseline`
3. build a reusable keyframe track from the active camera reference preset
4. sample that track for `4.0 s` at `60 FPS`
5. run `gimbal_step` on each generated frame step
6. capture selected milestones such as `end`

PASS means every step delta stayed inside configured continuity ranges.
FAIL means any step exceeded max range or failed minimum expected motion.

Continuity also supports visual debug mode:

- large top-center overlay with active camera type and segment progress
- fixed frame pacing (`visual_debug_target_fps`) so camera time is human-readable
- compact authored JSON that stays in camera-domain and is reusable outside `61_UI`

## Build and run

Build this example first:

```powershell
cmake --build build_vs2026/examples_tests/61_UI --config Debug --target 61_ui -- /m:1
```

Run manually from executable directory:

```powershell
./61_ui_d.exe --script app_resources/cameraz_smoke_all.json --script-log
```

For CI-style exit with automatic screenshot/capture behavior:

```powershell
./61_ui_d.exe --ci --script app_resources/cameraz_continuity.json --script-log --script-visual-debug
```

Notes:

- continuity visual run takes about `47 s`
- the authored continuity JSON is compact and segment-based rather than frame-by-frame
- if `visual_debug` is present in json, CLI flag is optional

## CTest entries

`CMakeLists.txt` registers two dedicated tests:

- `NBL_61_UI_CAMERA_SMOKE`
- `NBL_61_UI_CAMERA_CONTINUITY`

Run from `build_vs2026/examples_tests/61_UI`:

```powershell
ctest -C Debug --output-on-failure -R NBL_61_UI_CAMERA_
```
