# 61_UI Cameraz

This example demonstrates interactive camera control in the ImGui-based UI sample.
It contains a scripted-input harness that can drive camera actions in CI and validate behavior with frame-based checks.

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

Each planar uses one of the configured controller mappings and can be switched at runtime by scripted `action` events.

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

Per camera sequence:

1. select planar
2. store `baseline`
3. hold camera segment for `3.0 s` (`180` frames at `60 FPS`)
4. apply 3 small `imguizmo` movement steps spread over the segment
5. run `gimbal_step` after each step

PASS means every step delta stayed inside configured continuity ranges.
FAIL means any step exceeded max range or failed minimum expected motion.

Continuity also supports visual debug mode:

- large top-center overlay with active camera type and segment progress
- fixed frame pacing (`visual_debug_target_fps`) so camera time is human-readable

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

- continuity visual run takes about `33 s` (`11` cameras × `3 s`)
- if `visual_debug` is present in json, CLI flag is optional

## CTest entries

`CMakeLists.txt` registers two dedicated tests:

- `NBL_61_UI_CAMERA_SMOKE`
- `NBL_61_UI_CAMERA_CONTINUITY`

Run from `build_vs2026/examples_tests`:

```powershell
ctest -C Debug --output-on-failure -R NBL_61_UI_CAMERA_
```
