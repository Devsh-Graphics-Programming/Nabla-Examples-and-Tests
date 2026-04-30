# 40_PathTracer

`40_PathTracer` is a scripted Mitsuba path tracer example with an `argparse` based CLI. It can render one scene, render a text list of scenes, run headless, emit EXR artifacts and optionally compare them against reference EXRs in-process.

The example also accepts the old Ditt launcher flags, so existing automation can call the new executable without a wrapper script.

## CLI

```text
40_pathtracer.exe --scene <scene.xml|scene.zip> [--scene-entry <scene.xml>] [--process-sensors <mode>] [--sensor <id>] [--headless] [--defer-denoise] [--output-dir <dir>] [--report-dir <dir>] [--reference-dir <dir>]
```

Use `--scene-list <file>` instead of `--scene` to render multiple scenes into one report payload.

Examples:

```bat
40_pathtracer.exe --scene ..\media\mitsuba\shapetest.xml --process-sensors RenderAllThenTerminate --headless
40_pathtracer.exe --scene ..\media\mitsuba\kitchen.zip --scene-entry scene.xml --process-sensors RenderAllThenTerminate --sensor 0 --headless
40_pathtracer.exe --scene "..\media\mitsuba\my scene.zip" --scene-entry "shots\camera_a.xml" --defer-denoise --output-dir renders
40_pathtracer.exe --scene-list scenes.txt --process-sensors RenderAllThenTerminate --headless --output-dir out\current\renders --report-dir out\current --reference-dir out\reference
```

## Sensor Workflows

`--process-sensors` accepts these modes:

| Mode | Behavior |
| --- | --- |
| `RenderAllThenInteractive` | Render every sensor starting at `--sensor` and keep the window open afterwards. |
| `RenderAllThenTerminate` | Render every sensor starting at `--sensor` and exit when the queue is finished. |
| `RenderSensorThenInteractive` | Render only the selected sensor and keep the window open afterwards. |
| `InteractiveAtSensor` | Start from the selected sensor without queueing the full sensor range. |

`--sensor` defaults to `0`.

## Ditt Compatibility

These legacy flags are translated before `argparse` validation:

| Legacy flag | New CLI equivalent |
| --- | --- |
| `-SCENE=...` | `--scene ...` and optionally `--scene-entry ...` |
| `-PROCESS_SENSORS <mode> [id]` | `--process-sensors <mode>` and optionally `--sensor <id>` |
| `-PROCESS_SENSORS=<mode> [id]` | `--process-sensors <mode>` and optionally `--sensor <id>` |
| `-DEFER_DENOISE` | `--defer-denoise` |
| `-TERMINATE` | `--process-sensors RenderAllThenTerminate` |

Examples:

```bat
40_pathtracer.exe -SCENE=..\media\mitsuba\kitchen.zip scene.xml -PROCESS_SENSORS RenderAllThenTerminate 0
40_pathtracer.exe -SCENE="..\media\mitsuba\my good kitchen.zip scene.xml" -PROCESS_SENSORS RenderSensorThenInteractive 1
40_pathtracer.exe -SCENE="..\media\mitsuba\extracted folder\scene.xml" -PROCESS_SENSORS InteractiveAtSensor 2 -DEFER_DENOISE
```

## Runtime Config

CMake generates one config file per build configuration:

```text
bin/config/pt.<config>.json
```

The executable loads the matching file at startup. Values inside `cli` behave as default CLI arguments and are resolved relative to `bin/config`. Explicit command line arguments override generated defaults.

The generated defaults write to:

```text
bin/out/<config>/renders
bin/out/<config>
```

## Runtime Package

The example exposes a runtime-only CMake install component. Install it into the
same prefix as the shared runtime component:

```bat
cmake --install <build-dir> --config Release --prefix <install-dir> --component Runtimes
cmake --install <build-dir>\examples_tests --config Release --prefix <install-dir> --component EX40Runtime
```

After both install commands, the package contains the path tracer executable,
generated runtime config, app resources, shader outputs, Nabla and DXC runtime
DLLs and the static report viewer. It also writes `EX40Runtime.json`, which
records the portable paths used by CI and other launchers.

The component intentionally does not install scene media, private assets or reference EXRs. Those inputs should be mounted or materialized separately by the caller.

## Scene Lists

`--scene-list` reads one scene command per line. Empty lines and lines starting with `;` are ignored. Each line may contain a scene path, an optional ZIP entry and per-scene comparison overrides:

```text
scene.xml --errpixel 0.05 --errcount 0.0001
scene.zip scene.xml --abs --errcount 105
```

Supported per-line overrides:

| Option | Meaning |
| --- | --- |
| `--errpixel <value>` | Relative per-channel threshold. |
| `--epsilon <value>` | Absolute epsilon before relative comparison. |
| `--errcount <value>` | Allowed bad pixels. Interpreted as ratio after `--rel` and as count after `--abs`. |
| `--errssim <value>` | Maximum allowed `1 - SSIM` for denoised output. |
| `--rel` | Interpret `--errcount` as a ratio of image pixels. |
| `--abs` | Interpret `--errcount` as an absolute bad-pixel count. |
| `--defer-denoise` or `-DEFER_DENOISE` | Queue postprocess until shutdown for that scene. |

## Outputs

For every completed sensor render the executable writes EXR files and emits one JSON record to stdout:

```json
{
  "output_tonemap": "Render_scene_Sensor_0.exr",
  "output_rwmc_cascades": "Render_scene_Sensor_0_rwmc_cascades.exr",
  "output_albedo": "Render_scene_Sensor_0_albedo.exr",
  "output_normal": "Render_scene_Sensor_0_normal.exr",
  "output_denoised": "Render_scene_Sensor_0_denoised.exr"
}
```

The record is wrapped in `[JSON]` and `[ENDJSON]`.

`output_tonemap` keeps the old CI key name. `output_rwmc_cascades` is a debug export of the raw RWMC cascade resource. It may look close to the final screenshot in simple single-cascade scenes. AOV resources are exported separately.

`output_denoised` is currently produced by an internal no-op postprocess copy from `output_tonemap`. This keeps the CLI and artifact shape stable until real denoising lands.

## Comparison Report Payload

`--report-dir` enables writing `summary.json`, copied references and difference EXRs. `--reference-dir` enables pass/fail comparison against reference EXRs.

Tonemap, RWMC cascades, albedo and normal outputs use pixel-error comparison. Denoised output uses `1 - SSIM` and reports it as `Difference (SSIM)`.

Comparison options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--compare-error-threshold` | `0.05` | Relative per-channel threshold used when both compared values are above epsilon. |
| `--compare-epsilon` | `0.00001` | Absolute epsilon used before switching to relative comparison. |
| `--compare-allowed-error-ratio` | `0.0001` | Allowed ratio of pixels that may exceed the threshold before an image fails. |
| `--compare-allowed-error-count` | not set | Absolute allowed count of pixels that may exceed the threshold. |
| `--compare-ssim-threshold` | `0.001` | Maximum allowed `1 - SSIM` for `output_denoised`. |

Reference lookup first checks `<reference-dir>/<scene>/<filename>` and then `<reference-dir>/<filename>`.

The runtime report metadata uses portable paths, so the payload can be moved or published without embedding the original workspace path.

## Report Bundle Compare

Completed report bundles can be compared without rendering scenes again:

```bat
40_pathtracer.exe --compare-reports --candidate-report out\amd --candidate-name AMD --baseline-report out\nvidia --baseline-name NVIDIA --comparison-name "AMD vs NVIDIA" --report-dir out\compare\amd_vs_nvidia
```

For multiple inputs, write a small manifest and compare each input against one baseline:

```json
{
  "name": "Release compare smoke",
  "baseline": "nvidia",
  "inputs": [
    { "id": "nvidia", "name": "NVIDIA", "reportDir": "nvidia" },
    { "id": "amd", "name": "AMD", "reportDir": "amd" },
    { "id": "intel", "name": "Intel", "reportDir": "intel" }
  ]
}
```

```bat
40_pathtracer.exe --compare-report-set out\compare_vendor_smoke\set.json --report-dir out\compare_vendor_smoke\set
```

The compare payload is self-contained and relocatable. Pair reports copy the compared EXR artifacts into the compare directory, so the output can be served or uploaded without the source report directories.

Run the local smoke from the example directory:

```bat
python report\compareSetSmoke.py --exe bin\40_pathtracer.exe --output-dir bin\out\compare_vendor_smoke_local
```

Inside an installed package, run it from the installed executable directory and
provide a scene path:

```bat
python report\compareSetSmoke.py --scene <scene.xml> --output-dir out\compare_vendor_smoke_local
```
