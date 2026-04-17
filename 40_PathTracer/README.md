# EX40 Path Tracer CLI

`40_PathTracer` now exposes a proper `argparse`-based CLI while keeping the old `ditt` launch flow used by Jenkins for the old pair:

- `22.RaytracedAO`
- `39.DenoiserTonemapper`

The current wiring comes from the old `ditt` Jenkins and helper scripts:

- `tests/22.RaytracedAO/Jenkinsfile` builds and runs the renderer plus denoiser
- `tests/22.RaytracedAO/test.py` launches the renderer with `-SCENE=... -PROCESS_SENSORS RenderAllThenTerminate 0`
- `22.RaytracedAO/denoiser_hook.bat` shows the old denoiser CLI shape

## Supported workflows

`--process-sensors` accepts the same high-level modes as the old renderer:

| Mode | Behavior |
| --- | --- |
| `RenderAllThenInteractive` | Render every sensor starting at `--sensor` and keep the window open afterwards. |
| `RenderAllThenTerminate` | Render every sensor starting at `--sensor` and exit when the queue is finished. |
| `RenderSensorThenInteractive` | Render only the selected sensor and keep the window open afterwards. |
| `InteractiveAtSensor` | Start from the selected sensor without queueing the full sensor range. |

`--sensor` defaults to `0`.

## New CLI

```text
40_pathtracer.exe --scene <scene.xml|scene.zip> [--scene-entry <scene.xml>] [--process-sensors <mode>] [--sensor <id>] [--headless] [--defer-denoise] [--output-dir <dir>] [--denoiser-exe <path>]
```

Examples:

```bat
40_pathtracer.exe --scene ..\media\mitsuba\kitchen.zip --scene-entry scene.xml --process-sensors RenderAllThenTerminate --sensor 0 --headless
40_pathtracer.exe --scene ..\media\mitsuba\ditt\render_2160p.xml --process-sensors RenderSensorThenInteractive --sensor 1
40_pathtracer.exe --scene "..\media\mitsuba\my scene.zip" --scene-entry "shots\camera_a.xml" --defer-denoise --output-dir renders
```

## Ditt Compatibility

The following `ditt` flags are translated to the new parser before validation:

- `-SCENE=...`
- `-PROCESS_SENSORS <mode> [id]`
- `-PROCESS_SENSORS=<mode> [id]`
- `-DEFER_DENOISE`
- `-TERMINATE`

Examples:

```bat
40_pathtracer.exe -SCENE=..\media\mitsuba\kitchen.zip scene.xml -PROCESS_SENSORS RenderAllThenTerminate 0
40_pathtracer.exe -SCENE="..\media\mitsuba\my good kitchen.zip scene.xml" -PROCESS_SENSORS RenderSensorThenInteractive 1
40_pathtracer.exe -SCENE="..\media\mitsuba\extraced folder\scene.xml" -PROCESS_SENSORS InteractiveAtSensor 2 -DEFER_DENOISE
```

## Outputs

For every completed sensor render the example emits a JSON record to stdout:

```json
{
  "output_tonemap": "Render_scene_Sensor_0.exr",
  "output_albedo": "Render_scene_Sensor_0_albedo.exr",
  "output_normal": "Render_scene_Sensor_0_normal.exr",
  "output_denoised": "Render_scene_Sensor_0_denoised.exr"
}
```

The record is wrapped in:

- `[JSON]`
- `[ENDJSON]`

This matches the old CI convention from `tests/22.RaytracedAO/test.py`.

## Denoiser

Perspective renders export raw beauty, albedo and normal EXRs and then invoke `39_DenoiserTonemapper`.

- Immediate mode: denoise right after the sensor finishes.
- Deferred mode: queue denoise jobs and execute them during shutdown with `--defer-denoise`.
- `--denoiser-exe` overrides the executable path explicitly.

Each build generates a runtime config next to the executable:

- `bin/config/pt.debug.json`
- `bin/config/pt.release.json`
- `bin/config/pt.relwithdebinfo.json`

The generated JSON stores CLI defaults under a `cli` object. `--denoiser-exe` is emitted there as a path relative to the `config` directory, so the path stays config-aware without relying on the current working directory.
