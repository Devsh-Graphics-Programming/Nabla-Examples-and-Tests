# EX40 Path Tracer CLI

`40_PathTracer` exposes an `argparse`-based CLI for scripted Mitsuba scene rendering. It also accepts the `ditt` launcher flags used by the old Jenkins flow, so automation can move to EX40 without shell glue or a separate postprocess executable.

## Supported workflows

`--process-sensors` accepts:

| Mode | Behavior |
| --- | --- |
| `RenderAllThenInteractive` | Render every sensor starting at `--sensor` and keep the window open afterwards. |
| `RenderAllThenTerminate` | Render every sensor starting at `--sensor` and exit when the queue is finished. |
| `RenderSensorThenInteractive` | Render only the selected sensor and keep the window open afterwards. |
| `InteractiveAtSensor` | Start from the selected sensor without queueing the full sensor range. |

`--sensor` defaults to `0`.

## CLI

```text
40_pathtracer.exe --scene <scene.xml|scene.zip> [--scene-entry <scene.xml>] [--process-sensors <mode>] [--sensor <id>] [--headless] [--defer-denoise] [--output-dir <dir>]
```

Examples:

```bat
40_pathtracer.exe --scene ..\media\mitsuba\kitchen.zip --scene-entry scene.xml --process-sensors RenderAllThenTerminate --sensor 0 --headless
40_pathtracer.exe --scene ..\media\mitsuba\ditt\render_2160p.xml --process-sensors RenderSensorThenInteractive --sensor 1
40_pathtracer.exe --scene "..\media\mitsuba\my scene.zip" --scene-entry "shots\camera_a.xml" --defer-denoise --output-dir renders
```

## Ditt Compatibility

The following flags are translated before `argparse` validation:

- `-SCENE=...`
- `-PROCESS_SENSORS <mode> [id]`
- `-PROCESS_SENSORS=<mode> [id]`
- `-DEFER_DENOISE`
- `-TERMINATE`

Examples:

```bat
40_pathtracer.exe -SCENE=..\media\mitsuba\kitchen.zip scene.xml -PROCESS_SENSORS RenderAllThenTerminate 0
40_pathtracer.exe -SCENE="..\media\mitsuba\my good kitchen.zip scene.xml" -PROCESS_SENSORS RenderSensorThenInteractive 1
40_pathtracer.exe -SCENE="..\media\mitsuba\extracted folder\scene.xml" -PROCESS_SENSORS InteractiveAtSensor 2 -DEFER_DENOISE
```

## Outputs

For every completed sensor render the example emits a JSON record to stdout:

```json
{
  "output_tonemap": "Render_scene_Sensor_0.exr",
  "output_rwmc_cascades": "Render_scene_Sensor_0_rwmc_cascades.exr",
  "output_albedo": "Render_scene_Sensor_0_albedo.exr",
  "output_normal": "Render_scene_Sensor_0_normal.exr",
  "output_denoised": "Render_scene_Sensor_0_denoised.exr"
}
```

The record is wrapped in:

- `[JSON]`
- `[ENDJSON]`

`output_tonemap` keeps the old CI key name. The current implementation exports the rendered RWMC cascade view as the final screenshot, exports the AOV resources separately, and writes `output_denoised` through the internal postprocess hook.

## Postprocess Hook

EX40 does not launch an external denoiser. The finalization step lives inside the example.

- Immediate mode runs finalization after each sensor finishes.
- Deferred mode queues finalization until shutdown with `--defer-denoise` or `-DEFER_DENOISE`.
- Current finalization is a no-op copy from `output_tonemap` to `output_denoised`.

This keeps the CLI and JSON shape ready for the denoise, tonemap, bloom and Beauty resolve work without adding another executable boundary.
