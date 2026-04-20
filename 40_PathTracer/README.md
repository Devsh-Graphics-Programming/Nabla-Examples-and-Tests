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
40_pathtracer.exe --scene <scene.xml|scene.zip> [--scene-entry <scene.xml>] [--process-sensors <mode>] [--sensor <id>] [--headless] [--defer-denoise] [--output-dir <dir>] [--report-dir <dir>] [--reference-dir <dir>]
```

Examples:

```bat
40_pathtracer.exe --scene ..\media\mitsuba\kitchen.zip --scene-entry scene.xml --process-sensors RenderAllThenTerminate --sensor 0 --headless
40_pathtracer.exe --scene ..\media\mitsuba\ditt\render_2160p.xml --process-sensors RenderSensorThenInteractive --sensor 1
40_pathtracer.exe --scene "..\media\mitsuba\my scene.zip" --scene-entry "shots\camera_a.xml" --defer-denoise --output-dir renders
40_pathtracer.exe --scene ..\media\mitsuba\shapetest.xml --process-sensors RenderAllThenTerminate --headless --output-dir out\current\renders --report-dir out\current --reference-dir out\reference\renders
```

Comparison options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--compare-error-threshold` | `0.05` | Relative per-channel threshold used when both compared values are above epsilon. |
| `--compare-epsilon` | `0.00001` | Absolute epsilon used before switching to relative comparison. |
| `--compare-allowed-error-ratio` | `0.0001` | Allowed ratio of pixels that may exceed the threshold before an image fails. |
| `--compare-ssim-threshold` | `0.001` | Maximum allowed `1 - SSIM` difference for `output_denoised`. |

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

`output_tonemap` keeps the old CI key name. The current implementation resolves RWMC cascades into the `Beauty` image and exports that image as the final screenshot. `output_rwmc_cascades` is a debug export of the raw RWMC cascade resource, so it can match the final screenshot in single-cascade scenes. AOV resources are exported separately, and `output_denoised` is written through the internal postprocess hook. The example emits EXR artifacts only.

## Report Bundle

The path tracer writes the runtime report payload only. The static viewer lives in `report/` and reads `summary.json` plus the referenced EXR artifacts. With the generated CMake runtime config the default layout is:

```text
report/
  index.html
  server.py
  css/
    report.css
  js/
    report.js
    exrPreview.js
bin/
  config/
    pt.<config>.json
  out/
    <config>/
      summary.json
      renders/
        Render_<scene>.exr
        Render_<scene>_rwmc_cascades.exr
        Render_<scene>_albedo.exr
        Render_<scene>_normal.exr
        Render_<scene>_denoised.exr
      references/
        <scene>/
      diff_images/
        <scene>/
```

`--report-dir` selects where `summary.json`, copied references and difference images are written. `--output-dir` selects where the current render images are written. Paths from the generated `pt.<config>.json` are resolved relative to `bin/config`, so the default executable run writes to `bin/out/<config>`.

When `--reference-dir` is not provided, the report is generated with `not-checked` status and no pass/fail image comparison. When it is provided, the path tracer loads reference EXRs through Nabla's asset pipeline, compares them in-process, writes EXR diff images, copies reference images into the report bundle, and exits with failure if any compared image exceeds the configured threshold. Tonemap, RWMC cascades, albedo and normal outputs use pixel-error comparison. Denoised output uses `1 - SSIM` and reports it as `Difference (SSIM)`. Reference lookup first checks `<reference-dir>/<scene>/<filename>` and then `<reference-dir>/<filename>`.

Report metadata is written with portable paths, so the payload can be moved or published without embedding the original host workspace path.

Run the local viewer once while iterating on report UI:

```bat
python report\server.py
```

The server serves the `40_PathTracer` directory on `127.0.0.1` and opens `/report/`. It does not scan or know about build configurations, output folders or report payloads. The viewer chooses a payload only from the browser URL fragment. Editing `report/index.html`, `report/css/report.css` or `report/js/*.js` affects the live report after a browser refresh without rebuilding or regenerating runtime output.

The selected payload is encoded only in the browser URL fragment as a path relative to the served root. For example, `/report/#/bin/out/my-run` loads `40_PathTracer/bin/out/my-run/summary.json`. Multiple arbitrary report directories can be inspected in one server session by changing only the URL fragment. The fragment is client-side state and is not handled by `server.py`.

The report starts with a compact scene overview table for multi-scene CI runs. Each row links to a collapsible detail section with render, reference and difference EXR links.

The report viewer decodes raw OpenEXR files directly in browser JavaScript and draws them into a canvas. It fetches the `.exr` links over the local HTTP server. The viewer supports fit, 1:1 zoom, wheel zoom, drag panning and compact raw pixel value inspection.

No external image conversion tool is required.

## Postprocess Hook

EX40 does not launch an external denoiser. The finalization step lives inside the example.

- Immediate mode runs finalization after each sensor finishes.
- Deferred mode queues finalization until shutdown with `--defer-denoise` or `-DEFER_DENOISE`.
- Current finalization is a no-op copy from `output_tonemap` to `output_denoised`.

This keeps the CLI and JSON shape ready for the denoise, tonemap and bloom work without adding another executable boundary.
