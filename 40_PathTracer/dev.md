# 40_PathTracer Developer Workflow

This file covers local validation, report UI work and publishing the generated report payload. Runtime CLI usage lives in `README.md`.

## Build

Build only the path tracer example from the examples build tree:

```bat
cmake --build <build-dir>\examples_tests\40_PathTracer --target 40_pathtracer --config RelWithDebInfo
```

The executable is produced under:

```text
40_PathTracer/bin
```

The generated runtime config files are under:

```text
40_PathTracer/bin/config/pt.debug.json
40_PathTracer/bin/config/pt.release.json
40_PathTracer/bin/config/pt.relwithdebinfo.json
```

## Local Report Viewer

The C++ runtime writes only `summary.json` and EXR artifacts. The static viewer is committed under `report/` and reads a selected report payload from the browser URL fragment.

Start the viewer once:

```bat
cd 40_PathTracer
python report\server.py
```

Open any payload by changing only the fragment:

```text
http://127.0.0.1:8040/report/#/bin/out/ci_public
http://127.0.0.1:8040/report/#/bin/out/ci_private
http://127.0.0.1:8040/report/#/bin/out/my_test_run
```

`server.py` is config-agnostic. It serves the whole example directory and does not scan build configurations. Editing `report/index.html`, `report/css/report.css` or `report/js/*.js` affects the live report after a browser refresh without rebuilding or regenerating runtime output.

The viewer decodes raw OpenEXR files in browser JavaScript. It supports fit, 1:1 zoom, wheel zoom, drag panning and raw pixel inspection.

## CI-Like Public Run

Run from `40_PathTracer/bin`:

```bat
40_pathtracer_rwdi.exe --scene-list ..\..\media\mitsuba\public_test_scenes.txt --process-sensors RenderAllThenTerminate --headless --output-dir out\ci_public\renders --report-dir out\ci_public --reference-dir ..\..\..\ci\22.RaytracedAO\references\public
```

Output payload:

```text
40_PathTracer/bin/out/ci_public
```

## CI-Like Private Run

Run from `40_PathTracer/bin`:

```bat
40_pathtracer_rwdi.exe --scene-list ..\..\media\Ditt-Reference-Scenes\private_test_scenes.txt --process-sensors RenderAllThenTerminate --headless --output-dir out\ci_private\renders --report-dir out\ci_private --reference-dir ..\..\..\ci\22.RaytracedAO\references\private
```

Output payload:

```text
40_PathTracer/bin/out/ci_private
```

## Expected Validation State

The report should always contain all reference images that exist in the selected reference directory, even if current rendering fails. A missing render should be reported as `missing-render`. A missing expected reference should be reported as `missing-reference`.

Do not update references during normal validation runs. Reference refresh is a separate intentional operation.

## Report Smoke Iteration

For fast UI and summary logic iteration, use a small temporary report directory instead of rerunning the full path tracer set. The viewer only needs:

```text
<payload>/summary.json
<payload>/renders/*.exr
<payload>/references/**/*.exr
<payload>/diff_images/**/*.exr
```

Keep smoke payloads out of source control.

## S3 Publish

`report/publishS3.py` uploads a finished payload directory to an S3-compatible bucket. It skips files that are already up to date by size and modification time. Use `--checksum` only when exact content verification is needed.

Public:

```bat
python report\publishS3.py --source bin\out\ci_public --bucket devsh-store-prod --prefix ditt/public --jobs 16
```

Private:

```bat
python report\publishS3.py --source bin\out\ci_private --bucket devsh-store-prod --prefix ditt/private --jobs 16
```

Dry run:

```bat
python report\publishS3.py --source bin\out\ci_public --bucket devsh-store-prod --prefix ditt/public --dry-run --jobs 16
```

The script expects S3 credentials in environment variables such as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` or the matching Scaleway aliases. Endpoint and region can be passed with `--endpoint` and `--region`. It does not manage retention and does not know about CI concepts.

After upload, verify:

```text
https://store.devsh.eu/ditt/public/
https://store.devsh.eu/ditt/private/
```

Private reports are protected at the web server layer.

## Review Checklist

Before pushing a review checkpoint:

1. Build `40_pathtracer` in `RelWithDebInfo`.
2. Run a small report smoke payload if report code changed.
3. Run public and private CI-like payload generation when validating the full pipeline.
4. Publish public and private payloads to S3.
5. Open both store URLs and verify the report, summary counts, preview switching and EXR links.
