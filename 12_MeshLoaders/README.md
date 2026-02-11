# 12_MeshLoaders

Example for loading and writing `OBJ`, `PLY` and `STL` meshes.

## At a glance
- Default input list: `inputs.json`
- Default mode: `batch`
- Default tuning: `heuristic`
- Loader content hashes: enabled by default
- Output meshes: `saved/`
- Output screenshots: `screenshots/`

## Mode cheat sheet
- `batch`
  - Uses test list and runs normal workflow.
  - If test list has `row_view: true`, renders all cases in one scene.
- `interactive`
  - Opens file dialog and loads one model.
- `ci`
  - Runs strict pass/fail validation per case.

## Common workflows
- Quick visual check:
  - run default `batch`
- Inspect one local model:
  - run with `--interactive`
- Validate load/write correctness:
  - run with `--ci`
- Refresh geometry references:
  - run with `--update-references` (usually with `--ci`)

## CLI
- `--ci`
  - strict validation run
- `--interactive`
  - file-dialog run
- `--testlist <path>`
  - custom JSON list
- `--savegeometry`
  - keep writing output meshes
- `--savepath <path>`
  - force output path
- `--row-add <path>`
  - add model to row view at startup
- `--row-duplicate <count>`
  - duplicate last row-view case
- `--loader-perf-log <path>`
  - redirect loader diagnostics
- `--runtime-tuning <none|heuristic|hybrid>`
  - IO runtime tuning mode
- `--loader-content-hashes`
  - compatibility switch; already enabled by default
- `--update-references`
  - regenerate `references/*.geomhash`

## Controls (non-CI)
- Arrow keys: move camera
- Left mouse drag: rotate camera
- `Home`: reset view
- `A`: add model to row view
- `R`: reload row view from test list

## Input list format (`inputs.json`)
```json
{
  "row_view": true,
  "cases": [
    "../media/yellowflower.obj",
    { "name": "spanner", "path": "../media/ply/Spanner-ply.ply" },
    { "path": "../media/Stanford_Bunny.stl" }
  ]
}
```

Rules:
- `cases` is required and must be an array
- case item can be string path or object with `path` and optional `name`
- relative paths resolve against JSON file directory

## What CI validates
- Per-case geometry hash:
  - deterministic `BLAKE3` hash compared with `references/*.geomhash`
- Per-case image consistency:
  - `*_loaded.png` vs `*_written.png` byte diff
  - thresholds come from `MaxImageDiffBytes` and `MaxImageDiffValue` in `MeshLoadersApp.hpp`
- Any mismatch ends with non-zero exit code

## Performance logs to trust
- `Asset load call perf` for `getAsset`
- `Asset write call perf` for `writeAsset`

Internal loader stage logs are diagnostics only.
