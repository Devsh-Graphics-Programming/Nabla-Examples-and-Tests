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

## Optional benchmark datasets via CMake
- Use this when you want larger/public inputs downloaded automatically.
- Public dataset repository:
  - `https://github.com/Devsh-Graphics-Programming/Nabla-Benchmark-Datasets`
- Configure options:
  - `NBL_MESHLOADERS_ENABLE_BENCHMARK_DATASETS=ON`
  - `NBL_MESHLOADERS_DEFAULT_START_WITH_BENCHMARK_TESTLIST=ON|OFF` (default: `OFF`)
  - `NBL_MESHLOADERS_BENCHMARK_DATASET_DIR=<path>` (optional, default: build dir)
  - `NBL_MESHLOADERS_BENCHMARK_DATASET_REPO=<git-url>` (optional, default: public repo above)
  - `NBL_MESHLOADERS_BENCHMARK_PAYLOAD_RELATIVE_PATH=<path>` (optional, default: `inputs_benchmark.json`)
- What CMake does:
  - fetches/clones dataset repo during configure via CMake `FetchContent` (if payload file is missing)
  - resolves committed payload JSON from repo:
  - `<dataset_dir>/<payload_relative_path>`
  - verifies payload is a regular Git file (not an LFS pointer)
- Run benchmark list with:
  - `--testlist <dataset_dir>/<payload_relative_path>`
- Default startup behavior when benchmark datasets are enabled:
  - `NBL_MESHLOADERS_DEFAULT_START_WITH_BENCHMARK_TESTLIST=OFF`: still starts from local `inputs.json` (3 models)
  - `NBL_MESHLOADERS_DEFAULT_START_WITH_BENCHMARK_TESTLIST=ON`: starts from benchmark payload test list
- Run benchmark CI directly via `ctest`:
  - `ctest --output-on-failure -C Debug -R NBL_MESHLOADERS_CI_BENCHMARK`
  - runs both benchmark CI modes: `heuristic` and `hybrid`
  - benchmark CTest uses `--update-references` for payload-driven case names
- Run default CI directly via `ctest` (no benchmark datasets enabled):
  - `ctest --output-on-failure -C Debug -R ^NBL_MESHLOADERS_CI$`
  - uses default `inputs.json` (3 inputs)

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
- `X`: clear row view (empty scene)
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
