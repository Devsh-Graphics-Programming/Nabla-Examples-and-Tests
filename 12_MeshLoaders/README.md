# 12_MeshLoaders

Example for loading and writing `OBJ`, `PLY` and `STL` meshes.

## At a glance
- Default input list: `inputs.json`
- Default mode: `batch`
- Default tuning: `heuristic`
- Output meshes: `saved/`
- Output screenshots: `screenshots/`
- Default startup resolves `inputs.json` from the example directory layout and is not tied to the current working directory

## Mode cheat sheet
- `batch`
  - Uses test list and runs normal workflow.
  - If test list has `row_view: true`, assets are laid out in one inspection scene.
- `interactive`
  - Opens file dialog and loads one model.
- `ci`
  - Runs strict pass/fail validation per case.

## Row view concept
- `row_view` means one inspection scene containing all cases from the test list.
- `geometry` and `geometry collection` assets are normalized and laid out left-to-right so camera framing is stable for comparisons.
- `scene` assets are laid out as one row tile while keeping their authored internal instance transforms.

## Common workflows
- Quick visual check:
  - run default `batch`
- Inspect one local model:
  - run with `--interactive`
- Validate load/write correctness:
  - run with `--ci`

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
  - first tries payload from `examples_tests/media/<NBL_MESHLOADERS_MEDIA_PAYLOAD_RELATIVE_PATH>`
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
  - when benchmark datasets are enabled, benchmark `ctest` also writes structured performance run artifacts
  - if a matching reference exists for the current workload and machine profile, strict comparison is enabled automatically
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
  - relative JSON path resolves against local input CWD
  - relative case paths inside the JSON resolve against the JSON file directory
- `--savegeometry`
  - keep writing output meshes
- `--savepath <path>`
  - force output path
- `--row-add <path>`
  - add model to row view at startup
  - scene assets added this way keep their internal transforms inside one row tile
- `--row-duplicate <count>`
  - duplicate last row-view case
- `--loader-perf-log <path>`
  - redirect loader diagnostics
- `--loader-content-hashes`
  - keep loader content hashes enabled
  - this is already the default for this example
- `--runtime-tuning <sequential|heuristic|hybrid>`
  - IO runtime tuning mode
- `--perf-dump-dir <path>`
  - write structured performance run JSON artifacts
- `--perf-ref-dir <path>`
  - lookup directory for structured performance references
- `--perf-strict`
  - fail on performance regression only when a matching reference exists
  - if no matching reference exists, the run stays record-only
- `--perf-profile-override <name>`
  - override the automatically derived machine profile id

## Controls (non-CI)
- Arrow keys: move camera
- Left mouse drag: rotate camera
- `Home`: reset view
- `A`: add model to row view
- `X`: clear row view inspection scene
- `R`: reload current test list or interactive model

## Input list format (`inputs.json`)
```json
{
  "row_view": true,
  "cases": [
    "../media/cornell_box_multimaterial.obj",
    { "name": "spanner", "path": "../media/ply/Spanner-ply.ply" },
    { "path": "../media/Stanford_Bunny.stl" }
  ]
}
```

Rules:
- `cases` is required and must be an array
- case item can be string path or object with `path` and optional `name`
- relative paths resolve against JSON file directory
- default startup uses `inputs.json` resolved from the example directory layout rather than the process working directory
- `row_view: true` keeps geometry assets in direct row layout and places each scene asset as one row tile with authored internal transforms preserved inside that tile.

## What CI validates
- Per-case image consistency:
  - `*_loaded.png` vs `*_written.png` code-unit diff
  - thresholds come from `MaxImageDiffCodeUnits` and `MaxImageDiffCodeUnitValue` in `App.hpp`
- Any mismatch ends with non-zero exit code

## Structured Perf Runs
- Structured performance output is keyed by:
  - `workload_id`
    - derived from the benchmark/test input definition and runtime mode
  - `profile_id`
    - derived from the current CPU-centric machine/runtime profile or overridden explicitly
- Structured performance artifacts also carry provenance:
  - `created_at_utc`
  - `nabla_commit`
  - `nabla_dirty`
  - `examples_commit`
  - `examples_dirty`
- Reference lookup uses:
  - `<perf-ref-dir>/<workload_id>/<profile_id>.json`
- If no matching reference exists:
  - no comparison is performed
  - the run only writes its current JSON artifact
- If a matching reference exists:
  - per-case `original_load`, `write`, and `written_load` stage metrics are compared
  - strict mode fails only on actual regression, not on missing references
- JSON artifacts avoid host-specific absolute paths and store portable case/test-list identifiers instead

## Performance logs to trust
- `Asset load call perf` for `getAsset`
- `Asset write call perf` for `writeAsset`

Internal loader stage logs are diagnostics only.
