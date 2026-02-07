# 12_MeshLoaders

Loads and writes OBJ, PLY, and STL meshes. Default run reads `meshloaders_inputs.json` from this folder. Relative paths in that file resolve against the JSON file location.

Modes
- Default: row view if `row_view` is true in the JSON
- `--interactive`: single file dialog
- `--ci`: sequential load, write, reload, hash and image compare, then exit

Controls (non CI)
- Arrow keys: move camera
- Left mouse drag: rotate
- Home: reset view
- A: add a model to row view
- R: reload test list for row view

Test list
- `cases` can be a list of strings. Each string is a file path relative to the JSON file.

Args
- `--testlist <path>`
- `--savegeometry`
- `--savepath <path>`
- `--row-add <path>`
- `--row-duplicate <count>`

Performance (Debug, Win11, Ryzen 5 5600G, RTX 4070, 64 GiB RAM)
- Dataset:
  - `yellowflower.obj` (104416 bytes)
  - `Spanner-ply.ply` (5700266 bytes)
  - `Stanford_Bunny.stl` (5620184 bytes)
- Method:
  - 9 sequential runs per format
  - compared `master_like_oldalgo` vs `latest_optimized`
  - measured `getAsset` and `writeAsset` call times from example logs

Median summary

| Asset | Load old ms | Load latest ms | Load speedup x | Write old ms | Write latest ms | Write speedup x |
|---|---:|---:|---:|---:|---:|---:|
| `yellowflower.obj` | 31.657 | 25.988 | 1.22 | 543.659 | 156.585 | 3.47 |
| `Spanner-ply.ply` | 1020.151 | 132.630 | 7.69 | 45.458 | 41.828 | 1.09 |
| `Stanford_Bunny.stl` | 36153.774 | 23.387 | 1545.89 | 17324.853 | 209.200 | 82.81 |

Why old path was slow
- STL loader used tiny scalar reads in binary path (`4` bytes per float), which amplified IO call overhead.
- STL writer emitted many small writes per triangle (`normal + v0 + v1 + v2 + attr`).
- OBJ/PLY writers performed incremental small writes while building text output.
- IO strategy was hardcoded per loader/writer, without one shared policy for tuning.

Why current path is better
- One shared `SFileIOPolicy` is available in load/write params for all formats.
- Strategy is explicit (`Auto`, `WholeFile`, `Chunked`) with one resolution path and limits.
- `Auto` can use whole-file for small payloads and chunked IO for larger ones.
- Loader perf logs include requested/effective strategy and timing breakdown.

Raw benchmark data (full per-run tables)
- `tmp/master_vs_latest_debug.md`
- `tmp/bench_masterlike_vs_latest_debug_2026-02-07_v2/raw_runs.csv`
- `tmp/bench_masterlike_vs_latest_debug_2026-02-07_v2/paired_runs.csv`

https://github.com/user-attachments/assets/6f779700-e6d4-4e11-95fb-7a7fddc47255

