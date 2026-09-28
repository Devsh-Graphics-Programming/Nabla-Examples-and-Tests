# PT Bench Viewer

A small GUI to compare 40_PathTracer benchmark runs: it reads the `bench.json` +
`*.exr` that a `--benchmark` run drops under `bin/benchmarks/`, groups the
apples-to-apples variants, and scores them with FLIP (vs a converged reference),
a reference-free self-noise metric, mean luma (energy), and an efficiency number.

## 1. Install Python + dependencies

Python 3.10+ (3.13 recommended; **not** a bare 3.14 unless you install the wheels
into it). Then:

```
pip install numpy opencv-python pillow flip-evaluator
```

`flip-evaluator` is optional: without it the FLIP / noise columns stay blank but
everything else (preview, luma, diff heatmap) still works.

## 2. Put the reference folders in place

Copy the `*_reference` folders you were given into:

```
40_PathTracer/bin/benchmarks/
```

So you end up with, for example:

```
40_PathTracer/bin/benchmarks/
  render_720p_s0_reference/      <- reference (ground truth)
  render_720p_s0_20260603_.../   <- a benchmark run you produced
  daily_pt_s0_reference/
  ...
```

**How references are matched:** a run dir named `render_720p_s0_<timestamp>` uses
the reference dir `render_720p_s0_reference` (same prefix, with `_reference`
appended). The viewer averages the EXRs in that reference dir to form the ground
truth that FLIP is measured against.

**Per-mode runs** (when a run compares MIS modes Both / NEEOnly / BxDFOnly) expect
the reference split into one subfolder per mode, because each mode converges to a
different image:

```
render_720p_s0_reference/
  Both/       *.exr
  NEEOnly/    *.exr
  BxDFOnly/   *.exr
```

The viewer reads `<prefix>_reference/<mode>/`. Older single-mode runs just use the
flat `<prefix>_reference/*.exr` and need no subfolders.

> The first time a reference is read it is averaged and cached as
> `.combined_reference.npy` inside that folder, so later launches are fast. The
> cache auto-invalidates if the EXRs change.

## 3. Run

```
python scripts/bench_viewer.py bin/benchmarks
```

(Run it from the `40_PathTracer` directory, or pass an absolute path to the
benchmarks dir. With no argument it defaults to `./benchmarks`.) You can also use
the **Open Folder...** button to point it at any benchmarks dir, and **Refresh**
to re-scan after producing new runs.

## Using it

- The left tree lists runs, then groups (one per mode), then variant rows. Columns:
  `ps/sample`, `ms total`, `dispatches`, `FLIP vs ref`, `noise` (reference-free),
  `luma` (absolute Rec.709 mean), `FLIP²*ms` (efficiency, lower=better). FLIP/noise/
  luma fill in the first time you select a group.
- Selecting a **group** shows a side-by-side composite (tonemap | FLIP heatmap |
  diff vs reference | self-noise heatmap) per variant.
- Preview controls: mouse wheel = zoom, drag = pan, double-click = fit, EV slider =
  exposure.
- With a variant selected: **Left** = the variant, **Right** = its reference,
  **N** = toggle the bilateral-denoised version. Zoom/pan stays fixed so flicking
  between them reveals differences in place.
- **Right-click** a row or the image: open the EXR in your default viewer, or open
  the run folder.
- Notes per variant autosave to `notes.json` next to `bench.json`.
