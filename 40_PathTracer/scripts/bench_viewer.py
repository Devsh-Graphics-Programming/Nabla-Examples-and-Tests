"""
Bench viewer / comparator for 40_PathTracer.

Reads benchmarks/<run-dir>/{bench.json,*.exr,notes.json} produced by
runBenchmarkOnce in main.cpp. Within each apples-to-apples group (rows whose
"name" arrays agree on every entry except the LAST), it:
  * picks the bench winner (lowest ps_per_sample)
  * picks the noise winner via FLIP, scoring each variant against the OTHER
    variant as reference (symmetric, no ground truth needed)

UI: tkinter (stdlib) + Pillow for preview + OpenCV for EXR load + flip-evaluator
for the noise metric. Per-run notes are saved next to bench.json as notes.json.

Run:
    python bench_viewer.py [benchmarks_dir]

Optional deps (install whichever are missing):
    pip install pillow opencv-python numpy flip-evaluator
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

# Must be set BEFORE `import cv2`, OpenCV registers (or skips) the OpenEXR
# codec at import time and won't recheck this env var later. Force-set
# unconditionally (setdefault leaves a pre-existing "0" alone, which is the
# whole problem we hit).
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# --- optional deps with helpful errors ---------------------------------------
try:
    import numpy as np
except ImportError:
    print("missing numpy. install with: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import cv2  # OpenEXR enabled via the OPENCV_IO_ENABLE_OPENEXR set above
except ImportError:
    print("missing opencv. install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("missing pillow. install with: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    import flip_evaluator as flip  # flip-evaluator pip package
    _HAS_FLIP = True
except ImportError:
    _HAS_FLIP = False


def open_path(path: Path):
    """Open a folder or file with its Windows default handler (Explorer for dirs,
    the default image app for an .exr)."""
    try:
        os.startfile(str(path))                                      # type: ignore[attr-defined]  # Windows-only
    except Exception as e:
        messagebox.showerror("open", f"{path}: {e}")


# --- data --------------------------------------------------------------------
@dataclass
class BenchRow:
    name: list[str]            # full name tuple, e.g. ["PathTracer","Beauty","daily_pt/sensor0","nee-alias"]
    ps_per_sample: float
    gsamples_per_s: float
    ms_total: float
    regs: int = 0
    code_bytes: int = 0
    dispatches: int = 0
    exr_path: Optional[Path] = None

    @property
    def variant(self) -> str:
        return self.name[-1] if self.name else "?"

    @property
    def mode(self) -> Optional[str]:
        # MIS mode is its own name segment (Both/NEEOnly/BxDFOnly), inserted before
        # the technique. Present only in the 5-segment layout; legacy 4-segment runs
        # (technique-only) have no mode and return None.
        return self.name[-2] if len(self.name) >= 5 else None

    @property
    def group_key(self) -> tuple[str, ...]:
        # apples-to-apples: everything except the LAST name segment. With a mode
        # segment this keys the group on mode, so techniques compare within a mode.
        return tuple(self.name[:-1])


# Run-dir timestamp: "_YYYYMMDD_HHMMSS", optionally followed by a free-form tag
# like "_64spp". Everything from the timestamp on is stripped to get the prefix
# used to locate the matching "<prefix>_reference" dir.
_TIMESTAMP_TAIL_RE = re.compile(r"_(\d{8})_(\d{6})(?:_.*)?$")


def run_prefix(run_dir_name: str) -> Optional[str]:
    """Return '<prefix>' for 'render_720p_s0_20260521_134058' or
    'render_720p_s0_20260618_040122_64spp' -> 'render_720p_s0'.
    None if the dir doesn't contain our timestamp pattern."""
    m = _TIMESTAMP_TAIL_RE.search(run_dir_name)
    return run_dir_name[:m.start()] if m else None


def is_reference_dir(p: Path) -> bool:
    return p.is_dir() and p.name.endswith("_reference")


def _file_sig(p: Optional[Path]) -> tuple:
    """(mtime_ns, size) for change detection, or a sentinel if absent."""
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except (OSError, AttributeError):
        return (None, None)


def reference_dir_signature(ref_dir: Optional[Path]) -> tuple:
    if ref_dir is None:
        return ()
    return tuple((p.name, _file_sig(p)) for p in sorted(ref_dir.glob("*.exr")))


def group_reference_dir(reference_dir: Optional[Path], rows: list[BenchRow]) -> Optional[Path]:
    """The reference dir for one group, mirroring the run's subfolder structure: the
    name segments after the fixed 3-seg prefix, minus the selector. So a 6-segment
    (family/mode) group reads <prefix>_reference/<family>/<mode>/, a 5-segment reads
    <prefix>_reference/<mode>/, and a legacy 4-segment uses the flat <prefix>_reference/.
    Symmetric with the run-dir EXR layout, so the reference dir is just a converged run."""
    if reference_dir is None or not rows:
        return reference_dir
    sub = rows[0].name[3:-1]
    return reference_dir.joinpath(*sub) if sub else reference_dir


def group_signature(rows: list[BenchRow], ref_dir: Optional[Path]) -> tuple:
    """Changes whenever any input to analyze_group changes, so a matching
    signature means the cached verdict is still valid. ref_dir is the group's
    resolved (per-mode) reference dir."""
    exrs = tuple((r.variant, _file_sig(r.exr_path)) for r in sorted(rows, key=lambda r: r.variant))
    return (exrs, reference_dir_signature(ref_dir))


_REF_CACHE_NPY = ".combined_reference.npy"      # persisted mean, next to the *.exr
_REF_CACHE_SIG = ".combined_reference.sig.json"  # signature it was built from


def _sig_jsonable(sig: tuple) -> list:
    """reference_dir_signature() -> JSON-roundtrippable form for on-disk comparison."""
    return [[name, fs[0], fs[1]] for (name, fs) in sig]


def _compute_reference_mean(ref_dir: Path) -> Optional[np.ndarray]:
    """Mean of all *.exr in a reference dir (asymptotically every variant converges
    to the same image, so averaging is safe and denoises if several are present)."""
    exrs = sorted(ref_dir.glob("*.exr"))
    if not exrs:
        return None
    imgs = []
    shape = None
    for p in exrs:
        img = load_exr_rgb(p)
        if img is None:
            continue
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            print(f"reference shape mismatch in {ref_dir}: {p.name} {img.shape} != {shape}", file=sys.stderr)
            continue
        imgs.append(img)
    if not imgs:
        return None
    return np.mean(np.stack(imgs, axis=0), axis=0).astype(np.float32)


def load_reference_image(ref_dir: Path) -> Optional[np.ndarray]:
    """Combined (averaged) reference, persisted to a .npy sidecar so cold starts
    skip the decode-and-average. Recomputed only when the dir's *.exr set changes."""
    sig = reference_dir_signature(ref_dir)
    if not sig:
        return None
    npy  = ref_dir / _REF_CACHE_NPY
    sigp = ref_dir / _REF_CACHE_SIG
    if npy.is_file() and sigp.is_file():
        try:
            if json.loads(sigp.read_text(encoding="utf-8")) == _sig_jsonable(sig):
                return np.load(npy).astype(np.float32, copy=False)
        except Exception as e:
            print(f"reference cache read failed in {ref_dir}: {e}", file=sys.stderr)

    mean = _compute_reference_mean(ref_dir)
    if mean is None:
        return None
    try:
        np.save(npy, mean)
        sigp.write_text(json.dumps(_sig_jsonable(sig)), encoding="utf-8")
    except Exception as e:
        print(f"reference cache write failed in {ref_dir}: {e}", file=sys.stderr)
    return mean


@dataclass
class Run:
    dir: Path
    rows: list[BenchRow] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)   # keyed by variant name
    notes_path: Path = field(init=False)
    notes_dirty: bool = False
    reference_dir: Optional[Path] = None

    def __post_init__(self):
        self.notes_path = self.dir / "notes.json"

    def load(self):
        bench = self.dir / "bench.json"
        if not bench.is_file():
            return False
        try:
            doc = json.loads(bench.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{bench}: parse error: {e}", file=sys.stderr)
            return False

        self.rows.clear()
        for r in doc.get("results", []):
            name = r.get("name", [])
            if isinstance(name, str):
                name = [name]
            variant = name[-1] if name else ""
            # The first 3 name segments (PathTracer / RenderMode / scene-sensor) are a
            # fixed prefix; everything after maps to nested run-dir subfolders, with the
            # last segment the filename. So a 4-seg name is flat runDir/<variant>.exr, a
            # 5-seg adds <mode>/, a 6-seg adds <family>/<mode>/, etc.
            exr = self.dir.joinpath(*name[3:-1], f"{variant}.exr") if len(name) >= 4 else (self.dir / f"{variant}.exr")
            self.rows.append(BenchRow(
                name=list(name),
                ps_per_sample=float(r.get("ps_per_sample", 0.0)),
                gsamples_per_s=float(r.get("gsamples_per_s", 0.0)),
                ms_total=float(r.get("ms_total", 0.0)),
                regs=int(r.get("regs", 0)),
                code_bytes=int(r.get("code_bytes", 0)),
                dispatches=int(r.get("bench_dispatches", 0)),
                exr_path=exr if exr.is_file() else None,
            ))

        if self.notes_path.is_file():
            try:
                self.notes = json.loads(self.notes_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"{self.notes_path}: parse error: {e}", file=sys.stderr)
        return True

    def save_notes(self):
        if not self.notes_dirty:
            return
        try:
            self.notes_path.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")
            self.notes_dirty = False
        except Exception as e:
            messagebox.showerror("save notes", f"{self.notes_path}: {e}")


# --- EXR + FLIP --------------------------------------------------------------
def load_exr_rgb(path: Path) -> Optional[np.ndarray]:
    """Returns HxWx3 float32 linear RGB, or None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    # OpenCV gives BGR
    img = img[:, :, ::-1].astype(np.float32, copy=False)
    return np.ascontiguousarray(img)


def encode_for_display(rgb: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Image as-is: exposure scale, then hard-clip to [0,1] and gamma 2.2 encode for
    the screen. No tone curve, so values >1 clip to white rather than being
    compressed; the gamma is display encoding, not tonemapping. Returns uint8 RGB."""
    if rgb is None:
        return None
    s = rgb * float(2.0 ** exposure)
    s = np.where(np.isfinite(s), s, 0.0)
    s = np.clip(s, 0.0, 1.0) ** (1.0 / 2.2)    # gamma (display encoding only)
    return (s * 255.0 + 0.5).astype(np.uint8)


def flip_eval(test: np.ndarray, ref: np.ndarray) -> tuple[Optional[float], Optional[np.ndarray]]:
    """flip-evaluator -> (mean, errorMap). Returns (None, None) if unavailable or shape mismatch.
    errorMap is HxW float32 in [0,1]; suitable for flip_heatmap()."""
    if not _HAS_FLIP or test is None or ref is None or test.shape != ref.shape:
        return None, None
    try:
        t = np.clip(test, 0.0, 1.0).astype(np.float32)
        r = np.clip(ref,  0.0, 1.0).astype(np.float32)
        result = flip.evaluate(t, r, "HDR")
        if isinstance(result, tuple) and len(result) >= 2:
            err_map = result[0]
            mean    = float(result[1])
            if err_map is not None:
                err_map = np.asarray(err_map, dtype=np.float32)
                if err_map.ndim == 3:
                    err_map = err_map[..., 0]
            return mean, err_map
        return None, None
    except Exception as e:
        print(f"FLIP error: {e}", file=sys.stderr)
        return None, None


def bilateral_denoise(img: np.ndarray) -> Optional[np.ndarray]:
    """Edge-preserving denoise of a [0,1]-clipped copy of the image. sigmaColor is in
    intensity units: differences below ~0.08 are smoothed (treated as noise), larger
    jumps (edges) are preserved. Returns HxWx3 float32 in [0,1]."""
    if img is None:
        return None
    x = np.clip(img, 0.0, 1.0).astype(np.float32)
    x = np.where(np.isfinite(x), x, 0.0)
    return cv2.bilateralFilter(x, d=5, sigmaColor=0.08, sigmaSpace=5.0)


def flip_self_noise(img: np.ndarray) -> tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Reference-free noise via FLIP against a bilateral-denoised copy of the image.
    The difference is dominated by the variant's own noise, not its real edges.
    -> (mean, errorMap, denoisedImage)."""
    if not _HAS_FLIP or img is None:
        return None, None, None
    try:
        den = bilateral_denoise(img)
        mean, err = flip_eval(img, den)
        return mean, err, den
    except Exception as e:
        print(f"self-noise FLIP error: {e}", file=sys.stderr)
        return None, None, None


def flip_heatmap(err_map: np.ndarray) -> Optional[np.ndarray]:
    """Map FLIP error [0,1] to a colourised heatmap (HxWx3 uint8 RGB)."""
    if err_map is None:
        return None
    e8 = np.clip(err_map * 255.0, 0.0, 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(e8, cv2.COLORMAP_MAGMA)
    return bgr[..., ::-1]  # BGR -> RGB


# --- group analysis ----------------------------------------------------------
# Rec.709 luma weights for the energy/bias gate (ported from mean_ratio.py).
_LUMA = np.array([0.2126, 0.7152, 0.0722])
# A variant whose luma-mean ratio vs the basis is more than this off 1.0 is flagged BIASED. Bias is
# dangerous: the estimator looks like it's converging but to the WRONG image, so it must be loud.
_BIAS_TOL = 0.02


def safe_channel_mean(img: np.ndarray) -> np.ndarray:
    """Per-channel mean over FINITE pixels only, drops firefly inf/NaN (zeroing them would bias the
    mean down). Returns length-3; NaN for a channel with no finite pixels."""
    out = np.empty(3)
    for c in range(3):
        v = img[..., c]
        finite = v[np.isfinite(v)]
        out[c] = finite.mean() if finite.size else np.nan
    return out


@dataclass
class GroupVerdict:
    bench_winner_variant: Optional[str] = None
    bench_winner_ps: Optional[float] = None
    flip_scores: dict[str, float] = field(default_factory=dict)            # variant -> mean FLIP vs reference_basis
    noise_scores: dict[str, float] = field(default_factory=dict)           # variant -> FLIP vs bilateral-denoised self (no reference)
    noise_winner_variant: Optional[str] = None
    rgb_imgs: dict[str, np.ndarray] = field(default_factory=dict)          # variant -> HxWx3 linear RGB
    err_maps: dict[str, np.ndarray] = field(default_factory=dict)          # variant -> HxW FLIP error [0,1]
    noise_maps: dict[str, np.ndarray] = field(default_factory=dict)        # variant -> HxW self-noise FLIP error [0,1]
    denoised_imgs: dict[str, np.ndarray] = field(default_factory=dict)      # variant -> HxWx3 bilateral-denoised [0,1] (N-key preview)
    diff_maps: dict[str, np.ndarray] = field(default_factory=dict)         # variant -> HxW abs-diff vs reference
    diff_max: float = 0.0                                                   # shared scale for the group's diff heatmaps
    reference_img: Optional[np.ndarray] = None                              # ground truth if provided (else None)
    reference_basis: str = "mean-of-others"                                 # "reference" if external GT was used
    efficiency: dict[str, float] = field(default_factory=dict)              # variant -> FLIP^2 * ms_total (MC variance x time, lower better)
    efficiency_winner_variant: Optional[str] = None
    mean_luma: dict[str, float] = field(default_factory=dict)               # variant -> finite-pixel luma mean (energy)
    mean_ratio: dict[str, float] = field(default_factory=dict)              # variant -> luma mean / basis luma mean (bias gate, ~1.0 unbiased)


def analyze_group(rows: list[BenchRow], reference: Optional[np.ndarray] = None) -> GroupVerdict:
    v = GroupVerdict()
    if not rows:
        return v

    # bench winner: lowest ps_per_sample (skip rows with 0/NaN)
    timed = [r for r in rows if r.ps_per_sample > 0]
    if timed:
        w = min(timed, key=lambda r: r.ps_per_sample)
        v.bench_winner_variant = w.variant
        v.bench_winner_ps      = w.ps_per_sample

    # FLIP / diff basis selection: prefer an external reference (converged ground
    # truth) when provided. Falls back to "mean of OTHER variants" for legacy
    # symmetric A/B comparison.
    imgs = {r.variant: load_exr_rgb(r.exr_path) for r in rows if r.exr_path}
    imgs = {k: img for k, img in imgs.items() if img is not None}
    v.rgb_imgs = imgs
    # Energy / bias gate (ported from mean_ratio.py): per-variant finite-pixel luma mean. The ratio
    # vs the FLIP basis is filled in the per-variant loop below; unbiased estimators match the basis
    # so the ratio must be ~1.0, a deviation is energy bias (e.g. MIS double-counting) that FLIP
    # can smear but a global mean ratio catches cleanly.
    for v_name, im in imgs.items():
        v.mean_luma[v_name] = float(safe_channel_mean(im) @ _LUMA)
    use_ref   = reference is not None
    v.reference_img    = reference if use_ref else None
    v.reference_basis  = "reference" if use_ref else "mean-of-others"

    # Reference-free per-image noise (FLIP vs bilateral-denoised self). Independent
    # of any reference or sibling variant, so it runs whenever an image is present.
    if _HAS_FLIP:
        for v_name, im in imgs.items():
            ns, nmap, den = flip_self_noise(im)
            if ns is not None:
                v.noise_scores[v_name] = ns
            if nmap is not None:
                v.noise_maps[v_name] = nmap
            if den is not None:
                v.denoised_imgs[v_name] = den

    if imgs and (use_ref or len(imgs) >= 2):
        variants = list(imgs.keys())
        for v_name in variants:
            if use_ref:
                ref = reference if reference.shape == imgs[v_name].shape else None
            else:
                others = [imgs[o] for o in variants if o != v_name and imgs[o].shape == imgs[v_name].shape]
                ref    = np.mean(np.stack(others, axis=0), axis=0) if others else None
            if ref is None:
                continue

            # Bias gate: this variant's luma mean / the basis luma mean (want ~1.0 if unbiased).
            ref_luma = float(safe_channel_mean(ref) @ _LUMA)
            if np.isfinite(ref_luma) and ref_luma > 0.0 and np.isfinite(v.mean_luma.get(v_name, np.nan)):
                v.mean_ratio[v_name] = v.mean_luma[v_name] / ref_luma

            # Raw HDR abs diff (max across channels). Independent of FLIP, so it
            # works even without flip-evaluator installed.
            diff = np.abs(imgs[v_name] - ref).max(axis=-1)
            v.diff_maps[v_name] = diff
            v.diff_max = max(v.diff_max, float(diff.max()))

            if _HAS_FLIP:
                mean, err = flip_eval(imgs[v_name], ref)
                if mean is not None:
                    v.flip_scores[v_name] = mean
                if err is not None:
                    v.err_maps[v_name] = err
        if v.flip_scores:
            v.noise_winner_variant = min(v.flip_scores, key=v.flip_scores.get)

    # Efficiency: FLIP * ms_total (perceptual error x wall-clock cost). Lower wins.
    ms_by_variant = {r.variant: r.ms_total for r in rows if r.ms_total > 0}
    for variant, flip in v.flip_scores.items():
        ms = ms_by_variant.get(variant)
        if ms is not None:
            v.efficiency[variant] = flip * ms
    if v.efficiency:
        v.efficiency_winner_variant = min(v.efficiency, key=v.efficiency.get)

    return v


def diff_heatmap(diff_map: np.ndarray, vmax: float) -> Optional[np.ndarray]:
    """Map an HDR abs-diff map to a colourised heatmap (HxWx3 uint8 RGB) using a
    shared vmax so multiple variants in a group are visually comparable."""
    if diff_map is None:
        return None
    scale = 1.0 / vmax if vmax > 1e-12 else 0.0
    e8 = np.clip(diff_map * (scale * 255.0), 0.0, 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(e8, cv2.COLORMAP_VIRIDIS)
    return bgr[..., ::-1]


def verdict_summary(verdict: GroupVerdict) -> str:
    lines = ["=== Verdict ==="]
    if verdict.bench_winner_variant:
        lines.append(f"Bench winner : {verdict.bench_winner_variant}   ({verdict.bench_winner_ps:.2f} ps/sample)")
    else:
        lines.append("Bench winner : (no timing data)")
    if verdict.noise_winner_variant:
        lines.append(f"Noise winner : {verdict.noise_winner_variant}   (FLIP mean {verdict.flip_scores[verdict.noise_winner_variant]:.4f} vs {verdict.reference_basis})")
        for var, score in sorted(verdict.flip_scores.items(), key=lambda kv: kv[1]):
            marker = "*" if var == verdict.noise_winner_variant else " "
            lines.append(f"   {marker} {var:<20s} FLIP mean = {score:.4f}")
    else:
        lines.append("Noise winner : (FLIP unavailable or no reference)")
    if verdict.noise_scores:
        best = min(verdict.noise_scores, key=verdict.noise_scores.get)
        lines.append(f"Self-noise   : {best}   (reference-free, FLIP vs denoised self; lower=cleaner)")
        for var, score in sorted(verdict.noise_scores.items(), key=lambda kv: kv[1]):
            marker = "*" if var == best else " "
            lines.append(f"   {marker} {var:<20s} noise = {score:.4f}")
    if verdict.efficiency_winner_variant:
        lines.append(f"Efficiency   : {verdict.efficiency_winner_variant}   (FLIP * ms = {verdict.efficiency[verdict.efficiency_winner_variant]:.2f}, lower=better)")
        for var, score in sorted(verdict.efficiency.items(), key=lambda kv: kv[1]):
            marker = "*" if var == verdict.efficiency_winner_variant else " "
            lines.append(f"   {marker} {var:<20s} FLIP^2 * ms = {score:.2f}")
    if verdict.mean_ratio:
        lines.append(f"Bias gate    : luma mean / {verdict.reference_basis} (want 1.000 +/- noise; off = energy bias, e.g. MIS double-count)")
        for var, ratio in sorted(verdict.mean_ratio.items()):
            flag = "" if abs(ratio - 1.0) <= _BIAS_TOL else "   <-- BIASED"
            lines.append(f"     {var:<20s} ratio = {ratio:.4f}  (luma {verdict.mean_luma.get(var, float('nan')):.6g}){flag}")
    elif verdict.mean_luma:
        # No basis for a ratio (single variant, no reference), still surface the raw energy.
        lines.append("Bias gate    : (no basis for a ratio; luma means below)")
        for var, lm in sorted(verdict.mean_luma.items()):
            lines.append(f"     {var:<20s} luma = {lm:.6g}")
    return "\n".join(lines)


# --- UI ----------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk, root_dir: Path):
        self.root = root
        self.root.title("PT Bench Viewer")
        self.root.geometry("1400x850")

        self.root_dir = root_dir
        self.runs: list[Run] = []
        self.current_run: Optional[Run] = None
        self.current_row: Optional[BenchRow] = None
        self.current_group_key: Optional[tuple] = None         # set when a group node is selected
        self.current_group_rows: list[BenchRow] = []
        self.current_verdict: Optional[GroupVerdict] = None    # cached per selection
        # Persisted across Refresh: only groups whose EXR/reference files changed
        # are recomputed (EXR decode + FLIP is the expensive part). Value is
        # (file_signature, verdict) so a stale signature triggers a recompute.
        self._verdict_cache: dict = {}                         # (run_dir, group_key) -> (sig, GroupVerdict)
        self._ref_cache: dict = {}                             # ref_dir -> (sig, mean image)
        # (run_dir, group_key) -> list[(row_iid, BenchRow)], so a lazily-computed
        # verdict can backfill the FLIP/eff columns and winner colours of a group's
        # rows after the user first selects it.
        self._group_row_iids: dict = {}
        self._preview_cache: Optional[ImageTk.PhotoImage] = None
        self._exposure = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._scan(root_dir)

        if not _HAS_FLIP:
            messagebox.showwarning(
                "FLIP missing",
                "flip-evaluator not installed; noise-winner column will be empty.\n"
                "Install with:  pip install flip-evaluator",
            )

    # --- layout
    def _build_ui(self):
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="Open Folder...", command=self._pick_folder).pack(side=tk.LEFT)
        ttk.Button(bar, text="Refresh",        command=lambda: self._scan(self.root_dir)).pack(side=tk.LEFT, padx=(6, 0))
        self._dir_lbl = ttk.Label(bar, text=str(self.root_dir))
        self._dir_lbl.pack(side=tk.LEFT, padx=10)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # left: runs tree
        left = ttk.Frame(main)
        main.add(left, weight=1)
        self._tree = ttk.Treeview(left, columns=("export", "variant", "ps", "ms", "disp", "flip", "noise", "luma", "eff"), show="tree headings")
        self._tree.heading("#0", text="Run / Group")
        self._tree.heading("export",  text="")
        self._tree.heading("variant", text="Variant")
        self._tree.heading("ps",      text="ps/sample")
        self._tree.heading("ms",      text="ms total")
        self._tree.heading("disp",    text="dispatches")
        self._tree.heading("flip",    text="FLIP vs ref")
        self._tree.heading("noise",   text="noise")
        self._tree.heading("luma",    text="luma")
        self._tree.heading("eff",     text="FLIP*ms")
        self._tree.column("#0",      width=240, stretch=True)
        self._tree.column("export",  width=70,  anchor=tk.CENTER)
        self._tree.column("variant", width=110)
        self._tree.column("ps",      width=90,  anchor=tk.E)
        self._tree.column("ms",      width=80,  anchor=tk.E)
        self._tree.column("disp",    width=80,  anchor=tk.E)
        self._tree.column("flip",    width=80,  anchor=tk.E)
        self._tree.column("noise",   width=70,  anchor=tk.E)
        self._tree.column("luma",    width=80,  anchor=tk.E)
        self._tree.column("eff",     width=90,  anchor=tk.E)
        self._tree.pack(fill=tk.BOTH, expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Button-1>", self._on_tree_left_click, add="+")
        self._tree.bind("<Button-3>", self._on_tree_right_click)
        # Highlights: bench-time winner (green), noise winner (blue),
        # efficiency winner (purple). Cell collisions stack onto a mixed colour.
        self._tree.tag_configure("bench_win",        background="#dfffd6")
        self._tree.tag_configure("noise_win",        background="#d6ecff")
        self._tree.tag_configure("eff_win",          background="#e6d6ff")
        self._tree.tag_configure("bench_noise_win",  background="#fff3cc")
        self._tree.tag_configure("bench_eff_win",    background="#e6ffd6")
        self._tree.tag_configure("noise_eff_win",    background="#d6e6ff")
        self._tree.tag_configure("all_win",          background="#ffd6f0")
        # Bias OVERRIDES every winner highlight: a biased estimator converges to the WRONG image, so a
        # biased "winner" is the most dangerous case. Strong red + white text so it can't be missed.
        self._tree.tag_configure("biased",           background="#ff3b30", foreground="#ffffff")

        # right: preview + details + notes
        right = ttk.Panedwindow(main, orient=tk.VERTICAL)
        main.add(right, weight=3)

        prev_frame = ttk.Frame(right)
        right.add(prev_frame, weight=3)
        ctrl = ttk.Frame(prev_frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        ttk.Label(ctrl, text="EV").pack(side=tk.LEFT)
        ttk.Scale(ctrl, from_=-6.0, to=6.0, variable=self._exposure, orient=tk.HORIZONTAL,
                  command=lambda _=None: self._refresh_preview()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._exp_lbl = ttk.Label(ctrl, text="+0.0")
        self._exp_lbl.pack(side=tk.LEFT)
        self._exposure.trace_add("write", lambda *_: self._exp_lbl.configure(text=f"{self._exposure.get():+.1f}"))
        ttk.Button(ctrl, text="Fit",  width=4, command=self._fit_zoom).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(ctrl, text="1:1",  width=4, command=self._zoom_one).pack(side=tk.LEFT, padx=(2, 0))
        self._zoom_lbl = ttk.Label(ctrl, text="100%", width=6)
        self._zoom_lbl.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(ctrl, text="wheel=zoom  drag=pan  dbl-click=fit  <-=variant  ->=ref  N=denoised  G=prev variant").pack(side=tk.LEFT, padx=(10, 0))
        # Shows which image is on the canvas during a Left/Right A/B flip.
        self._ab_lbl = ttk.Label(ctrl, text="", width=22, anchor=tk.E)
        self._ab_lbl.pack(side=tk.RIGHT)

        self._canvas = tk.Canvas(prev_frame, background="#222", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        # Zoom/pan state.
        self._source_array: Optional[np.ndarray] = None   # current displayable RGB uint8
        self._source_pil: Optional[Image.Image] = None    # same data as PIL, built once per source
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._canvas_image_id: Optional[int] = None
        self._fit_scale = 1.0                              # canvas-px per source-px at fit
        self._zoom      = 1.0                              # user multiplier on top of fit
        self._view_x    = 0                                # canvas coord of source (0,0)
        self._view_y    = 0
        self._pan_origin: Optional[tuple] = None
        self._preview_mode = "variant"                     # leaf preview: variant | reference | denoised
        # Cross-variant A/B (G key): tree iids of the last two selected leaf rows, so
        # G re-selects the previous variant (selection follows) holding zoom/pan fixed.
        self._last_leaf_iid: Optional[str] = None
        self._prev_leaf_iid: Optional[str] = None
        self._keep_view_on_refresh = False
        self._canvas.bind("<Configure>",     self._on_canvas_resize)
        self._canvas.bind("<MouseWheel>",    self._on_wheel)        # Windows / Mac
        self._canvas.bind("<Button-4>",      self._on_wheel)        # Linux scroll up
        self._canvas.bind("<Button-5>",      self._on_wheel)        # Linux scroll down
        self._canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self._canvas.bind("<B1-Motion>",     self._on_pan_move)
        self._canvas.bind("<Double-Button-1>", lambda _e: self._fit_zoom())
        self._canvas.bind("<Button-3>", self._on_canvas_right_click)
        # Left=variant, Right=reference, N=toggle denoised-self. Bound on the tree
        # too (it has focus after a row click); returning "break" there suppresses
        # the default expand/collapse only when we actually drove a flip.
        for w in (self._canvas, self._tree):
            w.bind("<Left>",  lambda _e: self._on_mode_key("variant"))
            w.bind("<Right>", lambda _e: self._on_mode_key("reference"))
            w.bind("<n>",     lambda _e: self._on_mode_key("toggle_denoise"))
            w.bind("<N>",     lambda _e: self._on_mode_key("toggle_denoise"))
            w.bind("<g>",     lambda _e: self._select_previous_variant())
            w.bind("<G>",     lambda _e: self._select_previous_variant())

        details = ttk.Frame(right)
        right.add(details, weight=1)
        self._details = scrolledtext.ScrolledText(details, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self._details.pack(fill=tk.BOTH, expand=True)

        notes_frame = ttk.LabelFrame(right, text="Notes (autosave on selection change)")
        right.add(notes_frame, weight=1)
        self._notes = scrolledtext.ScrolledText(notes_frame, height=6, wrap=tk.WORD)
        self._notes.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._notes.bind("<<Modified>>", self._on_notes_modified)
        self._notes_loading = False

    # --- scanning
    def _reference_img(self, ref_dir: Optional[Path]) -> Optional[np.ndarray]:
        """Cached reference-mean load; reloads only if the ref dir's EXRs changed."""
        if ref_dir is None:
            return None
        sig    = reference_dir_signature(ref_dir)
        cached = self._ref_cache.get(ref_dir)
        if cached is not None and cached[0] == sig:
            return cached[1]
        img = load_reference_image(ref_dir)
        self._ref_cache[ref_dir] = (sig, img)
        return img

    def _pick_folder(self):
        d = filedialog.askdirectory(initialdir=str(self.root_dir), title="Pick benchmarks dir")
        if d:
            self.root_dir = Path(d)
            self._dir_lbl.configure(text=str(self.root_dir))
            self._scan(self.root_dir)

    def _scan(self, root_dir: Path):
        # commit any pending notes before reload
        self._commit_notes_if_dirty()

        self.runs.clear()
        self._tree.delete(*self._tree.get_children())
        self.current_run = None
        self.current_row = None
        self._preview_cache = None
        self._clear_source()
        self._set_details("")
        self._set_notes("")

        if not root_dir.is_dir():
            messagebox.showerror("scan", f"not a directory: {root_dir}")
            return

        # Two passes: collect reference dirs first so we can resolve them when
        # building bench runs.
        ref_dirs: dict[str, Path] = {}
        for sub in sorted(root_dir.iterdir()):
            if is_reference_dir(sub):
                ref_dirs[sub.name[: -len("_reference")]] = sub

        for sub in sorted(root_dir.iterdir()):
            if not sub.is_dir() or is_reference_dir(sub):
                continue
            run = Run(sub)
            if not run.load():
                continue
            prefix = run_prefix(sub.name)
            if prefix and prefix in ref_dirs:
                run.reference_dir = ref_dirs[prefix]
            self.runs.append(run)

        # Newest run first. Prefer bench.json's mtime (the write that finished the
        # run) over the dir's, which later sidecars (notes.json, ref cache) bump.
        def _run_mtime(r: Run) -> float:
            try:
                return (r.dir / "bench.json").stat().st_mtime
            except OSError:
                return 0.0
        self.runs.sort(key=_run_mtime, reverse=True)

        # _verdict_cache / _ref_cache persist across Refresh on purpose, only
        # changed files get recomputed below. The tree-iid indices don't.
        self._row_index.clear()
        self._group_index.clear()
        self._run_index.clear()
        self._group_row_iids.clear()
        self._last_leaf_iid = self._prev_leaf_iid = None   # tree iids go stale on rescan

        # Build the tree only, no analyze_group here. EXR decode + FLIP (the
        # ~0.4s/group cost) runs lazily the first time a group is selected, so
        # Refresh stays instant. FLIP/eff columns and winner colours fill in then.
        for run in self.runs:
            run_label = run.dir.name + (f"   [REF: {run.reference_dir.name}]" if run.reference_dir else "")
            run_id = self._tree.insert("", tk.END, text=run_label, values=("[export]",), open=True)
            self._run_index[run_id] = run
            groups: dict[tuple, list[BenchRow]] = {}
            for r in run.rows:
                groups.setdefault(r.group_key, []).append(r)
            for gkey, rows in groups.items():
                glabel = " / ".join(gkey) if gkey else "(ungrouped)"
                gid = self._tree.insert(run_id, tk.END, text=glabel, values=("[export]",), open=True)
                self._group_index[gid] = (run, gkey, rows)
                row_iids = []
                for r in rows:
                    iid = self._tree.insert(
                        gid, tk.END, text="",
                        values=("", r.variant, f"{r.ps_per_sample:.2f}", f"{r.ms_total:.2f}",
                                f"{r.dispatches}" if r.dispatches else "", "", "", "", ""),
                    )
                    self._row_index[iid] = (run, r)
                    row_iids.append((iid, r))
                self._group_row_iids[(run.dir, gkey)] = row_iids
                # If a valid verdict is already cached (unchanged group selected in
                # a previous session of this process), paint its columns now, free.
                cached = self._verdict_cache.get((run.dir, gkey))
                ref_dir = group_reference_dir(run.reference_dir, rows)
                if cached is not None and cached[0] == group_signature(rows, ref_dir):
                    self._apply_verdict_to_tree((run.dir, gkey), cached[1])

    _WINNER_TAG = {
        (True,  True,  True ): "all_win",
        (True,  True,  False): "bench_noise_win",
        (True,  False, True ): "bench_eff_win",
        (False, True,  True ): "noise_eff_win",
        (True,  False, False): "bench_win",
        (False, True,  False): "noise_win",
        (False, False, True ): "eff_win",
        (False, False, False): "",
    }

    def _ensure_verdict(self, run: Run, gkey: tuple, rows: list[BenchRow]) -> GroupVerdict:
        """Return the group's verdict, computing (decode + FLIP) and caching it on
        first request. Backfills the tree row columns/colours when freshly computed."""
        key     = (run.dir, gkey)
        ref_dir = group_reference_dir(run.reference_dir, rows)
        sig     = group_signature(rows, ref_dir)
        cached  = self._verdict_cache.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        verdict = analyze_group(rows, reference=self._reference_img(ref_dir))
        self._verdict_cache[key] = (sig, verdict)
        self._apply_verdict_to_tree(key, verdict)
        return verdict

    def _apply_verdict_to_tree(self, key: tuple, verdict: GroupVerdict):
        for iid, r in self._group_row_iids.get(key, []):
            bw = verdict.bench_winner_variant      == r.variant
            nw = verdict.noise_winner_variant      == r.variant
            ew = verdict.efficiency_winner_variant == r.variant
            tag    = self._WINNER_TAG[(bw, nw, ew)]
            # Bias gate: flag + recolor a row whose energy ratio is off 1.0. This OVERRIDES the winner
            # tag on purpose, bias is a correctness failure, not a quality ranking.
            ratio  = verdict.mean_ratio.get(r.variant)
            biased = ratio is not None and abs(ratio - 1.0) > _BIAS_TOL
            if biased:
                tag = "biased"
            flip_s  = verdict.flip_scores.get(r.variant)
            noise_s = verdict.noise_scores.get(r.variant)
            luma_s  = verdict.mean_luma.get(r.variant)
            eff_s   = verdict.efficiency.get(r.variant)
            self._tree.item(
                iid,
                values=(
                    "",  # export column (run/group rows only)
                    (f"⚠ {r.variant}  (x{ratio:.3f})" if biased else r.variant),
                    f"{r.ps_per_sample:.2f}",
                    f"{r.ms_total:.2f}",
                    f"{r.dispatches}" if r.dispatches else "",
                    f"{flip_s:.4f}"  if flip_s  is not None else "",
                    f"{noise_s:.4f}" if noise_s is not None else "",
                    f"{luma_s:.4g}"  if luma_s  is not None else "",
                    f"{eff_s:.2f}"   if eff_s   is not None else "",
                ),
                tags=(tag,) if tag else (),
            )

    # --- selection
    _row_index: dict[str, tuple[Run, BenchRow]] = {}
    _group_index: dict[str, tuple[Run, tuple, list[BenchRow]]] = {}
    _run_index: dict[str, Run] = {}

    def _busy(self, fn):
        """Run fn with a wait cursor. First selection of a group decodes EXRs and
        runs FLIP (~0.4s); this signals the brief stall instead of silently hanging."""
        try:
            self.root.configure(cursor="watch")
            self.root.update_idletasks()
            return fn()
        finally:
            self.root.configure(cursor="")

    def _on_tree_left_click(self, evt):
        """Click the [export] cell of a run or group row -> write a results JSON.
        Other cells fall through to normal selection (this is an additive binding)."""
        if self._tree.identify_region(evt.x, evt.y) != "cell":
            return
        if self._tree.identify_column(evt.x) != "#1":   # "export" is the first data column
            return
        iid = self._tree.identify_row(evt.y)
        if iid in self._run_index:
            self._busy(lambda: self._export_run(self._run_index[iid]))
        elif iid in self._group_index:
            run, gkey, rows = self._group_index[iid]
            self._busy(lambda: self._export_group(run, gkey, rows))

    def _bench_meta(self, run: Run) -> dict:
        try:
            d = json.loads((run.dir / "bench.json").read_text(encoding="utf-8"))
            return {"version": d.get("version"), "device": d.get("device")}
        except Exception:
            return {}

    @staticmethod
    def _variant_export(r: BenchRow, verdict: GroupVerdict) -> dict:
        return {
            "name": list(r.name),
            "variant": r.variant,
            "mode": r.mode,
            "ps_per_sample": r.ps_per_sample,
            "gsamples_per_s": r.gsamples_per_s,
            "ms_total": r.ms_total,
            "dispatches": r.dispatches,
            "regs": r.regs,
            "code_bytes": r.code_bytes,
            "flip_vs_ref": verdict.flip_scores.get(r.variant),
            "noise_self_denoised": verdict.noise_scores.get(r.variant),
            "luma_mean": verdict.mean_luma.get(r.variant),
            "luma_ratio_vs_ref": verdict.mean_ratio.get(r.variant),
            "efficiency_flip2_ms": verdict.efficiency.get(r.variant),
            "exr": str(r.exr_path) if r.exr_path else None,
        }

    def _group_export(self, run: Run, gkey: tuple, rows: list[BenchRow], verdict: GroupVerdict) -> dict:
        ref = group_reference_dir(run.reference_dir, rows)
        return {
            "group_key": list(gkey),
            "mode": rows[0].mode if rows else None,
            "reference_basis": verdict.reference_basis,
            "reference_dir": str(ref) if ref else None,
            "winners": {
                "bench_lowest_ps": verdict.bench_winner_variant,
                "noise_lowest_flip": verdict.noise_winner_variant,
                "efficiency_lowest_flip2_ms": verdict.efficiency_winner_variant,
            },
            "variants": [self._variant_export(r, verdict) for r in rows],
        }

    def _write_export(self, run: Run, tag: str, groups: list[dict]):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "results"
        out  = run.dir / f"results_{safe}.json"
        doc  = {"run_dir": str(run.dir), "bench_meta": self._bench_meta(run), "groups": groups}
        try:
            out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("export", f"{out}: {e}")
            return
        messagebox.showinfo("export", f"Wrote {len(groups)} group(s) to:\n{out}")

    def _export_group(self, run: Run, gkey: tuple, rows: list[BenchRow]):
        verdict = self._ensure_verdict(run, gkey, rows)
        self._write_export(run, "_".join(gkey[3:]) or "group", [self._group_export(run, gkey, rows, verdict)])

    def _export_run(self, run: Run):
        groups: dict[tuple, list[BenchRow]] = {}
        for r in run.rows:
            groups.setdefault(r.group_key, []).append(r)
        out = [self._group_export(run, gkey, rows, self._ensure_verdict(run, gkey, rows))
               for gkey, rows in groups.items()]
        self._write_export(run, "all", out)

    def _on_tree_right_click(self, evt):
        """Right-click a run/group/row -> context menu to open the EXR (rows) or the
        run folder (any node) with its Windows default handler."""
        iid = self._tree.identify_row(evt.y)
        if not iid:
            return
        run = self._run_index.get(iid)
        if run is None and iid in self._group_index:
            run = self._group_index[iid][0]
        row = None
        if iid in self._row_index:
            run, row = self._row_index[iid]
        if run is None:
            return
        menu = tk.Menu(self.root, tearoff=0)
        if row is not None and row.exr_path is not None:
            menu.add_command(label="Open EXR", command=lambda p=row.exr_path: open_path(p))
        menu.add_command(label="Open run folder", command=lambda p=run.dir: open_path(p))
        try:
            menu.tk_popup(evt.x_root, evt.y_root)
        finally:
            menu.grab_release()

    def _on_canvas_right_click(self, evt):
        """Right-click the preview -> open the current variant's EXR (leaf) or the
        run folder (group composite) with its Windows default handler."""
        menu = tk.Menu(self.root, tearoff=0)
        row = self.current_row
        if row is not None and row.exr_path is not None:
            menu.add_command(label="Open EXR", command=lambda p=row.exr_path: open_path(p))
        if self.current_run is not None:
            menu.add_command(label="Open run folder", command=lambda p=self.current_run.dir: open_path(p))
        if menu.index("end") is None:
            return
        try:
            menu.tk_popup(evt.x_root, evt.y_root)
        finally:
            menu.grab_release()

    def _on_select(self, _evt=None):
        # commit notes for previous selection first
        self._commit_notes_if_dirty()

        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]

        # group node?
        ginfo = self._group_index.get(iid)
        if ginfo:
            run, gkey, rows = ginfo
            self.current_run = run
            self.current_row = None
            self.current_group_key = gkey
            self.current_group_rows = rows
            self._busy(lambda: self._ensure_verdict(run, gkey, rows))
            verdict = self.current_verdict = self._verdict_cache[(run.dir, gkey)][1]
            self._show_group(run, gkey, rows, verdict)
            return

        # leaf row?
        rinfo = self._row_index.get(iid)
        if rinfo:
            run, row = rinfo
            # Track the last two distinct leaf selections so G can flip back.
            if iid != self._last_leaf_iid:
                self._prev_leaf_iid = self._last_leaf_iid
                self._last_leaf_iid = iid
            rows = [r for r in run.rows if r.group_key == row.group_key]
            self._busy(lambda: self._ensure_verdict(run, row.group_key, rows))
            verdict = self.current_verdict = self._verdict_cache[(run.dir, row.group_key)][1]
            self.current_run = run
            self.current_row = row
            self.current_group_key = row.group_key
            self.current_group_rows = rows
            self._show_row(run, row, verdict)
            return

        # run-level node: clear preview, blank details
        self.current_run = None
        self.current_row = None
        self.current_group_key = None
        self.current_verdict = None
        self._set_details("")
        self._set_notes("")
        self._clear_source()

    def _select_previous_variant(self):
        """G: re-select the previously-selected leaf row (tree selection follows, so
        it's clear which variant is shown), holding zoom/pan fixed for in-place A/B.
        The two rows then ping-pong, since each selection records the other as prev."""
        prev = self._prev_leaf_iid
        if not prev or prev not in self._row_index:
            return "break"
        self._keep_view_on_refresh = True
        self._tree.selection_set(prev)
        self._tree.focus(prev)
        self._tree.see(prev)
        return "break"

    def _show_row(self, run: Run, row: BenchRow, verdict: Optional[GroupVerdict]):
        d = []
        if verdict:
            d.append(verdict_summary(verdict))
            d.append("")
        d.append("=== Selected row ===")
        d.append(f"Run dir : {run.dir}")
        d.append(f"Name    : {' / '.join(row.name)}")
        d.append(f"Variant : {row.variant}")
        d.append(f"ps/samp : {row.ps_per_sample:.3f}")
        d.append(f"Gsmp/s  : {row.gsamples_per_s:.4f}")
        d.append(f"ms total: {row.ms_total:.3f}")
        d.append(f"dispatch: {row.dispatches}")
        d.append(f"regs    : {row.regs}")
        d.append(f"code B  : {row.code_bytes}")
        d.append(f"EXR     : {row.exr_path if row.exr_path else '(missing)'}")
        if verdict and row.variant in verdict.flip_scores:
            d.append(f"FLIP    : {verdict.flip_scores[row.variant]:.4f} (vs {verdict.reference_basis})")
        if verdict and row.variant in verdict.noise_scores:
            d.append(f"Noise   : {verdict.noise_scores[row.variant]:.4f} (reference-free, FLIP vs denoised self)")
        if verdict and row.variant in verdict.mean_luma:
            d.append(f"Luma    : {verdict.mean_luma[row.variant]:.6g} (Rec.709 mean, finite pixels, absolute/unclipped)")
        self._set_details("\n".join(d))

        self._set_notes(run.notes.get(row.variant, ""))
        self._refresh_preview()

    def _show_group(self, run: Run, gkey: tuple, rows: list[BenchRow], verdict: GroupVerdict):
        d = [verdict_summary(verdict), ""]
        d.append("=== Group ===")
        d.append(f"Run dir : {run.dir}")
        d.append(f"Group   : {' / '.join(gkey) if gkey else '(ungrouped)'}")
        d.append(f"Variants: {', '.join(r.variant for r in rows)}")
        self._set_details("\n".join(d))
        # No per-variant notes when on the group node, clear the box so the
        # previous selection's notes don't look like they apply to the group.
        self._set_notes("")
        self._refresh_preview()

    # --- A/B flicker compare: Left=variant, Right=reference, N=denoised self, G=last-selected variant
    def _on_mode_key(self, target: str):
        # Return "break" only when we drove a flip, so the tree's default arrow
        # behaviour still works on group/run nodes or when the image is missing.
        if target == "toggle_denoise":
            mode = "variant" if self._preview_mode == "denoised" else "denoised"
        else:
            mode = target
        return "break" if self._show_mode(mode) else None

    def _mode_image(self, mode: str, row: BenchRow, strict: bool = False):
        """(rgb, label) for a preview mode. When not strict, an unavailable mode
        falls back to the variant image (used on row change); when strict, returns
        (None, '') so a key press for a missing image is a no-op."""
        v = self.current_verdict
        var = v.rgb_imgs.get(row.variant) if v else None
        if var is None:
            return None, ""
        if mode == "reference":
            ref = v.reference_img
            if ref is not None and ref.shape == var.shape:
                return ref, "REFERENCE"
            if strict:
                return None, ""
        elif mode == "denoised":
            den = v.denoised_imgs.get(row.variant)
            if den is not None and den.shape == var.shape:
                return den, f"{row.variant}  (denoised)"
            if strict:
                return None, ""
        extras = []
        if v.reference_img is not None:
            extras.append("<-/->=ref")
        if row.variant in v.denoised_imgs:
            extras.append("N=denoise")
        if self._prev_leaf_iid:
            extras.append("G=prev")
        return var, f"{row.variant}" + (("   " + "  ".join(extras)) if extras else "")

    def _show_mode(self, mode: str) -> bool:
        """Switch the canvas to variant/reference/denoised, holding zoom/pan fixed
        so the swap reveals differences in place. No-op (False) unless a leaf row is
        shown and the requested image exists at the same resolution."""
        if self.current_row is None or self.current_verdict is None:
            return False
        rgb, label = self._mode_image(mode, self.current_row, strict=True)
        if rgb is None:
            return False
        self._preview_mode = mode
        self._set_source(encode_for_display(rgb, exposure=self._exposure.get()), keep_view=True)
        self._ab_lbl.configure(text=label)
        return True

    # --- preview
    def _refresh_preview(self):
        # group selected -> side-by-side composite (image + FLIP heatmap + diff row)
        if self.current_row is None and self.current_verdict is not None:
            self._render_group_composite()
            return

        # leaf selected -> single display-encoded EXR (zoomable)
        row = self.current_row
        if row is None:
            self._clear_source()
            return
        # analyze_group already decoded every variant's EXR into the verdict;
        # reuse it so selecting a row (or dragging the exposure slider) re-encodes
        # without re-reading the EXR from disk. Fall back to a direct load only if
        # the verdict has no image for this variant.
        # Keep whichever mode the last Left/Right/N press chose, so moving between
        # rows stays on REFERENCE/denoised instead of snapping back to the variant.
        rgb, label = self._mode_image(self._preview_mode, row)
        if rgb is None and row.exr_path is not None:
            rgb, label = load_exr_rgb(row.exr_path), row.variant
        if rgb is None:
            self._clear_source()
            return
        # A G-flip keeps zoom/pan so the two variants align; a normal selection fits.
        keep = self._keep_view_on_refresh
        self._keep_view_on_refresh = False
        self._set_source(encode_for_display(rgb, exposure=self._exposure.get()), keep_view=keep)
        self._ab_lbl.configure(text=label)

    def _render_group_composite(self):
        self._ab_lbl.configure(text="")
        verdict = self.current_verdict
        rows    = self.current_group_rows
        if verdict is None or not rows:
            self._canvas.configure(image="")
            self._preview_cache = None
            return

        ev = self._exposure.get()
        # Side-by-side composite is built off-screen at full source resolution;
        # the Canvas zoom/pan path then handles fit, wheel zoom and dragging.
        # per variant: image | FLIP heatmap (if available) | abs-diff heatmap
        # (if available). If a reference image is attached, prepend a "reference"
        # column so the eye has a ground-truth anchor next to the variants.
        cells: list[dict] = []
        if verdict.reference_img is not None:
            cells.append({
                "variant": "REFERENCE",
                "tone":    encode_for_display(verdict.reference_img, exposure=ev),
                "heat":    None,
                "diff":    None,
                "is_ref":  True,
            })
        for r in rows:
            rgb = verdict.rgb_imgs.get(r.variant)
            if rgb is None:
                continue
            tone  = encode_for_display(rgb, exposure=ev)
            heat  = flip_heatmap(verdict.err_maps.get(r.variant)) if r.variant in verdict.err_maps else None
            diff  = diff_heatmap(verdict.diff_maps.get(r.variant), verdict.diff_max) if r.variant in verdict.diff_maps else None
            noise = flip_heatmap(verdict.noise_maps.get(r.variant)) if r.variant in verdict.noise_maps else None
            cells.append({"variant": r.variant, "tone": tone, "heat": heat, "diff": diff, "noise": noise, "is_ref": False})
        if not cells:
            self._canvas.configure(image="")
            self._preview_cache = None
            return

        has_heat  = any(c["heat"] is not None for c in cells)
        has_diff  = any(c["diff"] is not None for c in cells)
        has_noise = any(c.get("noise") is not None for c in cells)
        rows_per_cell = 1 + (1 if has_heat else 0) + (1 if has_diff else 0) + (1 if has_noise else 0)

        label_h = 22
        gap     = 6
        n       = len(cells)
        ih, iw  = cells[0]["tone"].shape[:2]

        # Composite at FULL source resolution, zoom/pan are handled by the
        # canvas later, so we want max detail available.
        cell_w = iw
        cell_h = ih
        comp_h = rows_per_cell * (cell_h + label_h) + (rows_per_cell - 1) * gap
        comp_w = n * cell_w + (n - 1) * gap
        comp   = np.full((comp_h, comp_w, 3), 34, dtype=np.uint8)

        from PIL import ImageDraw, ImageFont
        pil_comp = Image.fromarray(comp)
        draw     = ImageDraw.Draw(pil_comp)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        def row_y(row_idx: int) -> tuple[int, int]:
            top = row_idx * (cell_h + label_h + gap)
            return top, top + label_h

        for i, c in enumerate(cells):
            x = i * (cell_w + gap)
            variant = c["variant"]

            lbl_y, img_y = row_y(0)
            pil_comp.paste(Image.fromarray(c["tone"]), (x, img_y))
            bw_mark = " (bench winner)"      if variant == verdict.bench_winner_variant      else ""
            nw_mark = " (noise winner)"      if variant == verdict.noise_winner_variant      else ""
            ew_mark = " (efficiency winner)" if variant == verdict.efficiency_winner_variant else ""
            score   = verdict.flip_scores.get(variant)
            noise   = verdict.noise_scores.get(variant)
            eff     = verdict.efficiency.get(variant)
            score_s = f"  FLIP {score:.4f}" if score is not None else ""
            noise_s = f"  noise {noise:.4f}" if noise is not None else ""
            eff_s   = f"  FLIP^2*ms {eff:.2f}" if eff   is not None else ""
            draw.text((x + 4, lbl_y + 2), f"{variant}{score_s}{noise_s}{eff_s}{bw_mark}{nw_mark}{ew_mark}",
                      fill=(230, 230, 230), font=font)

            row_idx = 1
            if has_heat:
                lbl_y, img_y = row_y(row_idx)
                draw.text((x + 4, lbl_y + 2), f"FLIP error vs {verdict.reference_basis}, {variant}",
                          fill=(255, 200, 120), font=font)
                if c["heat"] is not None:
                    pil_comp.paste(Image.fromarray(c["heat"]), (x, img_y))
                row_idx += 1

            if has_diff:
                lbl_y, img_y = row_y(row_idx)
                draw.text((x + 4, lbl_y + 2),
                          f"|delta| vs {verdict.reference_basis}, {variant}   (scale 0..{verdict.diff_max:.3f})",
                          fill=(180, 220, 255), font=font)
                if c["diff"] is not None:
                    pil_comp.paste(Image.fromarray(c["diff"]), (x, img_y))
                row_idx += 1

            if has_noise:
                lbl_y, img_y = row_y(row_idx)
                draw.text((x + 4, lbl_y + 2), f"noise (FLIP vs denoised self), {variant}",
                          fill=(150, 255, 180), font=font)
                if c.get("noise") is not None:
                    pil_comp.paste(Image.fromarray(c["noise"]), (x, img_y))

        self._set_source(np.array(pil_comp))

    # --- zoom / pan engine
    def _clear_source(self):
        self._source_array = None
        self._source_pil = None
        if self._canvas_image_id is not None:
            self._canvas.delete(self._canvas_image_id)
            self._canvas_image_id = None
        self._photo = None
        self._zoom_lbl.configure(text="--")
        self._ab_lbl.configure(text="")

    def _set_source(self, arr: np.ndarray, keep_view: bool = False):
        """Hand the canvas a new source image. Resets to fit-to-canvas zoom unless
        keep_view and the dimensions match the current image (the A/B flip case)."""
        prev = self._source_array
        self._source_array = arr
        # Build the PIL view once; _redraw reuses it (and crops via box=) so a
        # pan/zoom never re-copies the full source array.
        self._source_pil = Image.fromarray(arr)
        if keep_view and prev is not None and prev.shape == arr.shape and self._canvas_image_id is not None:
            self._redraw()
        else:
            self._fit_zoom()

    def _fit_zoom(self):
        if self._source_array is None:
            return
        cw = max(self._canvas.winfo_width(),  16)
        ch = max(self._canvas.winfo_height(), 16)
        ih, iw = self._source_array.shape[:2]
        # never enlarge above 1:1 when fitting, enlargement is the wheel's job
        self._fit_scale = min(cw / iw, ch / ih, 1.0)
        self._zoom      = 1.0
        eff = self._fit_scale * self._zoom
        # center
        self._view_x = (cw - iw * eff) / 2.0
        self._view_y = (ch - ih * eff) / 2.0
        self._redraw()

    def _zoom_one(self):
        """Snap to a 1 source-pixel = 1 canvas-pixel view, centred at canvas centre."""
        if self._source_array is None:
            return
        old_eff = self._fit_scale * self._zoom
        new_eff = 1.0
        self._zoom = new_eff / self._fit_scale if self._fit_scale > 0 else 1.0
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        # keep canvas centre anchored
        cx = cw / 2.0
        cy = ch / 2.0
        src_x = (cx - self._view_x) / old_eff if old_eff > 0 else 0
        src_y = (cy - self._view_y) / old_eff if old_eff > 0 else 0
        self._view_x = cx - src_x * new_eff
        self._view_y = cy - src_y * new_eff
        self._redraw()

    def _on_canvas_resize(self, _evt=None):
        # Refit only when the user hasn't zoomed in, otherwise a window resize
        # would yank them out of their zoom.
        if self._source_array is None:
            return
        if abs(self._zoom - 1.0) < 1e-6:
            self._fit_zoom()
        else:
            self._redraw()

    def _on_wheel(self, evt):
        if self._source_array is None:
            return
        # Windows/Mac use evt.delta; Linux uses Button-4/5.
        if getattr(evt, "num", None) == 4:
            factor = 1.25
        elif getattr(evt, "num", None) == 5:
            factor = 1.0 / 1.25
        else:
            factor = 1.25 if evt.delta > 0 else 1.0 / 1.25
        old_eff = self._fit_scale * self._zoom
        new_zoom = max(0.05, min(self._zoom * factor, 64.0))
        new_eff  = self._fit_scale * new_zoom
        # anchor zoom around the cursor: keep the same source pixel under the cursor
        src_x = (evt.x - self._view_x) / old_eff if old_eff > 0 else 0
        src_y = (evt.y - self._view_y) / old_eff if old_eff > 0 else 0
        self._view_x = evt.x - src_x * new_eff
        self._view_y = evt.y - src_y * new_eff
        self._zoom   = new_zoom
        self._redraw()

    def _on_pan_start(self, evt):
        if self._source_array is None:
            return
        self._pan_origin = (evt.x, evt.y, self._view_x, self._view_y)

    def _on_pan_move(self, evt):
        if self._source_array is None or self._pan_origin is None:
            return
        ox, oy, vx, vy = self._pan_origin
        self._view_x = vx + (evt.x - ox)
        self._view_y = vy + (evt.y - oy)
        self._redraw()

    def _redraw(self):
        src = self._source_pil
        if src is None:
            return
        eff = self._fit_scale * self._zoom
        iw, ih = src.size
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)

        # Only the slice of the source that lands inside the canvas needs to be
        # resampled. At high zoom this bounds the work to ~canvas-sized output
        # instead of resizing the entire (possibly huge) source every event.
        # BILINEAR samples across the crop edge, so pad the crop by 1px (NEAREST
        # doesn't need it) to avoid a shimmering seam while panning.
        nearest = eff >= 4.0
        pad = 0 if nearest else 1
        sx0 = max(0,  int(np.floor((0  - self._view_x) / eff)) - pad)
        sy0 = max(0,  int(np.floor((0  - self._view_y) / eff)) - pad)
        sx1 = min(iw, int(np.ceil ((cw - self._view_x) / eff)) + pad)
        sy1 = min(ih, int(np.ceil ((ch - self._view_y) / eff)) + pad)

        self._zoom_lbl.configure(text=f"{eff*100:.0f}%")

        # Fully panned off-screen: nothing visible, hide the image.
        if sx1 <= sx0 or sy1 <= sy0:
            if self._canvas_image_id is not None:
                self._canvas.itemconfigure(self._canvas_image_id, state=tk.HIDDEN)
            return

        target_w = max(1, int(round((sx1 - sx0) * eff)))
        target_h = max(1, int(round((sy1 - sy0) * eff)))
        resample = Image.NEAREST if nearest else Image.BILINEAR
        pil = src.resize((target_w, target_h), resample, box=(sx0, sy0, sx1, sy1))
        self._photo = ImageTk.PhotoImage(pil)

        place_x = self._view_x + sx0 * eff
        place_y = self._view_y + sy0 * eff
        if self._canvas_image_id is None:
            self._canvas_image_id = self._canvas.create_image(
                place_x, place_y, anchor=tk.NW, image=self._photo)
        else:
            self._canvas.itemconfigure(self._canvas_image_id, image=self._photo, state=tk.NORMAL)
            self._canvas.coords(self._canvas_image_id, place_x, place_y)

    # --- text helpers
    def _set_details(self, text: str):
        self._details.configure(state=tk.NORMAL)
        self._details.delete("1.0", tk.END)
        self._details.insert(tk.END, text)
        self._details.configure(state=tk.DISABLED)

    def _set_notes(self, text: str):
        self._notes_loading = True
        self._notes.delete("1.0", tk.END)
        self._notes.insert(tk.END, text)
        self._notes.edit_modified(False)
        self._notes_loading = False

    def _on_notes_modified(self, _evt=None):
        if self._notes_loading:
            return
        # Notes are per-variant, only persist when a leaf row is selected.
        if self.current_run is None or self.current_row is None:
            return
        if not self._notes.edit_modified():
            return
        self._notes.edit_modified(False)
        new = self._notes.get("1.0", tk.END).rstrip("\n")
        existing = self.current_run.notes.get(self.current_row.variant, "")
        if new != existing:
            self.current_run.notes[self.current_row.variant] = new
            self.current_run.notes_dirty = True

    def _commit_notes_if_dirty(self):
        if self.current_run is not None:
            self.current_run.save_notes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks_dir", nargs="?", default="benchmarks",
                    help="root directory containing one subdir per benchmark run")
    args = ap.parse_args()

    root = tk.Tk()
    app = App(root, Path(args.benchmarks_dir).resolve())

    def on_close():
        app._commit_notes_if_dirty()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
