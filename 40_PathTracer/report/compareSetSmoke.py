#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_workdir(root: Path) -> Path:
    for exe_name in ("40_pathtracer.exe", "40_pathtracer_rwdi.exe"):
        if (root / exe_name).is_file():
            return root
    return root / "bin"


def resolve_under(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def relative_to_workdir(path: Path, workdir: Path) -> str:
    return os.path.relpath(path.resolve(), workdir.resolve())


def run_step(command: list[str], workdir: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=workdir, check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_payload(report_dir: Path) -> dict:
    summary = load_json(report_dir / "summary.json")
    for summary_path in report_dir.rglob("summary.json"):
        payload = load_json(summary_path)
        base = summary_path.parent
        for relative in ("index.html", "css/report.css", "js/report.js"):
            require((base / relative).is_file(), f"Missing report UI file referenced by {summary_path}: {relative}")
        for scene in payload.get("results", []):
            for image in scene.get("array", []):
                for key in ("render", "reference", "difference"):
                    artifact = image.get(key)
                    if artifact:
                        require((base / artifact).is_file(), f"Missing {key} artifact referenced by {summary_path}: {artifact}")
    return summary


def copy_report_template(template_dir: Path, report_dir: Path) -> None:
    require(template_dir.is_dir(), f"Report template does not exist: {template_dir}")
    for item in template_dir.iterdir():
        destination = report_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def verify_no_host_paths(report_dir: Path, project_root: Path) -> None:
    project_text = str(project_root.resolve()).replace("\\", "/")
    for summary_path in report_dir.rglob("summary.json"):
        text = summary_path.read_text(encoding="utf-8").replace("\\", "/")
        require(project_text not in text, f"Summary leaks the local project path: {summary_path}")


def main() -> int:
    root = project_dir()
    default_workdir_path = default_workdir(root)
    default_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Generate a local EX40 compare-set smoke with three labeled reports.")
    parser.add_argument("--workdir", default=str(default_workdir_path), help="Directory used as the executable working directory.")
    parser.add_argument("--exe", default="40_pathtracer.exe", help="Path to the EX40 executable. Relative paths resolve against --workdir.")
    parser.add_argument("--scene", default="../../media/mitsuba/shapetest.xml", help="Scene path passed to EX40. Relative paths resolve in --workdir.")
    parser.add_argument("--output-dir", default=f"out/compare_vendor_smoke_{default_stamp}", help="Output directory. Relative paths resolve against --workdir.")
    parser.add_argument("--set-name", default="Release compare smoke", help="Human-readable compare-set label.")
    parser.add_argument("--baseline", default="nvidia", help="Baseline input id.")
    parser.add_argument("--inputs", nargs="+", default=["nvidia=NVIDIA", "amd=AMD", "intel=Intel"], help="Input specs in id=label form.")
    parser.add_argument("--skip-render", action="store_true", help="Reuse existing input report directories under --output-dir.")
    args = parser.parse_args()

    workdir = resolve_under(Path.cwd(), args.workdir)
    exe_path = resolve_under(workdir, args.exe)
    output_dir = resolve_under(workdir, args.output_dir)
    scene_path = resolve_under(workdir, args.scene)
    report_template = root / "report"

    require(workdir.is_dir(), f"Workdir does not exist: {workdir}")
    require(exe_path.is_file(), f"Executable does not exist: {exe_path}")
    require(scene_path.is_file(), f"Scene does not exist: {scene_path}")
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")

    inputs: list[tuple[str, str]] = []
    for spec in args.inputs:
        require("=" in spec, f"Input spec must use id=label form: {spec}")
        input_id, label = spec.split("=", 1)
        require(input_id, f"Input id is empty in spec: {spec}")
        require(label, f"Input label is empty in spec: {spec}")
        require(all(input_id != existing_id for existing_id, _ in inputs), f"Duplicate input id: {input_id}")
        inputs.append((input_id, label))
    require(len(inputs) >= 2, "At least two inputs are required.")
    require(any(input_id == args.baseline for input_id, _ in inputs), f"Baseline id was not found: {args.baseline}")

    output_dir.mkdir(parents=True)
    exe_arg = str(exe_path)
    output_rel = Path(relative_to_workdir(output_dir, workdir))

    if not args.skip_render:
        scene_arg = relative_to_workdir(scene_path, workdir)
        for input_id, _label in inputs:
            report_rel = output_rel / input_id
            run_step([
                exe_arg,
                "--scene", scene_arg,
                "--process-sensors", "RenderAllThenTerminate",
                "--headless",
                "--output-dir", str(report_rel / "renders"),
                "--report-dir", str(report_rel),
            ], workdir)

    manifest = {
        "name": args.set_name,
        "baseline": args.baseline,
        "inputs": [
            {
                "id": input_id,
                "name": label,
                "reportDir": input_id,
            }
            for input_id, label in inputs
        ],
    }
    manifest_path = output_dir / "set.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    set_rel = output_rel / "set"
    run_step([
        exe_arg,
        "--compare-report-set", str(output_rel / "set.json"),
        "--report-dir", str(set_rel),
    ], workdir)

    set_dir = output_dir / "set"
    copy_report_template(report_template, set_dir)
    for pair_summary in (set_dir / "pairs").glob("*/summary.json"):
        copy_report_template(report_template, pair_summary.parent)
    summary = verify_payload(set_dir)
    require(summary.get("pass_status") == "passed", f"Compare set did not pass: {summary.get('pass_status')}")
    pairs = summary.get("comparison", {}).get("pairs", [])
    require(len(pairs) == len(inputs) - 1, "Pair count does not match input count.")

    relocated_dir = output_dir / "set_relocated"
    shutil.copytree(set_dir, relocated_dir)
    verify_payload(relocated_dir)
    verify_no_host_paths(set_dir, root)
    verify_no_host_paths(relocated_dir, root)

    route = relative_to_workdir(set_dir, root).replace("\\", "/")
    relocated_route = relative_to_workdir(relocated_dir, root).replace("\\", "/")
    print(json.dumps({
        "status": "passed",
        "set": route,
        "relocated": relocated_route,
        "inputs": [{"id": input_id, "name": label} for input_id, label in inputs],
        "pairs": pairs,
        "url": "http://127.0.0.1:8040/report/#/" + route,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}", file=sys.stderr)
        raise SystemExit(error.returncode)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
