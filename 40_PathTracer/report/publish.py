#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WEB_EXTENSIONS = {".css", ".html", ".js", ".mjs", ".wasm"}


def clean_name(value):
    parts = [part for part in value.replace("\\", "/").strip("/").split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise argparse.ArgumentTypeError(f"invalid relative path: {value}")
    return "/".join(parts)


def report(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return clean_name(name), path


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Ditt path tracer report bundles with rsync.")
    parser.add_argument("--destination", required=True, help="rsync destination")
    parser.add_argument("--rsync-exe", default="rsync")
    parser.add_argument("--rsync-arg", action="append", default=[])
    parser.add_argument("--report", action="append", type=report, default=[], metavar="NAME=PATH")
    parser.add_argument("--public")
    parser.add_argument("--private")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path, root):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    return cwd_path if cwd_path.exists() else (root / path).resolve()


def collect(files, source, destination=Path(), extensions=None):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for item in sorted(source.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts:
            continue
        if extensions is not None and item.suffix.lower() not in extensions:
            continue
        files.append((item, destination / item.relative_to(source)))


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def rsync_path(value):
    text = str(value).replace("\\", "/").rstrip("/\\")
    if os.name == "nt" and len(text) > 2 and text[1] == ":":
        text = f"/{text[0].lower()}{text[2:]}"
    return text + "/"


def run_rsync(args, source):
    exe = shutil.which(args.rsync_exe) or (str(Path(args.rsync_exe)) if Path(args.rsync_exe).is_file() else None)
    if not exe:
        raise RuntimeError("rsync executable not found. Install rsync or pass --rsync-exe.")

    command = [exe, "-az"]
    command += ["--delete"] if args.delete else []
    command += ["--checksum"] if args.checksum else []
    command += ["--dry-run"] if args.dry_run else []
    command += args.rsync_arg
    command += [rsync_path(source), rsync_path(args.destination)]

    print(subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    report_dir = Path(__file__).resolve().parent
    example_dir = report_dir.parent

    reports = list(args.report)
    reports += [("public", args.public)] if args.public else []
    reports += [("private", args.private)] if args.private else []
    if not reports:
        raise RuntimeError("pass at least one --report, --public, or --private")

    files = []
    if not args.no_viewer:
        collect(files, report_dir, extensions=WEB_EXTENSIONS)
    for name, path in reports:
        collect(files, resolve(path, example_dir), Path(name))

    with tempfile.TemporaryDirectory(prefix="pt-report-stage-") as tmp:
        stage_dir = Path(tmp)
        for source, destination in files:
            link_or_copy(source, stage_dir / destination)
        run_rsync(args, stage_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
