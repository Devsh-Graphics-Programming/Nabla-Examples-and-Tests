#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import hmac
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


WEB_EXTENSIONS = {".css", ".html", ".js", ".mjs", ".wasm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a Ditt report directory to S3.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="https://s3.fr-par.scw.cloud")
    parser.add_argument("--region", default="fr-par")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def env(name, fallback=None):
    value = os.environ.get(name) or (os.environ.get(fallback) if fallback else None)
    if not value:
        raise RuntimeError(f"missing {name}" + (f" or {fallback}" if fallback else ""))
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sign(key, message):
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret_key, date, region):
    key = sign(("AWS4" + secret_key).encode("utf-8"), date)
    key = sign(key, region)
    key = sign(key, "s3")
    return sign(key, "aws4_request")


def content_type(path):
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/aces", ".exr")
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def collect(source, destination=Path(), extensions=None):
    result = []
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        result.append((path, destination / path.relative_to(source)))
    return result


def put_object(endpoint, region, access_key, secret_key, bucket, key, path, dry_run):
    data = path.read_bytes()
    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    host = urllib.parse.urlparse(endpoint).netloc
    quoted_key = urllib.parse.quote(key.replace("\\", "/"), safe="/-_.~")
    canonical_uri = f"/{bucket}/{quoted_key}"
    url = endpoint.rstrip("/") + canonical_uri
    payload_hash = digest(data)
    headers = {
        "content-type": content_type(path),
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
    canonical_request = "\n".join([
        "PUT",
        canonical_uri,
        "",
        canonical_headers,
        signed_headers,
        payload_hash,
    ])
    scope = f"{date}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        scope,
        digest(canonical_request.encode("utf-8")),
    ])
    signature = hmac.new(signing_key(secret_key, date, region), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    if dry_run:
        print(f"upload {key} ({len(data)} bytes)")
        return

    request = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(request) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"{key}: HTTP {response.status}")
    print(f"uploaded {key}")


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.is_dir():
        raise RuntimeError(f"not a directory: {source}")

    access_key = env("AWS_ACCESS_KEY_ID", "SCW_ACCESS_KEY")
    secret_key = env("AWS_SECRET_ACCESS_KEY", "SCW_SECRET_KEY")
    prefix = args.prefix.strip("/")
    report_dir = Path(__file__).resolve().parent
    upload_files = collect(report_dir, extensions=WEB_EXTENSIONS)
    upload_files += collect(source)
    upload_files = sorted(upload_files, key=lambda item: (item[1].name == "index.html", item[1].as_posix()))

    for path, destination in upload_files:
        put_object(args.endpoint, args.region, access_key, secret_key, args.bucket, f"{prefix}/{destination.as_posix()}", path, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, urllib.error.HTTPError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
