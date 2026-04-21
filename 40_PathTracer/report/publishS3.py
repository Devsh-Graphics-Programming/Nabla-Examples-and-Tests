#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import hmac
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


WEB_EXTENSIONS = {".css", ".html", ".js", ".mjs", ".wasm"}
CHUNK_SIZE = 8 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a Ditt report directory to S3.")
    parser.add_argument("--source")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="https://s3.fr-par.scw.cloud")
    parser.add_argument("--region", default="fr-par")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checksum", action="store_true", help="Hash every file before deciding whether to upload.")
    parser.add_argument("--jobs", type=int, default=16)
    return parser.parse_args()


def env(name, fallback=None):
    value = os.environ.get(name) or (os.environ.get(fallback) if fallback else None)
    if not value:
        raise RuntimeError(f"missing {name}" + (f" or {fallback}" if fallback else ""))
    return value


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hashes(path):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


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


def signed_headers(endpoint, region, access_key, secret_key, bucket, key, method, payload_hash, extra_headers=None):
    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    host = urllib.parse.urlparse(endpoint).netloc
    quoted_key = urllib.parse.quote(key.replace("\\", "/"), safe="/-_.~")
    canonical_uri = f"/{bucket}/{quoted_key}"
    headers = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if extra_headers:
        headers.update(extra_headers)
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
    canonical_request = "\n".join([
        method,
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
        sha256_bytes(canonical_request.encode("utf-8")),
    ])
    signature = hmac.new(signing_key(secret_key, date, region), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return canonical_uri, headers


def signed_url_request(endpoint, region, access_key, secret_key, bucket, key, method):
    uri, headers = signed_headers(endpoint, region, access_key, secret_key, bucket, key, method, sha256_bytes(b""))
    return urllib.request.Request(endpoint.rstrip("/") + uri, headers=headers, method=method)


def head_object(endpoint, region, access_key, secret_key, bucket, key):
    request = signed_url_request(endpoint, region, access_key, secret_key, bucket, key, "HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            return {
                "size": int(response.headers.get("Content-Length", "-1")),
                "etag": response.headers.get("ETag", "").strip('"'),
                "mtime_ns": response.headers.get("x-amz-meta-source-mtime-ns"),
            }
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def should_checksum(path, force):
    return force or path.suffix.lower() in (WEB_EXTENSIONS | {".json"})


def remote_matches(remote, path, checksum):
    if not remote:
        return False

    stat = path.stat()
    size = stat.st_size
    if remote["size"] != size:
        return False

    if remote.get("mtime_ns") == str(stat.st_mtime_ns):
        return True

    if not checksum:
        return False

    _, local_md5 = file_hashes(path)
    return remote["etag"] == local_md5


def put_stream(endpoint, region, access_key, secret_key, bucket, key, path, sha256):
    parsed = urllib.parse.urlparse(endpoint)
    uri, headers = signed_headers(
        endpoint,
        region,
        access_key,
        secret_key,
        bucket,
        key,
        "PUT",
        sha256,
        {
            "content-length": str(path.stat().st_size),
            "content-type": content_type(path),
            "x-amz-meta-source-mtime-ns": str(path.stat().st_mtime_ns),
        }
    )
    connection = http.client.HTTPSConnection(parsed.netloc) if parsed.scheme == "https" else http.client.HTTPConnection(parsed.netloc)
    with path.open("rb") as file:
        connection.request("PUT", uri, body=file, headers=headers)
    response = connection.getresponse()
    response.read()
    connection.close()
    if response.status not in (200, 201):
        raise RuntimeError(f"{key}: HTTP {response.status}")


def put_object(endpoint, region, access_key, secret_key, bucket, key, path, dry_run, checksum):
    remote = head_object(endpoint, region, access_key, secret_key, bucket, key)
    if remote_matches(remote,path,checksum):
        return f"skip {key} ({path.stat().st_size} bytes)"
    if dry_run:
        return f"upload {key} ({path.stat().st_size} bytes)"

    local_sha256, _ = file_hashes(path)
    put_stream(endpoint, region, access_key, secret_key, bucket, key, path, local_sha256)
    return f"uploaded {key}"


def main():
    args = parse_args()
    source = Path(args.source).resolve() if args.source else None
    if not args.static_only:
        if source is None:
            raise RuntimeError("missing --source")
        if not source.is_dir():
            raise RuntimeError(f"not a directory: {source}")

    access_key = env("AWS_ACCESS_KEY_ID", "SCW_ACCESS_KEY")
    secret_key = env("AWS_SECRET_ACCESS_KEY", "SCW_SECRET_KEY")
    prefix = args.prefix.strip("/")
    report_dir = Path(__file__).resolve().parent
    upload_files = collect(report_dir, extensions=WEB_EXTENSIONS)
    if not args.static_only:
        upload_files += collect(source)
    upload_files = sorted(upload_files, key=lambda item: (item[1].name == "index.html", item[1].as_posix()))

    def upload(item):
        path, destination = item
        return put_object(
            args.endpoint,
            args.region,
            access_key,
            secret_key,
            args.bucket,
            f"{prefix}/{destination.as_posix()}",
            path,
            args.dry_run,
            should_checksum(path,args.checksum)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.jobs)) as executor:
        for line in executor.map(upload,upload_files):
            print(line)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, urllib.error.HTTPError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
