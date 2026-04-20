#!/usr/bin/env python3
import argparse
import functools
import http.server
import socket
import sys
import webbrowser
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the Ditt static report viewer.")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8040, help="First port to try.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    return parser.parse_args()


def port_is_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host, first_port):
    for port in range(first_port, first_port + 64):
        if port_is_available(host, port):
            return port
    raise RuntimeError(f"No free port found from {first_port} to {first_port + 63}")


def main():
    args = parse_args()
    report_dir = Path(__file__).resolve().parent
    example_root = report_dir.parent
    port = choose_port(args.bind, args.port)
    url = f"http://{args.bind}:{port}/report/"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(example_root))
    server = http.server.ThreadingHTTPServer((args.bind, port), handler)

    print(f"Serving Ditt report root: {example_root}")
    print(f"Open: {url}")
    print(f"Open a payload with: {url}#/bin/out/<report-directory>")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
