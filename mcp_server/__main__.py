"""Command line entry point: python -m mcp_server --port 8090"""

from __future__ import annotations

import argparse
import signal
import sys
import threading


def main() -> int:
    parser = argparse.ArgumentParser(description="supercar-bench MCP tool server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # Tools read the shared mock data through car_va_data_manager; make sure
    # it is initialized before the first call.
    from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager

    car_va_data_manager.initialize()

    from mcp_server.server import create_server

    server = create_server(args.host, args.port)
    print(f"supercar-bench MCP server listening on http://{args.host}:{args.port}")

    stop = threading.Event()

    def _shutdown(signum, frame):  # noqa: ANN001
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while not stop.is_set():
            stop.wait(1.0)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
