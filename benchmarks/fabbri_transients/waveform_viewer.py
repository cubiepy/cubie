"""Serve switch waveforms: ``waveform_viewer.py RUN_DIR [PORT]``."""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

PAGE = Path(__file__).with_name("waveform_viewer.html")
MAX_POINTS = 20000


def nan_to_none(values):
    """Return a list with NaN as ``None``."""
    return [float(v) if np.isfinite(v) else None for v in values]


class Run:
    """Memory-mapped waveforms and grid of one ``waveforms.py`` run."""

    def __init__(self, run_dir: Path):
        self.voltage = np.load(run_dir / "voltage.npy", mmap_mode="r")
        self.time = np.load(run_dir / "time.npy")
        self.failed = np.load(run_dir / "failed.npy")
        grid = np.load(run_dir / "grid.npz")
        self.src_ach = np.unique(grid["source_ach"])
        self.src_iso = np.unique(grid["source_iso"])
        self.tgt_ach = np.unique(grid["target_ach"])
        self.tgt_iso = np.unique(grid["target_iso"])
        self.meta = {
            "source_ach": self.src_ach.tolist(),
            "source_iso": self.src_iso.tolist(),
            "target_ach": self.tgt_ach.tolist(),
            "target_iso": self.tgt_iso.tolist(),
            "source_cl": nan_to_none(grid["source_cl"]),
            "target_cl": nan_to_none(grid["target_cl"]),
            "source_prev_peak": nan_to_none(grid["source_prev_peak"]),
            "failed": self.failed.astype(int).tolist(),
            "duration": float(self.time[-1]),
            "n_samples": int(self.time.size),
        }

    def traces(self, sources, targets, t0, t1):
        """Return times and voltages; long spans keep block min/max."""
        lo, hi = np.searchsorted(self.time, [t0, t1])
        hi = max(hi, lo + 2)
        block = self.voltage[sources, targets, lo:hi].astype(np.float32)
        time = self.time[lo:hi]
        stride = int(np.ceil(block.shape[1] / MAX_POINTS))
        if stride <= 2:
            return time, block
        n_blocks = block.shape[1] // stride
        block = block[:, : n_blocks * stride].reshape(
            block.shape[0], n_blocks, stride
        )
        low, high = block.min(axis=2), block.max(axis=2)
        min_first = block.argmin(axis=2) < block.argmax(axis=2)
        first = np.where(min_first, low, high)
        second = np.where(min_first, high, low)
        starts = time[: n_blocks * stride : stride]
        out_time = np.stack([starts, starts + 0.5 * stride * (
            time[1] - time[0])], axis=1).ravel()
        out = np.stack([first, second], axis=2).reshape(block.shape[0], -1)
        return out_time, out


def make_handler(run: Run):
    """Return a request handler bound to ``run``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/meta":
                self.send(json.dumps(run.meta).encode(), "application/json")
            elif url.path == "/traces":
                sources = np.array(
                    [int(v) for v in query["sources"].split(",")]
                )
                targets = np.array(
                    [int(v) for v in query["targets"].split(",")]
                )
                time, block = run.traces(
                    sources, targets, float(query["t0"]), float(query["t1"])
                )
                header = np.array(block.shape, dtype="<u4").tobytes()
                body = (
                    header + time.astype("<f8").tobytes()
                    + block.astype("<f4").tobytes()
                )
                self.send(body, "application/octet-stream")
            else:
                self.send_error(404)

    return Handler


def main():
    run = Run(Path(sys.argv[1]))
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(run))
    print(f"serving http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
