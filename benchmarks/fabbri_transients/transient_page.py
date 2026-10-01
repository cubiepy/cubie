"""Write the transient surfaces page: ``transient_page.py OUT_DIR RUN_DIR``."""

import base64
import json
import sys
from pathlib import Path

import numpy as np

import common

STRIDE = 6

PAGE = (Path(__file__).parent / "transient_page.html").read_text(
    encoding="utf-8"
)


def nan_to_none(values, digits):
    """Return a list with NaN as ``None`` and values rounded."""
    return [
        round(float(v), digits) if np.isfinite(v) else None for v in values
    ]


def main():
    out_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = json.loads((run_dir / "settings.json").read_text())
    cadence = settings["cadence_s"]
    data = np.load(common.RESULTS / settings["input"])
    camp = np.load(
        common.RESULTS / settings["input"].replace(".npz", "_camp.npz")
    )
    n = data["ach"].size
    side = int(np.sqrt(n))
    peaks = np.load(run_dir / "peak_times.npy", mmap_mode="r")
    char = float(data["char"])

    # Last peak before the switch, in seconds relative to the switch.
    char_peaks = data["peak_times"]
    has_peak = np.isfinite(char_peaks).any(axis=1)
    prev = np.full(n, np.nan)
    prev[has_peak] = np.nanmax(char_peaks[has_peak], axis=1) - char

    diagonal = np.array([peaks[s, s] for s in range(n)], dtype=np.float64)
    # A peak on the switch is missed on both sides; it sits at 0 s.
    on_switch = (diagonal[:, 0] - prev) > 1.5 * data["steady_cl"]
    prev[on_switch] = 0.0
    expected = np.stack([
        diagonal[:, 0] - prev,
        diagonal[:, 1] - diagonal[:, 0],
        diagonal[:, 2] - diagonal[:, 1],
    ]) * 1e3

    axis_index = np.arange(STRIDE // 2, side, STRIDE)
    sources = (axis_index[None, :] + side * axis_index[:, None]).ravel()
    counts = np.zeros((sources.size, n, 3), dtype=np.uint16)
    for k, s in enumerate(sources):
        times = np.asarray(peaks[s], dtype=np.float64)
        counts[k] = np.where(
            np.isfinite(times), np.rint(times / cadence), 0
        ).astype(np.uint16)
    raw = counts.astype("<u2").tobytes()
    half = (len(raw) // 4) * 2
    for part, chunk in enumerate((raw[:half], raw[half:])):
        (out_dir / f"switch_peaks_{part}.txt").write_text(
            base64.b64encode(chunk).decode("ascii")
        )

    beats = np.isfinite(data["peak_times"]).sum(axis=1) >= 3
    payload = {
        "ach": data["ach"].reshape(side, side)[0].round(4).tolist(),
        "iso": data["iso"].reshape(side, side)[:, 0].round(4).tolist(),
        "cycle": nan_to_none(
            np.where(beats, data["steady_cl"] * 1e3, np.nan), 2
        ),
        "camp": nan_to_none(camp["median"].astype(np.float64), 6),
        "expected": [nan_to_none(row, 3) for row in expected],
        "cadence": cadence,
        "src_ach_index": axis_index.tolist(),
        "src_iso_index": axis_index.tolist(),
        "src_prev_peak": nan_to_none(prev[sources], 6),
    }
    page = (
        PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__CADENCE__", f"{cadence:.6f}")
        .replace("__STRIDE__", str(STRIDE))
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {out_dir}; switch data {len(raw) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
