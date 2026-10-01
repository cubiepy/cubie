"""Write the transient page: ``transient_page.py OUT NAME=RUN_DIR ...``."""

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


START_LABELS = {
    "final_state": "Settled state",
    "peak_state": "Voltage peak",
    "trough_state": "Trough",
    "mid_state": "Halfway between peaks",
}


def last_peak(data, settings, n):
    """Return each run's last peak before its start, in seconds."""
    char_peaks = data["peak_times"]
    has_peak = np.isfinite(char_peaks).any(axis=1)
    prev = np.full(n, np.nan)
    prev[has_peak] = np.nanmax(char_peaks[has_peak], axis=1) - float(
        data["char"]
    )
    key = settings.get("state_key") or "final_state"
    if key != "final_state":
        phases = np.load(common.RESULTS / settings["states"])
        found = phases["found"]
        phase = key.replace("_state", "")
        prev[found] = (
            phases["peak_time"][found] - phases[f"{phase}_time"][found]
        )
    return prev


def start_set(name, run_dir, data, sources, out_dir):
    """Write one start state's switch data; return its page entry."""
    settings = json.loads((run_dir / "settings.json").read_text())
    cadence = settings["cadence_s"]
    n = data["ach"].size
    peaks = np.load(run_dir / "peak_times.npy", mmap_mode="r")
    prev = last_peak(data, settings, n)
    diagonal = np.array([peaks[s, s] for s in range(n)], dtype=np.float64)
    # A peak on the start is missed on both sides; it sits at 0 s.
    on_start = (diagonal[:, 0] - prev) > 1.5 * data["steady_cl"]
    prev[on_start] = 0.0
    expected = np.stack([
        diagonal[:, 0] - prev,
        diagonal[:, 1] - diagonal[:, 0],
        diagonal[:, 2] - diagonal[:, 1],
    ]) * 1e3
    counts = np.zeros((sources.size, n, 3), dtype=np.uint16)
    for k, s in enumerate(sources):
        times = np.asarray(peaks[s], dtype=np.float64)
        counts[k] = np.where(
            np.isfinite(times), np.rint(times / cadence), 0
        ).astype(np.uint16)
    raw = counts.astype("<u2").tobytes()
    half = (len(raw) // 4) * 2
    files = []
    for part, chunk in enumerate((raw[:half], raw[half:])):
        files.append(f"switch_peaks_{name}_{part}.txt")
        (out_dir / files[-1]).write_text(
            base64.b64encode(chunk).decode("ascii")
        )
    label = START_LABELS[settings.get("state_key") or "final_state"]
    return cadence, {
        "label": label,
        "files": files,
        "expected": [nan_to_none(row, 3) for row in expected],
        "src_prev_peak": nan_to_none(prev[sources], 6),
    }


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [arg.split("=", 1) for arg in sys.argv[2:]]
    first = json.loads((Path(runs[0][1]) / "settings.json").read_text())
    data = np.load(common.RESULTS / first["input"])
    camp = np.load(
        common.RESULTS / first["input"].replace(".npz", "_camp.npz")
    )
    side = int(np.sqrt(data["ach"].size))
    axis_index = np.arange(STRIDE // 2, side, STRIDE)
    sources = (axis_index[None, :] + side * axis_index[:, None]).ravel()
    sets = {}
    for name, run_dir in runs:
        cadence, sets[name] = start_set(
            name, Path(run_dir), data, sources, out_dir
        )
    beats = np.isfinite(data["peak_times"]).sum(axis=1) >= 3
    payload = {
        "ach": data["ach"].reshape(side, side)[0].round(4).tolist(),
        "iso": data["iso"].reshape(side, side)[:, 0].round(4).tolist(),
        "cycle": nan_to_none(
            np.where(beats, data["steady_cl"] * 1e3, np.nan), 2
        ),
        "camp": nan_to_none(camp["median"].astype(np.float64), 6),
        "cadence": cadence,
        "src_ach_index": axis_index.tolist(),
        "src_iso_index": axis_index.tolist(),
        "sets": sets,
    }
    page = (
        PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__CADENCE__", f"{cadence:.6f}")
        .replace("__STRIDE__", str(STRIDE))
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {out_dir} with start states {', '.join(sets)}")


if __name__ == "__main__":
    main()
