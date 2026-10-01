"""Write the transient surfaces page: ``transient_page.py OUT_DIR RUN_DIR``."""

import json
import sys
from pathlib import Path

import numpy as np

import common

STRIDE = 6

PAGE = """<title>Fabbri Switch Transients</title>
<style>
/* Layout: steady-state pair, controls, then a three-panel transient row. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --fg: #1a2230; --muted: #5a6577;
  --line: #d9dee6; --accent: #256abf;
  --font-body: "IBM Plex Sans", system-ui, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11151c; --surface: #171c25; --fg: #e4e8ef; --muted: #98a2b3;
  --line: #2a3140; --accent: #5598e7; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11151c; --surface: #171c25; --fg: #e4e8ef; --muted: #98a2b3;
  --line: #2a3140; --accent: #5598e7; color-scheme: dark }
body { background: var(--bg); color: var(--fg);
  font-family: var(--font-body); padding-inline: 16px;
  padding-block: 20px 32px; }
main { max-width: 1500px; margin: 0 auto; display: grid; gap: 14px; }
h1 { font-size: 1.35rem; margin: 0; text-wrap: balance;
  font-weight: 600; }
h2 { font-size: 1rem; margin: 6px 0 0; font-weight: 600; }
p { margin: 0; color: var(--muted); line-height: 1.5; max-width: 80ch; }
.controls { display: flex; flex-wrap: wrap; gap: 8px;
  align-items: center; }
.controls span { font-size: .78rem; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted); }
.gap { margin-left: 12px; }
button, select { font: inherit; font-size: .9rem; padding: 6px 10px;
  border-radius: 6px; border: 1px solid var(--line);
  background: var(--surface); color: var(--fg); }
button { cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--accent);
  color: var(--accent); font-weight: 600; }
button:focus-visible, select:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
.row { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.panel { min-width: 0; display: grid; gap: 4px; }
.panel h3 { margin: 0; font-size: .85rem; font-weight: 600;
  color: var(--muted); }
.plot { width: 100%; height: 440px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 8px; }
.stats { font-family: var(--font-data); font-variant-numeric: tabular-nums;
  font-size: .85rem; color: var(--muted); }
.stats b { color: var(--fg); font-weight: 600; }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
  <h1>Fabbri-Linder switch transients</h1>
  <p>Every settled grid point (source) was relaunched with every grid
  point's parameters (target), and the first three peaks after the switch
  were recorded. Peaks are sampled every __CADENCE__ s. Interval 1 runs
  from the source's last peak before the switch to the first peak after
  it; intervals 2 and 3 are the next two. Expected intervals come from
  relaunching each grid point with its own parameters, measured the same
  way. This page holds every __STRIDE__th source on each axis.</p>
  <h2>Steady state</h2>
  <div class="row">
    <div class="panel"><h3>Cycle length (ms)</h3>
      <div id="p-cl" class="plot" role="img"
        aria-label="Steady-state cycle length"></div></div>
    <div class="panel"><h3>cAMP median (mM)</h3>
      <div id="p-camp" class="plot" role="img"
        aria-label="Steady-state cAMP median"></div></div>
  </div>
  <h2>Switch from one source</h2>
  <div class="controls">
    <span>Source ACh (nM)</span><select id="sach"></select>
    <span class="gap">Source Iso (nM)</span><select id="siso"></select>
    <span class="gap">Interval</span>
    <button id="i1" aria-pressed="true">1</button>
    <button id="i2" aria-pressed="false">2</button>
    <button id="i3" aria-pressed="false">3</button>
    <span class="gap">View</span>
    <button id="v3d" aria-pressed="true">3D surface</button>
    <button id="v2d" aria-pressed="false">Top-down map</button>
  </div>
  <div id="source" class="stats"></div>
  <div class="row">
    <div class="panel"><h3 id="t-exp">Expected interval (ms)</h3>
      <div id="p-exp" class="plot" role="img"
        aria-label="Expected interval per target"></div></div>
    <div class="panel"><h3 id="t-meas">Measured interval (ms)</h3>
      <div id="p-meas" class="plot" role="img"
        aria-label="Measured interval per target"></div></div>
    <div class="panel"><h3 id="t-ratio">Measured / expected</h3>
      <div id="p-ratio" class="plot" role="img"
        aria-label="Measured over expected interval"></div></div>
  </div>
  <div id="status" class="stats">Loading switch data...</div>
</main>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
const D = __DATA__;
const ramp = [[0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"],
  [0.5, "#3987e5"], [0.67, "#256abf"], [0.83, "#184f95"],
  [1, "#0d366b"]];
const diverge = [[0, "#1c5cab"], [0.5, "#e4e6ea"], [1, "#b2462e"]];
const N = D.ach.length, S = D.src_ach_index.length;
let interval = 0, view = "3d", src = [Math.floor(S / 2), 0], peaks = null;
const css = n => getComputedStyle(document.documentElement)
  .getPropertyValue(n).trim();
const ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];
function grid(flat, f) {
  const rows = [];
  for (let r = 0; r < N; r++) {
    const row = [];
    for (let c = 0; c < N; c++) {
      const v = flat[r * N + c];
      row.push(v === null || !isFinite(v) ? null : (f ? f(v) : v));
    }
    rows.push(row);
  }
  return rows;
}
function plot(id, z, label, hover, scale) {
  const fg = css("--fg"), muted = css("--muted"), line = css("--line");
  const surf = css("--surface");
  const axis = t => ({ title: { text: t, font: { color: muted } },
    tickfont: { color: muted }, gridcolor: line, zerolinecolor: line,
    type: "log", tickvals: ticks, ticktext: ticks.map(String) });
  const trace = { type: view === "3d" ? "surface" : "heatmap",
    x: D.ach, y: D.iso, z: z, connectgaps: false,
    colorscale: scale || ramp,
    hovertemplate: "ACh %{x:.2f} nM<br>Iso %{y:.2f} nM<br>" + hover +
      "<extra></extra>",
    colorbar: { title: { text: label, side: "right",
      font: { color: muted } }, tickfont: { color: muted },
      outlinewidth: 0, len: 0.75, thickness: 12 } };
  if (scale) { trace.cmid = 1; }
  const layout = { paper_bgcolor: surf, plot_bgcolor: surf,
    font: { family: "IBM Plex Sans, system-ui, sans-serif", color: fg,
      size: 11 },
    hoverlabel: { font: { family: "IBM Plex Mono, monospace" } } };
  if (view === "3d") {
    const box = t => Object.assign(axis(t),
      { backgroundcolor: surf, showbackground: true });
    layout.margin = { l: 0, r: 0, t: 0, b: 0 };
    layout.scene = { xaxis: box("ACh (nM)"), yaxis: box("Iso (nM)"),
      zaxis: { title: { text: label, font: { color: muted } },
        tickfont: { color: muted }, gridcolor: line,
        backgroundcolor: surf, showbackground: true },
      camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
      aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.7 } };
  } else {
    layout.margin = { l: 55, r: 10, t: 10, b: 45 };
    layout.xaxis = axis("ACh (nM)");
    layout.yaxis = axis("Iso (nM)");
  }
  Plotly.react(id, [trace], layout, { responsive: true, displaylogo: false });
}
function measured() {
  // Sample counts k -> time k * cadence after the switch; 0 = no peak.
  const s = src[1] * S + src[0], base = s * N * 3, out = new Float64Array(N);
  const prev = D.src_prev_peak[s];
  for (let t = 0; t < N; t++) {
    const k = [0, 1, 2].map(j => peaks[base + t * 3 + j]);
    let a, b;
    if (interval === 0) { a = prev; b = k[0] ? k[0] * D.cadence : NaN; }
    else {
      a = k[interval - 1] ? k[interval - 1] * D.cadence : NaN;
      b = k[interval] ? k[interval] * D.cadence : NaN;
    }
    out[t] = (b - a) * 1e3;
  }
  return out;
}
function drawSteady() {
  plot("p-cl", grid(D.cycle), "ms", "cycle %{z:.1f} ms");
  plot("p-camp", grid(D.camp), "mM", "cAMP %{z:.5f} mM");
}
function drawSwitch() {
  const k = interval + 1;
  const expected = D.expected[interval];
  document.getElementById("t-exp").textContent =
    `Expected interval ${k} (ms)`;
  document.getElementById("t-meas").textContent =
    `Measured interval ${k} after the switch (ms)`;
  document.getElementById("t-ratio").textContent =
    `Measured / expected, interval ${k}`;
  const si = D.src_ach_index[src[0]], sj = D.src_iso_index[src[1]];
  const s = src[1] * S + src[0];
  const prev = D.src_prev_peak[s];
  document.getElementById("source").innerHTML =
    `source <b>ACh ${D.ach[si].toFixed(2)} nM, Iso ` +
    `${D.iso[sj].toFixed(2)} nM</b>; steady cycle <b>` +
    `${isFinite(D.cycle[sj * N + si]) && D.cycle[sj * N + si] !== null
      ? D.cycle[sj * N + si].toFixed(1) + " ms" : "none"}</b>; last peak ` +
    `<b>${isFinite(prev) && prev !== null
      ? (-prev * 1e3).toFixed(1) + " ms" : "none"}</b> before the switch`;
  plot("p-exp", grid(expected), "ms", "expected %{z:.1f} ms");
  if (!peaks) return;
  const m = measured();
  plot("p-meas", grid(m), "ms", "measured %{z:.1f} ms");
  const ratio = m.map((v, t) => v / expected[t]);
  plot("p-ratio", grid(ratio), "ratio", "ratio %{z:.3f}", diverge);
}
const $ = id => document.getElementById(id);
function fill(sel, idx, axis, start) {
  idx.forEach((v, n) => { const o = document.createElement("option");
    o.value = n; o.textContent = axis[v].toPrecision(3);
    if (n === start) o.selected = true; sel.appendChild(o); });
}
fill($("sach"), D.src_ach_index, D.ach, src[0]);
fill($("siso"), D.src_iso_index, D.iso, src[1]);
$("sach").onchange = e => { src[0] = +e.target.value; drawSwitch(); };
$("siso").onchange = e => { src[1] = +e.target.value; drawSwitch(); };
function group(ids, set) {
  ids.forEach((id, n) => { $(id).onclick = () => { set(n);
    ids.forEach((o, m) => $(o).setAttribute("aria-pressed",
      String(m === n))); drawSteady(); drawSwitch(); }; });
}
group(["i1", "i2", "i3"], n => { interval = n; });
group(["v3d", "v2d"], n => { view = n ? "2d" : "3d"; });
const redraw = () => { drawSteady(); drawSwitch(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
new MutationObserver(redraw).observe(document.documentElement,
  { attributes: true, attributeFilter: ["data-theme"] });
drawSteady();
drawSwitch();
fetch("switch_peaks.bin").then(r => {
  if (!r.ok) throw new Error(`switch data request returned ${r.status}`);
  return r.arrayBuffer();
}).then(buf => {
  peaks = new Uint16Array(buf);
  $("status").textContent = "";
  drawSwitch();
}).catch(err => {
  $("status").textContent = `Switch data did not load: ${err.message}.`;
});
</script>
"""


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
    counts.tofile(out_dir / "switch_peaks.bin")

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
    print(f"wrote {out_dir}; switch data "
          f"{(out_dir / 'switch_peaks.bin').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
