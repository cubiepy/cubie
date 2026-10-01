"""Write a steady-state surface page: ``cycle_surface.py OUT [NPZ]``."""

import json
import sys

import numpy as np

import common

PAGE = """<title>Fabbri Steady State Surfaces</title>
<style>
/* Layout: one column; controls above a full-width plot, stats below. */
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
main { max-width: 1100px; margin: 0 auto; display: grid; gap: 14px; }
h1 { font-size: 1.35rem; margin: 0; text-wrap: balance;
  font-weight: 600; }
p { margin: 0; color: var(--muted); line-height: 1.5; max-width: 70ch; }
.controls { display: flex; flex-wrap: wrap; gap: 8px;
  align-items: center; }
.controls span { font-size: .78rem; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted); }
.controls span + span, .gap { margin-left: 12px; }
button { font: inherit; font-size: .9rem; padding: 6px 12px;
  border-radius: 6px; border: 1px solid var(--line);
  background: var(--surface); color: var(--fg); cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--accent);
  color: var(--accent); font-weight: 600; }
button:focus-visible { outline: 2px solid var(--accent);
  outline-offset: 2px; }
#plot { width: 100%; height: min(78vh, 760px); min-height: 420px;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 8px; }
.stats { display: flex; flex-wrap: wrap; gap: 18px;
  font-family: var(--font-data); font-variant-numeric: tabular-nums;
  font-size: .85rem; color: var(--muted); }
.stats b { color: var(--fg); font-weight: 600; }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
  <h1>Steady state over the ACh / Iso grid</h1>
  <p>Fabbri-Linder SAN model, autonomic cascade on (ANS = 1). Each point
  is the median cycle length over 10 s after settling 1200 s from the
  CellML initial state (rosenbrock23, atol = rtol = 1e-6, float32).
  Peak times are sampled every 0.000244 s. Gaps in cycle length are runs
  that do not beat.__CAMP_NOTE____NOTE__</p>
  <div class="controls">
    <span>Quantity</span>
    <button id="qcl" aria-pressed="true">Cycle length</button>
    __CAMP_BUTTON__
    <span class="gap">Height scale</span>
    <button id="lin" aria-pressed="true">Linear</button>
    <button id="log" aria-pressed="false">Log</button>
    <span class="gap">ACh / Iso axes</span>
    <button id="axlin" aria-pressed="true">Linear</button>
    <button id="axlog" aria-pressed="false">Log</button>
    <span class="gap">View</span>
    <button id="v3d" aria-pressed="true">3D surface</button>
    <button id="v2d" aria-pressed="false">Top-down map</button>
  </div>
  <div id="plot" role="img"
    aria-label="Steady-state quantity against ACh and Iso"></div>
  <div class="stats">
    <div>beating runs <b>__NBEAT__ / __NRUNS__</b></div>
    <div>cycle length <b>__MIN__ to __MAX__ ms</b></div>
    <div>grid <b>__GRID__</b></div>
  </div>
</main>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
const D = __DATA__;
const ramp = [[0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"],
  [0.5, "#3987e5"], [0.67, "#256abf"], [0.83, "#184f95"],
  [1, "#0d366b"]];
const toLog = z => z.map(r => r.map(v => v === null ? null : Math.log10(v)));
function logTicks(z) {
  const v = z.flat().filter(x => x !== null);
  const lo = Math.min(...v), hi = Math.max(...v), vals = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) {
      const t = m * 10 ** e;
      if (t >= lo && t <= hi) vals.push(t);
    }
  return { tickvals: vals.map(Math.log10), ticktext: vals.map(String) };
}
let scale = "lin", view = "3d", axes = D.log ? "log" : "lin";
let quantity = "cycle";
const css = n => getComputedStyle(document.documentElement)
  .getPropertyValue(n).trim();
function draw() {
  const fg = css("--fg"), muted = css("--muted"), line = css("--line");
  const surf = css("--surface");
  const log = scale === "log";
  const Q = D.layers[quantity];
  const zTicks = log ? logTicks(Q.z) : {};
  // Log axes cannot place zero; drop the zero row and column.
  const cut = axes === "log" && !D.log ? 1 : 0;
  const trim = rows => rows.slice(cut).map(r => r.slice(cut));
  const trace = {
    type: view === "3d" ? "surface" : "heatmap",
    x: D.ach.slice(cut), y: D.iso.slice(cut),
    z: trim(log ? toLog(Q.z) : Q.z), customdata: trim(Q.z),
    colorscale: ramp, connectgaps: false,
    hovertemplate: "ACh %{x:.2f} nM<br>Iso %{y:.2f} nM<br>" +
      Q.hover + "<extra></extra>",
    colorbar: Object.assign({
      title: { text: Q.label, side: "right",
        font: { color: muted } },
      tickfont: { color: muted }, outlinewidth: 0, len: 0.8 },
      zTicks)
  };
  const axis = t => ({ title: { text: t, font: { color: muted } },
    tickfont: { color: muted }, gridcolor: line, zerolinecolor: line });
  const ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];
  const xy = t => Object.assign(axis(t), axes === "log"
    ? { type: "log", tickvals: ticks, ticktext: ticks.map(String) }
    : { type: "linear" });
  const layout = { paper_bgcolor: surf, plot_bgcolor: surf,
    font: { family: "IBM Plex Sans, system-ui, sans-serif", color: fg },
    hoverlabel: { font: { family: "IBM Plex Mono, monospace" } } };
  if (view === "3d") {
    const box = t => Object.assign(axis(t),
      { backgroundcolor: surf, showbackground: true });
    layout.margin = { l: 0, r: 0, t: 0, b: 0 };
    layout.scene = {
      xaxis: Object.assign(box("ACh_cas (nM)"), xy("ACh_cas (nM)")),
      yaxis: Object.assign(box("Iso_cas (nM)"), xy("Iso_cas (nM)")),
      zaxis: Object.assign(box(Q.label), zTicks),
      camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
      aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.7 } };
  } else {
    layout.margin = { l: 60, r: 10, t: 10, b: 50 };
    layout.xaxis = xy("ACh_cas (nM)");
    layout.yaxis = xy("Iso_cas (nM)");
  }
  Plotly.react("plot", [trace], layout,
    { responsive: true, displaylogo: false });
}
const $ = id => document.getElementById(id);
function pick(on, off, set) {
  on.onclick = () => { set(); on.setAttribute("aria-pressed", "true");
    off.setAttribute("aria-pressed", "false"); draw(); };
}
pick($("lin"), $("log"), () => { scale = "lin"; });
if ($("qcamp")) {
  pick($("qcl"), $("qcamp"), () => { quantity = "cycle"; });
  pick($("qcamp"), $("qcl"), () => { quantity = "camp"; });
}
pick($("log"), $("lin"), () => { scale = "log"; });
pick($("v3d"), $("v2d"), () => { view = "3d"; });
pick($("axlin"), $("axlog"), () => { axes = "lin"; });
pick($("axlog"), $("axlin"), () => { axes = "log"; });
if (D.log) {
  $("axlog").setAttribute("aria-pressed", "true");
  $("axlin").setAttribute("aria-pressed", "false");
}
pick($("v2d"), $("v3d"), () => { view = "2d"; });
matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement,
  { attributes: true, attributeFilter: ["data-theme"] });
draw();
</script>
"""


def grid_text(data, side, log_grid):
    """Describe the grid's size, spacing and ranges."""
    ach = data["ach"].reshape(side, side)[0]
    iso = data["iso"].reshape(side, side)[:, 0]
    spacing = "log-spaced" if log_grid else "linear"
    return (
        f"{side} x {side} {spacing}, ACh {ach[0]:g}-{ach[-1]:g} nM, "
        f"Iso {iso[0]:g}-{iso[-1]:g} nM"
    )


def layer(values, side, label, hover, digits):
    """Return one plotted quantity as rows with gaps as ``None``."""
    rows = [
        [round(float(v), digits) if np.isfinite(v) else None for v in row]
        for row in values.reshape(side, side)
    ]
    return {"z": rows, "label": label, "hover": hover}


def main():
    name = sys.argv[2] if len(sys.argv) > 2 else "steady_state.npz"
    data = np.load(common.RESULTS / name)
    log_grid = bool(data["log_grid"]) if "log_grid" in data else False
    side = int(np.sqrt(data["ach"].size))
    beats = np.isfinite(data["peak_times"]).sum(axis=1) >= 3
    cycle = np.where(beats, data["steady_cl"] * 1e3, np.nan)
    layers = {"cycle": layer(
        cycle, side, "cycle length (ms)",
        "cycle %{customdata:.1f} ms", 2,
    )}
    camp_path = common.RESULTS / name.replace(".npz", "_camp.npz")
    if camp_path.exists():
        camp = np.load(camp_path)
        layers["camp"] = layer(
            camp["median"].astype(np.float64), side, "cAMP median (mM)",
            "cAMP %{customdata:.5f} mM", 6,
        )
        camp_note = (
            f" cAMP is the median of samples every "
            f"{float(camp['save_every']):.6f} s over "
            f"{float(camp['duration']):g} s from the settled state."
        )
        camp_button = (
            '<button id="qcamp" aria-pressed="false">cAMP median</button>'
        )
    else:
        camp_note = camp_button = ""
    payload = json.dumps(
        {
            "ach": data["ach"].reshape(side, side)[0].round(4).tolist(),
            "iso": data["iso"].reshape(side, side)[:, 0].round(4).tolist(),
            "layers": layers,
            "log": log_grid,
        },
        separators=(",", ":"),
    )
    page = (
        PAGE.replace("__DATA__", payload)
        .replace("__NBEAT__", str(int(beats.sum())))
        .replace("__NRUNS__", str(int(beats.size)))
        .replace("__MIN__", f"{np.nanmin(cycle):.0f}")
        .replace("__MAX__", f"{np.nanmax(cycle):.0f}")
        .replace("__GRID__", grid_text(data, side, log_grid))
        .replace("__CAMP_NOTE__", camp_note)
        .replace("__CAMP_BUTTON__", camp_button)
        .replace("__NOTE__", "" if log_grid else (
            " Log axes omit the zero-concentration row and column."))
    )
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        handle.write(page)


if __name__ == "__main__":
    main()
