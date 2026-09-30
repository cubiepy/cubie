"""Write a steady-state cycle-length surface page: ``cycle_surface.py OUT``."""

import json
import sys

import numpy as np

import common

PAGE = """<title>Fabbri Steady Cycle Length</title>
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
  <h1>Steady-state cycle length over the ACh / Iso grid</h1>
  <p>Fabbri-Linder SAN model, autonomic cascade on (ANS = 1). Each point
  is the median cycle length over 10 s after settling 1200 s from the
  CellML initial state (rosenbrock23, atol = rtol = 1e-6, float32).
  Peak times are sampled every 0.000244 s. Gaps are runs that do not
  beat.</p>
  <div class="controls">
    <span>Height scale</span>
    <button id="lin" aria-pressed="true">Linear</button>
    <button id="log" aria-pressed="false">Log</button>
    <span class="gap">View</span>
    <button id="v3d" aria-pressed="true">3D surface</button>
    <button id="v2d" aria-pressed="false">Top-down map</button>
  </div>
  <div id="plot" role="img"
    aria-label="Steady-state cycle length against ACh and Iso"></div>
  <div class="stats">
    <div>beating runs <b>__NBEAT__ / 65536</b></div>
    <div>cycle length <b>__MIN__ to __MAX__ ms</b></div>
    <div>grid step <b>ACh 0.392 nM, Iso 0.784 nM</b></div>
  </div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<script>
const D = __DATA__;
const ramp = [[0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"],
  [0.5, "#3987e5"], [0.67, "#256abf"], [0.83, "#184f95"],
  [1, "#0d366b"]];
const zLog = D.z.map(r => r.map(v => v === null ? null : Math.log10(v)));
const logTicks = { tickvals: [2.5, 2.7, 3, 3.3, 3.5],
  ticktext: ["316", "501", "1000", "1995", "3162"] };
let scale = "lin", view = "3d";
const css = n => getComputedStyle(document.documentElement)
  .getPropertyValue(n).trim();
function draw() {
  const fg = css("--fg"), muted = css("--muted"), line = css("--line");
  const surf = css("--surface");
  const log = scale === "log";
  const trace = {
    type: view === "3d" ? "surface" : "heatmap",
    x: D.ach, y: D.iso, z: log ? zLog : D.z, customdata: D.z,
    colorscale: ramp, connectgaps: false,
    hovertemplate: "ACh %{x:.2f} nM<br>Iso %{y:.2f} nM<br>" +
      "cycle %{customdata:.1f} ms<extra></extra>",
    colorbar: Object.assign({
      title: { text: "cycle length (ms)", side: "right",
        font: { color: muted } },
      tickfont: { color: muted }, outlinewidth: 0, len: 0.8 },
      log ? logTicks : {})
  };
  const axis = t => ({ title: { text: t, font: { color: muted } },
    tickfont: { color: muted }, gridcolor: line, zerolinecolor: line });
  const layout = { paper_bgcolor: surf, plot_bgcolor: surf,
    font: { family: "IBM Plex Sans, system-ui, sans-serif", color: fg },
    hoverlabel: { font: { family: "IBM Plex Mono, monospace" } } };
  if (view === "3d") {
    const box = t => Object.assign(axis(t),
      { backgroundcolor: surf, showbackground: true });
    layout.margin = { l: 0, r: 0, t: 0, b: 0 };
    layout.scene = { xaxis: box("ACh_cas (nM)"),
      yaxis: box("Iso_cas (nM)"),
      zaxis: Object.assign(box("cycle length (ms)"), log ? logTicks : {}),
      camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
      aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.7 } };
  } else {
    layout.margin = { l: 60, r: 10, t: 10, b: 50 };
    layout.xaxis = axis("ACh_cas (nM)");
    layout.yaxis = axis("Iso_cas (nM)");
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
pick($("log"), $("lin"), () => { scale = "log"; });
pick($("v3d"), $("v2d"), () => { view = "3d"; });
pick($("v2d"), $("v3d"), () => { view = "2d"; });
matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement,
  { attributes: true, attributeFilter: ["data-theme"] });
draw();
</script>
"""


def main():
    data = np.load(common.RESULTS / "steady_state.npz")
    side = int(np.sqrt(data["ach"].size))
    beats = np.isfinite(data["peak_times"]).sum(axis=1) >= 3
    cycle = np.where(beats, data["steady_cl"] * 1e3, np.nan)
    cycle = cycle.reshape(side, side)
    rows = [
        [round(float(v), 2) if np.isfinite(v) else None for v in row]
        for row in cycle
    ]
    payload = json.dumps(
        {
            "ach": data["ach"].reshape(side, side)[0].round(4).tolist(),
            "iso": data["iso"].reshape(side, side)[:, 0].round(4).tolist(),
            "z": rows,
        },
        separators=(",", ":"),
    )
    page = (
        PAGE.replace("__DATA__", payload)
        .replace("__NBEAT__", str(int(beats.sum())))
        .replace("__MIN__", f"{np.nanmin(cycle):.0f}")
        .replace("__MAX__", f"{np.nanmax(cycle):.0f}")
    )
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        handle.write(page)


if __name__ == "__main__":
    main()
