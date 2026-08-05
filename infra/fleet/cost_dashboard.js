const API_TOKEN = document.querySelector(
  'meta[name="api-token"]'
).content;
const C = {
  wait: '#c7ccd1',
  boot: '#4c78a8',
  steps: '#54a24b',
  shutdown: '#e45756'
};
const PALETTE = [
  '#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2',
  '#eeca3b', '#b279a2', '#ff9da6', '#9d755d', '#bab0ac',
  '#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#98df8a',
  '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
];
const charts = {};
const runTitles = new Map();
let accountRequestVersion = 0;

function chart(id) {
  if (!charts[id]) {
    charts[id] = echarts.init(document.getElementById(id));
  }
  return charts[id];
}

window.addEventListener('resize', () => {
  Object.values(charts).forEach(item => item.resize());
});

function escapeHTML(value) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[character]
  );
}

function mins(seconds) {
  return seconds == null ? null : seconds / 60;
}

// A leg whose runner died mid-step has no archived log; instance type
// and spot price are recovered from CloudTrail, but queue wait is not.
function legLabel(leg) {
  return leg.log_known === false ? `${leg.label} (no log)` : leg.label;
}

function renderSharedLegend(id, entries) {
  const legend = document.getElementById(id);
  legend.replaceChildren();
  entries.forEach(entry => {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.backgroundColor = entry.color;
    item.append(swatch, document.createTextNode(entry.name));
    legend.appendChild(item);
  });
}

function localDateValue(value) {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, '0');
  const day = String(value.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function localAccountTime(value) {
  const hourly = value.includes('T');
  const instant = new Date(hourly ? value : `${value}T00:00:00Z`);
  const options = hourly
    ? {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'}
    : {year: 'numeric', month: '2-digit', day: '2-digit'};
  return instant.toLocaleString(undefined, options);
}

function axisTooltip(parameters, suffix, digits = 1) {
  if (!parameters.length) {
    return '';
  }
  const lines = parameters
    .filter(item => item.value != null && item.seriesName !== 'off')
    .map(item => {
      const value = Number(item.value).toFixed(digits);
      return `${item.marker}${escapeHTML(item.seriesName)}: ` +
        `${value}${suffix}`;
    });
  return `<b>${escapeHTML(parameters[0].axisValueLabel)}</b><br>` +
    lines.join('<br>');
}

async function requestJSON(url, method = 'GET') {
  const response = await fetch(url, {
    method,
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {[TOKEN_HEADER]: API_TOKEN}
  });
  const payload = await response.json();
  if (!response.ok || (payload && payload.error)) {
    throw new Error(payload.detail || payload.error || response.statusText);
  }
  return payload;
}

const TOKEN_HEADER = 'X-Cubie-Dashboard-Token';

function renderGantt(payload) {
  const legs = payload.legs;
  const categories = legs.map(legLabel);
  const offsets = legs.map(leg => mins(leg.offset_s));
  const makeSeries = (name, key, color) => ({
    name,
    type: 'bar',
    stack: 'timeline',
    color,
    emphasis: {focus: 'series'},
    data: legs.map(leg => mins(leg[key]))
  });
  chart('cGantt').setOption({
    grid: {left: 150, right: 16, top: 30, bottom: 36},
    legend: {top: 0, data: ['wait', 'boot', 'CI steps', 'shutdown']},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, ' min')
    },
    xAxis: {
      type: 'value',
      name: 'minutes from first job scheduled',
      nameLocation: 'middle',
      nameGap: 24
    },
    yAxis: {
      type: 'category',
      data: categories,
      inverse: true,
      axisLabel: {fontSize: 10}
    },
    series: [
      {
        name: 'off',
        type: 'bar',
        stack: 'timeline',
        data: offsets,
        silent: true,
        itemStyle: {color: 'transparent'},
        tooltip: {show: false}
      },
      makeSeries('wait', 'wait_s', C.wait),
      makeSeries('boot', 'boot_s', C.boot),
      makeSeries('CI steps', 'steps_s', C.steps),
      makeSeries('shutdown', 'shutdown_s', C.shutdown)
    ]
  }, true);
}

function completeSum(legs, key) {
  if (legs.some(leg => leg[key] == null)) {
    return null;
  }
  return legs.reduce((total, leg) => total + leg[key], 0);
}

function renderGanttAggregate(payload) {
  const bands = [
    ['wait', 'wait_s', C.wait],
    ['boot', 'boot_s', C.boot],
    ['CI steps', 'steps_s', C.steps],
    ['shutdown', 'shutdown_s', C.shutdown]
  ];
  chart('cGanttAgg').setOption({
    grid: {left: 44, right: 10, top: 30, bottom: 36},
    legend: {show: false},
    tooltip: {
      trigger: 'item',
      formatter: item => item.value == null
        ? `${escapeHTML(item.seriesName)}: unknown`
        : `${escapeHTML(item.seriesName)}: ` +
          `${Number(item.value).toFixed(1)} min`
    },
    xAxis: {type: 'category', data: ['run total']},
    yAxis: {type: 'value', name: 'minutes'},
    series: bands.map(([name, key, color]) => ({
      name,
      type: 'bar',
      stack: 'timeline',
      color,
      data: [mins(completeSum(payload.legs, key))],
      label: {
        show: true,
        formatter: item => item.value != null && item.value >= 6 ? name : '',
        color: '#fff',
        fontSize: 10
      }
    }))
  }, true);
}

function stepTotals(legs) {
  const totals = {};
  legs.forEach(leg => leg.steps.forEach(step => {
    totals[step.name] = (totals[step.name] || 0) + step.dur_s;
  }));
  return Object.keys(totals).sort(
    (first, second) => totals[second] - totals[first]
  );
}

function renderSteps(payload) {
  const legs = payload.legs;
  const order = stepTotals(legs);
  const stepSeries = order.map((name, index) => ({
    name,
    type: 'bar',
    stack: 'steps',
    color: PALETTE[index % PALETTE.length],
    emphasis: {focus: 'series'},
    data: legs.map(leg => {
      const step = leg.steps.find(item => item.name === name);
      return step ? mins(step.dur_s) : 0;
    })
  }));
  renderSharedLegend('stepsLegend', [
    {name: 'wait', color: C.wait},
    ...order.map((name, index) => ({
      name,
      color: PALETTE[index % PALETTE.length]
    }))
  ]);
  chart('cSteps').setOption({
    grid: {left: 44, right: 16, top: 16, bottom: 96},
    legend: {show: false},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, ' min', 2)
    },
    xAxis: {
      type: 'category',
      data: legs.map(legLabel),
      axisLabel: {rotate: 40, fontSize: 10, interval: 0}
    },
    yAxis: {type: 'value', name: 'minutes'},
    series: [
      {
        name: 'wait',
        type: 'bar',
        color: C.wait,
        data: legs.map(leg => mins(leg.wait_s))
      },
      ...stepSeries
    ]
  }, true);
}

function renderStepsAggregate(payload) {
  const order = stepTotals(payload.legs);
  const totals = {};
  payload.legs.forEach(leg => leg.steps.forEach(step => {
    totals[step.name] = (totals[step.name] || 0) + step.dur_s;
  }));
  chart('cStepsAgg').setOption({
    grid: {left: 44, right: 10, top: 16, bottom: 96},
    legend: {show: false},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, ' min', 2)
    },
    xAxis: {type: 'category', data: ['run total']},
    yAxis: {type: 'value', name: 'minutes'},
    series: [
      {
        name: 'wait',
        type: 'bar',
        color: C.wait,
        data: [mins(completeSum(payload.legs, 'wait_s'))]
      },
      ...order.map((name, index) => ({
        name,
        type: 'bar',
        stack: 'steps',
        color: PALETTE[index % PALETTE.length],
        data: [mins(totals[name] || 0)]
      }))
    ]
  }, true);
}

// Windows and Linux are separate spot markets for the same instance
// type, so a bar that merged them would show a rate neither leg paid.
function typeKey(leg) {
  return leg.product ? `${leg.type} · ${leg.product}` : leg.type;
}

function renderType(payload) {
  const legs = payload.legs.filter(leg => leg.type);
  const types = [...new Set(legs.map(typeKey))].sort();
  const aggregates = types.map(type => {
    const matching = legs.filter(leg => typeKey(leg) === type);
    const complete = matching.every(
      leg => leg.cost != null && leg.billed_hours != null
    );
    if (!complete) {
      return {cost: null, billedHours: null};
    }
    return {
      cost: matching.reduce((total, leg) => total + leg.cost, 0),
      billedHours: matching.reduce(
        (total, leg) => total + leg.billed_hours,
        0
      )
    };
  });
  const costs = aggregates.map(aggregate => aggregate.cost);
  const costColor = '#25324b';
  const usageSeries = legs.map((leg, index) => ({
    name: leg.label,
    type: 'bar',
    stack: 'usage',
    color: PALETTE[index % PALETTE.length],
    emphasis: {focus: 'series'},
    data: types.map(type => {
      if (typeKey(leg) !== type) {
        return null;
      }
      return leg.billed_hours == null ? null : leg.billed_hours * 60;
    })
  }));
  renderSharedLegend('typeLegend', [
    ...legs.map((leg, index) => ({
      name: leg.label,
      color: PALETTE[index % PALETTE.length]
    })),
    {name: 'cost', color: costColor}
  ]);
  chart('cType').setOption({
    grid: {left: 56, right: 56, top: 16, bottom: 40},
    legend: {show: false},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => {
        if (!parameters.length) {
          return '';
        }
        const typeIndex = parameters[0].dataIndex;
        const type = types[typeIndex];
        const markers = new Map(
          parameters.map(item => [item.seriesIndex, item.marker])
        );
        const lines = [];
        legs.forEach((leg, index) => {
          if (typeKey(leg) !== type) {
            return;
          }
          const value = leg.billed_hours == null
            ? 'unknown'
            : `${(leg.billed_hours * 60).toFixed(1)} min`;
          lines.push(
            `${markers.get(index) || ''}${escapeHTML(leg.label)}: ${value}`
          );
        });
        const cost = costs[typeIndex];
        const costMarker = markers.get(legs.length) || '';
        const costText = cost == null
          ? 'unknown'
          : `$${cost.toFixed(4)}`;
        lines.push(`${costMarker}cost: ${costText}`);
        return `<b>${escapeHTML(type)}</b><br>${lines.join('<br>')}`;
      }
    },
    xAxis: {type: 'category', data: types},
    yAxis: [
      {type: 'value', name: 'billed minutes'},
      {type: 'value', name: 'USD', position: 'right'}
    ],
    series: [
      ...usageSeries,
      {
        name: 'cost',
        type: 'bar',
        color: costColor,
        yAxisIndex: 1,
        data: costs,
        label: {
          show: true,
          position: 'top',
          fontSize: 10,
          formatter: item => {
            const aggregate = aggregates[item.dataIndex];
            if (aggregate.cost == null || !aggregate.billedHours) {
              return '';
            }
            const rate = aggregate.cost / aggregate.billedHours;
            return `~$${rate.toFixed(3)}/h`;
          }
        }
      }
    ]
  }, true);
}

function renderRun(payload) {
  const empty = document.getElementById('runEmpty');
  const chartArea = document.getElementById('runCharts');
  const title = runTitles.get(String(payload.run_id)) ||
    `Run ${payload.run_id}`;
  if (!payload.legs.length) {
    const started = payload.started_leg_count || 0;
    const outcome = payload.conclusion || payload.status || 'unknown';
    document.getElementById('runMeta').textContent =
      `${title} · ${started} started GPU ` +
      `leg${started === 1 ? '' : 's'} · ${outcome}`;
    empty.textContent = payload.status === 'completed'
      ? 'This run started GPU work, but no GPU leg completed.'
      : 'GPU work has started, but no GPU leg has completed yet.';
    empty.classList.remove('hidden');
    chartArea.classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  chartArea.classList.remove('hidden');
  const known = payload.legs.filter(leg => leg.cost != null);
  const knownCost = known.reduce((total, leg) => total + leg.cost, 0);
  const missing = payload.legs.length - known.length;
  const costText = missing
    ? `known compute cost $${knownCost.toFixed(3)}; ` +
      `${missing} leg${missing === 1 ? '' : 's'} missing spot price ` +
      'or termination telemetry'
    : `compute cost $${knownCost.toFixed(3)}`;
  const logless = payload.legs.filter(leg => leg.log_known === false).length;
  const logText = logless
    ? ` · ${logless} leg${logless === 1 ? '' : 's'} left no archived log ` +
      '(runner lost mid-step)'
    : '';
  document.getElementById('runMeta').textContent =
    `${title} · ${payload.legs.length} GPU legs · ${costText}${logText}`;
  renderGantt(payload);
  renderGanttAggregate(payload);
  renderSteps(payload);
  renderStepsAggregate(payload);
  renderType(payload);
}

function renderStack(id, legendId, times, series, yName) {
  renderSharedLegend(
    legendId,
    series.map((item, index) => ({
      name: item.name,
      color: PALETTE[index % PALETTE.length]
    }))
  );
  chart(id).setOption({
    grid: {left: 56, right: 16, top: 16, bottom: 70},
    legend: {show: false},
    graphic: series.length ? [] : [{
      type: 'text',
      left: 'center',
      top: 'middle',
      style: {
        text: 'No available Cost Explorer data has a non-zero value ' +
          'in this window.',
        fill: '#7f8c8d',
        font: '14px sans-serif',
        textAlign: 'center'
      }
    }],
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, '', 4)
    },
    xAxis: {
      type: 'category',
      data: times.map(localAccountTime),
      axisLabel: {rotate: 60, fontSize: 9}
    },
    yAxis: {type: 'value', name: yName},
    series: series.map((item, index) => ({
      name: item.name,
      type: 'bar',
      stack: 'account',
      color: PALETTE[index % PALETTE.length],
      data: item.data
    }))
  }, true);
}

function resetAccountChart(wrapperId, chartId) {
  if (charts[chartId]) {
    charts[chartId].dispose();
    delete charts[chartId];
  }
  const chartElement = document.createElement('div');
  chartElement.id = chartId;
  chartElement.className = 'chart';
  document.getElementById(wrapperId).replaceChildren(chartElement);
}

async function loadAccount(force) {
  const requestVersion = ++accountRequestVersion;
  const status = document.getElementById('acctStatus');
  const start = document.getElementById('acctFrom').value;
  const end = document.getElementById('acctTo').value;
  const granularity = document.getElementById('acctGran').value;
  const query = new URLSearchParams({
    start,
    end,
    gran: granularity,
    tz: String(-new Date().getTimezoneOffset())
  });
  status.textContent = 'fetching…';
  try {
    const route = force ? '/api/account/refresh' : '/api/account';
    const payload = await requestJSON(
      `${route}?${query.toString()}`,
      force ? 'POST' : 'GET'
    );
    if (requestVersion !== accountRequestVersion) return;
    resetAccountChart('acctUsageWrap', 'cAcctUsage');
    resetAccountChart('acctCostWrap', 'cAcctCost');
    renderStack(
      'cAcctUsage',
      'acctUsageLegend',
      payload.times,
      payload.usage,
      'EC2 hours'
    );
    renderStack(
      'cAcctCost',
      'acctCostLegend',
      payload.times,
      payload.cost,
      'USD (gross usage)'
    );
    status.textContent = '';
  } catch (error) {
    if (requestVersion === accountRequestVersion) {
      status.textContent = `error: ${error.message}`;
    }
  }
}

async function loadRun(id) {
  const status = document.getElementById('status');
  status.textContent = `fetching run ${id} …`;
  try {
    const payload = await requestJSON(`/api/run?id=${encodeURIComponent(id)}`);
    renderRun(payload);
    status.textContent = '';
    return payload;
  } catch (error) {
    status.textContent = `error: ${error.message}`;
    return null;
  }
}

async function init() {
  const now = new Date();
  const accountStart = new Date(now);
  accountStart.setDate(accountStart.getDate() - 2);
  document.getElementById('acctTo').value = localDateValue(now);
  document.getElementById('acctFrom').value = localDateValue(accountStart);
  document.getElementById('acctFrom').onchange = () => loadAccount(false);
  document.getElementById('acctTo').onchange = () => loadAccount(false);
  document.getElementById('acctGran').onchange = () => loadAccount(false);
  document.getElementById('acctForce').onclick = () => loadAccount(true);
  loadAccount(false);
  const selector = document.getElementById('runSelect');
  try {
    const runs = await requestJSON('/api/runs');
    selector.replaceChildren();
    runs.forEach(run => {
      const title = run.title || `Run ${run.id}`;
      const count = run.started_leg_count;
      const outcome = run.conclusion || run.status || 'unknown';
      runTitles.set(String(run.id), title);
      const option = document.createElement('option');
      option.value = run.id;
      option.textContent =
        `${new Date(run.created_at).toLocaleString()} · ` +
        `${title} · ${count} started leg${count === 1 ? '' : 's'} · ` +
        outcome;
      selector.appendChild(option);
    });
    selector.onchange = () => loadRun(selector.value);
    if (runs.length) {
      selector.value = String(runs[0].id);
      await loadRun(runs[0].id);
    } else {
      document.getElementById('runMeta').textContent =
        'No recent runs found.';
      document.getElementById('runCharts').classList.add('hidden');
    }
  } catch (error) {
    selector.replaceChildren(new Option('error'));
    document.getElementById('status').textContent = error.message;
  }
}

init();
