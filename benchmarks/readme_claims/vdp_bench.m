function vdp_bench(nSample, blockSize)
% Time ode45 levers on the 1024 x 1024 van der Pol grid (1,048,576 runs).
%
% Usage:  vdp_bench(20000, 256)

if nargin < 1 || isempty(nSample),  nSample  = 256; end
if nargin < 2 || isempty(blockSize), blockSize = 256; end

nX0 = 1024; nMu = 1024; nRuns = nX0 * nMu;
duration = 20.0;
x0Values = linspace(1.0, 2.0, nX0);
muValues = linspace(1.0, 3.0, nMu);

rng(0);
flat = randperm(nRuns, nSample);
x0s = x0Values(floor((flat - 1) / nMu) + 1);
mus = muValues(mod(flat - 1, nMu) + 1);

% OutputFcn [] keeps ode45 from ever reaching for odeplot.
opts = odeset('RelTol', 1e-3, 'AbsTol', 1e-6, 'Refine', 1, ...
              'NormControl', 'off', 'OutputFcn', [], 'Stats', 'off');

fprintf('sample of %d solves from the %d-run grid\n\n', nSample, nRuns);
last = zeros(1, nSample);   % keeps each solve's result live
[~, ~] = ode45(@(t, y) vdp(t, y, 1.5), [0 duration], [1; 0], opts);

% --- serial, anonymous RHS -------------------------------------------
tic;
for k = 1:nSample
    mu = mus(k);
    [~, y] = ode45(@(t, y) [y(2); mu * (1 - y(1)^2) * y(2) - y(1)], ...
                   [0 duration], [x0s(k); 0], opts);
    last(k) = y(end, 1);
end
report('serial', toc, nSample, nRuns);

% --- serial, local-function RHS --------------------------------------
tic;
for k = 1:nSample
    [~, y] = ode45(@(t, y) vdp(t, y, mus(k)), ...
                   [0 duration], [x0s(k); 0], opts);
    last(k) = y(end, 1);
end
report('nested', toc, nSample, nRuns);

% --- stacked block ----------------------------------------------------
tic;
for start = 1:blockSize:nSample
    stop = min(start + blockSize - 1, nSample);
    xb = x0s(start:stop).';
    mb = mus(start:stop).';
    y0 = [xb; zeros(size(xb))];
    [~, y] = ode45(@(t, y) vdpBlock(t, y, mb), [0 duration], y0, opts);
    last(start:stop) = y(end, 1:numel(mb));
end
report(sprintf('mega/%d', blockSize), toc, nSample, nRuns);

% --- parfor -----------------------------------------------------------
pool = gcp('nocreate');
if isempty(pool)
    % Profile default caps workers below the core count; raise it in memory.
    cluster = parcluster('Processes');
    cluster.NumWorkers = feature('numcores');
    pool = parpool(cluster, cluster.NumWorkers);
end
warm = zeros(1, pool.NumWorkers);   % first parfor JITs on every worker
parfor k = 1:pool.NumWorkers
    [~, y] = ode45(@(t, y) vdp(t, y, 1.5), [0 duration], [1; 0], opts);
    warm(k) = y(end, 1);
end
tic;
parfor k = 1:nSample
    [~, y] = ode45(@(t, y) vdp(t, y, mus(k)), ...
                   [0 duration], [x0s(k); 0], opts);
    last(k) = y(end, 1);
end
report(sprintf('parfor/%d', pool.NumWorkers), toc, nSample, nRuns);

% --- parfor over chunks, one ode45 loop per iteration ------------------
nChunks = pool.NumWorkers * 4;
edges = round(linspace(1, nSample + 1, nChunks + 1));
chunkX = cell(1, nChunks); chunkMu = cell(1, nChunks);
for c = 1:nChunks
    chunkX{c} = x0s(edges(c):edges(c + 1) - 1);
    chunkMu{c} = mus(edges(c):edges(c + 1) - 1);
end
chunkLast = cell(1, nChunks);
tic;
parfor c = 1:nChunks
    xs = chunkX{c}; ms = chunkMu{c};
    out = zeros(1, numel(xs));
    for k = 1:numel(xs)
        [~, y] = ode45(@(t, y) vdp(t, y, ms(k)), ...
                       [0 duration], [xs(k); 0], opts);
        out(k) = y(end, 1);
    end
    chunkLast{c} = out;
end
report(sprintf('parchunk/%d', pool.NumWorkers), toc, nSample, nRuns);
last = [chunkLast{:}];

fprintf('checksum %.6f\n', sum(last));
end

function dy = vdp(~, y, mu)
dy = [y(2); mu * (1 - y(1)^2) * y(2) - y(1)];
end

function dy = vdpBlock(~, y, mu)
n = numel(mu);
x = y(1:n); v = y(n+1:end);
dy = [v; mu .* (1 - x.^2) .* v - x];
end

function report(name, elapsed, nSample, nRuns)
per = elapsed / nSample;
fprintf('%-10s %9.3f ms/solve   %d -> %9.2f min\n', ...
        name, per * 1e3, nRuns, per * nRuns / 60);
end
