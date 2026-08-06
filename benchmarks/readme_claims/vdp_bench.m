function vdp_bench(nSample, blockSize)
% Time ode45 on the 1024 x 1024 van der Pol grid (1,048,576 runs).
%
% Levers: anonymous-function RHS, local-function RHS, one ode45 over a
% stacked block, and parfor.  Timed on a random subsample and scaled to the
% full grid.
%
% Usage:  vdp_bench(4096, 256)

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

opts = odeset('RelTol', 1e-3, 'AbsTol', 1e-6, 'Refine', 1, ...
              'NormControl', 'off');

fprintf('sample of %d solves from the %d-run grid\n\n', nSample, nRuns);

% --- serial, anonymous RHS -------------------------------------------
tic;
for k = 1:nSample
    mu = mus(k);
    ode45(@(t, y) [y(2); mu * (1 - y(1)^2) * y(2) - y(1)], ...
          [0 duration], [x0s(k); 0], opts);
end
report('serial', toc, nSample, nRuns);

% --- serial, local-function RHS --------------------------------------
tic;
for k = 1:nSample
    ode45(@(t, y) vdp(t, y, mus(k)), [0 duration], [x0s(k); 0], opts);
end
report('nested', toc, nSample, nRuns);

% --- stacked block ----------------------------------------------------
tic;
for start = 1:blockSize:nSample
    stop = min(start + blockSize - 1, nSample);
    xb = x0s(start:stop).';
    mb = mus(start:stop).';
    y0 = [xb; zeros(size(xb))];
    ode45(@(t, y) vdpBlock(t, y, mb), [0 duration], y0, opts);
end
report(sprintf('mega/%d', blockSize), toc, nSample, nRuns);

% --- parfor -----------------------------------------------------------
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local');   % start it outside the timed region
end
tic;
parfor k = 1:nSample
    ode45(@(t, y) vdp(t, y, mus(k)), [0 duration], [x0s(k); 0], opts);
end
report(sprintf('parfor/%d', pool.NumWorkers), toc, nSample, nRuns);
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
