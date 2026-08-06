function vdp_crosscheck(nSample)
% Write matlab_crosscheck.csv: x0, mu, steps, x_end, v_end per sampled run.

if nargin < 1 || isempty(nSample), nSample = 64; end

nX0 = 1024; nMu = 1024; nRuns = nX0 * nMu;
duration = 20.0;
x0Values = linspace(1.0, 2.0, nX0);
muValues = linspace(1.0, 3.0, nMu);

rng(0);
flat = randperm(nRuns, nSample);
x0s = x0Values(floor((flat - 1) / nMu) + 1);
mus = muValues(mod(flat - 1, nMu) + 1);

opts = odeset('RelTol', 1e-3, 'AbsTol', 1e-6, 'Refine', 1, ...
              'NormControl', 'off', 'OutputFcn', []);

rows = zeros(nSample, 5);
for k = 1:nSample
    [t, y] = ode45(@(t, y) vdp(t, y, mus(k)), ...
                   [0 duration], [x0s(k); 0], opts);
    rows(k, :) = [x0s(k), mus(k), numel(t) - 1, y(end, 1), y(end, 2)];
end

writematrix(rows, 'matlab_crosscheck.csv');
fprintf('median steps %g, mean steps %g\n', ...
        median(rows(:, 3)), mean(rows(:, 3)));
end

function dy = vdp(~, y, mu)
dy = [y(2); mu * (1 - y(1)^2) * y(2) - y(1)];
end
