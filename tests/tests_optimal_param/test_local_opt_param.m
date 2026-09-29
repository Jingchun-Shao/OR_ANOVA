%TEST_LOCAL_OPT_PARAM_EXPERIMENT
% Compare Newton and both sextic eigenvalue methods with the BE/SDP reference.
%
% This script is intentionally small-scale because the BE reference forms
% dense matrices and requires CVX.

clear;
clc;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'core'));
set_path();

if exist('cvx_begin', 'file') ~= 2
    error(['CVX was not found on the MATLAB path. ', ...
        'Install CVX and run cvx_setup before using the BE reference.']);
end

rng(1);

configs = [
    6, 2, 1, 0.80;
    6, 2, 2, 0.80;
    7, 2, 2, 0.80;
    7, 2, 1, 0.80;
    8, 2, 1, 0.80;
    9, 2, 1, 0.80;
];


eps_model = 0.04;
eta_noise = 0.03;
num_samples = 100;
power_iter = 5000;

num_tests = size(configs, 1);
results = table( ...
    'Size', [num_tests, 15], ...
    'VariableTypes', repmat({'double'}, 1, 15), ...
    'VariableNames', {'d', 'n', 's', 'N', 'M', ...
        'num_samples', 'power_iter', ...
        'tau_newton', 'tau_sextic_power', 'tau_sextic_svd', 'tau_sdp', ...
        'lwce_newton_sq', 'lwce_sextic_power_sq', ...
        'lwce_sextic_svd_sq', 'wce_sdp_sq'} ...
);

fprintf('\n================ Local setting: optimal-parameter test ================\n');
fprintf('model mismatch bound = %.3e, relative noise level = %.3e\n', ...
    eps_model, eta_noise);

for k = 1:num_tests
    d = configs(k, 1);
    n = configs(k, 2);
    s = configs(k, 3);
    rho = configs(k, 4);

    sz = n * ones(1, d);
    N = prod(sz);
    M = max(1, round(rho * N));

    P_V_perp = @(x) x - cluster_expansion(x, s, sz);

    idx = randperm(N, M);
    S = sparse(1:M, idx, 1, M, N);

    x_model = cluster_expansion(randn(N, 1), s, sz);
    z = P_V_perp(randn(N, 1));
    z = eps_model * z / max(norm(z), 1e-14);
    x_true = x_model + z;

    eta_bound = eta_noise * norm(x_true);
    noise = randn(M, 1);
    noise = eta_bound * noise / max(norm(noise), 1e-14);
    y = S * x_true + noise;

    fprintf('\nTest %d/%d: d=%d, n=%d, s=%d, N=%d, M=%d, eta=%.3g\n', ...
        k, num_tests, d, n, s, N, M, eta_bound);

    eta_bound = 1.1 * eta_bound;
    eps_bound = 1.1 * eps_model;

    opts = struct('method', 'Newton', 'tol', 1e-6, 'verbose', false, ...
        'num_samples', num_samples, 'power_iter', power_iter);
    probe_state = rng;
    [tau_newton, info_newton] = solve_opt_param( ...
        S, y, s, sz, eps_bound, eta_bound, opts);

    rng(probe_state); % Use the same power probes as Newton.
    opts.method = 'sextic';
    opts.eigenvalue_method = 'power';
    [tau_sextic_power, info_sextic_power] = solve_opt_param( ...
        S, y, s, sz, eps_bound, eta_bound, opts);

    opts.eigenvalue_method = 'svd';
    [tau_sextic_svd, info_sextic_svd] = solve_opt_param( ...
        S, y, s, sz, eps_bound, eta_bound, opts);

    opts.method = 'BE';
    [tau_sdp, info_sdp] = solve_opt_param( ...
        S, y, s, sz, eps_bound, eta_bound, opts);

    results{k, :} = [d, n, s, N, M, num_samples, power_iter, ...
        tau_newton, tau_sextic_power, tau_sextic_svd, tau_sdp, ...
        info_newton.lwce_sq, info_sextic_power.lwce_sq, ...
        info_sextic_svd.lwce_sq, info_sdp.wce_sq];

    disp(results(k, :));
end

disp(results)
