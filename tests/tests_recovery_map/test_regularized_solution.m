%TEST_REGULARIZED_SOLUTION  Accuracy of the explicit recovery map.
%
% This script tests the FFT implementation of the explicit regularized
% recovery formula by measuring recovery accuracy on the unobserved entries:
%
%     ||f_tau|_{O^c} - f|_{O^c}||_F / ||f|_{O^c}||_F.
%
% The experiment uses the global setting, so any tau in (0, 1) gives the
% optimal recovery map. We fix tau = 0.2, model mismatch eps = 0.05, and
% measurement noise eta = 0.02.

clear;
clc;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'core'));
set_path();

rng(1);

%% Experiment setup
configs = [
    7,  2, 2,  80;
    8,  2, 2, 120;
    10, 2, 2, 200;
    6,  3, 2, 200;
    7,  3, 2, 250;
];

tau = 0.2;
eps_model = 0.05;
eta_noise = 0.02;
verbose = false;

num_tests = size(configs, 1);
results = table( ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    tau * ones(num_tests, 1), ...
    eps_model * ones(num_tests, 1), ...
    eta_noise * ones(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    'VariableNames', {'d', 'n', 's', 'N', 'M', 'tau', ...
        'eps_model', 'eta_noise', 'model_dim', 'unobserved_count', ...
        'solve_time', 'rel_unobserved_error', 'rel_observed_residual'} ...
);

fprintf('\n================ Recovery accuracy on unobserved entries ================\n');
fprintf('tau = %.3f, model mismatch = %.3e, noise level = %.3e\n', ...
    tau, eps_model, eta_noise);

for k = 1:num_tests
    d = configs(k, 1);
    n = configs(k, 2);
    s = configs(k, 3);
    M = configs(k, 4);

    sz = n * ones(1, d);
    N = prod(sz);

    if M >= N
        error('Number of samples M=%d must be strictly smaller than N=%d.', M, N);
    end

    P_V = @(x) cluster_expansion(x, s, sz);
    P_V_perp = @(x) x - P_V(x);

    obs_idx = randperm(N, M);
    unobs_idx = setdiff(1:N, obs_idx);
    S = sparse(1:M, obs_idx, 1, M, N);

    x_model = P_V(randn(N, 1));
    z = P_V_perp(randn(N, 1));
    z = eps_model * z / max(norm(z), 1e-14);
    x_true = x_model + z;

    noise = randn(M, 1);
    noise = eta_noise * noise / max(norm(noise), 1e-14);
    y = S * x_true + noise;

    fprintf('\nTest %d/%d: d=%d, n=%d, s=%d, N=%d, M=%d\n', ...
        k, num_tests, d, n, s, N, M);

    solve_timer = tic;
    [f_tau, info] = regularized_solution(S, y, s, tau, sz, verbose);
    solve_time = toc(solve_timer);

    unobserved_error = norm(f_tau(unobs_idx) - x_true(unobs_idx));
    unobserved_norm = norm(x_true(unobs_idx));
    rel_unobserved_error = unobserved_error / max(unobserved_norm, 1e-14);

    observed_residual = norm(S * f_tau - y);
    rel_observed_residual = observed_residual / max(norm(y), 1e-14);

    results.d(k) = d;
    results.n(k) = n;
    results.s(k) = s;
    results.N(k) = N;
    results.M(k) = M;
    results.model_dim(k) = info.r;
    results.unobserved_count(k) = numel(unobs_idx);
    results.solve_time(k) = solve_time;
    results.rel_unobserved_error(k) = rel_unobserved_error;
    results.rel_observed_residual(k) = rel_observed_residual;

    fprintf('Model dimension r:              %d\n', info.r);
    fprintf('Unobserved entries |O^c|:       %d\n', numel(unobs_idx));
    fprintf('Solve time:                     %.4f s\n', solve_time);
    fprintf('Relative unobserved error:      %.4e\n', rel_unobserved_error);
    fprintf('Relative observed residual:     %.4e\n', rel_observed_residual);
end

disp(results);
