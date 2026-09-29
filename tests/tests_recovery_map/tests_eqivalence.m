%TESTS_EQUIVALENCE Compare regularized_solution with full-tensor CG.

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
max_iter = 1e5;
tol_rel = 1e-12;

num_tests = size(configs, 1);
results = zeros(num_tests, 8);

for k = 1:num_tests
    d = configs(k, 1);
    n = configs(k, 2);
    s = configs(k, 3);
    M = configs(k, 4);

    sz = n * ones(1, d);
    N = prod(sz);

    if M > N
        error('Number of samples M=%d exceeds ambient dimension N=%d.', M, N);
    end

    P_V = @(x) cluster_expansion(x, s, sz);
    P_V_perp = @(x) x - P_V(x);

    idx = randperm(N, M);
    S = sparse(1:M, idx, 1, M, N);

    x_model = P_V(randn(N, 1));
    z = P_V_perp(randn(N, 1));
    z = eps_model * z / max(norm(z), 1e-14);
    x_true = x_model + z;

    noise = randn(M, 1);
    noise = eta_noise * noise / max(norm(noise), 1e-14);
    y = S * x_true + noise;

    regularized_timer = tic;
    [f_regularized, info] = regularized_solution( ...
        S, y, s, tau, sz, false, 1000, 1e-10);
    regularized_time = toc(regularized_timer);

    cg_timer = tic;
    f_cg = optimal_recovery_CG(S, y, s, tau, sz, ...
        zeros(N, 1), max_iter, tol_rel, false);
    cg_time = toc(cg_timer);

    obj_regularized = objective_value(f_regularized, S, y, P_V_perp, tau);
    obj_cg = objective_value(f_cg, S, y, P_V_perp, tau);
    rel_discrepancy = norm(f_regularized - f_cg) / max(norm(f_cg), 1e-14);
    objective_gap = abs(obj_regularized - obj_cg) / max(abs(obj_cg), 1e-14);

    assert(info.converged, 'Regularized coefficient CG did not converge.');
    assert(rel_discrepancy < 1e-6, ...
        'Regularized recovery and full-tensor CG disagree.');
    assert(objective_gap < 1e-8, ...
        'Regularized recovery and full-tensor CG objectives disagree.');

    results(k, :) = [N, M, info.r, regularized_time, cg_time, ...
        info.iterations, rel_discrepancy, objective_gap];
end

results = array2table(results, 'VariableNames', ...
    {'N', 'M', 'model_dim', 'regularized_time', 'cg_time', ...
    'regularized_iterations', 'relative_difference', 'objective_gap'});
disp(results);

%% Local helper
function val = objective_value(x, S, y, P_V_perp, tau)
    val = tau * norm(S * x - y)^2 + ...
        (1 - tau) * norm(P_V_perp(x))^2;
end
