%TEST_LOCAL_POWER_METHOD_EXPERIMENT
% Standalone numerical check for the randomized lambda_min estimator.

clear;
clc;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'core'));
set_path();

rng(1);

configs = [
    40, 0.01, 1.00, 1e-2, 1e-2;
    80, 0.01, 1.00, 1e-2, 1e-2;
    120, 0.01, 1.00, 1e-2, 1e-2;
    160, 0.01, 1.00, 1e-2, 1e-2;
];

num_tests = size(configs, 1);
results = table( ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    zeros(num_tests, 1), ...
    'VariableNames', {'m', 'lambda_min_true', 'lambda_max', ...
        'target_rel_error', 'failure_prob', 'num_samples', ...
        'iterations', 'lambda_min_est', 'rel_error'} ...
);

fprintf('\n================ Local setting: power-method test ================\n');

for k = 1:num_tests
    m = configs(k, 1);
    lambda_min = configs(k, 2);
    lambda_max = configs(k, 3);
    target_rel_error = configs(k, 4);
    failure_prob = configs(k, 5);

    [Q, ~] = qr(randn(m));
    lambda = linspace(lambda_min, lambda_max, m)';
    A = Q * diag(lambda) * Q';
    A_func = @(x) A * x;

    [num_samples, iterations] = power_estimator_parameters( ...
        m, failure_prob, target_rel_error, lambda_min);

    timer = tic;
    lambda_min_est = estimate_lambda_power(A_func, m, num_samples, iterations);
    elapsed = toc(timer);

    rel_error = abs(lambda_min_est - lambda_min) / lambda_min;

    results.m(k) = m;
    results.lambda_min_true(k) = lambda_min;
    results.lambda_max(k) = lambda_max;
    results.target_rel_error(k) = target_rel_error;
    results.failure_prob(k) = failure_prob;
    results.num_samples(k) = num_samples;
    results.iterations(k) = iterations;
    results.lambda_min_est(k) = lambda_min_est;
    results.rel_error(k) = rel_error;

    fprintf('\nTest %d/%d\n', k, num_tests);
    fprintf('m                  = %d\n', m);
    fprintf('true lambda_min    = %.6e\n', lambda_min);
    fprintf('target rel error   = %.4e\n', target_rel_error);
    fprintf('failure prob delta = %.4e\n', failure_prob);
    fprintf('num samples        = %d\n', num_samples);
    fprintf('iterations         = %d\n', iterations);
    fprintf('estimated lambda_min = %.6e\n', lambda_min_est);
    fprintf('relative error     = %.4e\n', rel_error);
    fprintf('elapsed time       = %.3f s\n', elapsed);
end

disp(results);
