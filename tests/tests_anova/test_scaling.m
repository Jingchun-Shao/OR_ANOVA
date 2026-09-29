%% ------------------------------------------------------------
% Tests on fast ANOVA
% Part 1: accuracy on several tensors
% Part 2: timing/scaling comparison

clear; clc;
rng(1);

%% ============================================================
% Part 1: Accuracy test on several random tensors
% =============================================================

fprintf('\n================ Accuracy test ================\n');

accuracy_configs = [
    6,  4, 2;
    8,  4, 2;
    10, 4, 2;
    8,  5, 3;
    10, 5, 3;
];

num_accuracy_tests = size(accuracy_configs, 1);

accuracy_results = table( ...
    zeros(num_accuracy_tests,1), ...
    zeros(num_accuracy_tests,1), ...
    zeros(num_accuracy_tests,1), ...
    zeros(num_accuracy_tests,1), ...
    zeros(num_accuracy_tests,1), ...
    'VariableNames', {'d','n','s','N','rel_error'} ...
);

for k = 1:num_accuracy_tests
    d = accuracy_configs(k,1);
    n = accuracy_configs(k,2);
    s = accuracy_configs(k,3);

    fprintf('\nAccuracy test %d/%d: d=%d, n=%d, s=%d\n', ...
        k, num_accuracy_tests, d, n, s);

    g = randn(n * ones(1, d));
    N = numel(g);

    % Fast ANOVA projector
    f_fast = cluster_expansion(g, s);

    % Recursive ANOVA reconstruction
    D = anova_recursively(g);
    D.decompose(s);
    f_recursive = D.reconstruct();

    % Relative error
    rel_error = norm(f_fast(:) - f_recursive(:)) / norm(f_recursive(:));

    accuracy_results.d(k) = d;
    accuracy_results.n(k) = n;
    accuracy_results.s(k) = s;
    accuracy_results.N(k) = N;
    accuracy_results.rel_error(k) = rel_error;

    fprintf('Relative error: %.2e\n', rel_error);
end

disp(accuracy_results);
% writetable(accuracy_results, 'fast_anova_accuracy_results.csv');


%% ============================================================
% Part 2: Scaling test
% =============================================================

fprintf('\n================ Scaling test ================\n');

% Keep n and s fixed, increase d.
% Choose sizes where the recursive method is still feasible.
n = 4;
s = 2;
d_list = 4:12;

num_scaling_tests = numel(d_list);

scaling_results = table( ...
    zeros(num_scaling_tests,1), ...
    zeros(num_scaling_tests,1), ...
    zeros(num_scaling_tests,1), ...
    zeros(num_scaling_tests,1), ...
    zeros(num_scaling_tests,1), ...
    zeros(num_scaling_tests,1), ...
    'VariableNames', {'d','n','s','N','t_fast','t_recursive'} ...
);

for k = 1:num_scaling_tests
    d = d_list(k);

    fprintf('\nScaling test %d/%d: d=%d, n=%d, s=%d\n', ...
        k, num_scaling_tests, d, n, s);

    g = randn(n * ones(1, d));
    N = numel(g);

    fast_pipeline = @() cluster_expansion(g, s);
    recursive_pipeline = @() anova_pipeline(g, s);

    t_fast = timeit(fast_pipeline);
    t_recursive = timeit(recursive_pipeline);

    scaling_results.d(k) = d;
    scaling_results.n(k) = n;
    scaling_results.s(k) = s;
    scaling_results.N(k) = N;
    scaling_results.t_fast(k) = t_fast;
    scaling_results.t_recursive(k) = t_recursive;

    fprintf('Fast time:      %.4f s\n', t_fast);
    fprintf('Recursive time: %.4f s\n', t_recursive);
    fprintf('Speedup:        %.2fx\n', t_recursive / t_fast);
end

scaling_results.speedup = ...
    scaling_results.t_recursive ./ scaling_results.t_fast;

disp(scaling_results);
% writetable(scaling_results, 'fast_anova_scaling_results.csv');



%% Plot running time in log scale with respect to dimension d

figure;

semilogy(scaling_results.d, scaling_results.t_fast, '-o', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Fast ANOVA');

hold on;

semilogy(scaling_results.d, scaling_results.t_recursive, '-s', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Recursive ANOVA');

hold off;

xlabel('Dimension d');
ylabel('Running time (seconds, log scale)');
title(sprintf('Scaling of ANOVA computation time, n=%d, s=%d', n, s));
legend('Location', 'northwest');
grid on;

%saveas(gcf, 'fast_anova_time_vs_dimension.png');


%% ============================================================
% Local helper
% =============================================================

function y = anova_pipeline(g, s)
    D = anova_recursively(g);
    D.decompose(s);
    y = D.reconstruct();
end