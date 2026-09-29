%TEST_ZERO_INTERSECTION_MIN_SAMPLES
% Estimate the empirical sample threshold for V_s cap ker(Lambda) = {0}.
%
% For each dimension d, this script estimates the smallest M for which
% Lambda * U has full column rank with high empirical probability, then
% compares it with the theoretical scale r log(r / eps).

clear;
clc;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'core'));
set_path();

rng(1);

%% Experiment setup
n = 2;
s = 2;
d_list = 8:14;
eps_theory = 0.01;
success_target = 0.995;
num_trials = 200;
rank_tol = 1e-10;

num_dims = numel(d_list);
results = table( ...
    zeros(num_dims, 1), ...
    n * ones(num_dims, 1), ...
    s * ones(num_dims, 1), ...
    zeros(num_dims, 1), ...
    zeros(num_dims, 1), ...
    zeros(num_dims, 1), ...
    zeros(num_dims, 1), ...
    zeros(num_dims, 1), ...
    zeros(num_dims, 1), ...
    'VariableNames', {'d', 'n', 's', 'N', 'r', ...
        'M_empirical', 'success_rate', 'r_log_r_over_eps', ...
        'theory_over_empirical'} ...
);

fprintf('\n================ Zero-intersection minimal sample test ================\n');
fprintf('n = %d, s = %d, trials = %d, target success = %.2f\n', ...
    n, s, num_trials, success_target);

for row = 1:num_dims
    d = d_list(row);
    sz = n * ones(1, d);
    N = prod(sz);

    active = build_cluster_mask_vector(sz, s);
    r = nnz(active);
    U = build_fourier_basis(active, sz);

    theory_M = ceil(r * log(r / eps_theory));

    % Search from the information-theoretic lower bound M >= r up to the
    % theoretical scale.  Use a small step to keep the run time reasonable.
    step = max(1, ceil(0.05 * r));
    M_grid = unique([r:step:min(N, theory_M), min(N, theory_M)]);

    M_empirical = NaN;
    empirical_success = NaN;

    fprintf('\nd = %d, N = %d, r = %d, theory M = %d\n', d, N, r, theory_M);

    for M = M_grid
        success_rate = estimate_success_rate(U, N, r, M, num_trials, rank_tol);
        fprintf('  M = %4d, M/r = %.2f, success = %.2f\n', ...
            M, M / r, success_rate);

        if success_rate >= success_target
            M_empirical = M;
            empirical_success = success_rate;
            break;
        end
    end

    results.d(row) = d;
    results.N(row) = N;
    results.r(row) = r;
    results.M_empirical(row) = M_empirical;
    results.success_rate(row) = empirical_success;
    results.r_log_r_over_eps(row) = theory_M;
    results.theory_over_empirical(row) = theory_M / M_empirical;
end

disp(results);

%% Plot
figure('Visible', 'off');
plot(results.d, results.M_empirical, '-o', 'LineWidth', 1.5, ...
    'DisplayName', 'Empirical minimal M');
hold on;
plot(results.d, results.r_log_r_over_eps, '-s', 'LineWidth', 1.5, ...
    'DisplayName', 'r log(r / epsilon)');
hold off;
grid on;
xlabel('Dimension d');
ylabel('Number of observations');
title(sprintf('Zero-intersection sample scaling, n=%d, s=%d', n, s));
legend('Location', 'northwest');

plot_path = fullfile(fileparts(mfilename('fullpath')), ...
    'zero_intersection_min_samples.png');
saveas(gcf, plot_path);
fprintf('\nSaved plot to %s\n', plot_path);

%% Local helpers
function success_rate = estimate_success_rate(U, N, r, M, num_trials, rank_tol)
    if M < r
        success_rate = 0;
        return;
    end

    successes = false(num_trials, 1);
    for trial = 1:num_trials
        idx = randi(N, M, 1);
        C = U(idx, :);
        singular_values = svd(C, 'econ');
        successes(trial) = singular_values(r) > rank_tol;
    end
    success_rate = mean(successes);
end

function active = build_cluster_mask_vector(sz, s)
    d = numel(sz);
    order = zeros(sz);

    grids = cell(1, d);
    for j = 1:d
        grids{j} = 1:sz(j);
    end

    [subscripts{1:d}] = ndgrid(grids{:}); 
    for j = 1:d
        order = order + (subscripts{j} > 1);
    end

    active = order(:) <= s;
end

function U = build_fourier_basis(active, sz)
    N = prod(sz);
    active_idx = find(active);
    r = numel(active_idx);
    U = zeros(N, r);

    for j = 1:r
        coeff = zeros(N, 1);
        coeff(active_idx(j)) = 1;
        basis_tensor = sqrt(N) * ifftn(reshape(coeff, sz));
        U(:, j) = basis_tensor(:);
    end
end
