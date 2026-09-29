% MULTINARY_OXIDES_MODEL_REGRESSION
% Fit multinary oxide responses in the low-cluster Fourier model space.

clear; clc;

%% ====================== User configuration ==============================
this_dir = fileparts(mfilename('fullpath'));
data_mat = fullfile(this_dir, 'multinary_oxides_tensor_dataset.mat');
result_mat = fullfile(this_dir, 'multinary_oxides_model_regression_result.mat');

s_values = [1, 2];
lambda_ratios = [ 0.02, 0.002, 0.0002];
train_ratio = 0.8;
min_count = 1;
seed = 1;
chunk_size = 1000;
max_iterations = 500;
solver_tolerance = 1e-6;
%% =======================================================================

rng(seed);

fprintf('[1/4] Loading %s ...\n', data_mat);
D = load(data_mat);

F = D.F;
pattern_count = D.pattern_count;
active_patterns = D.active_patterns;
response_names = D.response_names;
sz = D.domain_tensor_size;
N = prod(sz);

fprintf('[2/4] Splitting active tensor cells ...\n');
eligible_patterns = active_patterns(pattern_count(active_patterns) >= min_count);
eligible_patterns = eligible_patterns(:).';

perm = eligible_patterns(randperm(numel(eligible_patterns)));
num_train = round(train_ratio * numel(eligible_patterns));
train_patterns = perm(1:num_train);
test_patterns = perm(num_train + 1:end);

fprintf('    Tensor size: [%s] (%d ambient cells)\n', num2str(sz), N);
fprintf('    Eligible cells: %d\n', numel(eligible_patterns));
fprintf('    Train cells: %d\n', numel(train_patterns));
fprintf('    Test cells: %d\n', numel(test_patterns));

fprintf('[3/4] Running low-cluster Fourier model regression ...\n');
rows = {};
best = struct('rmse', inf, 'f_hat', []);
num_runs = numel(response_names) * numel(s_values) * numel(lambda_ratios);
run_id = 0;

for i = 1:numel(s_values)
    s = s_values(i);

    fprintf('    Building Fourier design for s = %d ...\n', s);
    model = build_low_cluster_fourier_model(train_patterns, sz, s, chunk_size);
    fprintf('    dim(M_s): %d\n', model.r);

    for r = 1:numel(response_names)
        response_name = response_names{r};
        y_train = F(r, train_patterns).';
        truth = F(r, test_patterns).';
        [coefficient_path, path_info] = solve_model_lasso_path( ...
            model, train_patterns, y_train, lambda_ratios, chunk_size, ...
            max_iterations, solver_tolerance);

        for j = 1:numel(lambda_ratios)
            run_id = run_id + 1;
            coefficients = coefficient_path(:, j);
            f_hat = apply_U_fourier(coefficients, model.active, sz);

            f_hat = real(f_hat(:)).';
            pred = f_hat(test_patterns).';

            rel_err = norm(pred - truth) / max(1e-12, norm(truth));
            rmse = sqrt(mean((pred - truth).^2));
            mae = mean(abs(pred - truth));

            fprintf(['        Run %d/%d: response = %s, s = %d, ' ...
                'lambda/lambda_max = %.3g | nnz = %d/%d | ' ...
                'rel_err = %.4e, rmse = %.4e, mae = %.4e\n'], ...
                run_id, num_runs, response_name, s, lambda_ratios(j), ...
                path_info.nnz(j), model.r, rel_err, rmse, mae);

            rows(end + 1, :) = {response_name, s, lambda_ratios(j), ...
                path_info.lambda(j), path_info.time_sec(j), ...
                path_info.iterations(j), path_info.nnz(j), ...
                rel_err, rmse, mae, model.r}; %#ok<SAGROW>

            if rmse < best.rmse
                best.rmse = rmse;
                best.f_hat = f_hat;
                best.response = response_name;
                best.s = s;
                best.lambda_ratio = lambda_ratios(j);
                best.lambda = path_info.lambda(j);
                best.coefficients = coefficients;
                best.model = rmfield(model, 'normal_matrix');
            end
        end
    end
end

results = cell2table(rows, 'VariableNames', ...
    {'response', 's', 'lambda_ratio', 'lambda', 'time_sec', ...
    'iterations', 'num_nonzero', 'test_rel_err', ...
    'test_rmse', 'test_mae', 'model_dim'});
results = sortrows(results, {'response', 'test_rel_err'});

fprintf('[4/4] Results on held-out tensor cells ...\n');
disp(results);

save(result_mat, ...
    'results', 'best', 'train_patterns', 'test_patterns', ...
    'response_names', 's_values', 'lambda_ratios', 'train_ratio', ...
    'min_count', 'seed', 'max_iterations', 'solver_tolerance', ...
    '-v7');

fprintf('Saved %s\n', result_mat);

function model = build_low_cluster_fourier_model(patterns, sz, s, chunk_size)
%BUILD_LOW_CLUSTER_FOURIER_MODEL Build normal equations for S U_s b ~= y.

    % M_s is the low-cluster expansion space: retain Fourier modes with
    % at most s non-DC coordinates.
    mask = build_low_cluster_support(sz, s);
    active = mask(:);
    freq_patterns = find(active);
    freq_subs = linear_to_subscripts(freq_patterns, sz);
    r = low_cluster_model_dimension(sz, s);
    assert(nnz(active) == r, ...
        'Low-cluster support does not match dim(M_s).');

    normal_matrix = zeros(r, r);

    for first = 1:chunk_size:numel(patterns)
        last = min(first + chunk_size - 1, numel(patterns));
        rows = first:last;

        C = fourier_design_rows(patterns(rows), freq_subs, sz);
        normal_matrix = normal_matrix + C' * C;
    end

    model = struct();
    model.sz = sz;
    model.s = s;
    model.r = r;
    model.active = active;
    model.freq_subs = freq_subs;
    model.normal_matrix = normal_matrix;
end

function [coefficient_path, info] = solve_model_lasso_path( ...
        model, patterns, y, lambda_ratios, chunk_size, max_iterations, tolerance)
%SOLVE_MODEL_LASSO_PATH Solve
%   min_b (1/M) ||S U_s b - y||_2^2 + lambda ||b_nonDC||_1.
% Lambda is specified as a fraction of lambda_max, the smallest penalty
% that leaves only the unpenalized DC coefficient active.

    y = y(:);
    M = numel(y);
    rhs = zeros(model.r, 1);

    for first = 1:chunk_size:M
        last = min(first + chunk_size - 1, M);
        rows = first:last;

        C = fourier_design_rows(patterns(rows), model.freq_subs, model.sz);
        rhs = rhs + C' * y(rows);
    end

    G = model.normal_matrix / M;
    c = rhs / M;
    penalized = true(model.r, 1);
    penalized(1) = false;

    coefficients = zeros(model.r, 1);
    coefficients(1) = c(1) / G(1, 1);
    intercept_gradient = 2 * (G * coefficients - c);
    lambda_max = max(abs(intercept_gradient(penalized)));
    lambda_values = lambda_ratios(:).' * lambda_max;

    lipschitz = 2 * estimate_largest_eigenvalue(G, 50);
    lipschitz = max(lipschitz, eps);

    coefficient_path = zeros(model.r, numel(lambda_values));
    info.lambda = lambda_values;
    info.lambda_max = lambda_max;
    info.iterations = zeros(size(lambda_values));
    info.nnz = zeros(size(lambda_values));
    info.time_sec = zeros(size(lambda_values));

    for j = 1:numel(lambda_values)
        lambda = lambda_values(j);
        extrapolated = coefficients;
        momentum = 1;
        timer = tic;

        for iteration = 1:max_iterations
            gradient = 2 * (G * extrapolated - c);
            smooth_at_extrapolated = smooth_objective(G, c, extrapolated);

            while true
                candidate = complex_soft_threshold( ...
                    extrapolated - gradient / lipschitz, ...
                    lambda / lipschitz, penalized);
                step = candidate - extrapolated;
                quadratic_bound = smooth_at_extrapolated + ...
                    real(gradient' * step) + ...
                    0.5 * lipschitz * norm(step)^2;

                if smooth_objective(G, c, candidate) <= ...
                        quadratic_bound + 1e-12
                    break;
                end
                lipschitz = 2 * lipschitz;
            end

            if norm(candidate - coefficients) <= ...
                    tolerance * max(1, norm(coefficients))
                coefficients = candidate;
                break;
            end

            next_momentum = (1 + sqrt(1 + 4 * momentum^2)) / 2;
            extrapolated = candidate + ...
                ((momentum - 1) / next_momentum) * ...
                (candidate - coefficients);
            coefficients = candidate;
            momentum = next_momentum;
        end

        coefficient_path(:, j) = coefficients;
        info.iterations(j) = iteration;
        info.time_sec(j) = toc(timer);
        cutoff = 1e-10 * max(1, max(abs(coefficients)));
        info.nnz(j) = nnz(abs(coefficients) > cutoff);
    end
end

function value = smooth_objective(G, c, coefficients)
%SMOOTH_OBJECTIVE Quadratic data-fit term without its constant y''*y/M.

    value = real(coefficients' * (G * coefficients)) - ...
        2 * real(coefficients' * c);
end

function x = complex_soft_threshold(z, threshold, penalized)
%COMPLEX_SOFT_THRESHOLD Proximal map for sum(abs(z)).

    x = z;
    magnitude = abs(z(penalized));
    shrinkage = max(0, 1 - threshold ./ max(magnitude, eps));
    x(penalized) = z(penalized) .* shrinkage;
end

function eigenvalue = estimate_largest_eigenvalue(A, iterations)
%ESTIMATE_LARGEST_EIGENVALUE Deterministic power iteration for Hermitian A.

    x = ones(size(A, 1), 1) / sqrt(size(A, 1));

    for k = 1:iterations
        Ax = A * x;
        Ax_norm = norm(Ax);
        if Ax_norm == 0
            eigenvalue = 0;
            return;
        end
        x = Ax / Ax_norm;
    end

    eigenvalue = max(0, real(x' * (A * x)));
end
function C = fourier_design_rows(patterns, freq_subs, sz)
%FOURIER_DESIGN_ROWS Build rows of the orthonormal Fourier basis U_s.

    sample_subs = linear_to_subscripts(patterns, sz);
    phase = zeros(numel(patterns), size(freq_subs, 1));

    for j = 1:numel(sz)
        phase = phase + ...
            (sample_subs(:, j) - 1) * ((freq_subs(:, j) - 1).' / sz(j));
    end

    C = exp(2i * pi * phase) / sqrt(prod(sz));
end

function x = apply_U_fourier(coefficients, active, sz)
    N = prod(sz);
    coeff_full = zeros(N, 1);
    coeff_full(active) = coefficients;
    Xhat = reshape(coeff_full, sz);
    X = sqrt(N) * ifftn(Xhat);
    x = X(:);
end

function subs = linear_to_subscripts(patterns, sz)
    d = numel(sz);
    cells = cell(1, d);
    [cells{:}] = ind2sub(sz, patterns(:));
    subs = zeros(numel(patterns), d);

    for j = 1:d
        subs(:, j) = cells{j};
    end
end

function mask = build_low_cluster_support(sz, s)
%BUILD_LOW_CLUSTER_SUPPORT Fourier support of the expansion M_s.

    d = numel(sz);
    cluster_order = zeros(sz);

    for j = 1:d
        shape = ones(1, d);
        shape(j) = sz(j);
        non_dc = reshape([0, ones(1, sz(j) - 1)], shape);
        cluster_order = cluster_order + non_dc;
    end

    mask = cluster_order <= min(s, d);
end

function dimension = low_cluster_model_dimension(sz, s)
%LOW_CLUSTER_MODEL_DIMENSION Compute dim(M_s) from the cluster expansion.
% dim(M_s) = sum_{|u| <= s} prod_{j in u} (n_j - 1).

    d = numel(sz);
    dimension = 1;

    for order = 1:min(s, d)
        subsets = nchoosek(1:d, order);
        subset_sizes = reshape(sz(subsets), size(subsets));
        dimension = dimension + sum(prod(subset_sizes - 1, 2));
    end
end