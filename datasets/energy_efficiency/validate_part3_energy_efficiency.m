%VALIDATE_PART3_ENERGY_EFFICIENCY Compare power and SVD in Part 3.
% Reproduces the Energy Efficiency setup in reproduce_OR_ANOVA.mlx.

clear; clc;
this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(this_dir));
addpath(fullfile(project_root, 'core'));
set_path();
data_file = fullfile(this_dir, 'data_set', 'ENB2012_data.xlsx');
result_file = fullfile(this_dir, 'part3_energy_efficiency_validation_result.mat');
csv_file = fullfile(this_dir, 'part3_energy_efficiency_validation_summary.csv');

%% Part 3 tensor and training split
seed = 1;
rng(seed);
T = readtable(data_file, 'VariableNamingRule', 'preserve');
geometry = T{:, {'X1', 'X2', 'X3', 'X4', 'X5'}};
assert(size(unique(geometry, 'rows'), 1) == numel(unique(T.X1)), ...
    'Part 3 requires one geometry per X1 value.');
response_names = {'Y1', 'Y2'};
Y = T{:, response_names};
[x1_levels, ~, x1_index] = unique(T.X1, 'stable');
[x6_levels, ~, x6_index] = unique(T.X6, 'stable');
[glazing_levels, ~, glazing_index] = unique( ...
    T{:, {'X7', 'X8'}}, 'rows', 'stable');
sz = [numel(x1_levels), numel(x6_levels), size(glazing_levels, 1)];
N = prod(sz);
cell_index = sub2ind(sz, x1_index, x6_index, glazing_index);
cell_count = accumarray(cell_index, 1, [N, 1]);
F = zeros(numel(response_names), N);
for j = 1:numel(response_names)
    F(j, :) = accumarray(cell_index, Y(:, j), [N, 1], @mean, 0).';
end
active_cells = find(cell_count > 0);
rng(seed + 1);
cells = active_cells(randperm(numel(active_cells)));
num_train = round(0.7 * numel(cells));
train_idx = cells(1:num_train);
test_idx = cells(num_train + 1:end);
S = sparse(1:num_train, train_idx, 1, num_train, N);
assert(isequal(sz, [12, 4, 16]) && numel(active_cells) == 768 && ...
    num_train == 538 && numel(test_idx) == 230, ...
    'The dataset or split differs from Part 3.');

%% Part 3 global recovery and assumed bounds
cases = table(["Y1"; "Y1"; "Y2"; "Y2"], [1; 2; 1; 2], ...
    [0.05; 0.02; 0.08; 0.06], 0.03 * ones(4, 1), ...
    'VariableNames', {'target', 's', 'mismatch_rel', 'noise_rel'});
n_cases = height(cases);
global_norm = zeros(n_cases, 1);
global_test_error = zeros(n_cases, 1);
epsilon_bound = zeros(n_cases, 1);
eta_bound = zeros(n_cases, 1);
for k = 1:n_cases
    response_index = find(strcmp(response_names, cases.target(k)), 1);
    y = F(response_index, train_idx).';
    truth = F(response_index, test_idx).';
    recovered = regularized_solution(S, y, cases.s(k), 0.5, sz);
    global_norm(k) = norm(recovered);
    global_test_error(k) = norm(recovered(test_idx) - truth) / ...
        max(norm(truth), eps);
    epsilon_bound(k) = cases.mismatch_rel(k) * global_norm(k);
    eta_bound(k) = cases.noise_rel(k) * norm(y);
end

%% GWCE: Part 3's zero-data local problem
power_iter = [4000, 10000];
power_options = struct('method', 'sextic', ...
    'eigenvalue_method', 'power', 'num_samples', 100);
svd_options = struct('method', 'sextic', 'eigenvalue_method', 'svd');
gwce_power = zeros(n_cases, 1);
gwce_svd = zeros(n_cases, 1);
gwce_power_seconds = zeros(n_cases, 1);
gwce_svd_seconds = zeros(n_cases, 1);
zero_power_info = cell(n_cases, 1);
zero_svd_info = cell(n_cases, 1);
% Keep the Part 3 random stream and power-call order.
rng(seed + 4);
for k = 1:n_cases
    opts = power_options;
    opts.power_iter = power_iter(cases.s(k));
    timer = tic;
    [gwce_power(k), zero_power_info{k}] = compute_gwce( ...
        S, cases.s(k), sz, epsilon_bound(k), eta_bound(k), opts);
    gwce_power_seconds(k) = toc(timer);
    timer = tic;
    [gwce_svd(k), zero_svd_info{k}] = compute_gwce( ...
        S, cases.s(k), sz, epsilon_bound(k), eta_bound(k), svd_options);
    gwce_svd_seconds(k) = toc(timer);
end

%% Local parameter, LWCE, and recovery
tau_power = zeros(n_cases, 1);
tau_svd = zeros(n_cases, 1);
lwce_power = zeros(n_cases, 1);
lwce_svd = zeros(n_cases, 1);
local_power_seconds = zeros(n_cases, 1);
local_svd_seconds = zeros(n_cases, 1);
local_power_test_error = zeros(n_cases, 1);
local_svd_test_error = zeros(n_cases, 1);
local_power_info = cell(n_cases, 1);
local_svd_info = cell(n_cases, 1);
for k = 1:n_cases
    response_index = find(strcmp(response_names, cases.target(k)), 1);
    y = F(response_index, train_idx).';
    truth = F(response_index, test_idx).';
    opts = power_options;
    opts.power_iter = power_iter(cases.s(k));
    timer = tic;
    [tau_power(k), local_power_info{k}] = solve_opt_param( ...
        S, y, cases.s(k), sz, epsilon_bound(k), eta_bound(k), opts);
    local_power_seconds(k) = toc(timer);
    timer = tic;
    [tau_svd(k), local_svd_info{k}] = solve_opt_param( ...
        S, y, cases.s(k), sz, epsilon_bound(k), eta_bound(k), svd_options);
    local_svd_seconds(k) = toc(timer);
    lwce_power(k) = sqrt(local_power_info{k}.lwce_sq);
    lwce_svd(k) = sqrt(local_svd_info{k}.lwce_sq);
    recovered_power = regularized_solution(S, y, cases.s(k), tau_power(k), sz);
    recovered_svd = regularized_solution(S, y, cases.s(k), tau_svd(k), sz);
    local_power_test_error(k) = norm(recovered_power(test_idx) - truth) / ...
        max(norm(truth), eps);
    local_svd_test_error(k) = norm(recovered_svd(test_idx) - truth) / ...
        max(norm(truth), eps);
end

%% Independent checks of the SVD sextic results
% Build the same symmetric operator as solve_sextic, once per ANOVA order.
P_perp = cell(2, 1);
for s = 1:2
    P_perp{s} = zeros(N, N);
    for j = 1:N
        e = zeros(N, 1);
        e(j) = 1;
        P_perp{s}(:, j) = e - cluster_expansion(e, s, sz);
    end
end
sample_projector = full(S' * S);
lambda_at_local_tau = zeros(n_cases, 1);
lambda_at_zero_tau = zeros(n_cases, 1);
lwce_direct = zeros(n_cases, 1);
gwce_direct = zeros(n_cases, 1);
tau_numeric = zeros(n_cases, 1);
gwce_tau_numeric = zeros(n_cases, 1);
lwce_numeric_direct = zeros(n_cases, 1);
gwce_numeric_direct = zeros(n_cases, 1);

for k = 1:n_cases
    s = cases.s(k);
    epsilon = epsilon_bound(k);
    eta = eta_bound(k);
    info = local_svd_info{k};
    zero_info = zero_svd_info{k};
    interval = sort([0.5, epsilon / (epsilon + eta)]);
    options = optimset('TolX', 1e-10);

    % Direct SVD at the reported sextic solutions checks lambda(tau).
    lambda_at_local_tau(k) = minimum_singular_value( ...
        P_perp{s}, sample_projector, tau_svd(k));
    lwce_direct(k) = sqrt(wce_sq(tau_svd(k), ...
        lambda_at_local_tau(k), epsilon, eta, info.delta));
    lambda_at_zero_tau(k) = minimum_singular_value( ...
        P_perp{s}, sample_projector, zero_info.tau_zero);
    gwce_direct(k) = sqrt(wce_sq(zero_info.tau_zero, ...
        lambda_at_zero_tau(k), epsilon, eta, 0));

    % One-dimensional minimization checks the sextic root independently.
    local_objective = @(t) wce_sq(t, lambda_from_c(t, info.C_est), ...
        epsilon, eta, info.delta);
    tau_numeric(k) = fminbnd(local_objective, ...
        interval(1), interval(2), options);
    lambda_numeric = minimum_singular_value( ...
        P_perp{s}, sample_projector, tau_numeric(k));
    lwce_numeric_direct(k) = sqrt(wce_sq(tau_numeric(k), ...
        lambda_numeric, epsilon, eta, info.delta));

    zero_objective = @(t) wce_sq(t, ...
        lambda_from_c(t, zero_info.C_est), epsilon, eta, 0);
    gwce_tau_numeric(k) = fminbnd(zero_objective, ...
        interval(1), interval(2), options);
    lambda_numeric_zero = minimum_singular_value( ...
        P_perp{s}, sample_projector, gwce_tau_numeric(k));
    gwce_numeric_direct(k) = sqrt(wce_sq(gwce_tau_numeric(k), ...
        lambda_numeric_zero, epsilon, eta, 0));
end

%% Comparison tables and saved results
local_validation = table(cases.target, cases.s, epsilon_bound, eta_bound, ...
    cellfun(@(x) x.lambda_test, local_power_info), ...
    cellfun(@(x) x.lambda_test, local_svd_info), ...
    tau_power, tau_svd, tau_numeric, lwce_power, lwce_svd, ...
    lwce_direct, lwce_numeric_direct, local_power_test_error, ...
    local_svd_test_error, local_power_seconds, local_svd_seconds, ...
    'VariableNames', {'target', 's', 'epsilon_bound', 'eta_bound', ...
    'lambda_power', 'lambda_svd', 'tau_power', 'tau_svd', ...
    'tau_numeric', 'lwce_power', 'lwce_svd', 'lwce_direct', ...
    'lwce_numeric_direct', 'test_error_power', 'test_error_svd', ...
    'seconds_power', 'seconds_svd'});
gwce_validation = table(cases.target, cases.s, ...
    cellfun(@(x) x.lambda_test, zero_power_info), ...
    cellfun(@(x) x.lambda_test, zero_svd_info), ...
    cellfun(@(x) x.tau_zero, zero_power_info), ...
    cellfun(@(x) x.tau_zero, zero_svd_info), gwce_tau_numeric, ...
    gwce_power, gwce_svd, gwce_direct, gwce_numeric_direct, ...
    gwce_power_seconds, gwce_svd_seconds, ...
    'VariableNames', {'target', 's', 'lambda_power', 'lambda_svd', ...
    'tau_power', 'tau_svd', 'tau_numeric', 'gwce_power', 'gwce_svd', ...
    'gwce_direct', 'gwce_numeric_direct', 'seconds_power', 'seconds_svd'});

local_validation.tau_power_minus_svd = tau_power - tau_svd;
local_validation.lwce_power_minus_svd = lwce_power - lwce_svd;
gwce_validation.gwce_power_minus_svd = gwce_power - gwce_svd;
local_validation.svd_checks_pass = ...
    abs(tau_svd - tau_numeric) <= 1e-5 & ...
    abs(lwce_svd - lwce_direct) <= 1e-5 .* max(1, lwce_direct) & ...
    abs(lwce_svd - lwce_numeric_direct) <= 1e-5 .* ...
        max(1, lwce_numeric_direct);
gwce_validation.svd_checks_pass = ...
    abs(gwce_validation.tau_svd - gwce_tau_numeric) <= 1e-5 & ...
    abs(gwce_svd - gwce_direct) <= 1e-5 .* max(1, gwce_direct) & ...
    abs(gwce_svd - gwce_numeric_direct) <= 1e-5 .* ...
        max(1, gwce_numeric_direct);

disp('Part 3 local parameter and LWCE validation:');
disp(local_validation(:, {'target', 's', 'tau_power', 'tau_svd', ...
    'lwce_power', 'lwce_svd', 'svd_checks_pass'}));
disp('Part 3 GWCE validation:');
disp(gwce_validation(:, {'target', 's', 'gwce_power', 'gwce_svd', ...
    'svd_checks_pass'}));

save(result_file, 'cases', 'sz', 'train_idx', 'test_idx', ...
    'global_norm', 'global_test_error', 'local_validation', ...
    'gwce_validation', 'local_power_info', 'local_svd_info', ...
    'zero_power_info', 'zero_svd_info', 'seed', 'power_iter', ...
    'lambda_at_local_tau', 'lambda_at_zero_tau');
summary = join(local_validation, ...
    removevars(gwce_validation, {'lambda_power', 'lambda_svd', ...
    'tau_power', 'tau_svd', 'tau_numeric', ...
    'seconds_power', 'seconds_svd', 'svd_checks_pass'}), ...
    'Keys', {'target', 's'});
writetable(summary, csv_file);
fprintf('Saved %s\nSaved %s\n', result_file, csv_file);

function lambda = minimum_singular_value(P_perp, sample_projector, tau)
    A = (1 - tau) * P_perp + tau * sample_projector;
    lambda = min(svd(A));
end

function lambda = lambda_from_c(tau, C)
    lambda = 0.5 * (1 - sqrt(1 - 4 * C * tau * (1 - tau)));
end

function value = wce_sq(tau, lambda, epsilon, eta, delta)
    value = ((1 - tau) * epsilon^2 + tau * eta^2 - ...
        (1 - tau) * tau * delta^2) / lambda;
end
