% ENERGY_EFFICIENCY_REGULARIZED_RECOVERY
% Global and local recovery of heating and cooling loads.

clear; clc;

%% Configuration
this_dir = fileparts(mfilename('fullpath'));
anova_root = fileparts(fileparts(this_dir));
data_file = fullfile(this_dir, 'data_set', 'ENB2012_data.xlsx');
result_file = fullfile(this_dir, 'energy_efficiency_regularized_recovery_result.mat');

response_names = {'Y1', 'Y2'};
global_orders = 1:3;
local_orders = 1:2;
tau_global = 0.5;
train_ratio = 0.8;
seed = 1;

% Illustrative bounds, scaled separately for each response and order.
mismatch_rel = 0.05;
noise_rel = 0.03;
svd_options = struct('method', 'sextic', 'eigenvalue_method', 'svd');
local_options = struct('method', 'sextic', 'eigenvalue_method', 'power', ...
    'num_samples', 100, 'power_iter', 10000);

addpath(fullfile(anova_root, 'core'));
set_path();

%% Import and tensorize the data
energy_table = readtable(data_file, 'VariableNamingRule', 'preserve');
geometry = energy_table{:, {'X1', 'X2', 'X3', 'X4', 'X5'}};
assert(size(unique(geometry, 'rows'), 1) == numel(unique(energy_table.X1)), ...
    'X1 must uniquely identify the geometry.');

% X1, X6, X7, and X8 index separate tensor axes.
[x1_levels, ~, x1_index] = unique(energy_table.X1, 'stable');
[x6_levels, ~, x6_index] = unique(energy_table.X6, 'stable');
[x7_levels, ~, x7_index] = unique(energy_table.X7, 'stable');
[x8_levels, ~, x8_index] = unique(energy_table.X8, 'stable');
sz = [numel(x1_levels), numel(x6_levels), numel(x7_levels), numel(x8_levels)];
cell_index = sub2ind(sz, x1_index, x6_index, x7_index, x8_index);
N = prod(sz);

Y = energy_table{:, response_names};
cell_count = accumarray(cell_index, 1, [N, 1]);
F = zeros(numel(response_names), N);
for r = 1:numel(response_names)
    F(r, :) = accumarray(cell_index, Y(:, r), [N, 1], @mean, 0).';
end
active_cells = find(cell_count > 0);

%% Shared training and test cells
rng(seed);
cells = active_cells(randperm(numel(active_cells)));
num_train = round(train_ratio * numel(cells));
train_idx = cells(1:num_train);
test_idx = cells(num_train + 1:end);
S = sparse(1:num_train, train_idx, 1, num_train, N);

fprintf('Tensor: [%s]; recorded: %d/%d; train: %d; test: %d\n', ...
    num2str(sz), numel(active_cells), N, num_train, numel(test_idx));

%% Global recovery and GWCE
global_rows = cell(numel(response_names) * numel(global_orders), 9);
row = 0;
for r = 1:numel(response_names)
    response = string(response_names{r});
    y = F(r, train_idx).';
    truth = F(r, test_idx).';

    for s = global_orders
        timer = tic;
        [f_hat, recovery_info] = regularized_solution( ...
            S, y, s, tau_global, sz, false);
        recovery_time = toc(timer);

        epsilon_bound = mismatch_rel * norm(f_hat);
        eta_bound = noise_rel * norm(y);
        gwce = compute_gwce(S, s, sz, epsilon_bound, eta_bound, svd_options);
        test_error = norm(f_hat(test_idx) - truth) / norm(truth);

        row = row + 1;
        global_rows(row, :) = {response, s, recovery_info.r, tau_global, ...
            test_error, gwce, epsilon_bound, eta_bound, recovery_time};
    end
end
global_results = cell2table(global_rows, 'VariableNames', ...
    {'response', 's', 'model_dim', 'tau', 'test_relative_L2_error', ...
    'gwce', 'epsilon_bound', 'eta_bound', 'recovery_time'});
disp(global_results(:, {'response', 's', 'test_relative_L2_error', 'gwce'}));

%% Local recovery: sextic parameter with power eigenvalue estimate
local_rows = cell(numel(response_names) * numel(local_orders), 8);
row = 0;
for r = 1:numel(response_names)
    response = string(response_names{r});
    y = F(r, train_idx).';
    truth = F(r, test_idx).';

    for s = local_orders
        global_row = global_results.response == response & global_results.s == s;
        epsilon_bound = global_results.epsilon_bound(global_row);
        eta_bound = global_results.eta_bound(global_row);

        timer = tic;
        [tau_local, parameter_info] = solve_opt_param( ...
            S, y, s, sz, epsilon_bound, eta_bound, local_options);
        parameter_time = toc(timer);

        timer = tic;
        [f_hat, recovery_info] = regularized_solution( ...
            S, y, s, tau_local, sz, false);
        recovery_time = toc(timer);
        test_error = norm(f_hat(test_idx) - truth) / norm(truth);

        row = row + 1;
        local_rows(row, :) = {response, s, recovery_info.r, tau_local, ...
            sqrt(parameter_info.lwce_sq), test_error, ...
            parameter_time, recovery_time};
    end
end
local_results = cell2table(local_rows, 'VariableNames', ...
    {'response', 's', 'model_dim', 'tau', 'lwce', 'test_relative_L2_error', ...
    'parameter_time', 'recovery_time'});
disp(local_results(:, ...
    {'response', 's', 'tau', 'lwce', 'test_relative_L2_error'}));

save(result_file, 'global_results', 'local_results', 'train_idx', 'test_idx', ...
    'sz', 'response_names', 'global_orders', 'local_orders', ...
    'tau_global', 'train_ratio', 'seed', 'mismatch_rel', 'noise_rel');
fprintf('Saved %s\n', result_file);
