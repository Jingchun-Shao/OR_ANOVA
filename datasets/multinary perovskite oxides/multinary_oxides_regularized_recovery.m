% MULTINARY_OXIDES_REGULARIZED_RECOVERY
% Global recovery of formation energy on recorded multinary-oxide compositions.

clear; clc;

%% Configuration
this_dir = fileparts(mfilename('fullpath'));
anova_root = fileparts(fileparts(this_dir));
data_mat = fullfile(this_dir, 'multinary_oxides_tensor_dataset.mat');
result_mat = fullfile(this_dir, 'multinary_oxides_regularized_recovery_result.mat');

s_values = [1, 2];
tau = 0.5;
train_ratio = 0.8;
seed = 1;

addpath(fullfile(anova_root, 'core'));
set_path();

%% Load the preprocessed tensor and split recorded cells
D = load(data_mat, 'F', 'active_patterns', 'response_names', 'domain_tensor_size');
F = D.F;
response_name = 'dH_formation_eV';
formation_idx = strcmp(D.response_names, response_name);
sz = D.domain_tensor_size;
N = prod(sz);
active_cells = D.active_patterns(:).';

rng(seed);
permutation = active_cells(randperm(numel(active_cells)));
num_train = round(train_ratio * numel(active_cells));
train_idx = permutation(1:num_train);
test_idx = permutation(num_train + 1:end);
S = sparse(1:num_train, train_idx, 1, num_train, N);

fprintf('Tensor: [%s]; recorded: %d/%d; train: %d; test: %d\n', ...
    num2str(sz), numel(active_cells), N, num_train, numel(test_idx));

%% Global recovery
rows = cell(numel(s_values), 6);
y = F(formation_idx, train_idx).';
truth = F(formation_idx, test_idx).';

% Testing entries are withheld during recovery; their relative error can
% be viewed as a QoI on unobserved entries.
for k = 1:numel(s_values)
    s = s_values(k);
    timer = tic;
    [f_hat, info] = regularized_solution(S, y, s, tau, sz, false);
    elapsed = toc(timer);

    rel_error_testing = norm(f_hat(test_idx) - truth) / norm(truth);
    rows(k, :) = {response_name, s, info.r, tau, rel_error_testing, elapsed};
    fprintf('%s, s=%d: rel_error_testing=%.4g\n', ...
        response_name, s, rel_error_testing);
end

results = cell2table(rows, 'VariableNames', ...
    {'response', 's', 'model_dim', 'tau', 'rel_error_testing', 'time_sec'});
disp(results);

save(result_mat, 'results', 'train_idx', 'test_idx', ...
    'response_name', 's_values', 'tau', 'train_ratio', 'seed');
fprintf('Saved %s\n', result_mat);
