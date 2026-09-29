% PREPROCESS_MULTINARY_OXIDES_TO_TENSOR
% Build the native A1 x A2 x B1 x B2 tensor for the MPContribs
% multinary perovskite oxides dataset.

clear; clc;

this_dir = fileparts(mfilename('fullpath'));
raw_data_file = fullfile(this_dir, 'data_set', 'multinary_oxides.csv');
output_mat = fullfile(this_dir, 'multinary_oxides_tensor_dataset.mat');

response_names = { ...
    'dH_formation_eV', ...
    'dH_decomposition_eV' ...
};

raw_table = readtable(raw_data_file, 'VariableNamingRule', 'preserve');
num_raw_rows = height(raw_table);

raw_feature_names = {'A1', 'A2', 'B1', 'B2'};
feature_names = {'A1_id', 'A2_id', 'B1_id', 'B2_id'};

raw_features = string(raw_table{:, raw_feature_names});
Y_raw = raw_table{:, response_names};
valid = all(~ismissing(raw_features), 2) & all(isfinite(Y_raw), 2);

raw_features = raw_features(valid, :);
Y = Y_raw(valid, :);
raw_table = raw_table(valid, :);

[grid_subscripts, levels] = encode_discrete_grid(raw_features);
X = grid_subscripts;
domain_tensor_size = cellfun(@numel, levels);

sub = num2cell(grid_subscripts, 1);
pattern_linear = sub2ind(domain_tensor_size, sub{:});

num_cells = prod(domain_tensor_size);
pattern_count = accumarray(pattern_linear, 1, [num_cells, 1], @sum, 0);
active_patterns = find(pattern_count > 0);

% This mapping relation will be recorded to the dataset.
A1_mapping = make_mapping_table(levels{1}, 'A1', 'A1_id');
A2_mapping = make_mapping_table(levels{2}, 'A2', 'A2_id');
B1_mapping = make_mapping_table(levels{3}, 'B1', 'B1_id');
B2_mapping = make_mapping_table(levels{4}, 'B2', 'B2_id');


fprintf('Multinary oxides tensor size: [%s]\n', num2str(domain_tensor_size));
fprintf('Active cells: %d/%d (%.2f%%)\n', ...
    numel(active_patterns), num_cells, 100 * numel(active_patterns) / num_cells);
fprintf('Rows used after removing missing/non-finite values: %d/%d\n', ...
    height(raw_table), num_raw_rows);
fprintf('Median samples per active cell: %.1f\n', ...
    median(pattern_count(active_patterns)));

F = zeros(numel(response_names), num_cells);
for r = 1:numel(response_names)
    response_sum = accumarray(pattern_linear, Y(:, r), [num_cells, 1], @sum, 0);
    F(r, active_patterns) = response_sum(active_patterns) ./ pattern_count(active_patterns);
end

pattern_count = pattern_count.';
active_patterns = active_patterns.';
pattern_linear = pattern_linear.';

save(output_mat, ...
    'F', 'pattern_count', 'active_patterns', 'pattern_linear', ...
    'grid_subscripts', 'X', 'Y', 'feature_names', 'raw_feature_names', ...
    'response_names', 'domain_tensor_size', ...
    'A1_mapping', 'A2_mapping', 'B1_mapping', 'B2_mapping', ...
    'levels', 'raw_data_file', ...
    '-v7.3');

fprintf('Saved %s\n', output_mat);

function [subs, levels] = encode_discrete_grid(X)
    subs = zeros(size(X));
    levels = cell(1, size(X, 2));

    % map the category to index column by column
    for j = 1:size(X, 2)
        levels{j} = unique(X(:, j), 'stable');
        [~, subs(:, j)] = ismember(X(:, j), levels{j}); 
    end
end

function mapping = make_mapping_table(values, symbol_name, id_name)
    ids = (1:numel(values)).';
    mapping = table(values(:), ids, 'VariableNames', {symbol_name, id_name});
end
