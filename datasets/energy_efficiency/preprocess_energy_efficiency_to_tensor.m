% PREPROCESS_ENERGY_EFFICIENCY_TO_TENSOR
% Build empirical heating/cooling load functions on a discrete tensor grid.

clear; clc;

this_dir = fileparts(mfilename('fullpath'));
data_file = fullfile(this_dir, 'data_set', 'ENB2012_data.xlsx');
output_mat = fullfile(this_dir, 'energy_efficiency_tensor_dataset.mat');

T = readtable(data_file, 'VariableNamingRule', 'preserve');

% X2, X3, and X4 are geometry-derived and strongly tied to X1/X5, so the
% default tensor keeps a compact independent design description.
feature_names = {'X1', 'X5', 'X6', 'X7', 'X8'};
response_names = {'Y1', 'Y2'};

X = T{:, feature_names};
Y = T{:, response_names};

[grid_subscripts, levels] = encode_discrete_grid(X);
domain_tensor_size = cellfun(@numel, levels);

sub = num2cell(grid_subscripts, 1);
pattern_linear = sub2ind(domain_tensor_size, sub{:});

num_cells = prod(domain_tensor_size);
pattern_count = accumarray(pattern_linear, 1, [num_cells, 1], @sum, 0);
active_patterns = find(pattern_count > 0);

F = zeros(numel(response_names), num_cells);
for r = 1:numel(response_names)
    response_sum = accumarray(pattern_linear, Y(:, r), [num_cells, 1], @sum, 0);
    F(r, active_patterns) = response_sum(active_patterns) ./ pattern_count(active_patterns);
end

fprintf('Energy efficiency tensor size: [%s]\n', num2str(domain_tensor_size));
fprintf('Active cells: %d/%d (%.2f%%)\n', ...
    numel(active_patterns), num_cells, 100 * numel(active_patterns) / num_cells);
fprintf('Median samples per active cell: %.1f\n', median(pattern_count(active_patterns)));

pattern_count = pattern_count.';
active_patterns = active_patterns.';
pattern_linear = pattern_linear.';

save(output_mat, ...
    'F', 'pattern_count', 'active_patterns', 'pattern_linear', ...
    'grid_subscripts', 'X', 'Y', 'feature_names', 'response_names', ...
    'domain_tensor_size', 'levels', 'data_file', ...
    '-v7.3');

fprintf('Saved %s\n', output_mat);

function [subs, levels] = encode_discrete_grid(X)
    subs = zeros(size(X));
    levels = cell(1, size(X, 2));

    for j = 1:size(X, 2)
        levels{j} = unique(X(:, j), 'stable');
        [~, subs(:, j)] = ismember(X(:, j), levels{j});
    end
end
