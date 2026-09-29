%% TEST_FOR_FUN
% Project f(x) = sin(x_1 + ... + x_d) onto low-order cluster spaces.

clear; clc;

anova_root = fullfile('E:', 'Foundations-of-Autonomous-Materials-Discovery', 'ANOVA');
addpath(genpath(fullfile(anova_root, 'core')));

%% Setup
d = 10;
n = 4;
sz = n * ones(1, d);
x_grid = linspace(0, 2*pi, n + 1);
x_grid(end) = [];

%% Build f(x) = sin(sum_j x_j)
theta = zeros(sz);
for j = 1:d
    shape = ones(1, d);
    shape(j) = n;
    theta = theta + reshape(x_grid, shape);
end

f = sin(theta);

%% Project to cluster spaces
% For this function, the Fourier support is a full d-way interaction.
% Low-order projections should therefore be nearly zero; s = d recovers f.
orders = 1:d;
rel_errors = zeros(size(orders));

fprintf('Testing f(x) = sin(x_1 + ... + x_d)\n');
fprintf('d = %d, n = %d, tensor size = %d\n\n', d, n, prod(sz));
fprintf(' order    relative error\n');
fprintf(' -----    --------------\n');

for k = 1:numel(orders)
    s = orders(k);
    f_proj = real(cluster_expansion(f, s));
    rel_errors(k) = norm(f(:) - f_proj(:)) / norm(f(:));

    fprintf(' %5d    %.6e\n', s, rel_errors(k));
end

f_proj = real(cluster_expansion(f, 1));
fprintf('norm of first-order projection %.6e\n', norm(f_proj, "fro"));
% Conclusion: so even simple function like sin(\sum x_i) could have a
% high effective dimension. Anova is just one kind of low complexity model

