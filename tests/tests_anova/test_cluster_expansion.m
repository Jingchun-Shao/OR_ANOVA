%% ------------------------------------------------------------
% Check the cluster_expansion.m by comparing to the recursive approach

clear; clc;

%% Setting parameters
d = 10; % dimension
n = 5; % grid size
s = 4; % interation order

% Ground truth
g = rand (n*ones(1, d)); 

%% Test 1: whether the output is the same as ANOVA
f_proj = cluster_expansion(g, s);

% Do Anova decomposition to s order
D = anova_recursively(g); % The ANOVA instance
D.decompose(s); % Do ANOVA decomposition to the order of s
y = D.reconstruct(); % Sum the component up


rel_error = norm(y(:) - f_proj(:)) / norm(y(:));
fprintf('[d=%d, s=%d] rel.error = %.2e\n', d, s, rel_error);


%%  Test 2: Compare the speed of recursion approach and fft approach

% Pipelines for the two approaches
fft_pipeline   = @() cluster_expansion(g, s);      % FFT + mask + IFFT
anova_pipeline1 = @() anova_pipeline(g, s);  % direct truncated ANOVA

% Stable timing
t_fft   = timeit(fft_pipeline);
t_anova = timeit(anova_pipeline1);



fprintf('--- SPEED comparison ---\n');
fprintf('d=%d, n=%d, s=%d\n', d, n, s);
fprintf('FFT-mask time:       %.4f s\n', t_fft);
fprintf('ANOVA (direct) time: %.4f s\n', t_anova);

function y = anova_pipeline(g, s)
    D = anova_recursively(g); % The ANOVA instance
    D.decompose(s); % Do ANOVA decomposition to the order of s
    y = D.reconstruct(); % Sum the component up
end

