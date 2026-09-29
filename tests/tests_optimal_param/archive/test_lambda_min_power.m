clc; clear; close all;
%rng(1);  

%% Problem setup
m = 80;          % dimension
delta = 0.01;     % failure probability
epsilon = 0.01;   % relative error tolerance on A


%% Construct PSD matrix A with spectrum in [0,1]
[Q, ~] = qr(randn(m));
% lambda = rand(m, 1);
lambda_min = 0.01;
lambda_max = 1.0;
lambda = linspace(lambda_min, lambda_max, m)';
A = Q * diag(lambda) * Q';

A_func = @(x) A * x;


%% parameter choice based on the theory
epsilon_prime = epsilon * min(lambda) / (1 - min(lambda)); % rel error for B = I-A

num_samples = ceil(25 * log(2 / delta));
iter = ceil(log(9 * m) / log(1 + epsilon_prime));

% num_samples = 1000;
% iter = 100;


fprintf('Dimension (m):      %d\n', m);
fprintf('Num samples (n):    %d\n', num_samples);
fprintf('Iterations (k):     %d\n', iter);
fprintf('--------------------------------------\n');

%% Run estimator
lambda_min_est = estimate_lambda_power(A_func, m, num_samples, iter);

%% Ground truth
lambda_min_true = min(lambda);
rel_error = abs(lambda_min_true - lambda_min_est) / lambda_min_true;

%% Report results
% fprintf('True min eigenvalue:       %.8f\n', lambda_min_true);
% fprintf('Estimated min eigenvalue:  %.8f\n', lambda_min_est);
% fprintf('--------------------------------------\n');
fprintf('Calculated Relative Error: %.4e\n', rel_error);
fprintf('Target Error Tolerance:    %.4e\n', epsilon);

if rel_error <= epsilon
    fprintf('Status: SUCCESS (Within Tolerance)\n');
else
    fprintf('Status: FAIL (Outside Tolerance)\n');
end

