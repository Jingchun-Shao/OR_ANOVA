% Setup
m = 100;                 % Dimension 
epsilon = 0.01;          % Relative error tolerance

%% Construct PSD matrix A with spectrum in [0,1]
[Q, ~] = qr(randn(m));
lambda_min = 0.01;
lambda_max = 1.0;
lambda = linspace(lambda_min, lambda_max, m)';
A = Q * diag(lambda) * Q';

% Helper functions
A_func = @(x) A * x;
B_func = @(x) x - A_func(x); 

% --- Parameters
k = 10000; 
n = 100;

fprintf('Dimension (m):      %d\n', m);
fprintf('Num samples (n):    %d\n', n);
fprintf('Iterations (k):     %d\n', k);
fprintf('--------------------------------------\n');

% --- Algorithm Implementation ---
% We look for the MAXIMUM of the transformed matrix B
max_Z_B = -inf; 

for j = 1:n
    % 1. Generate single Gaussian vector
    g = randn(m, 1);
    
    % 2. Normalize to unit sphere
    x = g / norm(g);
    
    % 3. Compute B^k * x iteratively
    v = x;
    for i = 1:k
        v = B_func(v); % Iterate using B = I - A
    end
    
    % 4. Compute Quadratic Form Z = x' * (B^k * x)
    Z_val = x' * v;
    
    % 5. Update Empirical Maximum for B
    if Z_val > max_Z_B
        max_Z_B = Z_val;
    end
end

% --- The Estimator ---
% 1. Estimate max eig of B
lambda_max_B_est = max_Z_B^(1/k);

% 2. Recover min eig of A: lambda_min(A) = 1 - lambda_max(B)
lambda_min_est = 1 - lambda_max_B_est;
lambda_min_true = min(lambda);

% --- Analysis ---
rel_error = abs(lambda_min_true - lambda_min_est) / lambda_min_true;

fprintf('True min eigenvalue:      %.5f\n', lambda_min_true);
fprintf('Estimated min eigenvalue: %.5f\n', lambda_min_est);
fprintf('--------------------------------------\n');
fprintf('Relative Error:        %.4e\n', rel_error);
