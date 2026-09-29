%%% Test on optimal_recovery_GD.m
%  min_f omega * ||f(idx) - y|| + (1-omega) * ||(I- P)f||. P is the
%  orthogonal projection to the subspace spanned by low-order cluster basis.


%% Setup
d = 10; % dimension
n = 2; % grid size
s = 2; % interation order
sz = n * ones(1, d); % size of the tensor
omega = 0.5; % weight in (0,1)
N = prod(sz);

% optimization setup
max_epoch = 1e5;
tol_grad  = 1e-8;
x_initial = zeros(N,1);  % Starting point

%% Ground truth
g = rand(n * ones(1, d)); 
g = cluster_expansion(g, s); % Project it to the s-order cluster space
x_true = reshape(g, [], 1); % Reshape the groundtruth tensor to

%% Measurements
N   = n^d;
rho = 0.2;                    % sampling rate 
M   = max(1, round(rho*N));   % number of measurements

% Selection matrix
idx = randperm(N, M);         % measurement indices
S = sparse(1:M, idx, 1, M, N);

% sigma = 0.001;                 % noise std 
sigma = 0;
y = S * x_true + sigma*randn(M,1);


%% GD
tic;   % start timing
x = optimal_recovery_GD(S, y, s, omega, sz, x_initial, max_epoch, tol_grad);
total_time = toc;    % end timing

fprintf('Total GD time: %.4f sec\n', total_time);


% analysis of result
rel_err = norm(x - x_true)/max(1e-12, norm(x_true));
fprintf('Relative reconstruction error: %.3e\n', rel_err);





