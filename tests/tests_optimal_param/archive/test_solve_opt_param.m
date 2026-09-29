% TEST_SOLVE_OPT_PARAM
% Compute the optimal parameter with Newton's method and BE optimazation
% program. Also test with lwce

% rng(1);

%% Setup
d = 7; % dimension
n = 2; % grid size
s = 2; % interation order
sz = n * ones(1, d); % size of the tensor
N = prod(sz); % The size of vector from reshaped tensor.
eta = 0.1;   % noisy level
eps = 0.05;   % model mismatch bound

%% The observation operator and the groundtruth
% projection to function class
P_V_perp = @(x) x - cluster_expansion(x, s, sz);  

% Observation operator
rho = 0.5;                    % sampling rate 
M   = max(1, round(rho*N));   % number of measurements
idx = randperm(N, M);         % measurement indices
S = sparse(1:M, idx, 1, M, N);

% Generate ground_truth 
% V direction
xV = cluster_expansion(randn(N,1), s, sz);  % model part (in V)
% V perp direction
z  = P_V_perp(randn(N,1));                  % pure V perp direction
z  = eps * z / norm(z);                     % scale to eps
% sum two components together
x  = xV + z;


% Generate observations
noise  = randn(M, 1);
noise = eta * noise / norm(noise);

y = S * x +  noise;


%% Compute the optimal parameter by Newton's method

% Setup for Newton's method
opts = struct(); % adopt the default setup
opts.method = 'Newton';
[tau_newton, info_newton] = solve_opt_param(S, y, s, sz, eps, eta, opts);
lwce_newton = info_newton.lwce_sq;

%% Compute the optimal parameter by the BE method
% Setup for Newton's method
opts = struct(); % adopt the default setup
opts.method = 'BE';
[tau_BE, info_BE] = solve_opt_param(S, y, s, sz, eps, eta, opts);
lwce_BE = info_BE.wce_sq;


%% Relative error of tau
rel_err_tau = abs(tau_newton - tau_BE) / abs(tau_BE);

%% Print results
fprintf('\n--- Results Comparison ---\n');
fprintf('Newton Method : tau* = %.6f, LWCE^2 = %.4e\n', tau_newton, lwce_newton);
fprintf('SDP Method    : tau* = %.6f, LWCE^2 = %.4e\n', tau_BE, lwce_BE);
fprintf('Relative error on tau = %.4e\n', rel_err_tau);

