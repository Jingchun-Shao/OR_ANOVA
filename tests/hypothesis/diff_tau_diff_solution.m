%%% Claim:
%%% In the noiseless case, the penalized optimization problem
%%%
%%%   min_f  omega * ||Lambda f - y||^2 + (1-omega) * ||P_V_perp f||^2
%%%
%%% degenerates to the constrained program
%%%
%%%   min_f ||P_V_perp f||    subject to Lambda f = y
%%%
%%% which corresponds to omega -> 1.
%%%
%%% We also compare this with the other limiting regime(omega -> 0)
%%%
%%%   min_{f in V} ||Lambda f - y||,
%%%
%%% and show that they give different answers.

%% Setup
d = 7;                  % dimension
n = 2;                  % grid size
s = 2;                  % interaction order
sz = n * ones(1, d);    % tensor size
N = prod(sz);           % ambient dimension after vectorization

eta = 0.0;              % noise level
eps = 0.05;             % model mismatch bound

%% Observation operator and ground truth
% Projection onto V^\perp
P_V_perp = @(x) x - cluster_expansion(x, s, sz);

% Observation operator Lambda, represented by the sampling matrix S
rho = 0.5;                          % sampling rate
M = max(1, round(rho * N));         % number of measurements
idx = randperm(N, M);               % sampled indices
S = sparse(1:M, idx, 1, M, N);

% Generate ground truth: x_true = xV + z, with xV in V and z in V^\perp
xV = cluster_expansion(randn(N,1), s, sz);   % component in V
z  = P_V_perp(randn(N,1));                   % component in V^\perp
z  = eps * z / norm(z);                      % scale to mismatch level
x_true = xV + z;

% Generate observations, no noise
y = S * x_true ;

% %% Check 1: compute the optimal parameter
% fprintf('\nCheck 1: compute the optimal parameter.\n');
% 
% [tau_Newton, info_Newton] = solve_opt_param(S, y, s, sz, eps, eta);
% omega_opt = tau_Newton;
% 
% fprintf('Optimal parameter omega = %.6e\n', omega_opt);

%% Check 2: compare different omega choices
fprintf('\nCheck 2: compare the recoveries for omega ~ 0 and omega ~ 1.\n');

omega_small = 1e-6;        % approx. min_{f in V} ||Lambda f - y||
omega_large = 1 - 1e-6;    % approx. min_f ||P_V_perp f|| s.t. Lambda f = y

% Optimization setup
max_epoch = 1e5;
tol_residue = 1e-8;
x_initial = zeros(N,1);

% Recoveries

x_small = optimal_recovery_CG( ...
    S, y, s, omega_small, sz, x_initial, max_epoch, tol_residue);

x_large = optimal_recovery_CG( ...
    S, y, s, omega_large, sz, x_initial, max_epoch, tol_residue);

% Relative difference between the two recoveries
rel_diff = norm(x_small - x_large) / norm(x_large);

fprintf('\nRelative difference between the two recoveries:\n');
fprintf('  ||x_small - x_large|| / ||x_large|| = %.6e\n', rel_diff);

%% Check 2.2: compare recovery on the unobserved entries
% This tests the quantity of interest: recovery away from the sampled set.
unobs_mask = true(N,1);
unobs_mask(idx) = false;

rel_diff_unobs = norm(x_small(unobs_mask) - x_large(unobs_mask)) ...
    / norm(x_large(unobs_mask));

fprintf('\nExtra check: relative difference on "unobserved entries".\n');
fprintf('  ||Q(x_small) - Q(x_large)|| / ||Q(x_large)|| = %.6e\n', rel_diff_unobs);
fprintf('  Here Q is the restriction operator onto the unobserved entries.\n');

