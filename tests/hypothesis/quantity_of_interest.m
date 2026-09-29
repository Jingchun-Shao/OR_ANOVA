%% This script tests whether the squared global worst-case error of the
%  quantity of interest Gamma depends on tau in the tensor completion
%  setting.
%
%  Here Gamma extracts the unobserved entries

clear; clc; close all;
rng(1);

%% Setup
d  = 7;                 % dimension
n  = 2;                 % grid size in each mode
s  = 2;                 % effective dimension
sz = n * ones(1, d);    % tensor size
N  = prod(sz);          % ambient dimension after vectorization

eta = 0.1;             % noise level
epsilon = 0.5;             % model mismatch bound

%% Observation operator and ground truth
% Orthogonal projector onto V^\perp
P_V_perp = @(x) x - cluster_expansion(x, s, sz);

% Observation operator Lambda, represented by the sampling matrix S
rho = 0.5;                          % sampling rate
M   = max(1, round(rho * N));       % number of measurements
idx = randperm(N, M);               % sampled coordinates
S   = sparse(1:M, idx, 1, M, N);

% Ground truth: x_true = xV + z, with xV in V and z in V^\perp
xV = cluster_expansion(randn(N,1), s, sz);
z  = P_V_perp(randn(N,1));
z  = epsilon * z / max(norm(z), 1e-14);
x_true = xV + z;

% Noisy observations
e_noise = randn(M,1);
e_noise = eta * e_noise / max(norm(e_noise), 1e-14);
y = S * x_true + e_noise; 

%% Check: does gwce^2(Gamma o Delta^tau) depend on tau?
fprintf('\nCheck: compute gwce^2(Gamma o Delta^tau) and compare with the SDP bound.\n');

% Build the matrix of P_{V^\perp}
P = build_operator_matrix(P_V_perp, N);
T = P / epsilon; % \|Pf\|<epsilon <-> \|Tf\|\leq 1

% fprintf('Sanity check on projector matrix:\n');
% fprintf('  ||P^2 - P||_F = %.6e\n', norm(P * P - P, 'fro'));
% fprintf('  ||P - P^T||_F = %.6e\n', norm(P - P', 'fro'));

%% Quantity of interest: restriction to the unobserved entries
unobs_mask = true(N,1);
unobs_mask(idx) = false;
unobs_idx = find(unobs_mask);
Nq = numel(unobs_idx);


% Gamma x = x restricted to the unobserved coordinates

% % Case 1: unobserved entries
% Gamma = sparse(1:Nq, unobs_idx, 1, Nq, N);

% % Case 2: random linear map
% Gamma = randn(Nq, N);

% % Case 3: random linear map satisfying orthogonal observation
% mat = randn(N, Nq); 
% [Q, ~] = qr(mat, "econ"); 
% Gamma = Q';
% 
% % check orthogonal observation
% error_norm = norm(Gamma * Gamma' - eye(Nq));
% disp(['Orthogonality check error (should be near 1e-15): ', num2str(error_norm)]);

% % Case 4: random entries, satisfying orthogonal observation
% % Pick Nq unique, unrepeated random column indices from 1 to N
% random_idx = randperm(N, Nq);
% Gamma = sparse(1:Nq, random_idx, 1, Nq, N);
% 
% gamma_case = 'random orthogonal observation';

% fprintf('\nQuantity of interest:\n');
% fprintf('  Gamma extracts %d unobserved entries out of N = %d.\n', Nq, N);

% Check 5: The observed entries
% Gamma = S;
% It changes with different tau

% % Check 6: Identity map
% Gamma = eye(N);


% % Check 7: orthonormal map applying to unobserved
% Gamma0 = sparse(1:Nq, unobs_idx, 1, Nq, N);   % size Nq x N
% 
% % Choose output dimension r < Nq
% r = round(0.5 * Nq);   % example; choose what you want
% 
% % Build a wide row-orthonormal matrix M: r x Nq
% A = randn(Nq, r);
% [Q, ~] = qr(A, "econ");     % Q is Nq x r and Q'Q = I_r
% 
% M = Q';                % M is r x Nq and M M' = I_r
% 
% % New Gamma
% Gamma = M * Gamma0;    % size r x N




%% Check 8: mixed observed/unobserved row-orthonormal QoI
% Gamma = [M_obs * Lambda_obs;
%          M_unobs * Lambda_unobs]

% Observed selection operator
Gamma_obs0 = S;              % size Mobs x N
Mobs = size(S, 1);

% Unobserved selection operator
Gamma_unobs0 = sparse(1:Nq, unobs_idx, 1, Nq, N);   % size Nq x N

% Choose compressed dimensions
r_obs   = round(0.5 * Mobs);   % observed sketch dimension
r_unobs = round(0.5 * Nq);     % unobserved sketch dimension

% Build row-orthonormal M_obs: r_obs x Mobs
Aobs = randn(Mobs, r_obs);
[Qobs, ~] = qr(Aobs, "econ");
M_obs = Qobs';

% Build row-orthonormal M_unobs: r_unobs x Nq
Aunobs = randn(Nq, r_unobs);
[Qunobs, ~] = qr(Aunobs, "econ");
M_unobs = Qunobs';

% special case: some observed + unobserved
num_picking = 10;
random_idx = randperm(M, num_picking);
P = sparse(1:num_picking, random_idx, 1, num_picking, M);

% Mixed quantity of interest
%Gamma_obs   = M_obs   * Gamma_obs0;
Gamma_obs =  P * Gamma_obs0;
%Gamma_unobs = M_unobs * Gamma_unobs0;
Gamma_unobs = Gamma_unobs0;

Gamma = [Gamma_obs;
         Gamma_unobs];

gamma_case = sprintf('mixed QoI: r_obs=%d, r_unobs=%d', r_obs, r_unobs);

% Diagnostics
err_Gamma  = norm(Gamma * Gamma' - eye(size(Gamma, 1)), 'fro');

fprintf('  ||Gamma Gamma^T - I||_F       = %.3e\n', err_Gamma);





%% Optimal benchmark from the reduced SDP
%   min c + d eta^2
%   s.t. c T'*T + d S''S >= Gamma''Gamma
[~, c_sharp, d_sharp] = solve_optimal_param(T, S, Gamma, eta);
tau_sharp = d_sharp / (c_sharp + d_sharp);

% then compute the gwce^2
[Delta_sharp, ~] = optimal_recovery_map(T, S, Gamma, tau_sharp);
gwce_sharp_sq = solve_gwce_sq(T, S, Gamma, Delta_sharp, eta);
fprintf('\n For the optimal parameter:\n');
fprintf('  optimal gwce^2 = %.12e\n', gwce_sharp_sq);
fprintf('  tau_sharp      = %.12e\n', tau_sharp);
% fprintf('  c_sharp        = %.12e\n', c_sharp);
% fprintf('  d_sharp        = %.12e\n', d_sharp);

%% Sweep tau to see whether gwce depends on tau
tau_list = linspace(1e-3, 1 - 1e-3, 31);
gwce_tau_sq = zeros(size(tau_list));
fprintf('\nSweeping tau:\n');
for k = 1:numel(tau_list)
    tau = tau_list(k);
    [Delta_tau, ~] = optimal_recovery_map(T, S, Gamma, tau);
    gwce_tau_sq(k) = solve_gwce_sq(T, S, Gamma, Delta_tau, eta);
    fprintf('  tau = %.4f,   gwce^2 = %.12e\n', tau, gwce_tau_sq(k));
end
% plot
figure;
plot(tau_list, gwce_tau_sq, 'o-', 'LineWidth', 1.2); hold on;
yline(gwce_sharp_sq, '--', 'LineWidth', 1.2);
xline(tau_sharp, ':', 'LineWidth', 1.2);
legend('gwce^2(\Gamma \circ \Delta^\tau)', ...
       'reduced SDP optimum', '\tau^#', ...
       'Location', 'best');
grid on;
xlabel('\tau');
ylabel('squared global worst-case error');
title(['GWCE sweep, \Gamma = ', gamma_case]);
%% ============================================================
%% Local functions
%% ============================================================
function A = build_operator_matrix(func_handle, N)
% Build the matrix of a linear operator from its action on basis vectors.
    A = zeros(N, N);
    I = eye(N);
    for j = 1:N
        A(:,j) = func_handle(I(:, j));
    end
end

function [Delta_tau, X_tau] = optimal_recovery_map(T, S, Gamma, tau)
% State recovery map:
%   x_tau = argmin_x (1-tau)||T x||^2 + tau ||Sx-y||^2
%
% Hence
%   X_tau = ((1-tau)(T'*T) + tau S'S)^{-1} (tau S')
%
% Quantity-of-interest map:
%   Delta_tau y = Gamma x_tau
    A_tau = (1 - tau) * (T' * T) + tau * (S' * S);
    rhs   = tau * S';
    X_tau = A_tau \ rhs; % optimal recovery map for Identity
    Delta_tau = Gamma * X_tau; % Optimal recovery map for the QOI
end

function [optval, c_opt, d_opt] = solve_optimal_param(T, S, Gamma, eta)
% Solve
%   min_{c,d >= 0} c + d eta^2
%   s.t. c T'*T + d S'S - Gamma'Gamma >= 0
    B = Gamma' * Gamma;

    cvx_begin sdp quiet
        variables c_opt d_opt
        minimize(c_opt + d_opt * eta^2)
        c_opt >= 0;
        d_opt >= 0;
        LMI = c_opt * (T' * T) + d_opt * (S' * S) - B;
        LMI = 0.5 * (LMI + LMI'); % to ensure symmetry
        LMI >= 0;
    cvx_end
    optval = cvx_optval;
end

function optval = solve_gwce_sq(T, S, Gamma, Delta, eta)
% For a fixed recovery map Delta, solve
%
%   gwce(Delta)^2
%   = min_{c,d >= 0} c + d eta^2
%     s.t.
%     [ c(T'*T) - (Gamma-Delta S)'(Gamma-Delta S),    (Gamma-Delta S)' Delta;
%       Delta' (Gamma-Delta S),                       dI - Delta' Delta ] >= 0
    M = size(S,1);
    E = Gamma - Delta * S;

    cvx_begin sdp quiet
        variables c_opt d_opt
        minimize(c_opt + d_opt * eta^2)
        c_opt >= 0;
        d_opt >= 0;
        LMI = [c_opt * (T' * T) - E' * E,   E' * Delta; ...
               Delta' * E,                  d_opt * eye(M) - Delta' * Delta];
        LMI = 0.5 * (LMI + LMI'); % Guarantee symmetry
        LMI >= 0;
    cvx_end
    optval = cvx_optval;
end