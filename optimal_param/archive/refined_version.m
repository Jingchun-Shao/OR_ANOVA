%%% Finding the optimal parameter for local setting. This version is based
%%% on Newton's method

%% Setup
d = 7; % dimension
n = 2; % grid size
s = 2; % interation order
sz = n * ones(1, d); % size of the tensor
N = prod(sz); % The size of vector from reshaped tensor.
eta = 0.2;   % noisy level
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


%% Estimate delta bound by Optimization method 
% Optimization parameters
max_epoch = 1e5;
tol_residue = 1e-12;
x_initial = zeros(N,1);

% 1. Limit omega -> 0: Forces f into V to estimate delta
omega_noise = 1e-6;
f_model = optimal_recovery_CG(S, y, s, omega_noise, sz, x_initial, max_epoch, tol_residue);
est_delta_model = norm(S * f_model - y);

% 2. Limit omega -> 1: Forces f to fit y to estimate delta
omega_model = 1 - 1e-6;
f_data = optimal_recovery_CG(S, y, s, omega_model, sz, x_initial, max_epoch, tol_residue);
est_delta_data = norm(P_V_perp(f_data));

% Check the consistance, and assign
fprintf('Checking consistence:\n')
fprintf('Omega ~ 0, delta = %.3e | Omega ~ 1, delta= %.3e\n', est_delta_model, est_delta_data);
delta = est_delta_model;

fprintf('Parameters: eta(noise) = %.4e, epsilon(mismatch)=%.4e, delta= %.4e\n', eta, eps, delta);


%% Define the helper functions for Newton's method
A_mult = @(tau, x) (1 - tau) * P_V_perp(x) + tau * S' * S * x;

% functions related to R(tau)
N_tau = @(tau) (1 - tau)^2 * eps^2 - tau^2 * eta^2;
D_tau = @(tau) (1 - tau) * eps^2 - tau * eta^2 + (1 - tau) * tau * (1 - 2 * tau) * delta^2;
R     = @(tau) N_tau(tau) / D_tau(tau);

N_tau_prime = @(tau) 2 * ((tau - 1) * eps^2 - tau * eta^2); % der for derivative
D_tau_prime = @(tau) -eps^2 - eta^2 + (6 * tau^2 - 6 * tau + 1) * delta^2;
R_prime = @(tau) (N_tau_prime(tau) * D_tau(tau) - N_tau(tau) * D_tau_prime(tau)) / (D_tau(tau)^2);

% function related to lambda_min(tau)
lambda_prime = @(lambda, tau) ...
    ((1 - 2*tau) / (tau * (1 - tau))) * ... 
    ((lambda * (1 - lambda)) / (1 - 2*lambda)); 

% function of F
F = @(lambda, tau)  lambda - R(tau);
F_prime = @(lambda, tau) lambda_prime(lambda, tau) - R_prime(tau);

%% Determine C in lambda(1-lambda) = C*tau(1-tau) to recover lambda
% Parameters
tau_test    = 0.3;        % test value of tau
num_samples = 150;        % number of random initializations
power_iter  = 10000;        % power iterations

A_mult_test = @(x) A_mult(tau_test, x); % Operator with fixed tau
lambda_test = lambda_min_power( A_mult_test, N, num_samples, power_iter);

% Compute constant C
C_est = lambda_test * (1 - lambda_test) / (tau_test * (1 - tau_test));

fprintf('tau = %.3f\n', tau_test);
fprintf('lambda_min ≈ %.4e\n', lambda_test);
fprintf('Estimated C ≈ %.4e\n', C_est);

% With this, we can determine a unique lambda_min for any tau
lambda_from_tau = @(tau) ...
    0.5 * (1 - sqrt(1 - 4 * C_est * tau * (1 - tau)));

%% Newton's method implentation
% Setup for Newton's method
max_newton_iter = 500;
newton_tol = 1e-6;

% Storage for tau_k history
tau_list = zeros(max_newton_iter+1, 1);

% Intial guss
tau_k = eps / (eps + eta);
%tau_k = 0.3;
tau_list(1) = tau_k;

disp('Starting Newton''s method for optimal tau');

for k = 1:max_newton_iter
    % Print every 10 iterations (when remainder of k/10 is 0)
    if mod(k, 10) == 0
        fprintf('Iteration %d: Current tau = %.6f\n', k, tau_k);
    end
    % Find the lambda_min(tau_k)
    lambda_k = lambda_from_tau(tau_k);

    % compute F and F'
    F_k = F(lambda_k, tau_k);
    F_prime_k = F_prime(lambda_k, tau_k);

    % Check for convergence based on F(tau) is near zero
    if abs(F_k) < newton_tol
        fprintf('Newton''s method converges after %d iterations.\n', k);
        break
    end


    % Updates
    tau_next = tau_k - F_k / F_prime_k;

    % Safety bounds (Prevent tau from hitting 0 or 1)
    epsilon_boundary = 0.01; 
    
    if tau_next < epsilon_boundary
        tau_next = epsilon_boundary;
    elseif tau_next > (1 - epsilon_boundary)
        tau_next = 1 - epsilon_boundary;
    end


    % Record the tau_k
    tau_list(k+1) = tau_next;
    tau_k = tau_next;

end

tau_star = tau_k;
fprintf('Optimal parameter tau*: %.4f\n', tau_star);


%% Compute the lwce at tau_star
% ---- lwce at tau_star (Newton solution) ----
lambda_star = lambda_from_tau(tau_star);

lwce_sq_star = ((1 - tau_star) / lambda_star) * eps^2 ...
             + (tau_star / lambda_star) * eta^2 ...
             - ((1 - tau_star) * tau_star / lambda_star) * delta^2;

% Error from: 1. power method and quadratic method to compute lambda_star 
% 2. CG(delta) 

% A question: When eta(noise std) is larger, both lambda and lwce is pretty
% accurate. What's the reason?

% lwce_star = sqrt(max(lwce_sq_star, 0));
lwce_star = lwce_sq_star;

fprintf('tau* = %.6f, lwce(tau*) = %.6e\n', tau_star, lwce_star);




%% --- Verification A:Verify LWCE using SDP ---
% Based on the provided 'BE' (Best Estimate) formulation

fprintf('\n--- Starting SDP Verification (CVX) ---\n');

% Construct Explicit Matrices (Required for CVX)
fprintf('Constructing explicit matrices for P_V_perp...\n');
I_mat = eye(N);
P_mat = zeros(N);
for i = 1:N
    P_mat(:, i) = P_V_perp(I_mat(:, i));
end


% We use the BE function to find the true optimal tau and error
[tau_sdp_opt, ~, wce_sdp_opt] = BE(eps, eta, y, P_mat, S);

fprintf('\n--- Results Comparison ---\n');
fprintf('Newton Method : tau* = %.6f, LWCE^2 = %.4e\n', tau_star, lwce_star);
fprintf('SDP Method    : tau* = %.6f, LWCE^2 = %.4e\n', tau_sdp_opt, wce_sdp_opt^2);

%% --- Verification B: Eigenvalue ---
fprintf('\n--- Eigenvalue Estimation Verification ---\n');

% 1. Construct the explicit matrix A(tau*)
A_explicit = (1 - tau_star) * P_mat + tau_star * (S' * S);

% 2. Compute the TRUE minimum eigenvalue directly
eigs_A = eig(full(A_explicit)); 
lambda_true = min(eigs_A);

% 3. Retrieve the estimated lambda from your Newton method
lambda_est = lambda_from_tau(tau_star);

% 4. Compare results
fprintf('Tau* parameter         : %.4f\n', tau_star);
fprintf('Lambda (Quad Estimate) : %.6e\n', lambda_est);
fprintf('Lambda (True Matrix)   : %.6e\n', lambda_true);
fprintf('Relative Error         : %.2f%%\n', ...
    abs(lambda_true - lambda_est)/lambda_true * 100);

% 5. Check "Constant C" consistency
% Determine what the constant C *should* have been at this specific tau*
C_actual = lambda_true * (1 - lambda_true) / (tau_star * (1 - tau_star));

fprintf('\n--- Constant C Consistency ---\n');
fprintf('C (estimated at tau=0.3): %.6f\n', C_est);
fprintf('C (actual at tau*)      : %.6f\n', C_actual);

if abs(C_est - C_actual) > 0.1
    fprintf('>> Note: The spectral shape deviates from the ellipse assumption.\n');
    fprintf('>> This explains why the LWCE prediction might differ from SDP.\n');
end



% % ---------------------------------------------------------
% % Verification C: Verify Curve with Finer Grid
% % ---------------------------------------------------------



%% --- Helper Function: Beck and Eldar ---
% Solves for the optimal regularizer and estimate simultaneously
function [tau,cheb_center,wce] = BE(epsilon,eta,y,P,L)
    N = size(P,1);

    % sedumi
    cvx_solver sedumi
    cvx_precision best

    cvx_begin quiet
    % cvx solver choice sedumi mosiac
        variable c nonnegative
        variable d nonnegative
        variable t nonnegative
        % Objective: Minimize the bound on the squared error
        minimize( epsilon^2*c - (norm(y)^2-eta^2)*d + t )
        subject to
            c*P + d*(L'*L) - eye(N) == hermitian_semidefinite(N);
            [c*P+d*(L'*L), -d*L'*y; -d*y'*L, t] == hermitian_semidefinite(N+1);
    cvx_end
    
    tau = d/(c+d);
    cheb_center = (c*P+d*L'*L)\(d*L'*y); % Actually I don't know the deduction
    wce = sqrt(cvx_optval);
end