%%% Finding the optimal parameter for local setting. This version is based
%%% on Newton's method

%% Setup
d = 10; % dimension
n = 2; % grid size
s = 2; % interation order
sz = n * ones(1, d); % size of the tensor
N = prod(sz); % The size of vector from reshaped tensor.
noise_std = 0.01;             % noise std (noisy level, must>0 in our case


%% The observation operator and the groundtruth
% projection to function class
P_V_perp = @(x) x - cluster_expansion(x, s, sz);  

% Observation operator
rho = 0.5;                    % sampling rate 
M   = max(1, round(rho*N));   % number of measurements
idx = randperm(N, M);         % measurement indices
S = sparse(1:M, idx, 1, M, N);

% Generate ground_truth 
eps = 0.01;   % model mismatch bound

% V direction
xV = cluster_expansion(randn(N,1), s, sz);  % model part (in V)
% V perp direction
z  = P_V_perp(randn(N,1));                  % pure V perp direction
z  = eps * z / norm(z);                     % scale to eps
% sum two components together
x  = xV + z;


% Generate observations
noise  = randn(M, 1);
noise = noise_std * noise / norm(noise);

y = S * x +  noise;


%% Bound setup

% noise bound
eta = noise_std;

% f_instance = S' * y; % One instance matches the evaluation, simple fill-in
% delta = norm(P_V_perp(f_instance)); % P projects on V^perp, for model dismatch
% % Todo: the true delta is smaller than this instance

% Based on the CG code
% optimization setup
max_epoch = 1e5;
tol_residue  = 1e-8;
x_initial = zeros(N,1);  % Starting point
omega = 1 - 1e-6;   % very close to 1

f_delta = optimal_recovery_CG( ...
    S, y, s, omega, sz, ...
    x_initial, max_epoch, tol_residue);
delta = norm(P_V_perp(f_delta));


% Compute the imcompatibility by the previous CG code

fprintf('eps_for_model = %.3e, eta_for_noise = %.3e, delta < %.3e\n', ...
        eps, eta, delta);

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
lambda_prime = @(lambda, tau) (1-2*tau) * tau * (1-tau) * lambda * (1-lambda) / (1-2*lambda);

% function of F
F = @(lambda, tau)  lambda - R(tau);
F_prime = @(lambda, tau) lambda_prime(lambda, tau) - R_prime(tau);

%% Determine C in lambda(1-lambda) = C*tau(1-tau) to recover lambda
% Parameters
tau_test    = 0.3;        % test value of tau
num_samples = 150;        % number of random initializations
power_iter  = 50000;        % power iterations

A_mult_test = @(x) A_mult(tau_test, x); % Operator with fixed tau
lambda_test = lambda_min_power(A_mult_test, N, num_samples, power_iter);

% Compute constant C
C_est = lambda_test * (1 - lambda_test) / (tau_test * (1 - tau_test));

fprintf('tau = %.3f\n', tau_test);
fprintf('lambda_min ≈ %.6e\n', lambda_test);
fprintf('Estimated C ≈ %.6e\n', C_est);

% With this, we can determine a unique lambda_min for any tau
lambda_from_tau = @(tau) ...
    0.5 * (1 - sqrt(1 - 4 * C_est * tau * (1 - tau)));

%% Newton's method implentation
% Setup for Newton's method
max_newton_iter = 100;
newton_tol = 1e-6;

% Storage for tau_k history
tau_list = zeros(max_newton_iter+1, 1);

% Intial guss
%tau_k = eps / (eps + eta);
tau_k = 0.86;
tau_list(1) = tau_k;

disp('Starting Newton''s method for optimal tau');

for k = 1:max_newton_iter
    fprintf('The current newton iteration is %d.\n', k)
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
    tau_next = max(0.0, min(1.0, tau_next)); % limit to [0,1]
    tau_k = tau_next;

    % Record the tau_k
    tau_list(k+1) = tau_next;

end

tau_star = tau_k;
fprintf('Optimal parameter tau*: %.4f\n', tau_star);


%% Check that tau_star minimizes lwce(tau)
% ---- lwce at tau_star (Newton solution) ----
lambda_star = lambda_from_tau(tau_star);

lwce_sq_star = ((1 - tau_star) / lambda_star) * eps^2 ...
             + (tau_star / lambda_star) * eta^2 ...
             - ((1 - tau_star) * tau_star / lambda_star) * delta^2;

% lwce_star = sqrt(max(lwce_sq_star, 0));
lwce_star = lwce_sq_star;

fprintf('tau* = %.6f, lwce(tau*) = %.6e\n', tau_star, lwce_star);

% ---- lwce over a grid of tau ----
tau_grid = 0:0.01:1;
lwce_vals = zeros(size(tau_grid));

for i = 1:numel(tau_grid)
    tau = tau_grid(i);
    lambda_tau = lambda_from_tau(tau);

    lwce_sq = ((1 - tau) / lambda_tau) * eps^2 ...
            + (tau / lambda_tau) * eta^2 ...
            - ((1 - tau) * tau / lambda_tau) * delta^2;

    % lwce_vals(i) = sqrt(max(lwce_sq, 0));
    lwce_vals(i) = lwce_sq;
end

% ---- plot ----
figure;
plot(tau_grid, lwce_vals, 'LineWidth', 2); hold on;
xline(tau_star, '--r', 'LineWidth', 1.5);
grid on;

% set(gca, 'YScale', 'log');

xlabel('\tau');
ylabel('lwce(\tau)');
title('Local Worst-Case Error vs \tau');
legend('lwce(\tau)', '\tau_* (Newton)', 'Location', 'best');






