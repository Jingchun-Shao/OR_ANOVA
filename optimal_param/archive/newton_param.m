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



% %% Checks before Newton's method
% % The matrix of interest
% make_A_func = @(tau) @(x) (1 - tau) * P_V_perp(x) + tau * (S' * (S * x));
% 
% A_tau = make_A_func(tau);
% tau = 0.01;
% A_tau = make_A_func(tau);
% 
% % 1. A_tau is PSD
% for i = 1:5
%     x = randn(N,1);
%     x = x / norm(x);
%     val = x' * A_tau(x);
%     fprintf('x^T A x = %.6e\n', val);
% end
% 
% % 2. The eigenvalue of A_tau
% % Build the matrix concretely
% A_mat = zeros(N, N);
% for j = 1:N
%     e = zeros(N,1);
%     e(j) = 1;
%     A_mat(:,j) = A_tau(e);
% end
% 
% % compute the eigenvalues
% eigvals = eig(A_mat);
% 
% fprintf('Min eigenvalue: %.6e\n', min(eigvals));
% fprintf('Max eigenvalue: %.6e\n', max(eigvals));
% 
% % fprintf('Eigenvalues:\n');
% % fprintf('%.6e\n', eigvals);





%% Bound setup

% 1.noise bound
eta = noise_std;

% 2. Compute delta
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




%% Newton's method implentation
% Setup for Newton's method
max_newton_iter = 100;
newton_tol = 1e-6;

% Setup for power method
num_samples = 150;
power_iter = 1000; % Number of itertions for finding the smallest eigenvalue

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
    A_tau_k_mult = @(x) A_mult(tau_k, x);
    lambda_k = smallest_eig_sym(A_tau_k_mult, power_iter, N);
    %lambda_k = lambda_min_power(A_tau_k_mult, N, num_samples, power_iter);

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
power_iter = 1000;

% ---- lwce at tau_star (Newton solution) ----
A_tau_mult = @(x) A_mult(tau_star, x);
lambda_star = smallest_eig_sym(A_tau_mult, power_iter, N);
% lambda_star = lambda_min_power(A_tau_mult, N, num_samples, power_iter);

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
    A_tau_mult = @(x) A_mult(tau_star, x);
    lambda_tau = smallest_eig_sym(A_tau_mult, power_iter, N);
    % lambda_tau = lambda_min_power(A_tau_mult, N, num_samples, power_iter);

    lwce_sq = ((1 - tau) / lambda_tau) * eps^2 ...
            + (tau / lambda_tau) * eta^2 ...
            - ((1 - tau) * tau / lambda_tau) * delta^2;

    % TODO: compute 

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








%% The power method to compute the smallest eigenvalue
function [lambda_min, u_min] = smallest_eig_sym(A_mult, maxit, N)
    % Since the spectrum ranges from[0, 1] and A is symmetric, finding the
    % smallest eigenvalue of A is equivalent to finding the largest
    % eigenvalue of (I - A)

    % Form helper function (I-A)
    B = @(x) x - A_mult(x);

    u0 = randn(N, 1);
    u = u0 / norm(u0); % normalize the initial vector

    % Use the power method to find the max e.v. of I-A
    for k = 1:maxit
        v  = B(u);  % B*u = (I - A)*u
        u  = v / norm(v);
    end

    mu_max_B = u' * B(u);   % largest eigenvalue of I-A 

    % Find the min e.v. of A
    lambda_min = 1 - mu_max_B;
    u_min = u;
end


