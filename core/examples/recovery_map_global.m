%RECOVERY_MAP_GLOBAL Example of global recovery with a fixed parameter.

rng(1);

%% Problem setup
d = 7;                  % tensor order
n = 2;                  % grid points per dimension
s = 2;                  % maximum ANOVA interaction order
sz = n * ones(1, d);
N = prod(sz);

tau = 0.5;              % fixed global recovery parameter
model_mismatch = 0.05;
eta = 0.02;             % noise bound

%% Ground truth and measurements
P_V_perp = @(x) x - cluster_expansion(x, s, sz);

M = round(0.5 * N);
observed = randperm(N, M);
S = sparse(1:M, observed, 1, M, N);

x_model = cluster_expansion(randn(N, 1), s, sz);
x_residual = P_V_perp(randn(N, 1));
x_residual = model_mismatch * x_residual / max(norm(x_residual), eps);
x_true = x_model + x_residual;

noise = randn(M, 1);
noise = eta * noise / max(norm(noise), eps);
y = S * x_true + noise;

%% Global recovery
x_recovered = optimal_recovery_CG( ...
    S, y, s, tau, sz, zeros(N, 1), 1e5, 1e-8, false);

relative_error = norm(x_recovered - x_true) / norm(x_true);
fprintf('Global parameter tau: %.4f\n', tau);
fprintf('Relative recovery error: %.3e\n', relative_error);
