function [tau_star, info] = solve_Newton(S, y, s, sz, eps, eta, opts)
%SOLVE_NEWTON Compute the optimal tau by Newton's method.
%
% Inputs:
%   S(selection operator), y(data), s(effective dim), sz(size of tensor),
%   eps(model mismatch), eta(noise)
%
% Optional fields in opts:
%   opts.tau_test     - default 0.3
%   opts.num_samples  - default from power_estimator_parameters
%   opts.power_iter   - default from power_estimator_parameters
%   opts.max_iter     - default 200
%   opts.tol          - default 1e-3
%   opts.tau0         - default midpoint of 0.5 and eps / (eps + eta)
%   opts.boundary     - default 1e-3
%   opts.verbose      - default false

    if nargin < 7 || isempty(opts)
        opts = struct();
    end

    N = prod(sz);
    [default_samples, default_iter] = power_estimator_parameters(N);

    if ~isfield(opts, 'tau_test'),    opts.tau_test = 0.3; end % one can change
    if ~isfield(opts, 'num_samples'), opts.num_samples = default_samples; end
    if ~isfield(opts, 'power_iter'),  opts.power_iter = default_iter; end
    if ~isfield(opts, 'max_iter'),    opts.max_iter = 200; end
    if ~isfield(opts, 'tol'),         opts.tol = 1e-3; end
    tau_ratio = eps / (eps + eta);
    if ~isfield(opts, 'tau0'),        opts.tau0 = 0.5 * (0.5 + tau_ratio); end
    if ~isfield(opts, 'boundary'),    opts.boundary = 1e-3; end % prevent newton method from hitting boundary
    if ~isfield(opts, 'verbose'),     opts.verbose = false; end % print while doing newton method
    % Projection onto V^\perp
    P_V_perp = @(x) x - cluster_expansion(x, s, sz);

    % Step 1: estimate delta using omega ~ 1
    delta = estimate_delta(S, y, s, sz, P_V_perp);

    %% Step 2: define A(tau)
    A_mult = @(tau, x) (1 - tau) * P_V_perp(x) + tau * (S' * (S * x));

    %% Step 3: estimate C from one test value tau_test
    tau_test = opts.tau_test;

    lambda_test = estimate_lambda_power( ...
        @(x) A_mult(tau_test, x), N, opts.num_samples, opts.power_iter);

    C_est = lambda_test * (1 - lambda_test) / (tau_test * (1 - tau_test));

    lambda_from_tau = @(tau) ...
        0.5 * (1 - sqrt(1 - 4 * C_est * tau * (1 - tau)) );
    % The solution is unique because lambda < 0.5

    %% Step 4: define Newton functions
    N_tau = @(tau) (1 - tau)^2 * eps^2 - tau^2 * eta^2;

    D_tau = @(tau) ...
        (1 - tau) * eps^2 - tau * eta^2 ...
        + (1 - tau) * tau * (1 - 2 * tau) * delta^2;

    R = @(tau) N_tau(tau) / D_tau(tau);

    N_tau_prime = @(tau) 2 * ((tau - 1) * eps^2 - tau * eta^2);

    D_tau_prime = @(tau) ...
        -eps^2 - eta^2 + (6 * tau^2 - 6 * tau + 1) * delta^2;

    R_prime = @(tau) ...
        (N_tau_prime(tau) * D_tau(tau) - N_tau(tau) * D_tau_prime(tau)) ...
        / (D_tau(tau)^2);

    lambda_prime = @(lambda, tau) ...
        ((1 - 2 * tau) / (tau * (1 - tau))) * ...
        ((lambda * (1 - lambda)) / (1 - 2 * lambda));

    %% Step 5: Newton's method
    tau = opts.tau0;
    if eps == eta
        tau = 0.5;
    end

    tau_history = zeros(opts.max_iter + 1, 1);
    F_history = zeros(opts.max_iter, 1);
    tau_history(1) = tau;

    % The equal-parameter case is already solved by tau = 0.5.
    converged = (eps == eta);
    used_iter = 0;

    if ~converged
        for k = 1:opts.max_iter
            lambda_k = lambda_from_tau(tau);
            F_k = lambda_k - R(tau);
            dF_k = lambda_prime(lambda_k, tau) - R_prime(tau);

            F_history(k) = F_k;
            used_iter = k;

            if abs(F_k) < opts.tol
                converged = true;
                tau_history(k + 1) = tau;
                break;
            end

            if ~isfinite(dF_k) || abs(dF_k) < 1e-14
                warning('solve_Newton:smallDerivative', ...
                    'Newton step stopped because derivative is too small.');
                tau_history(k + 1) = tau;
                break;
            end

            tau_next = tau - F_k / dF_k;
            tau_next = min(max(tau_next, opts.boundary), 1 - opts.boundary);

            tau_history(k + 1) = tau_next;
            tau = tau_next;

            if opts.verbose && mod(k, 10) == 0
                fprintf('iter = %d, tau = %.6f, F = %.3e\n', k, tau, F_k);
            end
        end
    end

    tau_star = tau;
    tau_lower = min(0.5, tau_ratio);
    tau_upper = max(0.5, tau_ratio);
    if ~converged || ~isreal(tau_star) || ~isfinite(tau_star)
        error('solve_Newton:InvalidOptimalParameter', ...
            'Newton did not converge to a finite real tau.');
    end

    within_interval = tau_star >= tau_lower - 1e-12 && ...
                      tau_star <= tau_upper + 1e-12;
    if ~within_interval
        warning('solve_Newton:OutsideInterval', ...
            ['Newton converged to tau = %.8g, outside [%.8g, %.8g]. ' ...
             'Check the eigenvalue approximation and whether eps and eta ' ...
             'are appropriate for the data.'], ...
            tau_star, tau_lower, tau_upper);
    end

    %% Step 6: compute LWCE at tau_star
    lambda_star = lambda_from_tau(tau_star);

    lwce_sq = compute_lwce(tau_star, lambda_star, eps, eta, delta);

    %% Output info
    info = struct();
    info.delta = delta;
    info.C_est = C_est;
    info.lambda_star = lambda_star;
    info.lwce_sq = lwce_sq;
    info.converged = converged;
    info.within_interval = within_interval;
    info.iterations = used_iter;
    info.tau_history = tau_history(1:used_iter + 1);
    info.F_history = F_history(1:used_iter);
end
