function [tau_star, info] = solve_sextic(S, y, s, sz, eps, eta, opts)
%SOLVE_SEXTIC Compute the optimal tau from the sextic equation.
% Requires Symbolic Math Toolbox for vpasolve.
% opts.eigenvalue_method is 'power' (default) or 'svd' (dense).

    if nargin < 7 || isempty(opts)
        opts = struct();
    end
    N = prod(sz);

    if ~isfield(opts, 'tau_test'),    opts.tau_test = 0.3; end
    if ~isfield(opts, 'eigenvalue_method'), opts.eigenvalue_method = 'power'; end

    P_V_perp = @(x) x - cluster_expansion(x, s, sz);
    delta = estimate_delta(S, y, s, sz, P_V_perp);

    tau_test = opts.tau_test;
    eigenvalue_method = string(opts.eigenvalue_method);
    switch eigenvalue_method
        case "power"
            [default_samples, default_iter] = power_estimator_parameters(N);
            if ~isfield(opts, 'num_samples'), opts.num_samples = default_samples; end
            if ~isfield(opts, 'power_iter'),  opts.power_iter = default_iter; end

            A_mult = @(x) (1 - tau_test) * P_V_perp(x) ...
                + tau_test * (S' * (S * x));
            lambda_test = estimate_lambda_power( ...
                A_mult, N, opts.num_samples, opts.power_iter);
        case "svd"
            A_mat = tau_test * full(S' * S);
            for j = 1:N
                e = zeros(N, 1);
                e(j) = 1;
                A_mat(:, j) = A_mat(:, j) + (1 - tau_test) * P_V_perp(e);
            end
            lambda_test = min(svd(A_mat));
            % fprintf('The minimal eigenvalue is: %f\n', lambda_test);
    end
    C_est = lambda_test * (1 - lambda_test) / (tau_test * (1 - tau_test));

    if C_est < 1e-12
        warning('solve_sextic:SmallSpectralGap', ...
            'The estimated spectral gap is very small.');
        tau_star = NaN;
        info = struct('delta', delta, 'C_est', C_est, ...
            'lambda_test', lambda_test, 'lwce_sq', Inf, 'converged', false);
        return;
    end

    lambda_from_tau = @(tau) ...
        0.5 * (1 - sqrt(1 - 4 * C_est * tau * (1 - tau)));

    tau_ratio = eps / (eps + eta);
    tau_lower = min(0.5, tau_ratio);
    tau_upper = max(0.5, tau_ratio);

    if eps == eta
        tau_candidates = 0.5;
    else
        syms tau
        c = C_est;
        eqn = c * ((1 - tau) * eps^2 - tau * eta^2 ...
            + (1 - tau) * tau * (1 - 2 * tau) * delta^2)^2 ...
            - (eps^2 - eta^2 + (1 - 2 * tau) * delta^2) ...
            * ((1 - tau)^2 * eps^2 - tau^2 * eta^2) == 0;

        tau_candidates = double(vpasolve(eqn, tau, [tau_lower, tau_upper]));
    end

    tau_candidates = unique(tau_candidates);
    if isempty(tau_candidates)
        error('solve_sextic:NoRoot', ...
            'No root found in [%.12g, %.12g].', tau_lower, tau_upper);
    end
    if numel(tau_candidates) > 1
        warning('solve_sextic:MultipleRoots', ...
            'Multiple roots found in [%.12g, %.12g]: %s', ...
            tau_lower, tau_upper, mat2str(tau_candidates.', 12));
    end

    lwce_candidates = arrayfun(@(t) ...
        compute_lwce(t, lambda_from_tau(t), eps, eta, delta), tau_candidates);
    [lwce_sq, best_index] = min(lwce_candidates);
    tau_star = tau_candidates(best_index);
    lambda_star = lambda_from_tau(tau_star);

    info = struct();
    info.delta = delta;
    info.eigenvalue_method = eigenvalue_method;
    info.lambda_test = lambda_test;
    info.C_est = C_est;
    info.lambda_star = lambda_star;
    info.lwce_sq = lwce_sq;
    info.converged = true;
    info.tau_candidates = tau_candidates;
end
