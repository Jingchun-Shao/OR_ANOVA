function lwce_sq = compute_lwce(tau, lambda_tau, eps, eta, delta)
%OPTIMAL_PARAM.COMPUTE_LWCE Compute squared local worst-case error bound.

    lwce_sq = ((1 - tau) / lambda_tau) * eps^2 ...
            + (tau / lambda_tau) * eta^2 ...
            - ((1 - tau) * tau / lambda_tau) * delta^2;
end
