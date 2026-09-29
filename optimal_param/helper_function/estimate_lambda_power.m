function lambda_min_est = estimate_lambda_power(A_func, m, num_samples, iter)
%OPTIMAL_PARAM.ESTIMATE_LAMBDA_POWER Estimate minimum eigenvalue of PSD A.
%
%   Uses randomized power method on B = I - A.
%   The sextic solver uses this estimate for its spectral constant.

    if nargin < 4
        error('optimal_param:estimate_lambda_power:NotEnoughInputs', ...
            'Expected A_func, m, num_samples, iter.');
    end

    B_func = @(x) x - A_func(x);
    sum_val = 0;

    for j = 1:num_samples
        g = randn(m, 1);
        v = g;
        for i = 1:iter
            v = B_func(v);
        end
        sum_val = sum_val + (v' * g);
    end

    f_kn_B = (5 * sum_val / num_samples)^(1 / iter);
    lambda_min_est = 1 - f_kn_B;
end
