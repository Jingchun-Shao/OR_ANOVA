function lambda_min_est = lambda_min_power(A_func, m, num_samples, iter)
% ESTIMATE_LAMBDA_MIN_POWER
%   Estimates lambda_min(A) for a PSD matrix A using a
%   randomized power method applied to B = I - A.
%
% INPUTS:
%   A_func      : function handle, y = A_func(x)
%   m           : dimension
%   num_samples : number of Gaussian probe vectors
%   iter        : number of power iterations
%
% OUTPUT:
%   lambda_min_est : estimated minimum eigenvalue of A

    % Define B = I - A
    B_func = @(x) x - A_func(x);

    % Draw Gaussian probes
    G = randn(m, num_samples);
    sum_val = 0;

    % Power method loop
    for j = 1:num_samples
        g = G(:, j);
        v = g;
        for i = 1:iter
            v = B_func(v);
        end
        sum_val = sum_val + (v' * g);
    end

    % Estimator for lambda_max(B)
    f_kn_B = (5 * sum_val / num_samples)^(1 / iter);

    % Recover lambda_min(A)
    lambda_min_est = 1 - f_kn_B;
end
