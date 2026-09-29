function [num_samples, power_iter] = power_estimator_parameters( ...
    N, failure_prob, target_rel_error, lambda_min_est)
%POWER_ESTIMATOR_PARAMETERS Settings for lambda_min(A) via B = I - A.
% Guaranteed target_rel_error requires lambda_min_est to be a lower bound.

    if nargin < 2 || isempty(failure_prob), failure_prob = 0.01; end
    if nargin < 3 || isempty(target_rel_error), target_rel_error = 0.01; end
    if nargin < 4 || isempty(lambda_min_est), lambda_min_est = 0.01; end

    rel_error_B = target_rel_error * lambda_min_est / (1 - lambda_min_est);
    num_samples = ceil(25 * log(2 / failure_prob));
    power_iter = ceil(log(9 * N) / log1p(rel_error_B));
end
