function [gwce, info] = compute_gwce(S, s, sz, epsilon, eta, opts)
%COMPUTE_GWCE Global worst-case error via the local problem at zero data.
% Uses the same model-mismatch and noise bounds as the recovery experiment.

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    opts.method = 'sextic';

    [tau_zero, info] = solve_opt_param( ...
        S, zeros(size(S, 1), 1), s, sz, epsilon, eta, opts);
    info.tau_zero = tau_zero;
    info.gwce_sq = info.lwce_sq;
    gwce = sqrt(info.gwce_sq);
end
