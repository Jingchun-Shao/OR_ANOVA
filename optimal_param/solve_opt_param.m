function [tau_star, info] = solve_opt_param(S, y, s, sz, eps, eta, opts)
%OPTIMAL_PARAM.SOLVE Unified entrypoint for optimal-parameter solvers.
%
%   [tau_star, info] = optimal_param.solve(S, y, s, sz, eps, eta, opts)
%
%   opts.method:
%       "Newton" (default) - Newton method with lambda estimation
%       "sextic"       - Solve the sextic equation (requires Symbolic Math Toolbox)
%       "BE"           - Beck-Eldar SDP (requires CVX)
%   For "sextic", opts.eigenvalue_method is "power" (default) or "svd".
%   The "svd" option forms a dense N-by-N matrix.

    if nargin < 7 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts, 'method')
        opts.method = "Newton";
    end

    switch string(opts.method)
        case "Newton"
            [tau_star, info] = solve_Newton(S, y, s, sz, eps, eta, opts);
        case "sextic"
            [tau_star, info] = solve_sextic(S, y, s, sz, eps, eta, opts);
        case "BE"
            [tau_star, info] = solve_BE(S, y, s, sz, eps, eta, opts);
        otherwise
            error('optimal_param:solve:UnknownMethod', ...
                'Unknown opts.method "%s". Use "Newton", "sextic", or "BE".', ...
                string(opts.method));
    end
end
