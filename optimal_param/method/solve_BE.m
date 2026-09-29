function [tau_star, info] = solve_BE(S, y, s, sz, eps, eta, opts)
%OPTIMAL_PARAM.SOLVE_BE Solve optimal tau via Beck-Eldar SDP.
%
% Requires CVX in MATLAB path.

    if nargin < 7 || isempty(opts), opts = struct(); end 

    N = prod(sz);
    P_V_perp = @(x) x - cluster_expansion(x, s, sz);

    I_mat = eye(N);
    P_mat = zeros(N);
    for i = 1:N
        P_mat(:, i) = P_V_perp(I_mat(:, i));
    end

    [tau_star, cheb_center, wce] = run_BE(eps, eta, y, P_mat, S);

    info = struct();
    info.method = "BE";
    info.converged = true;
    info.iterations = 1;
    info.wce = wce;
    info.wce_sq = wce^2;
    info.cheb_center = cheb_center;
end

function [tau, cheb_center, wce] = run_BE(epsilon, eta, y, P, L)
    % Check cvx and clear the enviroment
    if exist('cvx_begin', 'file') ~= 2
        error('optimal_param:solve_BE:CVXMissing', ...
            ['CVX is required for BE solver but was not found in path. ', ...
             'Install CVX and run cvx_setup before using opts.method="BE".']);
    end
    cvx_clear

    N = size(P, 1);

    % Core algorithm
    cvx_begin quiet
        cvx_solver sedumi
        cvx_precision best

        variable c nonnegative
        variable d nonnegative
        variable t nonnegative
        minimize( epsilon^2*c - (norm(y)^2 - eta^2)*d + t )
        subject to
            c*P + d*(L'*L) - eye(N) == hermitian_semidefinite(N);
            [c*P + d*(L'*L), -d*L'*y; -d*y'*L, t] == hermitian_semidefinite(N+1);
    cvx_end

    tau = d / (c + d);
    cheb_center = (c*P + d*(L'*L)) \ (d*L'*y);
    wce = sqrt(cvx_optval);
end
