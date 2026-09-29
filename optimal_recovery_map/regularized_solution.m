function [f_tau, info] = regularized_solution(S, y, s, tau, sz, verbose, max_iter, tol)
%REGULARIZED_SOLUTION Matrix-free regularized recovery using the model basis.
%
%   [f_tau, info] = REGULARIZED_SOLUTION(S, y, s, tau, sz)
%
%   Solves the regularized recovery problem
%
%       min_f tau*||S*f-y||^2 + (1-tau)*||(I-P_s)*f||^2
%
%   under orthonormal point observations. Following Foucart and Liao,
%   the solution has the form
%
%       f_tau = tau*S'*a + U_s*b,
%
%   where U_s is an orthonormal Fourier basis for V_s, C = S*U_s,
%   a = y-C*b, and b = (C'*C)^(-1)*C'*y when C has full column rank.
%   We compute b by CG on the equivalent least-squares problem,
%   applying C and C' without storing C.
%
%   Inputs:
%     S         : M-by-N selection matrix
%     y         : M-by-1 observation vector
%     s         : cluster order
%     tau       : regularization weight in [0,1]
%     sz        : tensor size vector
%     verbose   : optional true/false for printing dimensions (default: false)
%     max_iter  : optional maximum CG iterations (default: 1000)
%     tol       : optional CG stopping tolerance (default: 1e-10)
%
%   Outputs:
%     f_tau     : recovered N-by-1 vector
%     info      : dimensions, coefficients, and CG convergence details

    if nargin < 6 || isempty(verbose)
        verbose = false;
    end
    if nargin < 7 || isempty(max_iter)
        max_iter = 1000;
    end
    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    [M, N] = size(S); % S is a M by N selection matrix
    if N ~= prod(sz) || numel(y) ~= M
        error('S and y dimensions must match the tensor grid.');
    end
    
    % Build the matrix-vec multiplication function
    [apply_C, apply_Cstar, apply_U, mask] = ...
        build_fast_operator(S, s, sz);
    r = nnz(mask); % dimnesion of low interaction space V_s

    % Use CG to solve for b
    [b, converged, iterations] = cg_least_squares( ...
        apply_C, apply_Cstar, y, r, max_iter, tol);

    % Assemble the regularized solution
    f_model = apply_U(b);
    a = y - S * f_model;
    f_tau = f_model + tau * (S' * a);

    if verbose
        fprintf('Ambient dimension N = %d\n', N);
        fprintf('Model dimension r = %d\n', r);
        fprintf('Number of samples M = %d\n', M);
    end

    info.N = N;
    info.M = M;
    info.r = r;
    info.tau = tau;
    info.b = b;
    info.a = a;
    info.converged = converged;
    info.iterations = iterations;
    info.relative_normal_residual = ...
        norm(apply_Cstar(a)) / max(norm(apply_Cstar(y)), eps);
    info.relative_data_residual = norm(a) / max(norm(y), eps);
end

function [x, converged, iterations] = cg_least_squares(apply_C, apply_Cstar, y, model_dim, max_iter, tol)
%CG_LEAST_SQUARES Matrix-free CG for min_x ||C*x-y||_2.
% Solve H*x = b with H = C'*C and b = C'*y.

    H_apply = @(v) apply_Cstar(apply_C(v));
    b = apply_Cstar(y);
    x = zeros(model_dim, 1);
    r = b;  % x starts at zero
    p = r;
    rs_old = r' * r;
    b_norm = norm(b);
    converged = (b_norm == 0);
    iterations = 0;

    while ~converged && iterations < max_iter
        Hp = H_apply(p);
        alpha = rs_old / (p' * Hp);
        x = x + alpha * p;
        r = r - alpha * Hp;
        rs_new = r' * r;
        iterations = iterations + 1;

        converged = (sqrt(rs_new) <= tol * max(1, b_norm));
        if ~converged
            beta = rs_new / rs_old;
            p = r + beta * p;
            rs_old = rs_new;
        end
    end
end

function [apply_C, apply_Cstar, apply_U, mask] = build_fast_operator(S, s, sz)
%BUILD_FAST_OPERATOR Matrix-free actions for C = S*U_s and its adjoint.
% S must select distinct tensor cells, one per row.

    N = prod(sz);
    mask = build_cluster_mask(sz, s);
    mask = mask(:);

    apply_U = @U;
    apply_C = @(b) S * apply_U(b);
    apply_Cstar = @(sampled) U_adjoint(S' * sampled);

    function f = U(b)
        embedding = zeros(N, 1);
        embedding(mask) = b;
        f = sqrt(N) * ifftn(reshape(embedding, sz));
        f = f(:);
    end

    function b = U_adjoint(f)
        spectrum = fftn(reshape(f, sz)) / sqrt(N);
        b = spectrum(mask);
    end
end

function mask = build_cluster_mask(sz, s)
%BUILD_CLUSTER_MASK Keep Fourier modes involving at most s axes.

    count_power_vec = 1;
    for i = 1:numel(sz)
        vi = 2 * ones(sz(i), 1);
        vi(1) = 1;
        count_power_vec = kron(vi, count_power_vec);
    end
    mask = reshape(round(log2(count_power_vec)) <= s, sz);
end
