function [f_tau, info] = regularized_solution_direct(S, y, s, tau, sz, verbose)
%REGULARIZED_SOLUTION_DIRECT Archived direct recovery using the model basis.
%This version forms the matrices explicitly to solve b
%
%   [f_tau, info] = REGULARIZED_SOLUTION_DIRECT(S, y, s, tau, sz)
%
%   Solves the regularized recovery problem
%
%       min_f tau * ||S f - y||^2 + (1 - tau) * ||(I - P)f||^2
%
%   under orthonormal point observations. The solution has the form
%
%       f_tau = tau * S' a + U b,
%
%   where U is an orthonormal Fourier basis for the low-effective-dimension
%   model space V_s, C = S U, b solves C b ~= y, and a = y - C b.
%
%   Inputs:
%     S         : M-by-N sampling/selection matrix
%     y         : M-by-1 measurements vector
%     s         : effective dimension
%     tau       : regularization weight in [0,1]
%     sz        : tensor size vector, e.g. [n n ... n]
%     verbose   : optional true/false for printing dimensions
%
%   Outputs:
%     f_tau     : recovered N-by-1 vector
%     info      : intermediate quantities

    if nargin < 6 || isempty(verbose)
        verbose = true;
    end

    N = prod(sz);
    [M, N_S] = size(S);

    if N_S ~= N
        error('The number of columns in S must equal prod(sz).');
    end

    if numel(y) ~= M
        error('Length of y must match the number of rows in S.');
    end

    y = y(:);

    % Building basis for V_s
    mask = build_cluster_mask(sz, s);
    active = mask(:); % reshape into vector format
    r = nnz(active);

    U_apply = @(b) apply_U_fourier(b, active, sz);
    Ustar_apply = @(x) apply_Ustar_fourier(x, active, sz);

    if verbose
        fprintf('Ambient dimension N = %d\n', N);
        fprintf('Model dimension r = %d\n', r);
        fprintf('Number of samples M = %d\n', M);
    end

    Ct = zeros(r, M);
    for m = 1:M
        selector_m = full(S(m, :)');
        Ct(:, m) = Ustar_apply(selector_m);
    end

    C = Ct';

    b = C \ y;  % This computes the least square solution
    a = y - C * b;

    f_representer = S' * a;
    f_model = U_apply(b);
    f_tau = tau * f_representer + f_model;

    info = struct();
    info.N = N;
    info.M = M;
    info.r = r;
    info.tau = tau;
    info.active = active;
    info.C = C;
    info.a = a;
    info.b = b;
    info.f_representer = f_representer;
    info.f_model = f_model;
end

%% Helper function to build basis for V_s
function mask = build_cluster_mask(sz, s, r)
%BUILD_CLUSTER_MASK  Construct a boolean mask for cluster order <= s.

    if nargin < 3
        r = 2;
    end

    d = numel(sz);
    count_power_vec = 1;

    for i = 1:d
        vi = r * ones(sz(i), 1, 'double');
        vi(1) = 1;
        count_power_vec = kron(vi, count_power_vec);
    end

    count_order_vec = round(log(count_power_vec) / log(r));
    mask = reshape(count_order_vec <= s, sz);
end

function x = apply_U_fourier(b, active, sz)
%APPLY_U_FOURIER Apply U*b without explicitly forming U.

    N = prod(sz);
    b_full = zeros(N, 1);
    b_full(active) = b;
    B = reshape(b_full, sz);
    X = sqrt(N) * ifftn(B); % scaling to guarantee it's unitary
    x = X(:);
end

function b = apply_Ustar_fourier(x, active, sz)
%APPLY_USTAR_FOURIER Apply U^*x without explicitly forming U.

    N = prod(sz);

    if numel(x) ~= N
        error('Length of x must equal prod(sz).');
    end

    X = reshape(x, sz);
    Xhat = fftn(X) / sqrt(N); % scaling to guarantee it's unitary
    Xhat = Xhat(:);
    b = Xhat(active);
end
