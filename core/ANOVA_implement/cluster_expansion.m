
%%% Consider cluster expansion with Fourier basis. 
%%% Then the first basis is 1_m, with the rest acting on span{1_m}^{\perp}.
%%% 
%%% Remark: This is equivalent to Anova, in a sense that they are both doing 
%%% orthogonal projection onto the same tensor product subspace. 

function f = cluster_expansion(g, s, sz)

    if isvector(g)
        % Vector in, vector out
        assert(nargin >= 3, 'For vector input, you must pass sz (tensor size).');
        G     = reshape(g, sz);
        Gproj = tensor_cluster_expansion(G, s);
        f     = Gproj(:);             % vector-out
    else
        % tensor-in, tensor-out
        f = tensor_cluster_expansion(g, s);
    end
end

function f_proj = tensor_cluster_expansion(g, s)
    % FFT -> mask -> IFFT route
    % Output will be the sum of ANOVA terms up to s order
    F      = fftn(g);
    mask   = build_cluster_mask(size(F), s);
    F_proj = mask .* F;
    f_proj = ifftn(F_proj);
end


function mask = build_cluster_mask(sz, s, r)
%BUILD_CLUSTER_MASK  Construct a boolean mask that only keeps term with
%   cluster order less than s
%
%   mask = BUILD_CLUSTER_MASK(sz, s)
%   mask = BUILD_CLUSTER_MASK(sz, s, r)
%
%   Inputs:
%     sz : vector of tensor sizes, e.g. [n n ... n]
%     s  : maximum active dimension order to keep
%     r  : for counting the cluster order
%
%   Output:
%     mask : logical array of size sz, true where order <= s
%
%   Example:
%     F = fftn(rand(5,5,5));
%     mask = build_cluster_mask(size(F), 2);
%     F_masked = F .* mask;

    if nargin < 3
        r = 2;   % default base
    end

    d = numel(sz);
    count_power_vec = 1;   % kronecker product start

    for i = 1:d
        vi = r * ones(sz(i), 1, 'double');
        vi(1) = 1;                % setting vi= [1,r,...,r]
        count_power_vec = kron(vi, count_power_vec);
    end

    % compute interaction order (# of non-DC dimensions)
    count_order_vec = round(log(count_power_vec) / log(r));

    % build mask
    mask = reshape(count_order_vec <= s, sz);
end