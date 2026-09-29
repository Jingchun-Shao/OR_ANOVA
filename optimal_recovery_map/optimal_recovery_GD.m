%%% Recover the tensor by gradient descent 
%  min_f omega * ||f(idx) - y|| + (1-omega) * ||(I- P)f||. P is the
%  orthogonal projection to the subspace spanned by low-order cluster basis.

function x = optimal_recovery_GD(S, y, s, omega, sz, x_initial, max_epoch, tol_grad)
%OPTIMAL_RECOVERY_GD  Gradient descent solver for
%   min_f  omega * ||S f - y||^2 + (1-omega) * ||(I - P)f||^2,
% where P is the orthogonal projection onto the low-order cluster space.
%
% INPUTS:
%   S         : M-by-N sampling/selection matrix
%   y         : M-by-1 measurements vector
%   s         : cluster order used in cluster_expansion
%   omega     : weight in (0,1)
%   sz        : size vector of the tensor, e.g. [n n ... n] (1-by-d)
%   x_initial : (optional) initial N-by-1 guess (default: zeros(N,1))
%   max_epoch : (optional) max number of GD iterations (default: 1e3)
%   tol_grad  : (optional) gradient tolerance (default: 1e-8)
%
% OUTPUT:
%   x         : recovered N-by-1 vector

    % ---------------- Basic dimensions ----------------
    N = prod(sz);           % total number of tensor entries
    % d = numel(sz);       % dimension (currently not used)

    % ---------------- Defaults for optional inputs ----------------
    if nargin < 6 || isempty(x_initial)
        x_initial = zeros(N, 1);
    end

    if nargin < 7 || isempty(max_epoch)
        max_epoch = 1e3;
    end

    if nargin < 8 || isempty(tol_grad)
        tol_grad = 1e-8;
    end

    % ---------------- objective & gradient ------------------------------
    IminusP = @(x) x - cluster_expansion(x, s, sz);
    obj    = @(x) omega*norm(S*x - y)^2 + (1-omega)*norm(IminusP(x))^2;
    H_apply = @(v) omega*(S'*(S*v))   + (1-omega)*(IminusP(v)) ;
    grad = @(v) 2 * H_apply(v) - 2 * omega * S' * y;


    % ----------------Gradient descent----------------------
    % Starting point
    x = x_initial;
    fprintf('Init: obj=%.3e, ||grad||=%.3e\n', obj(x), norm(grad(x)));
    
    % Gradient descent
    for epoch = 1:max_epoch
        gk   = grad(x);
    
        % exact steepest-descent step for this quadratic: alpha = (g'g)/(g'(2H)g)
        Hgk  = H_apply(gk);
        denom = 2 * max(1e-16, gk'*Hgk);
        alpha = (gk'*gk) / denom;
    
        xnew = x - alpha * gk;
    
        %  Stop when grad < tol_grad
        if norm(gk) <= tol_grad*(1+norm(y))
            x = xnew;
            fprintf('Stop (grad) at %d: obj=%.3e, ||grad||=%.3e\n', epoch, obj(x), norm(gk));
            break;
        end

        x = xnew;
    
        % %Print out the intermediate values every 50 epochs
        % if mod(epoch,50)==0
        %     fprintf('iter %4d: obj=%.3e, ||grad||=%.3e, alpha=%.3e\n', ...
        %             epoch, obj(x), norm(gk), alpha);
        % end
    end
    
    if epoch==max_epoch
        fprintf('Reached max iters. obj=%.3e, ||grad||=%.3e\n', obj(x), norm(grad(x)));
    end
    
    
end



