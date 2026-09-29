%%% Recover the tensor by conjugate gradient 
%  min_f omega * ||f(idx) - y||^2 + (1-omega) * ||(I- P)f||^2
%  <=> solve H x = b with H = omega S'*S + (1-omega)(I-P), b = omega S'*y

function x = optimal_recovery_CG(S, y, s, omega, sz, x_initial, max_iter, tol_rel, verbose)
%OPTIMAL_RECOVERY_CG  Conjugate Gradient solver for
%   min_f  omega * ||S f - y||^2 + (1-omega) * ||(I - P)f||^2
%   <=> solve H x = b with
%       H = omega S'*S + (1-omega)(I - P),   b = omega S'*y,
% where P is the orthogonal projection onto the low-order cluster space.
%
% INPUTS:
%   S         : M-by-N sampling/selection matrix
%   y         : M-by-1 measurements vector
%   s         : cluster order used in cluster_expansion
%   omega     : weight in (0,1)
%   sz        : size vector of the tensor, e.g. [n n ... n] (1-by-d)
%   x_initial : (optional) initial N-by-1 guess (default: zeros(N,1))
%   max_iter  : (optional) max number of CG iterations (default: 1e3)
%   tol_rel   : (optional) tolerance on relative residual (default: 1e-8)
%   verbose   : (optional) true/false for printing progress (default: true)
%   
%
% OUTPUTS:
%   x         : recovered N-by-1 vector

    % ---------------- Basic dimensions ----------------
    N = prod(sz); % size of the vector reshaped from the tensor


    % ---------------- Optional inputs & initial guess ----------------
    if nargin < 6 || isempty(x_initial)
        x_initial = zeros(N, 1);
    end

    if nargin < 7 || isempty(max_iter)
        max_iter = 1e3;
    end

    if nargin < 8 || isempty(tol_rel)
        tol_rel = 1e-8;
    end

    if nargin < 9 || isempty(verbose)
        verbose = false;
    end

    % ---------------- objective & operator -------------------------
    IminusP = @(x) x - cluster_expansion(x, s, sz);  % (I - P)x
    obj     = @(x) omega*norm(S*x - y)^2 + (1-omega)*norm(IminusP(x))^2;
    
    % H * v = [omega S'S + (1-omega)(I-P)] v
    H_apply = @(v) omega*(S'*(S*v)) + (1-omega)*IminusP(v);
    
    % Right-hand side for the normal equations: Hx = b
    b = omega * (S' * y);

    % ---------------- Conjugate Gradient Setup ----------------
    x = x_initial;             % Starting point
    Hx = H_apply(x);
    r  = b - Hx;            % initial residual
    p  = r;                 % initial search direction
    rs_old = r' * r; 
    
    if verbose
        fprintf('Init: obj = %.3e, ||r|| = %.3e\n', obj(x), sqrt(rs_old));
    end
    
    for k = 1:max_iter
        Hp = H_apply(p);
        alpha = rs_old / (p' * Hp);
    
        % update x and residual
        x = x + alpha * p;
        r = r - alpha * Hp;
        rs_new = r' * r;
    
        % % (optional) print every 50 iterations
        % if mod(k,1)==0
        %     fprintf('iter %4d: obj=%.3e', ...
        %             k, obj(x));
        % end
    
        if sqrt(rs_new) < tol_rel * max(1, norm(b))
       
            if verbose
                fprintf('CG converged at iter %d: obj = %.3e\n', k, obj(x));
            end

            break;
        end
    
        % CG direction update
        beta = rs_new / rs_old;
        p    = r + beta * p;
        rs_old = rs_new;
    end
    
    if k == max_iter && verbose
        fprintf('CG reached max iters: obj=%.3e\n', ...
                obj(x) );
    end
    
    
end


