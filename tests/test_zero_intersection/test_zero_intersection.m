%% Setup
d = 7;                  % dimension
n = 2;                  % grid size
s = 2;                  % effective dimension
sz = n * ones(1, d);    % tensor size
N = prod(sz);           % dimension after vectorization

eta = 0.01;             % noise level 
eps = 0.05;             % model mismatch bound

%% Observation operator ( need to be refined)
P_V = @(x) cluster_expansion(x, s, sz);

% 1. Build the matrix for the projection onto V^\perp
P_V_mat = build_operator_matrix(P_V, N);

% 2. Compute an orthonormal basis of V via SVD
% Since P_V_mat = U * U',
% the singular vectors corresponding to singular value 1 span V.
[U_svd, S_svd, ~] = svd(P_V_mat, 'econ');
sing_vals = diag(S_svd);

tol = 1e-10;
idx_V = sing_vals > 1 - tol;   % singular values close to 1
V_basis = U_svd(:, idx_V);     % orthonormal basis of V
dim_V = size(V_basis, 2);


%% Observation operator S
rho = 0.5;                          % sampling rate
M = max(1, round(rho * N));         % number of measurements
idx = randperm(N, M);               % sampled indices
S = sparse(1:M, idx, 1, M, N);

%% Check the condition: V \cap ker(Lambda) = {0}

% Construct the Gram matrix for the restricted operator
Gram = V_basis' * S' * S * V_basis; 

% Check the rank of the Gram matrix
rank_Gram = rank(full(Gram));

fprintf('Dimension of V: %d\n', dim_V);
fprintf('Rank of Gram matrix: %d\n', rank_Gram);

if rank_Gram == dim_V
    disp('Success: V \cap ker(Lambda) = {0} is satisfied.');
else
    disp('Failed: V \cap ker(Lambda) contains non-zero vectors.');
end

%% Helper Functions
function A = build_operator_matrix(func_handle, N)
% Build the matrix of a linear operator from its action on basis vectors.
    A = zeros(N, N);
    I = eye(N); 
    for j = 1:N
        A(:,j) = func_handle(I(:, j));
    end
end