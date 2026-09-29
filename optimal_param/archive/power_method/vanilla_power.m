%%%%% Find the minimal eigenvalue of a symmetric matrix with variational
%%%%% formation. In the following I implicitly assume the eigenvalues
%%%%% range from [0,1]

% Settup
N = 100; % dim
max_iter = 10000;% Max iteration

% Todo: analysis for the minimum iterations to guarantee accuracy

% construct a symmetric matrix with spectrum in [0,1]
[Q, ~] = qr(randn(N));           % random orthogonal matrix
lambda = rand(N, 1); % eigenvalues in [0,1]
D = diag(lambda);
A = Q * D * Q';    % Symmetric matrix

% define matrix-vector product for A
A_func = @(x) A * x;

% The power method
[lambda_min_approx, u_min_approx] = smallest_eig_sym(A_func, max_iter, N);

% Step 4: compare with groundtruth
lambda_min_truth = min (lambda);

fprintf('Groundtruth smallest eigenvalue   : %.8f\n', lambda_min_truth);
fprintf('Approx smallest eigenvalue  : %.8f\n', lambda_min_approx);
fprintf('Relative error              : %.2e\n', ...
        abs(lambda_min_truth - lambda_min_approx) / lambda_min_truth);


function [lambda_min, u_min] = smallest_eig_sym(A_mult, maxit, N)
    % Since the spectrum ranges from[0, 1] and A is symmetric, finding the
    % smallest eigenvalue of A is equivalent to finding the largest
    % eigenvalue of (I - A)
    
    % Form helper function (I-A)
    B = @(x) x - A_mult(x);

    u0 = randn(N, 1);
    u = u0 / norm(u0); % normalize the initial vector
    
    % Use the power method to find the max e.v. of I-A
    for k = 1:maxit
        v  = B(u);  % B*u = (I - A)*u
        u  = v / norm(v);
    end

    mu_max_B = u' * B(u);   % largest eigenvalue of I-A 

    % Find the min e.v. of A
    lambda_min = 1 - mu_max_B;
    u_min = u;
end
