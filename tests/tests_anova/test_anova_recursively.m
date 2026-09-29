%%%%% This file aims to implement the Anova decomposition for tensor

% Generate a tensor
d = 7; 
n = 5; % grid size along each dimension
x = rand(n * ones(1, d)); 

% Compute ANOVA components
D = anova_recursively(x); % The ANOVA instance
D.decompose(d); % Do full ANOVA decomposition

y = D.reconstruct(); % Sum up to the order of s
terms = D.terms; % Retrieve all the terms


% 1. Check reconstruction error
err = norm(y(:) - x(:)) / norm(x(:));
fprintf('Relative reconstruction error = %.2e\n', err);


% List to record the errors
error_list = [];

% 2. Check orthogonality of ANOVA terms
for i = 2:2^d
    term_A = terms{i};
    for j = 1:i-1
        term_B = terms{j};

        % Compute inner product as mean over all dimensions
        Inner_product = mean_over(term_A .* term_B, 1:d);

        % Record
        error_list = [error_list; abs(Inner_product)]; %#ok<AGROW>

        % Optional: only warn if large
        if abs(Inner_product) > 1e-12
            warning('Terms %d and %d are NOT orthogonal (Inner product = %g)', i, j, Inner_product);
        end
    end
end

% Summary statistics
mean_err = mean(error_list);
max_err  = max(error_list);
fprintf('Orthogonality check complete.\n');
fprintf('Mean |Inner product| = %.2e\n', mean_err);
fprintf('Max  |Inner product| = %.2e\n', max_err);


function Amean = mean_over(A, dims)
%MEAN_OVER  Take mean over a list of dimensions, keeping singleton sizes.
%
%   Amean = MEAN_OVER(A, dims)
%   averages array A over all dimensions in the vector dims.
%   Each averaged dimension is replaced by a singleton (size 1).
%
%   Example:
%       A = rand(3,3,3);
%       B = mean_over(A, [2, 3]);  % average over 2nd and 3rd dims
%       size(B)   % returns [3 1 1]

    if isempty(dims)
        Amean = A; 
        return;
    end

    % sort descending to avoid index shifts as dimensions collapse
    dims = sort(dims, 'descend');  
    Amean = A;
    for d = dims
        Amean = mean(Amean, d);   % MATLAB mean() keeps singleton dim size
    end
end

