classdef anova_recursively < handle
    properties
        X           % original tensor
        dim         % number of dims
        s           % The highest interaction order in ANOVA decomposition
        terms       % cell(2^d, 1) storing tensors (or [])
        mu          % cached global mean (order 0)
        isDone = false
    end

    %
    methods
        function obj =  anova_recursively(X)
            % Building the object
            obj.X = X;
            obj.dim = ndims(X);
            obj.terms = cell(2^obj.dim, 1);
            obj.mu = [];
        end
    
        % Do Anova decomposition recursively to the order of s
        function decompose(obj, s)
            obj.s = s;  % Record the highest interaction order by this ANOVA

            % Retrieve the data
            x = obj.X;
            d = obj.dim;

            for p = 0:s
        
                if p == 0
                    % No active dims: the global mean
                    nonactiveIdx = 1:d;
                    obj.mu = obj.mean_over(x, nonactiveIdx);
                    mask = uint64(0);
                    obj.terms{double(mask)+1} = obj.mu;
                    continue
                end
        
                % All subsets of size s
                combos = nchoosek(1:d, p);
        
                for k = 1:size(combos,1)
                    activeIdx    = combos(k,:);                % active dimensions
                    nonactiveIdx = setdiff(1:d, activeIdx);    % average over others
        
                    % Start with the mean over nonactive dims
                    component = obj.mean_over(x, nonactiveIdx);
        
                    % Subtract contributions of all lower-order subsets of activeIdx
                    for i = 0:(p-1)
                        % Fix the bug that nchoosek cannot accept k=0
                        if i == 0
                            component = component - obj.mu;
                            continue
                        else
                            C = nchoosek(activeIdx, i);           % subsets of size i
                            for j = 1:size(C, 1)
                                activeIdx_P  = C(j, :);           % could be empty when i=0
                                maskInt_P    = obj.activeIdx2maskInt(activeIdx_P);
                                component    = component - obj.terms{double(maskInt_P)+1};
                            end
                            % Note: The subset can also be found by inner product
                            % of mask. See code based on Simon's code
                        end
                    end
        
                    % Store the s-way interaction component
                    maskInt = obj.activeIdx2maskInt(activeIdx);
                    obj.terms{double(maskInt)+1} = component;
                end
            end

            obj.isDone = true;
        end


        function y = reconstruct(obj)
            % Sum up all the ANOVA terms up to the order of s
            y = zeros(size(obj.X));

            for i = 1:2^obj.dim
                if ~isempty(obj.terms{i})
                    y = y + obj.terms{i};  % Add each component 
                end
            end
        end

    end

    % Private method
    methods (Static, Access = private)

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


        function maskInt = activeIdx2maskInt(activeIdx)
            maskInt = sum(2.^(activeIdx - 1));
        end

    end


end
    



 


