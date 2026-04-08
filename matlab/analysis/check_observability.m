function [is_observable, rank_O, cond_num, eigenvalues] = check_observability(O, tol)
% CHECK_OBSERVABILITY Check if the system is observable given the observability Gramian.
%
%   [is_observable, rank_O, cond_num, eigenvalues] = check_observability(O, tol)
%
%   Inputs:
%       O   - [3x3] Observability Gramian
%       tol - Numerical tolerance for rank determination (default: 1e-6)
%
%   Outputs:
%       is_observable - logical, true if rank(O) = 3
%       rank_O        - Numerical rank of O
%       cond_num      - Condition number (lambda_max / lambda_min)
%       eigenvalues   - [3x1] Eigenvalues sorted descending
%
%   See also: compute_observability_gramian

    if nargin < 2; tol = 1e-6; end

    eigenvalues = sort(eig(O), 'descend');
    rank_O = sum(eigenvalues > tol * eigenvalues(1));
    is_observable = (rank_O == 3);

    if eigenvalues(3) > 0
        cond_num = eigenvalues(1) / eigenvalues(3);
    else
        cond_num = Inf;
    end
end
