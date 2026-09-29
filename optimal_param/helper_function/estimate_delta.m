function delta = estimate_delta(S, y, s, sz, P_V_perp)
% Estimate delta using omega ~ 1:
% delta = ||P_V_perp(f)||

    if ~any(y)
        delta = 0;
        return;
    end

    max_epoch = 1e5;
    tol_residue = 1e-10;

    N = prod(sz);
    x0 = zeros(N, 1);
    omega = 1 - 1e-6;

    f = optimal_recovery_CG(S, y, s, omega, sz, x0, max_epoch, tol_residue, false);
    delta = norm(P_V_perp(f));
end
