function test_sparse_projected(cov_dir)
% TEST_SPARSE_PROJECTED  Acceptance tests for compute_weights_sparse_projected.
%
%   test_sparse_projected(cov_dir)
%
%   cov_dir must contain accept_cov_sub-01.mat / accept_cov_sub-03.mat with
%   C_xx, C_xy, D_xy, gamma (training fold) and C_xy_te, D_xy_te (held-out
%   fold), exported from the real peak-sphere data (N = 1600, fold 1; same
%   subsample as the 2026-08-31 diagnostic of the superseded solver).
%
%   Tests
%     T1  constraint:  |train confound| <= 1e-10 * |train signal| at every
%         sparsity level (the superseded solver: violations of 0.02-0.12)
%     T2  sparsity:    achieved fraction of zeros == nominal (within 1/p)
%         (superseded solver: 0% zeros at nominal 10-50% for sub-03)
%     T3  dense limit: at sparsity = 0, initialized at the dense CRM
%         solution (the intended usage), the solver stays there
%         (|C-metric cosine| >= 0.9999). A perturbed-init run is reported
%         alongside without pass/fail: like any biconvex alternating scheme
%         (incl. Witten's PMD) the iteration can settle on a partial
%         optimum from arbitrary starts -- that is why the solver is always
%         initialized from the dense solution in practice.
%     T4  report:      held-out signal/confound across sparsity levels,
%         side by side with the superseded solver (no pass/fail; the
%         diagnostic showed conf/sig ~0.96 for sub-03 = the "collapse")
%     T5  CCA guard:   synthetic problem with negligible D_xy runs in
%         cca_mode without error
%
%   Uses (does not modify) the existing toolbox for the dense init and the
%   superseded sparse solver.

arguments
    cov_dir (1,1) string
end

here     = fileparts(mfilename('fullpath'));            % .../crm/sandbox
crm_root = fileparts(here);                             % .../crm
addpath(genpath(fullfile(crm_root, 'crm')));            % dense + old sparse solver
addpath(genpath(fullfile(crm_root, 'utils')));

SPARSITIES = [0.1 0.3 0.5 0.7 0.9];
OLD_LEVELS = [0.3 0.7];       % superseded solver, for the T4 contrast only
n_fail = 0;

for sid = ["sub-01", "sub-03"]
    S = load(fullfile(cov_dir, sprintf('accept_cov_%s.mat', sid)));
    p = size(S.C_xx, 1);
    fprintf('\n================ %s (p = %d, gamma = %.4f) ================\n', ...
        sid, p, S.gamma);

    % Dense reference (toolbox solver), shared init for all sparse runs
    [wx_d, wy_d, lambda3] = compute_weights_init_rand( ...
        S.C_xx, S.C_xx, S.C_xy, S.D_xy, gamma=S.gamma, k=0);
    Cr = S.C_xx + S.gamma * eye(p);
    fprintf('dense ref: lambda3=%+.4f  train sig=%+.5f conf=%+.2e  test sig=%+.5f conf=%+.5f\n', ...
        lambda3, wx_d' * S.C_xy * wy_d, wx_d' * S.D_xy * wy_d, ...
        wx_d' * S.C_xy_te * wy_d, wx_d' * S.D_xy_te * wy_d);

    % ---- T1 + T2 + T4: sparse runs ----
    fprintf('\n%-8s %-10s %-12s %-12s %-11s %-11s %-6s %-5s %-6s %-4s\n', 'nominal', ...
        'zeros', 'train sig', 'train conf', 'test sig', 'test conf', ...
        'iters', 'conv', 'froze', 'flag');
    for s = SPARSITIES
        [wx, wy, info] = compute_weights_sparse_projected( ...
            S.C_xx, S.C_xx, S.C_xy, S.D_xy, sparsity=s, gamma=S.gamma, ...
            w_x0=wx_d, w_y0=wy_d);
        te_sig  = wx' * S.C_xy_te * wy;
        te_conf = wx' * S.D_xy_te * wy;
        fprintf('%-8.1f %-10.4f %+-12.5f %+-12.2e %+-11.5f %+-11.5f %-6d %-5d %-6d %-4d\n', ...
            s, info.zeros_x, info.train_signal, info.train_confound, ...
            te_sig, te_conf, info.iters, info.converged, info.froze_at, ...
            info.n_flagged);

        % T1: constraint
        if abs(info.train_confound) > 1e-10 * max(abs(info.train_signal), eps)
            fprintf('  FAIL T1 (constraint): |conf| = %.2e\n', abs(info.train_confound));
            n_fail = n_fail + 1;
        end
        % T2: achieved sparsity (allow n_expand rescued coordinates)
        if abs(info.zeros_x - s) > (1 + info.n_expand) / p
            fprintf('  FAIL T2 (sparsity): achieved %.4f vs nominal %.1f\n', ...
                info.zeros_x, s);
            n_fail = n_fail + 1;
        end
        % Convergence: must settle, if necessary via the support freeze
        if ~info.converged
            fprintf('  FAIL (no convergence even with frozen support)\n');
            n_fail = n_fail + 1;
        end
    end

    % ---- T3: dense limit, initialized at the dense solution ----
    [wx, wy, info] = compute_weights_sparse_projected( ...
        S.C_xx, S.C_xx, S.C_xy, S.D_xy, sparsity=0, gamma=S.gamma, ...
        w_x0=wx_d, w_y0=wy_d);
    cos_x = abs(wx' * Cr * wx_d);
    cos_y = abs(wy' * Cr * wy_d);
    fprintf('\nT3 dense limit (dense init): |cos_x| = %.6f, |cos_y| = %.6f, sig = %+.5f (ref %+.5f), %d iters\n', ...
        cos_x, cos_y, info.train_signal, wx_d' * S.C_xy * wy_d, info.iters);
    if min(cos_x, cos_y) < 0.9999
        fprintf('  FAIL T3 (dense limit not preserved)\n');
        n_fail = n_fail + 1;
    end

    % Perturbed-init behavior, reported only: biconvex alternation can
    % settle on a partial optimum from arbitrary starts (expected; same
    % status as Witten's PMD -- hence dense init / multi-init in practice).
    rng(7);
    pert = 0.5 * randn(p, 1);
    wx0 = wx_d + pert / sqrt(pert' * Cr * pert);
    wx0 = wx0 / sqrt(wx0' * Cr * wx0);
    [wx, wy, info] = compute_weights_sparse_projected( ...
        S.C_xx, S.C_xx, S.C_xy, S.D_xy, sparsity=0, gamma=S.gamma, ...
        w_x0=wx0, w_y0=wx0);
    fprintf(['T3 report (perturbed init): |cos_x| = %.6f, sig = %+.5f ' ...
             '(%.1f%% of ref) -- partial optimum, expected\n'], ...
        abs(wx' * Cr * wx_d), info.train_signal, ...
        100 * info.train_signal / (wx_d' * S.C_xy * wy_d));

    % ---- T4 contrast: superseded solver on the same matrices ----
    fprintf('\nsuperseded solver (frozen lambda3 + L1 budget), same matrices:\n');
    for s = OLD_LEVELS
        [wxo, wyo] = compute_weights_sparse_init_rand( ...
            S.C_xx, S.C_xx, S.C_xy, S.D_xy, sparsity=s, gamma=S.gamma, k=0);
        fprintf(['  s=%.1f: zeros=%.4f  train sig=%+.5f conf=%+.5f  ' ...
                 'test sig=%+.5f conf=%+.5f\n'], ...
            s, mean(wxo == 0), wxo' * S.C_xy * wyo, wxo' * S.D_xy * wyo, ...
            wxo' * S.C_xy_te * wyo, wxo' * S.D_xy_te * wyo);
    end
end

% ---- T5: CCA guard on a synthetic problem with negligible confound ----
rng(1);
ps = 40;
A = randn(ps); Cs = A * A' / ps + eye(ps);
Cxys = randn(ps) / ps;
Dneg = 1e-9 * randn(ps) / ps;                 % ||C||/||D|| >> 10
[wxs, wys, info] = compute_weights_sparse_projected( ...
    Cs, Cs, Cxys, Dneg, sparsity=0.5, gamma=0.1, ...
    w_x0=randn(ps,1), w_y0=randn(ps,1));
fprintf('\nT5 CCA guard: cca_mode=%d, zeros=%.2f, converged=%d\n', ...
    info.cca_mode, info.zeros_x, info.converged);
if ~info.cca_mode
    fprintf('  FAIL T5 (CCA guard not triggered)\n');
    n_fail = n_fail + 1;
end

fprintf('\n==== %d test failure(s) ====\n', n_fail);
end
