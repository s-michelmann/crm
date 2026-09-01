function [w_x, w_y, info] = compute_weights_sparse_projected( ...
        C_xx, C_yy, C_xy, D_xy, options)
% COMPUTE_WEIGHTS_SPARSE_PROJECTED  Sparse CRM by block ascent with
% support-restricted null-space projection (prototype, "Option 2").
%
%   [w_x, w_y, info] = compute_weights_sparse_projected(C_xx, C_yy, C_xy, D_xy, ...
%                          sparsity=0.7, gamma=g)
%
%   Solves, approximately,
%       max  w_x' C_xy w_y
%       s.t. w_x' D_xy w_y = 0,          (confound constraint)
%            w_x' C_xx w_x = 1,  w_y' C_yy w_y = 1,
%            frac. of zero entries of w_x, w_y  =  sparsity.
%
%   WHY THIS SOLVER EXISTS
%   The original sparse solver (compute_weights_sparse_init_rand) freezes the
%   constraint multiplier lambda3 at its dense-init value and runs Euclidean
%   power iteration with an L1 budget checked before normalization. Its fixed
%   points (i) violate the constraint by 3-4 orders of magnitude in-sample and
%   (ii) do not achieve the nominal sparsity (0% zeros at nominal 10-50% for
%   some data). This solver fixes both:
%     - block-coordinate ascent: each half-step solves
%           max_w  w' C_xy w_fixed   s.t.  w' D_xy w_fixed = 0,  w' C w = 1
%       in CLOSED FORM, with the multiplier recomputed from the current
%       iterate every half-step (no frozen lambda3);
%     - updates are preconditioned by (C + gamma*I)^-1, so without
%       thresholding the fixed point is the dense CRM solution (generalized
%       singular pair), not the Euclidean SVD pair;
%     - sparsity is enforced by soft top-k thresholding (keep the k largest
%       magnitudes, shrink them by the (k+1)-th), so the achieved fraction of
%       zeros equals the nominal one by construction;
%     - after thresholding, the constraint is re-enforced by a projection
%       RESTRICTED TO THE SUPPORT: only already-nonzero coordinates are
%       touched, so exact zeros survive and the constraint holds exactly.
%       This is the last optimization step before normalization
%       (normalization is a pure scaling and preserves both properties).
%
%   INPUTS
%     C_xx, C_yy : [p x p], [q x q] auto-covariance matrices
%     C_xy       : [p x q] signal cross-covariance
%     D_xy       : [p x q] confound cross-covariance
%
%   OPTIONS
%     sparsity  : target fraction of zero entries (0-1). 0 = dense. [0.7]
%     gamma     : ridge added to C_xx, C_yy (same metric as dense solver) [0]
%     w_x0,w_y0 : initial vectors. Empty -> dense CRM init via
%                 compute_weights_init_rand (must be on the path). [empty]
%     max_iter  : maximum outer iterations [500]
%     tol       : convergence tolerance on ||w - w_old|| [1e-8]
%     n_inner   : threshold->project passes per half-step [2]
%     cca_ratio : if ||C_xy||_F / ||D_xy||_F exceeds this, the confound is
%                 negligible and the constraint steps are skipped entirely
%                 (mirrors the dense solver's CCA fallback) [10]
%     freeze_frac : fraction of max_iter after which, if not yet converged,
%                 the supports are FROZEN at their current pattern and only
%                 the magnitudes keep iterating. Needed because top-k
%                 re-selection can cycle between two supports whose scores
%                 straddle the cut; with a fixed support the iteration is an
%                 alternation of linear maps and settles. [0.6]
%     guard_tol : relative tolerance for the degenerate-support guard [1e-8]
%     max_expand: max coordinates added when rescuing a degenerate support,
%                 as a fraction of the support size [0.1]
%
%   OUTPUTS
%     w_x, w_y : sparse weight vectors, unit norm in the (C + gamma*I) metric
%     info     : struct with fields
%       .iters            outer iterations used
%       .converged        true if tol reached before max_iter
%       .train_signal     w_x' C_xy w_y at the solution
%       .train_confound   w_x' D_xy w_y at the solution (should be ~1e-15)
%       .zeros_x,.zeros_y achieved fraction of zeros
%       .cca_mode         true if the CCA guard disabled the constraint
%       .n_expand         support-rescue expansions performed
%       .n_flagged        half-steps where the constraint could not be
%                         enforced within the (expanded) support; a nonzero
%                         count means .train_confound may be inflated
%       .froze_at         iteration at which the supports were frozen
%                         (0 = never needed)
%
%   NOTE ON INITIALIZATION: like all biconvex alternating schemes (including
%   Witten et al.'s PMD), this converges to a partial (coordinatewise)
%   optimum. The dense CRM solution is the joint optimum and a fixed point
%   of these dynamics at sparsity = 0, so ALWAYS initialize from the dense
%   solution (the default); from arbitrary inits the iteration can settle on
%   block-stationary pairs a few percent below the optimum. For robustness
%   across near-degenerate spectra, wrap with multiple dense inits
%   (compute_weights_multi_rand-style) and select confound-first.
%
%   See also compute_weights_init_rand (dense init),
%            compute_weights_sparse_init_rand (superseded prototype).

arguments
    C_xx double
    C_yy double
    C_xy double
    D_xy double
    options.sparsity  (1,1) double {mustBeInRange(options.sparsity, 0, 1)} = 0.7
    options.gamma     (1,1) double {mustBeNonnegative} = 0
    options.w_x0      double = []
    options.w_y0      double = []
    options.max_iter  (1,1) double = 500
    options.tol       (1,1) double = 1e-8
    options.n_inner   (1,1) double = 2
    options.cca_ratio (1,1) double = 10
    options.guard_tol (1,1) double = 1e-8
    options.max_expand(1,1) double = 0.1
    options.freeze_frac(1,1) double = 0.6
end

p = size(C_xx, 1);
q = size(C_yy, 1);

% Regularized metrics, factored once. All solves below use these Cholesky
% factors; the gamma ridge matches the dense solver so that at sparsity = 0
% the two solvers optimize the identical problem.
Lx = chol(C_xx + options.gamma * eye(p), 'lower');
Ly = chol(C_yy + options.gamma * eye(q), 'lower');

% Number of SURVIVING coordinates per vector. This is the user-facing
% sparsity semantics: nominal fraction of zeros = achieved fraction of zeros.
% (An L1 budget, as in the superseded solver, has no fixed relationship to
% the support size and produced 0% zeros at nominal 50% on real data.)
k_x = max(2, round((1 - options.sparsity) * p));   % >= 2: one coordinate can
k_y = max(2, round((1 - options.sparsity) * q));   % never satisfy the
                                                   % constraint non-trivially

% ---- GUARD 1 (CCA fallback) --------------------------------------------
% If the confound matrix is negligible relative to the signal matrix, the
% constraint is vacuous; enforcing it against numerical noise in D_xy is
% meaningless and the degenerate-support guard would trigger constantly.
% Mirror the dense solver: skip all constraint handling.
cca_mode = norm(C_xy, 'fro') / max(norm(D_xy, 'fro'), 1e-300) > options.cca_ratio;

% ---- Initialization -----------------------------------------------------
% Start from the dense CRM solution: it satisfies the constraint and is the
% fixed point of these dynamics at sparsity = 0, so the sparse solution is
% reached by deforming a valid solution rather than from an arbitrary point.
if isempty(options.w_x0)
    assert(exist('compute_weights_init_rand', 'file') == 2, ...
        'sparse_projected:noInit', ...
        'compute_weights_init_rand not on path and no w_x0/w_y0 given');
    [w_x, w_y] = compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy, ...
        gamma=options.gamma, k=0);
else
    w_x = options.w_x0;
    w_y = options.w_y0;
end

n_expand  = 0;
n_flagged = 0;
converged = false;
froze_at  = 0;
Ax_fix = [];                    % empty = support free (top-k re-selected)
Ay_fix = [];
freeze_iter = round(options.freeze_frac * options.max_iter);

for iter = 1:options.max_iter
    w_x_old = w_x;
    w_y_old = w_y;

    % Support freeze: top-k re-selection can cycle between two supports
    % whose entry magnitudes straddle the cut, preventing convergence while
    % the solution quality is already stable. If the iteration has not
    % converged by freeze_iter, lock the supports at their current pattern;
    % the remaining iterations only adjust magnitudes (an alternation of
    % linear maps, which settles), and the constraint projection continues
    % to operate within the frozen support.
    if iter == freeze_iter && ~converged
        Ax_fix = (w_x ~= 0);
        Ay_fix = (w_y ~= 0);
        froze_at = iter;
    end

    % ---- half-step 1: update w_x with w_y fixed ----
    [w_x, ex, fl] = block_update(C_xy * w_y, D_xy * w_y, Lx, k_x, ...
        cca_mode, options, Ax_fix);
    n_expand = n_expand + ex;  n_flagged = n_flagged + fl;

    % ---- half-step 2: update w_y with w_x fixed ----
    % w_y is updated LAST, so on exit the pair (w_x, w_y) satisfies the
    % constraint exactly: the final operation on w_y was the
    % support-restricted projection (followed only by scaling).
    [w_y, ex, fl] = block_update(C_xy' * w_x, D_xy' * w_x, Ly, k_y, ...
        cca_mode, options, Ay_fix);
    n_expand = n_expand + ex;  n_flagged = n_flagged + fl;

    if norm(w_x - w_x_old) + norm(w_y - w_y_old) < options.tol
        converged = true;
        break
    end
end

% Sign convention: signal non-negative (matches the dense solver).
if w_x' * C_xy * w_y < 0
    w_y = -w_y;
end

info = struct( ...
    'iters',          iter, ...
    'converged',      converged, ...
    'train_signal',   w_x' * C_xy * w_y, ...
    'train_confound', w_x' * D_xy * w_y, ...
    'zeros_x',        mean(w_x == 0), ...
    'zeros_y',        mean(w_y == 0), ...
    'cca_mode',       cca_mode, ...
    'n_expand',       n_expand, ...
    'n_flagged',      n_flagged, ...
    'froze_at',       froze_at);
end


% =========================================================================
function [w, n_expand, flagged] = block_update(c, d, L, k, cca_mode, opt, A_fix)
% One half-step: exact constrained target, then (threshold -> project)
% inner passes, then metric normalization.
%
%   c = C_xy * w_other   (correlation direction, data space)
%   d = D_xy * w_other   (constraint direction, data space)
%   L = chol(C + gamma*I)
%   A_fix = [] for free support (top-k re-selected each pass), or a logical
%           mask locking the support (see freeze_frac in the caller)
%
% Without thresholding this returns the exact maximizer of w'c subject to
% w'd = 0 and w'(C+gamma*I)w = 1 (Lagrangian stationarity gives
% w ~ Cinv*(c - lambda*d) with lambda fixed by the constraint), i.e. the
% multiplier is recomputed HERE, every half-step, from the current iterate —
% this is what the superseded solver's frozen lambda3 failed to do.

n_expand = 0;
flagged  = false;

% Sparsification operator: free support -> soft top-k (support re-selected
% from the current values); frozen support -> hard restriction to the locked
% pattern (magnitudes free, no shrinkage needed since the support is fixed).
if isempty(A_fix)
    sparsify = @(w) soft_topk(w, k);
else
    sparsify = @(w) w .* A_fix;
end

u = L' \ (L \ c);                       % Cinv * c  (two triangular solves)

if cca_mode
    % Constraint disabled (Guard 1): plain thresholded ascent direction.
    w = sparsify(u);
    w = renorm(w, L);
    return
end

v   = L' \ (L \ d);                     % Cinv * d
vd  = v' * d;                           % = d'Cinv d >= 0; ~0 only if d ~ 0
if vd < opt.guard_tol * max(d' * d, 1e-300)
    % d is numerically zero for this w_other (local CCA situation, e.g. a
    % fold where the confound pairing is empty): nothing to constrain.
    w = renorm(sparsify(u), L);
    return
end

lambda = (u' * d) / vd;                 % dynamic multiplier
w_t    = u - lambda * v;                % exact null-space target: w_t'd = 0

w = w_t;
for pass = 1:opt.n_inner
    % (a) Sparsify: soft top-k threshold (zero everything below the
    % (k+1)-th largest magnitude, shrink survivors by it -> exactly k
    % nonzeros, i.e. nominal sparsity = achieved sparsity), or, once the
    % support is frozen, hard restriction to the locked pattern.
    w = sparsify(w);

    % (b) Support-restricted projection — the LAST optimization step:
    % remove the residual constraint violation introduced by (a) using the
    % constraint direction v RESTRICTED to the current support. Off-support
    % coordinates of v are zeroed, so the projection touches only
    % already-nonzero entries: the zeros from (a) survive, and w'd = 0
    % holds exactly afterwards.
    A   = (w ~= 0);
    vA  = zeros(size(v));
    vA(A) = v(A);
    den = vA' * d;

    % ---- GUARD 2 (degenerate support) ----
    % den = sum_{i in A} v_i d_i can be ~0 if the surviving coordinates
    % barely overlap the constraint direction (tiny supports, adversarial
    % thresholding). Dividing by it would blow the solution up. Rescue by
    % expanding the support with the coordinates that contribute most to
    % |den| (largest |v_i .* d_i| off-support), a few at a time, up to
    % max_expand * k extra coordinates. If still degenerate, flag and skip
    % the projection: the constraint is then only approximate for this
    % half-step, and the caller sees it via info.n_flagged.
    if abs(den) < opt.guard_tol * max(vd, 1e-300)
        contrib = abs(v .* d);
        contrib(A) = -Inf;                       % only off-support candidates
        [~, order] = sort(contrib, 'descend');
        budget = max(1, round(opt.max_expand * k));
        for e = 1:budget
            A(order(e)) = true;
            vA(order(e)) = v(order(e));
            % re-seed the added coordinate from the unthresholded target so
            % it participates in the projection with a meaningful value
            w(order(e)) = w_t(order(e));
            den = vA' * d;
            n_expand = n_expand + 1;
            if abs(den) >= opt.guard_tol * max(vd, 1e-300)
                break
            end
        end
        if abs(den) < opt.guard_tol * max(vd, 1e-300)
            flagged = true;
            break                                % keep w as thresholded
        end
    end

    w = w - ((w' * d) / den) * vA;               % exact: w'd = 0
end

% Metric normalization LAST: pure scaling, preserves both the zeros and the
% constraint (w'd = 0 is homogeneous in w).
w = renorm(w, L);
end


function w = soft_topk(w, k)
% Keep the k largest-magnitude entries, soft-shrunk by the (k+1)-th largest.
% delta is data-adaptive, so exactly k entries survive (ties are measure
% zero); using soft (not hard) thresholding keeps the operator continuous,
% which the block iteration needs to converge.
if k >= numel(w)
    return
end
a = sort(abs(w), 'descend');
delta = a(k + 1);
w = sign(w) .* max(abs(w) - delta, 0);
end


function w = renorm(w, L)
% Unit norm in the (C + gamma*I) metric, via the Cholesky factor.
nrm = norm(L' * w);
if nrm < 1e-150
    error('sparse_projected:zeroVector', ...
        'Weight vector collapsed to zero (threshold too aggressive).');
end
w = w / nrm;
end


function mustBeNonnegative(x)
if any(x < 0)
    error('Value must be nonnegative.');
end
end
