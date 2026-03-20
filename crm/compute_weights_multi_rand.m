function [w_x_best, w_y_best, lambda3_best, Wxs, Wys, lambdas, corrs] = ...
    compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy, options)


    % COMPUTE_WEIGHTS_MULTI_RAND
    %
    %   Run multiple random initializations of the dense CRM solver
    %   (compute_weights_init_rand) and return the best solution based on
    %   the achieved cross‑covariance correlation.
    %
    %   This function is a wrapper that:
    %       1. Calls compute_weights_init_rand() n_init times
    %       2. Uses different random seeds (k = 0,1,2,...)
    %       3. Stores all solutions (w_x, w_y, lambda3)
    %       4. Computes the canonical correlation wx' * C_xy * wy
    %       5. Selects and returns the best solution
    %
    %   It is useful when the optimization landscape is non‑convex and
    %   different random starts may converge to different local optima.
    %
    % -------------------------------------------------------------------------
    %   SYNTAX
    %
    %   [w_x_best, w_y_best, lambda3_best, Wxs, Wys, lambdas, corrs] = ...
    %       compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy)
    %
    %   [...] = compute_weights_multi_rand(..., f=2, gamma=0.01, chlsky=true, n_init=20)
    %
    % -------------------------------------------------------------------------
    %   INPUTS
    %
    %   C_xx   : [p × p] SPD covariance matrix for X
    %   C_yy   : [q × q] SPD covariance matrix for Y
    %   C_xy   : [p × q] cross‑covariance matrix
    %   D_xy   : [p × q] confound matrix (to be minimized)
    %
    %   f      : number of canonical components (default = 1)
    %
    %   gamma  : ridge regularization strength passed to
    %            compute_weights_init_rand (default = 0)
    %
    %   chlsky : logical flag
    %            true  → use Cholesky‑based solver for initialization
    %            false → use standard solver
    %
    %   n_init : number of random initializations (default = 10)
    %
    %   rank_ratio_thresh : scalar (default = NaN → auto)
    %            Controls how the best init is selected.  The solver
    %            computes rank(C_xy) / rank(D_xy).  When this ratio exceeds
    %            the threshold, the signal subspace is much larger than the
    %            confound subspace, so strict confound‑first selection is
    %            used (low confound, then max signal).  Below the threshold,
    %            the subspaces nearly overlap, so diff‑based selection is
    %            used instead (max signal − |confound|), tolerating some
    %            confound for higher signal.
    %
    %            NaN (default): auto‑compute threshold as 2, i.e. switch to
    %            diff‑based selection (max signal − |confound|) when
    %            rank(C_xy) < 2 × rank(D_xy), meaning the excess rank
    %            (signal directions beyond the confound) is less than the
    %            confound rank itself.  The difference score avoids ratio
    %            blow‑up when confound fluctuates near zero.
    %
    % -------------------------------------------------------------------------
    %   OUTPUTS
    %
    %   w_x_best     : best canonical weight vector for X
    %   w_y_best     : best canonical weight vector for Y
    %   lambda3_best : best lambda3 value returned by the solver
    %
    %   Wxs          : [p × n_init] matrix of all w_x solutions
    %   Wys          : [q × n_init] matrix of all w_y solutions
    %   lambdas      : [1 × n_init] vector of all lambda3 values
    %   corrs        : [1 × n_init] vector of correlations wx' * C_xy * wy
    %
    % -------------------------------------------------------------------------
    %   NOTES
    %
    %   • Each initialization uses seed (k‑1)
    %
    %   • The "best" solution is selected adaptively based on the rank
    %     ratio of C_xy and D_xy (see rank_ratio_thresh).  When the signal
    %     subspace is large relative to the confound, confound‑first
    %     selection is used.  When data is limited and subspaces overlap,
    %     diff‑based selection (signal − |confound|) is used instead.
    %
    %   • All solutions are returned for inspection of variability
    %     across random starts.
    %
    % -------------------------------------------------------------------------
    %   SEE ALSO
    %       compute_weights_init_rand
    %       compute_weights_sparse_init_rand
    %       compute_weights_sparse_multi_rand

    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        options.f (1,1) double {mustBePositive} = 1
        options.gamma (1,1) double {mustBeNonnegative} = 0
        options.chlsky (1,1) logical = true
        options.n_init (1,1) double {mustBeInteger, mustBePositive} = 10
        options.rank_ratio_thresh (1,1) double = NaN   % NaN → auto (default = 2)
    end

    n_init = options.n_init;
    p = size(C_xx,1);
    q = size(C_yy,1);

    Wxs     = zeros(p, n_init);
    Wys     = zeros(q, n_init);
    lambdas = zeros(1, n_init);
    corrs   = zeros(1, n_init);

    fprintf('Running %d random initializations:\n', n_init);

    for k = 1:n_init

        % --- Call updated solver (k-1 preserves your original behavior) ---
        [wx, wy, lambda3] = compute_weights_init_rand( ...
            C_xx, C_yy, C_xy, D_xy, ...
            f=options.f, gamma=options.gamma, chlsky=options.chlsky, k=k-1);

        % --- Store ---
        Wxs(:,k)     = wx;
        Wys(:,k)     = wy;
        lambdas(k)   = lambda3;
        corrs(k)     = real(wx' * C_xy * wy);

        % --- Progress bar ---
        pct = k / n_init;
        barWidth = 30;
        nFilled = round(pct * barWidth);
        fprintf('\r[%s%s] %3.0f%%', ...
            repmat('#',1,nFilled), repmat('.',1,barWidth-nFilled), pct*100);
    end

    fprintf('\n');

    % --- Pick best: adaptive selection based on rank ratio ---
    %
    %  Compute numerical rank of C_xy and D_xy to assess how much room
    %  CRM has to satisfy the constraint without sacrificing signal.
    %
    %  High rank ratio (rank_cxy >> rank_dxy):
    %    Signal subspace much larger than confound → strict confound‑first
    %    selection.  Filter for low |confound|, then pick max signal.
    %
    %  Low rank ratio (rank_cxy ≈ rank_dxy):
    %    Subspaces nearly overlap → enforcing confound = 0 eliminates most
    %    signal directions.  Use diff‑based selection (max signal − |confound|)
    %    to tolerate some confound for better signal.
    confounds = arrayfun(@(k) abs(Wxs(:,k)' * D_xy * Wys(:,k)), 1:n_init);

    % Numerical rank (SVD, tolerance = max(size) * eps(norm))
    rank_cxy = rank(C_xy);
    rank_dxy = max(rank(D_xy), 1);
    rank_ratio = rank_cxy / rank_dxy;

    % Determine threshold
    if isnan(options.rank_ratio_thresh)
        rr_thresh = 2;   % auto: switch when excess rank < confound rank
    else
        rr_thresh = options.rank_ratio_thresh;
    end

    if rank_ratio >= rr_thresh
        % --- High capacity: confound‑first, then max signal ---
        min_conf = min(confounds);
        tol      = max(min_conf * 10, 1e-8);
        valid    = (confounds <= tol);

        score = corrs;
        score(~valid) = -Inf;
        select_mode = 'confound-first';
    else
        % --- Low capacity: max (signal − |confound|) ---
        %  Use difference rather than ratio to avoid blow‑up when confound
        %  fluctuates near zero.  Small confound differences in the noise
        %  band have proportionally small effect on the score; signal
        %  dominates when confound suppression is already good.
        score = corrs - confounds;
        select_mode = 'diff';
    end

    [~, idx_best] = max(score);

    w_x_best     = Wxs(:,idx_best);
    w_y_best     = Wys(:,idx_best);
    lambda3_best = lambdas(idx_best);

    fprintf('  rank(C_xy)=%d, rank(D_xy)=%d, ratio=%.1f → %s selection\n', ...
        rank_cxy, rank_dxy, rank_ratio, select_mode);
end

function mustBeSquareMatrix(M)
if ~ismatrix(M) || size(M,1) ~= size(M,2)
    error('Matrix must be square.');
end
end
function mustBeNonnegative(x)
if any(x < 0)
    error('Value must be nonnegative.');
end
end