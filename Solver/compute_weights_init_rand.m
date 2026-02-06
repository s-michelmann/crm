function [w_x, w_y, lambda3] = compute_weights_init_rand( ...
        C_xx, C_yy, C_xy, D_xy, f, gamma, chlsky, k)
    % COMPUTE_WEIGHTS_INIT_RAND
    %
    %   Compute dense canonical weight vectors (w_x, w_y) for CRM using
    %   either a standard solver or a Cholesky‑based solver, with optional
    %   random initialization of the confound parameter lambda3.
    %
    %   This function implements the core dense CRM optimization:
    %       • ridge‑regularizes C_xx and C_yy (gamma)
    %       • optionally uses Cholesky factors for numerical stability
    %       • searches over lambda3 using fminsearch
    %       • solves the generalized eigenproblem for the top component
    %       • enforces a consistent sign convention
    %
    %   It is the base solver used by:
    %       compute_weights_multi_rand
    %       compute_weights_sparse_init_rand
    %       compute_weights_sparse_multi_rand
    %
    % -------------------------------------------------------------------------
    %   SYNTAX
    %
    %   [w_x, w_y, lambda3] = compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy)
    %
    %   [...] = compute_weights_init_rand(..., f, gamma, chlsky, k)
    %
    % -------------------------------------------------------------------------
    %   INPUTS
    %
    %   C_xx   : [p × p] SPD covariance matrix for X
    %   C_yy   : [q × q] SPD covariance matrix for Y
    %   C_xy   : [p × q] cross‑covariance matrix
    %   D_xy   : [p × q] confound matrix (to be minimized)
    %
    %   f      : number of canonical component (default: 1, i.e., first)
    %
    %   gamma  : ridge regularization strength added to C_xx and C_yy
    %            (default = 0)
    %
    %   chlsky : logical flag
    %            true  → use Cholesky‑based solver
    %            false → use standard solver
    %
    %   k      : random seed index for initializing lambda3
    %            k = 0 → deterministic start (lambda3 = 0)
    %            k > 0 → random start in a heuristic range
    %
    % -------------------------------------------------------------------------
    %   OUTPUTS
    %
    %   w_x     : canonical weight vector for X
    %   w_y     : canonical weight vector for Y
    %   lambda3 : confound parameter minimizing |w_x' * D_xy * w_y|
    %
    % -------------------------------------------------------------------------
    %   NOTES
    %
    %   • If the signal‑to‑confound ratio is large (ratio > 10), the function
    %     defaults to classical CCA (lambda3 = 0).
    %
    %   • Otherwise, lambda3 is optimized via fminsearch using the objective:
    %         | w_x' * D_xy * w_y |
    %
    %   • The Cholesky path avoids explicit matrix inversion and is more
    %     numerically stable for ill‑conditioned covariance matrices.
    %
    %   • The returned vectors satisfy:
    %         w_x' * C_xx * w_x = 1
    %         w_y' * C_yy * w_y = 1
    %
    %   • A sign convention ensures w_x' * C_xy * w_y ≥ 0.
    %
    % -------------------------------------------------------------------------
    %   SEE ALSO
    %       compute_weights_multi_rand
    %       compute_weights_sparse_init_rand
    %       compute_weights_sparse_multi_rand
    %       choose_sparsity

    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        f (1,1) double {mustBePositive} = 1
        gamma (1,1) double {mustBeNonnegative} = 0
        chlsky (1,1) logical = false
        k (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    end

    % --- Regularization ---
    if gamma > 0
        C_xx = C_xx + gamma * eye(size(C_xx,1));
        C_yy = C_yy + gamma * eye(size(C_yy,1));
    end

    % --- Precompute Cholesky if requested ---
    if chlsky
        [Lx,pflag] = chol(C_xx,'lower');
        [Ly,qflag] = chol(C_yy,'lower');
        if pflag || qflag
            error('C_xx/C_yy not SPD. Increase gamma.');
        end
    end

    % --- Heuristic range for lambda3 ---
    ratio = norm(C_xy,'fro') / max(norm(D_xy,'fro'), 1e-12);
    maxrange = 10 * ratio;
    if ~isfinite(maxrange) || maxrange == 0
        maxrange = 1;
    end

    % --- Case: ratio > 10 → default to CCA ---
    if ratio > 10
        M = (C_xx \ C_xy) / C_yy * C_xy';
        [W, D] = eigs(M, f, 'lm');

        w_x = W(:,f);
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x);

        w_y = (C_yy \ (C_xy' * w_x)) / sqrt(D(f,f));

        if w_x' * C_xy * w_y < 0
            w_y = -w_y;
        end

        lambda3 = 0;
        return
    end

    % --- Case 2: CRM random start ---
    if k == 0
        lambda0 = 0;
    else
        rng(k)
        lambda0 = (2*rand - 1) * maxrange;
    end

    % --- No Cholesky path ---
    if ~chlsky
        fun = @(l) foo2(C_xx, C_yy, C_xy, D_xy, l, f);
        lambda3 = fminsearch(fun, lambda0);

        M = inv(C_xx) * (C_xy + lambda3 * D_xy) * ...
            inv(C_yy) * (C_xy + lambda3 * D_xy)';

        [W, D] = eigs(M, f, 'lm');
        w_x = W(:,f);
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x);

        lbd = sqrt(D(f,f));
        w_y = -inv(C_yy) * (C_xy + lambda3 * D_xy)' / lbd * w_x;

    else
        % --- Cholesky path ---
        fun = @(l) foo3(Lx, Ly, C_xy, D_xy, l, f);
        lambda3 = fminsearch(fun, lambda0);

        K = (Lx \ (C_xy + lambda3 * D_xy)) / Ly';
        M = K * K';

        opts.isreal = true;
        [W, D] = eigs(M, f, 'lm', opts);
        vx = W(:,end);

        w_x = Lx' \ vx;
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x);

        lbd = sqrt(D(end,end));
        rhs = (C_xy + lambda3 * D_xy)' * w_x;
        w_y = -(C_yy \ rhs) / lbd;
    end

    % --- Sign convention ---
    if w_x' * C_xy * w_y < 0
        w_y = -w_y;
    end
end

function [tst] = foo2(C_xx, C_yy, C_xy, D_xy, lbd3, f)
    % calculate the f largest eigenvalues only!
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
    [W,D] = eigs(M,f,'lm');
    w_x = W(:,f);
    w_x = w_x./sqrt(w_x'*C_xx*w_x);
    w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x;
    tst = abs(w_x'*D_xy*w_y);
end

function tst = foo3(Lx, Ly, C_xy, D_xy, lbd3, f)
    % --- Consistent eigen criterion and solves (no inv) ---
    % M = K*K' with K = Cxx^{-1/2}(Cxy+lD)Cyy^{-T/2}
    K = (Lx \ (C_xy + lbd3 * D_xy)) / Ly';
    M = K * K';

    opts.isreal = true;
    [W,D] = eigs(M, f, 'lm', opts);
    vx = W(:,end);

    % Map back to original coordinates
    w_x = Lx' \ vx;

    % C_xx-metric normalization (match foo2)
    C_xx = Lx * Lx';
    C_yy = Ly * Ly';

    w_x = w_x ./ sqrt(w_x' * C_xx * w_x);

    % w_y via solves (match foo2 logic without inv)
    rhs = (C_xy + lbd3 * D_xy)' * w_x;
    w_y = C_yy \ rhs;

    % Confound term to minimize
    tst = abs(w_x' * D_xy * w_y);
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