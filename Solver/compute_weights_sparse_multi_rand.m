function [w_x_best, w_y_best, Wxs, Wys, corrs] = ...
    compute_weights_sparse_multi_rand(C_xx, C_yy, C_xy, D_xy, ff, params)
    % COMPUTE_WEIGHTS_SPARSE_MULTI_RAND
    %
    %   Run multiple random initializations of the sparse CCA/CRM solver
    %   (compute_weights_sparse_init_rand) and return the best sparse
    %   canonical vectors based on the achieved cross‑covariance correlation.
    %
    %   This function is a wrapper that:
    %       1. Calls compute_weights_sparse_init_rand() n_init times
    %       2. Uses different random seeds (k = 0,1,2,...)
    %       3. Applies user‑specified or automatically chosen sparsity levels
    %       4. Stores all sparse solutions (w_x, w_y)
    %       5. Computes the canonical correlation wx' * C_xy * wy
    %       6. Selects and returns the best sparse solution
    %
    %   It is useful when the sparse optimization landscape is non‑convex
    %   and different random starts may converge to different local optima.
    %
    % -------------------------------------------------------------------------
    %   SYNTAX
    %
    %   [w_x_best, w_y_best, Wxs, Wys, corrs] = ...
    %       compute_weights_sparse_multi_rand(C_xx, C_yy, C_xy, D_xy, ff, params)
    %
    % -------------------------------------------------------------------------
    %   INPUTS
    %
    %   C_xx   : [p × p] SPD covariance matrix for X
    %   C_yy   : [q × q] SPD covariance matrix for Y
    %   C_xy   : [p × q] cross‑covariance matrix
    %   D_xy   : [p × q] confound matrix (to be minimized)
    %
    %   ff     : number of canonical components (usually 1)
    %
    %   params : struct of name‑value parameters passed to
    %            compute_weights_sparse_init_rand, including:
    %
    %       alpha     – step size for D_xy gradient
    %       beta      – step size for C_xy gradient
    %       theta_x   – L1 constraint for w_x (0 → auto‑select)
    %       theta_y   – L1 constraint for w_y (0 → auto‑select)
    %       gamma     – ridge regularization strength
    %       chlsky    – use Cholesky‑based initialization
    %       k         – random seed index (overridden internally)
    %       max_iter  – maximum number of gradient iterations
    %       tol       – convergence tolerance
    %       n_init    – number of random initializations
    %
    % -------------------------------------------------------------------------
    %   OUTPUTS
    %
    %   w_x_best : best sparse canonical vector for X
    %   w_y_best : best sparse canonical vector for Y
    %
    %   Wxs      : [p × n_init] matrix of all sparse w_x solutions
    %   Wys      : [q × n_init] matrix of all sparse w_y solutions
    %   corrs    : [1 × n_init] vector of correlations wx' * C_xy * wy
    %
    % -------------------------------------------------------------------------
    %   NOTES
    %
    %   • Each initialization uses seed (k‑1), matching the dense solver's
    %     behavior and ensuring reproducibility.
    %
    %   • Sparsity is enforced via soft‑thresholding inside
    %     compute_weights_sparse_init_rand.
    %
    %   • If theta_x/theta_y are zero, sparsity levels are chosen
    %     automatically using choose_sparsity(), following Witten et al.
    %
    %   • The "best" sparse solution is the one with the largest
    %         corr = real(w_x' * C_xy * w_y)
    %
    %   • All solutions are returned so the user can inspect variability
    %     across random starts or perform stability analysis.
    %
    % -------------------------------------------------------------------------
    %   SEE ALSO
    %       compute_weights_sparse_init_rand
    %       compute_weights_init_rand
    %       compute_weights_multi_rand
    %       choose_sparsity


    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        ff (1,1) double {mustBePositive}
        params.alpha (1,1) double {mustBePositive} = 0.001
        params.beta  (1,1) double {mustBePositive} = 0.001
        % If NOT provided init as zero 
        params.theta_x (1,1) double {mustBeNonnegative} = 0; 
        params.theta_y (1,1) double {mustBeNonnegative} = 0;
        params.gamma (1,1) double {mustBeNonnegative} = 0
        params.chlsky (1,1) logical = false
        params.k (1,1) double {mustBeInteger, mustBeNonnegative} = 0
        params.max_iter (1,1) double {mustBeInteger, mustBePositive} = 10000
        params.tol (1,1) double {mustBePositive} = 1e-6
        params.n_init (1,1) double {mustBePositive, mustBeInteger, mustBeNonnegative} = 10
    end

    n_init = params.n_init;
    
    p = size(C_xx,1);
    q = size(C_yy,1);

    Wxs   = zeros(p, n_init);
    Wys   = zeros(q, n_init);
    corrs = zeros(1, n_init);

    fprintf('Running %d sparse random initializations:\n', n_init);

    for k = 1:n_init

        % --- Set seed for this init ---
        params_k = rmfield(params, 'n_init');

        params_k.k = k-1; % always start at zero
        args = namedargs2cell(params_k);

        % --- Call sparse solver ---
        [wx, wy] = compute_weights_sparse_init_rand( ...
            C_xx, C_yy, C_xy, D_xy, ff, args{:});

        % --- Store ---
        Wxs(:,k) = wx;
        Wys(:,k) = wy;
        corrs(k) = real(wx' * C_xy * wy);

        % --- Progress bar ---
        pct = k / n_init;
        barWidth = 30;
        nFilled = round(pct * barWidth);
        fprintf('\r[%s%s] %3.0f%%', ...
            repmat('#',1,nFilled), repmat('.',1,barWidth-nFilled), pct*100);
    end

    fprintf('\n');

    % --- Pick best ---
    [~, idx_best] = max(corrs);
    w_x_best = Wxs(:,idx_best);
    w_y_best = Wys(:,idx_best);
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