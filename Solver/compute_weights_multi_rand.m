function [w_x_best, w_y_best, lambda3_best, Wxs, Wys, lambdas, corrs] = ...
    compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy, f, gamma, chlsky, n_init)


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
    %   [...] = compute_weights_multi_rand(..., f, gamma, chlsky, n_init)
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
    %   • The "best" solution is defined as the one with the largest
    %         corr = real(w_x' * C_xy * w_y)
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
        f (1,1) double {mustBePositive} = 1
        gamma (1,1) double {mustBeNonnegative} = 0
        chlsky (1,1) logical = false
        n_init (1,1) double {mustBeInteger, mustBePositive} = 10
    end

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
            C_xx, C_yy, C_xy, D_xy, f, gamma, chlsky, k-1);

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

    % --- Pick best ---
    [~, idx_best] = max(corrs);
    w_x_best     = Wxs(:,idx_best);
    w_y_best     = Wys(:,idx_best);
    lambda3_best = lambdas(idx_best);
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