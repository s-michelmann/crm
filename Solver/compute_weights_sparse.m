function [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy, params)
%COMPUTE_WEIGHTS_SPARSE  Sparse CRM (legacy interface).
%
%   [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy)
%   [w_x, w_y] = compute_weights_sparse(..., f=2, gamma=0.01, mu=Inf, ...)
%
%   Prefer compute_weights_sparse_init_rand for new code (supports random
%   init, automatic sparsity selection, Cholesky path).
%
%   mu defaults to lambda3 from the dense init (Inf → auto).
%   step_size defaults to 1/sigma_max(C_xy + mu*D_xy) (0 → auto).

    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        params.f (1,1) double {mustBePositive} = 1
        params.mu (1,1) double = Inf              % Inf → auto (use lambda3)
        params.step_size (1,1) double {mustBeNonnegative} = 0  % 0 → auto (1/L)
        params.sparsity (1,1) double = NaN        % NaN → auto; 0–1 = fraction of zeros
        params.theta_x (1,1) double {mustBeNonnegative} = 0;  % 0 → use sparsity or auto
        params.theta_y (1,1) double {mustBeNonnegative} = 0;
        params.gamma (1,1) double {mustBeNonnegative} = 0
        params.max_iter (1,1) double {mustBeInteger, mustBePositive} = 10000
        params.tol (1,1) double {mustBePositive} = 0.001
    end

    theta_x = params.theta_x;
    theta_y = params.theta_y;

    [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy, ...
        f=params.f, gamma=params.gamma);

    % --- Confound penalty weight ---
    if isinf(params.mu)
        mu = lambda3;
    else
        mu = params.mu;
    end

    % --- Step size ---
    if params.step_size == 0
        L = svds(C_xy + mu * D_xy, 1);
        step = 1 / max(L, eps);
    else
        step = params.step_size;
    end

    % --- Combined matrix (matches dense solver convention) ---
    M_xy = C_xy + mu * D_xy;

    % --- Regularized covariance (same metric as dense init) ---
    gamma = params.gamma;
    p = size(C_xx, 1);
    q = size(C_yy, 1);
    C_xx_r = C_xx + gamma * eye(p);
    C_yy_r = C_yy + gamma * eye(q);

    % --- Sparsity selection (priority: theta > sparsity ratio > auto) ---
    if theta_x > 0 && theta_y > 0
        % User provided explicit thresholds — use as-is

    elseif ~isnan(params.sparsity)
        % User provided sparsity ratio — compute theta from it
        [auto_tx, auto_ty] = theta_from_sparsity( ...
            params.sparsity, p, q, w_x, w_y);
        if theta_x == 0, theta_x = auto_tx; end
        if theta_y == 0, theta_y = auto_ty; end

    else
        % Auto-select (Donoho & Johnstone / Witten heuristic)
        warning('sparse_crm:no_sparsity', ...
            ['No sparsity setting provided (theta_x, theta_y, or sparsity). ' ...
             'Using automatic selection via Donoho & Johnstone heuristic.']);
        [auto_tx, auto_ty, frac_zeros_x, frac_zeros_y] = ...
            choose_sparsity(C_xx, C_yy, w_x, w_y);
        if theta_x == 0, theta_x = auto_tx; end
        if theta_y == 0, theta_y = auto_ty; end
        fprintf('  Auto-selected sparsity: %.1f%% zeros (x, p=%d), %.1f%% zeros (y, q=%d)\n', ...
            frac_zeros_x * 100, p, frac_zeros_y * 100, q);
    end

    % Iterative Solver
    for iter = 1:params.max_iter
        w_x_old = w_x;
        w_y_old = w_y;

        w_x = w_x + step * (M_xy * w_y);
        w_y = w_y + step * (M_xy' * w_x);

        w_x = apply_threshold(w_x, theta_x);
        w_y = apply_threshold(w_y, theta_y);

        nrm_x = w_x' * C_xx_r * w_x;
        if nrm_x < 1e-20
            w_x = w_x_old;
            break
        end
        w_x = w_x / sqrt(nrm_x);

        nrm_y = w_y' * C_yy_r * w_y;
        if nrm_y < 1e-20
            w_y = w_y_old;
            break
        end
        w_y = w_y / sqrt(nrm_y);

        if norm(w_x - w_x_old) + norm(w_y - w_y_old) < params.tol
            break;
        end
    end
end

function w_out = apply_threshold(w, theta)

    l1_norm = sum(abs(w));
    if l1_norm <= theta
        w_out = w;
        return;
    end

    delta_min = 0;
    delta_max = max(abs(w));

    for k = 1:50
        delta = (delta_min + delta_max) / 2;
        w_temp = sign(w) .* max(abs(w) - delta, 0);

        if sum(abs(w_temp)) > theta
            delta_min = delta;
        else
            delta_max = delta;
        end
    end
    w_out = sign(w) .* max(abs(w) - delta_max, 0);
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

function [theta_x, theta_y, frac_zeros_x, frac_zeros_y] = choose_sparsity(C_xx, C_yy, w_x0, w_y0, c)
    % CHOOSE_SPARSITY  Auto-select L1 thresholds for sparse CCA/CRM.
    %
    %   Uses the universal threshold heuristic: keep k = min(10%, c/sqrt(p)) * p
    %   non-zero entries, with per-entry magnitude estimated from the dense
    %   solution's median absolute weight.
    %
    %   References:
    %     Donoho & Johnstone (1994), "Ideal spatial adaptation by wavelet
    %       shrinkage", Biometrika 81(3), 425-455.
    %     Witten, Tibshirani & Hastie (2009), "A penalized matrix
    %       decomposition", Biostatistics 10(3), 515-534.

    if nargin < 5, c = 2; end

    p_x = size(C_xx,1);
    p_y = size(C_yy,1);

    frac_nonzero_x = min(0.1, c / sqrt(p_x));
    frac_nonzero_y = min(0.1, c / sqrt(p_y));

    k_x = max(1, round(frac_nonzero_x * p_x));
    k_y = max(1, round(frac_nonzero_y * p_y));

    med_x = median(abs(w_x0));
    med_y = median(abs(w_y0));

    theta_x = k_x * med_x;
    theta_y = k_y * med_y;

    frac_zeros_x = 1 - k_x / p_x;
    frac_zeros_y = 1 - k_y / p_y;
end

function [theta_x, theta_y] = theta_from_sparsity(sparsity, p_x, p_y, w_x0, w_y0)
    % THETA_FROM_SPARSITY  Compute L1 thresholds from a sparsity ratio.
    %
    %   sparsity : fraction of zeros (0-1). E.g. 0.9 = 90% zeros.

    frac_nonzero = 1 - sparsity;
    k_x = max(1, round(frac_nonzero * p_x));
    k_y = max(1, round(frac_nonzero * p_y));

    med_x = median(abs(w_x0));
    med_y = median(abs(w_y0));

    theta_x = k_x * med_x;
    theta_y = k_y * med_y;
end
