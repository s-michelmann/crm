function [w_x, w_y] = compute_weights_sparse_init_rand(C_xx, C_yy, C_xy, D_xy, ff, params)
    % COMPUTE_WEIGHTS_SPARSE_INIT_RAND % 
    % Compute sparse CRM weight vectors (w_x, w_y) using a 
    % gradient‑based update rule with optional automatic sparsity selection. 
    % 
    % This function:  
    % 1. Initializes (w_x, w_y) using compute_weights_init_rand 
    % 2. Chooses L1‑norm constraints (theta_x, theta_y) if not provided 
    % 3. Iteratively updates w_x, w_y using gradient ascent on C_xy
    %    and gradient descent on D_xy  
    % 4. Applies soft‑thresholding to enforce L1 constraints 
    % 5. Renormalizes w_x, w_y in the C_xx / C_yy metric 

    % % INPUTS % ------ 
    % C_xx : [p × p] SPD covariance matrix for X 
    % C_yy : [q × q] SPD covariance matrix for Y 
    % C_xy : [p × q] cross‑covariance matrix 
    % D_xy : [p × q] confound matrix (to be minimized) 
    % ff : number of canonical component (usually 1 = first) 
    % params : struct with fields: 
    %   alpha – step size for D_xy gradient (default 0.001) 
    %   beta – step size for C_xy gradient (default 0.001) 
    %   theta_x – L1 constraint for w_x. 
    %   If 0, automatically chosen via choose_sparsity(). 
    %   theta_y – L1 constraint for w_y. 
    %   If 0, automatically chosen via choose_sparsity(). 
    %   gamma – ridge regularization added to C_xx, C_yy 
    %           (passed to compute_weights_init_rand) 
    %   chlsky – whether to use Cholesky‑based solver for init 
    %   k – random seed index for initialization 
    %   max_iter – maximum number of gradient iterations 
    %   tol – stopping tolerance based on change in w_x, w_y 

    % % OUTPUTS % ------- 
    %   w_x : sparse canonical vector for X 
    %   w_y : sparse canonical vector for Y 
    
    % % NOTES % ----- % 
    %   If theta_x = theta_y = 0, sparsity is chosen automatically using 
    %     choose_sparsity(), which implements a Witten‑style L1 constraint 
    %     based on the magnitude of the dense initial solution. 
    %   Soft‑thresholding is applied using apply_threshold(), which finds 
    %   the shrinkage δ such that ||w||_1 = theta. 
    %   After each update, w_x and w_y are normalized so that: 
    %       w_x' * C_xx * w_x = 1 % w_y' * C_yy * w_y = 1 
    %   The algorithm alternates between: 
    %           -ascent on C_xy (signal) 
    %           -descent on D_xy (confound) 
    %   This function computes *one* sparse solution. For multiple random 
    %       starts, use compute_weights_sparse_multi_rand()%
    
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
    end
    
    alpha = params.alpha;
    beta = params.beta;
    theta_x = params.theta_x;
    theta_y = params.theta_y;
    gamma = params.gamma; % We can use the Ridge term
    chlsky = params.chlsky;
    k = params.k;
    [w_x, w_y, ~] =  compute_weights_init_rand(C_xx, C_yy,C_xy, D_xy, ff, ...
        gamma, chlsky, k);
    
    % did NOT provide sparsity, compute it automatically
    if params.theta_x == 0 && params.theta_y ==0
        [theta_x, theta_y] = choose_sparsity(C_xx, C_yy, w_x, w_y);
    elseif params.theta_x == 0
        [theta_x, ~] = choose_sparsity(C_xx, C_yy, w_x, w_y);
    elseif params.theta_y == 0
        [~, theta_y] = choose_sparsity(C_xx, C_yy, w_x, w_y);
    else
        theta_x = params.theta_x;
        theta_y = params.theta_y;
    end
    
    % Iterative Solver
    for iter = 1:params.max_iter
        w_x_old = w_x;
        w_y_old = w_y;
    
        g_x = D_xy * w_y;         % D gradient
        g_y = D_xy' * w_x;
        w_x = w_x - alpha * g_x;
        w_y = w_y - alpha * g_y;
    
        h_x = C_xy * w_y;         % C gradient
        h_y = C_xy' * w_x;
        w_x = w_x + beta * h_x;
        w_y = w_y + beta * h_y;
    
        w_x = apply_threshold(w_x, theta_x); % Threshold
        w_y = apply_threshold(w_y, theta_y);
    
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x); % norm
        w_y = w_y ./ sqrt(w_y' * C_yy * w_y);
    
        if norm(w_x - w_x_old) + norm(w_y - w_y_old) < params.tol % Stop if converged.
            break;
        end
    end
end

function w_out = apply_threshold(w, theta)
    % APPLY_THRESHOLD  Soft-threshold a vector so its L1 norm equals theta.
    
    %   w_out = apply_threshold(w, theta)
    %
    %   Uses binary search to find the shrinkage parameter δ such that:
    %       sum(abs( sign(w) .* max(abs(w) - δ, 0) )) ≈ theta
    %
    %   If the L1 norm is already ≤ theta, the vector is returned unchanged.
    
    arguments
        w (:,1) double
        theta (1,1) double {mustBeNonnegative}
    end
    
    % Current L1 norm
    current_norm = sum(abs(w));
    
    % If already below threshold, no shrinkage needed
    if current_norm <= theta
        w_out = w;
        return
    end
    
    % Binary search bounds for δ
    delta_min = 0;
    delta_max = max(abs(w));
    
    % Binary search iterations (log2(N) is enough, but 50 is safe & cheap)
    for iter = 1:50
        delta = 0.5 * (delta_min + delta_max);
    
        w_temp = sign(w) .* max(abs(w) - delta, 0);
        new_norm = sum(abs(w_temp));
    
        if new_norm > theta
            % δ too small → shrink more
            delta_min = delta;
        else
            % δ too large or just right → shrink less
            delta_max = delta;
        end
    end
    
    % Final shrinkage
    delta_final = delta_max;
    w_out = sign(w) .* max(abs(w) - delta_final, 0);
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

function [theta_x, theta_y] = choose_sparsity(C_xx, C_yy, w_x0, w_y0, c)
    % CHOOSE_SPARSITY  Compute principled L1 thresholds for sparse CCA.
    %
    %   Only used when user does not provide theta_x/theta_y.
    %   c = sparsity constant (default 2)
    
    % Witten et al. (2009)
    % Donoho & Johnstone soft‑threshold theory
    
    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        w_x0 (:,1) double
        w_y0 (:,1) double
        c (1,1) double {mustBePositive} = 2
    end
    
    p_x = size(C_xx,1);
    p_y = size(C_yy,1);
    
    % sparsity fraction = min(10%, c / sqrt(p))
    frac_x = min(0.1, c / sqrt(p_x));
    frac_y = min(0.1, c / sqrt(p_y));
    
    k_x = max(1, round(frac_x * p_x));
    k_y = max(1, round(frac_y * p_y));
    
    % robust magnitude estimate
    med_x = median(abs(w_x0));
    med_y = median(abs(w_y0));
    
    theta_x = k_x * med_x;
    theta_y = k_y * med_y;
end
