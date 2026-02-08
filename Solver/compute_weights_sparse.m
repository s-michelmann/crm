function [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy, ff, params)
%   [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy, params)

    % doing this neater. sparse
    alpha = params.alpha;
    beta = params.beta;
    theta_x = params.theta_x;
    theta_y = params.theta_y;
    gamma = params.gamma; % We can use the Ridge term
    
    [w_x, w_y, ~] = compute_weights(C_xx, C_yy, C_xy, D_xy, ff, gamma);

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

    l1_norm = sum(abs(w)); % If theta is very large, no thresholding is needed.
    if l1_norm <= theta
        w_out = w;
        return;
    end

    % Binary search for the delta that results in L1 norm close to theta
    % Another iterative optimization method.
    delta_min = 0;
    delta_max = max(abs(w));
    
    for k = 1:length(w)
        delta = (delta_min + delta_max) / 2;
        w_temp = sign(w) .* max(abs(w) - delta, 0);
        
        if sum(abs(w_temp)) == 0        % Threshold too high
            delta_max = delta;
        elseif sum(abs(w_temp)) > theta % Threshold too low
            delta_min = delta;
        else
            delta_max = delta;          % Good enough.
        end
    end
    w_out = sign(w) .* max(abs(w) - delta_max, 0);
end