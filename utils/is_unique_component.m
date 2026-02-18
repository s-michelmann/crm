function is_unique = is_unique_component(w_new, W_existing, tol)
    % is_unique = is_unique_component(w_new, W_existing, tol)
    %
    % Determines whether w_new is unique relative to the columns of W_existing,
    % using a strict elementwise test that is invariant to global sign.
    %
    % A vector is considered a duplicate if it matches an existing vector
    % up to sign, i.e. norm(w_new - w_old) < tol OR norm(w_new + w_old) < tol.

    if nargin < 3
        tol = sqrt(2/length(w_new));
    end

    if isempty(W_existing)
        is_unique = true;
        return
    end

    % Normalize new vector
    w_new = w_new(:) / norm(w_new);

    % Normalize existing vectors
    Wn = W_existing ./ vecnorm(W_existing);

    % Compute sign-invariant distances
    d1 = vecnorm(Wn - w_new, 2, 1);   % distance to +w_new
    d2 = vecnorm(Wn + w_new, 2, 1);   % distance to -w_new

    % A match occurs if either distance is below tolerance
    is_duplicate = (d1 < tol) | (d2 < tol);

    is_unique = ~any(is_duplicate);
end
