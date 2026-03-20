function [theta_x, theta_y] = choose_sparsity(C_xx, C_yy, percent_zeros_x, percent_zeros_y)
% CHOOSE_SPARSITY  Compute L1-norm targets for desired sparsity.
%
%   percent_zeros_x, percent_zeros_y are between 0 and 1.
%   Example: 0.5 means 50% zeros (50% non-zero).
%
%   This function sets theta_x and theta_y so that the L1 norm
%   after shrinkage corresponds to the desired number of non-zero
%   coefficients, assuming w is normalized such that w' C w = 1.

    % dimensionalities
    p_x = size(C_xx,1);
    p_y = size(C_yy,1);

    % convert percent zeros → percent non-zero
    frac_nonzero_x = 1 - percent_zeros_x;
    frac_nonzero_y = 1 - percent_zeros_y;

    % number of non-zero coefficients desired
    k_x = max(1, round(frac_nonzero_x * p_x));
    k_y = max(1, round(frac_nonzero_y * p_y));

    % typical magnitude after normalization: ~1/sqrt(p)
    typical_mag_x = 1 / sqrt(p_x);
    typical_mag_y = 1 / sqrt(p_y);

    % L1 norm target
    theta_x = k_x * typical_mag_x;
    theta_y = k_y * typical_mag_y;
end
