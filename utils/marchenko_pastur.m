function [threshold, is_significant] = marchenko_pastur(n, p, eigenvalues, sigma2)
%MARCHENKO_PASTUR  Marchenko–Pastur threshold for eigenvalue significance.
%
%   [threshold, is_significant] = marchenko_pastur(n, p, eigenvalues)
%   [threshold, is_significant] = marchenko_pastur(n, p, eigenvalues, sigma2)
%
%   Computes the upper edge of the Marchenko–Pastur distribution, which
%   gives the maximum eigenvalue expected from a random matrix with i.i.d.
%   entries of variance sigma^2.  Eigenvalues exceeding this threshold are
%   considered signal (not noise).
%
%   INPUTS
%
%   n           : number of samples (observations)
%   p           : number of features (variables)
%   eigenvalues : vector of eigenvalues to test
%   sigma2      : noise variance (default = median eigenvalue, via
%                 Marchenko–Pastur median approximation)
%
%   OUTPUTS
%
%   threshold      : upper edge lambda_+ of the MP distribution
%   is_significant : logical vector, true where eigenvalues > threshold
%
%   FORMULA
%
%   gamma = p / n
%   lambda_+ = sigma^2 * (1 + sqrt(gamma))^2
%   lambda_- = sigma^2 * (1 - sqrt(gamma))^2   (not returned)
%
%   REFERENCES
%
%   Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets
%     of random matrices", Mathematics of the USSR-Sbornik 1(4), 457–483.
%
%   SEE ALSO  eig, svd

    arguments
        n (1,1) double {mustBePositive, mustBeInteger}
        p (1,1) double {mustBePositive, mustBeInteger}
        eigenvalues (:,1) double
        sigma2 (1,1) double {mustBePositive} = median(eigenvalues)
    end

    gamma = p / n;
    threshold = sigma2 * (1 + sqrt(gamma))^2;
    is_significant = eigenvalues > threshold;
end
