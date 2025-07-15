function [Vx, Vy, lbd3] = crmXYtrain(X, Y, labelsx, labelsy, s1, s2, window_width)
%crmXY computes canonical representational mapping between variable
%sample-points s1 and s2 in the multivariate data X and Y.

%@X the data should be organized as follows:
% dim 1 (rows) are features
% dim 2 (columns) are observations
% dim 3 are samples
%@Y the data should be organized as follows:
% dim 1 (rows) are features
% dim 2 (columns) are observations
% dim 3 are samples

NtimeX = size(X,3);
NfeatX = size(X,1);
NobsX = size(X,2);

NtimeY = size(Y,3);
NfeatY = size(Y,1);
NobsY = size(Y,2);


assert(NobsX == size(labelsx,1)&NobsX == size(labelsx,1), ...
    'number of labels and number of observations need to match');

Vx = nan(numel(s1), numel(s2),NfeatX);
Vy = nan(numel(s1), numel(s2),NfeatY);
lbd3 = nan(numel(s1), numel(s2)) ;
% get the label-combinations:
[same_indices, diff_indices] = getCombsXY(labelsx, labelsy);

if nargin < 7; window_width = 0; end
if s1(1)-floor(window_width/2) <1 || s2(1)-floor(window_width/2) <1
    error('window reaches beyond 0. Consider including more data');
end
if s1(end)+floor(window_width/2) > NtimeX ||...
        s2(end)+floor(window_width/2) > NtimeX
    error('window reaches beyond end of data');
end

% loop through the first time-axis
lc1 = 1;
for t1 = s1
    % pick data in window
    dt1 = X(:,:,t1-floor(window_width/2):t1+floor(window_width/2));
    
    %demean for covariance computation
    if size(dt1,3)>1
        dt1 = bsxfun(@minus, dt1, nanmean(dt1,3));
    end
    
    % get the left half of the combinations
    dt1s = dt1(:,same_indices(:,1),:);
    dt1d = dt1(:,diff_indices(:,1),:);
    
    % get t1 covariance matrix
    Cxx = cov(dt1s(:,:)');
    % loop through the second time-axis
    lc2 = 1;
    for t2 = s2
        dt2 = Y(:,:,t2-floor(window_width/2):t2+floor(window_width/2));
        %demean for covariance computation
        if size(dt2,3)>1
            dt2 = bsxfun(@minus, dt2, nanmean(dt2,3));
        end
        % get the right half of the combinations
        dt2s = dt2(:,same_indices(:,2),:);
        dt2d = dt2(:,diff_indices(:,2),:);
        
        % get t2 covariance matrix
        Cyy = cov(dt2s(:,:)');
        % covariance between x and y to maximize (same labels)
        Cxy = dt1s(:,:)*dt2s(:,:)';
        % covariance between x and y to minimize(different labels)
        Dxy = dt1d(:,:)*dt2d(:,:)';
        
        [Vx(lc1, lc2,:), Vy(lc1, lc2,:), lbd3(lc1, lc2)] = ...
            compute_weights(Cxx,Cyy,Cxy, Dxy,1) ;
        lc2 = lc2 +1;
    end
    lc1 = lc1 + 1;
end
end