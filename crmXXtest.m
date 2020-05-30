function [corr_same, corr_diff] = crmXXtest(X, labels, s1, s2, window_width,Vx, Vy)

%@X the data should be organized as follows:
% dim 1 (rows) are features
% dim 2 (columns) are observations
% dim 3 are samples
%@ labels is a vector of integers that denote the type of observation
%@ s1 is a vector of sample points in X (can be a single integer)
%@ s2 is a vector of sample points in X (can be a single integer)
%@ window_width is a window in sample-points across which the
%correlation will be maximized. If window_width is empty or <= 1, the
%correlation will only be maximized across trials
Ntime = size(X,3);
Nfeat = size(X,1);
Nobs = size(X,2);

corr_same = nan(numel(s1), numel(s2));
corr_diff = nan(numel(s1), numel(s2));

% get the label-combinations:
[same_indices, diff_indices] = getCombsXX(labels);

if nargin < 5; window_width = 0; end
assert(Nobs == size(labels,1), ...
    'number of labels and number of observations need to match');

if s1(1)-floor(window_width/2) <1 || s2(1)-floor(window_width/2) <1
    error('window reaches beyond 0. Consider including more data');
end
if s1(end)+floor(window_width/2) > Ntime ||...
        s2(end)+floor(window_width/2) > Ntime
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
    
    
    % loop through the second time-axis
    lc2 = 1;
    for t2 = s2
        dt2 = X(:,:,t2-floor(window_width/2):t2+floor(window_width/2));
        %demean for covariance computation
        if size(dt2,3)>1
            dt2 = bsxfun(@minus, dt2, nanmean(dt2,3));
        end
        % get the right half of the combinations
        dt2s = dt2(:,same_indices(:,2),:);
        dt2d = dt2(:,diff_indices(:,2),:);
        
        x =  squeeze(Vx(lc1, lc2, :));
        y =  squeeze(Vy(lc1, lc2, :));
        dt1s_proj =  dt1s(:,:)'*x;
        dt1d_proj =  dt1d(:,:)'* x;
        dt2s_proj =  dt2s(:,:)'*y;
        dt2d_proj =  dt2d(:,:)'* y;
        
        corr_same(lc1, lc2) = corr(dt1s_proj, dt2s_proj);
        corr_diff(lc1, lc2) = corr(dt1d_proj, dt2d_proj);
        
        
        lc2 = lc2 +1;
    end
    lc1 = lc1 + 1;
end
end