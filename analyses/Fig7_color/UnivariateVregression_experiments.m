%% CRM

tic

m_long_c = reshape(m_cone(33:182,33:182,:), 150*150, 9750)';
l_long_c = reshape(l_cone(33:182,33:182,:), 150*150, 9750)';


X = tdfread("../../data/File130Gn.txt").x0x2D2;
X = circshift(X, -3)';
X = zscore(X);

Y = l_long_c(751:9750,:);
Y = zscore(Y)';

S = X;
T = l_long_c(751:9750,:) + m_long_c(751:9750,:);
T = zscore(T)';

%
Cyy = (Y * Y')./9750;
Cxy = (X * Y')./9750; 
Dxy = (S * T')./9750;
lambda_ridge = 1e-2 * trace(Cyy);

Cyy_inv = inv(Cyy + lambda_ridge * eye(size(Cyy)));

% do univariateCRM
lambda3 = - Dxy * Cyy_inv * Cxy' ./ (Dxy * Cyy_inv * Dxy');
w_y_raw = Cyy_inv * (Cxy + lambda3 * Dxy)';
w_y = w_y_raw ./ sqrt(w_y_raw' * Cyy * w_y_raw);

% For comparison, do standard regression.
w_standard = Cyy_inv * Cxy';
w_standard = w_standard / sqrt(w_standard' * Cyy * w_standard);

% Correlation with True Signal
corrcoef(w_y'*Y, X)  
Cxy * w_y

w_y'*(Cyy * w_y)

corrcoef(w_standard'*Y, X) 
Cxy * w_standard

Dxy * w_y % should be close to zero. wy is in the D nullspace
Dxy * w_standard % some number
 
c = redblue(256);

toc
%%
figure(1)

subplot(1,3,1)
imagesc(reshape(w_standard,150,150))
clim([-1.5,1.5].*max(abs(w_standard(:))))
title("CCA: M space")
colorbar;
daspect([1 1 1])

subplot(1,3,2)
imagesc(reshape(w_y,150,150))
title("CRM L-M in luminance null")
colormap(c)
clim([-1.5,1.5].*max(abs(w_y(:))))
colorbar;
daspect([1 1 1])

subplot(1,3,3)
d = reshape(w_y,150,150)-reshape(w_standard,150,150);
imagesc(d)
title("difference")
colormap(c)
clim([-1,1].*max(abs(d(:))))
colorbar;
daspect([1 1 1])
