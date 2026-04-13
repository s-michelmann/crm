%Run PreprocessVideo.ipynb to make the matlab files.

%% Plot example frames

load('matlab_flowershow_rawdic.mat')
load('matlab_flowershow_m.mat')
load('matlab_flowershow_l.mat')

%%
frameid = 1999;

frame =reshape(raw_image(frameid,:,:,:),216,216,3);
subplot(2,3,1)
frame = frame(:, :, 3:-1:1);
image(frame)
daspect([1 1 1])

subplot(2,3,2)
imagesc(l_cone(:,:,frameid))
daspect([1 1 1])
caxis([0,4])
colormap("parula")
freezeColors;
title("L - Cone")

subplot(2,3,3)
imagesc(m_cone(:,:,frameid))
daspect([1 1 1])
title("M - Cone")
colormap("parula")
freezeColors;
caxis([0,4])


subplot(2,3,5)
imagesc((l_cone(:,:,frameid)+m_cone(:,:,frameid))/2)
daspect([1 1 1])
title("L+M")
colormap("parula")
freezeColors;
caxis([0,4])

subplot(2,3,6)
c = redblue(256);
imagesc(l_cone(:,:,frameid)-m_cone(:,:,frameid))
colormap(c)
daspect([1 1 1])
title("L-M")
caxis([-0.6,0.6])

%% Plot example neuron

cell1 = tdfread("../../data/File110Ron.txt").x0x2D2;
cell2 = tdfread("../../data/File130Gn.txt").x0x2D2;
cell3 = tdfread("../../data/File258Mon.txt").x0x2D2;

dt = 1000./150; % in units ms.
shift = 0;

time_cell = (0:length(cell1)-1)*dt + 5000 - shift*6.66;
 
figure(1),clf;
plot(time_cell, 0 + cell1'./std(cell1), 'k')
hold on;
plot(time_cell, 1 + cell2'./std(cell2), 'k')
plot(time_cell, 2 + cell3'./std(cell3), 'k')

hold on;

l_test = reshape(mean(mean(l_cone,1),2),9750,1);
m_test = reshape(mean(mean(m_cone,1),2),9750,1);

plot((0:length(l_test)-1)*dt, l_test./std(l_test), 'r');
plot((0:length(m_test)-1)*dt, m_test./std(m_test), 'g');

xlim([54000,57000])
ylim([-10, 10])

figure(77)
scatter(l_test, m_test, 'k.')
xlim([0,2.5])
ylim([0,2.5])
xlabel("L")
ylabel("M")
daspect([1 1 1])
title(corrcoef(l_test, m_test))


%%
figure(777) % Standard regression in M

X = circshift(cell3, -3)';
X = zscore(X')';

Y = m_long_c(751:9750,:);
Y = zscore(Y)';

Cyy = (Y * Y')./9750;
Cxy = (X * Y')./9750; 
lambda_ridge = 1e-2 * trace(Cyy);

Cyy_inv = inv(Cyy + lambda_ridge * eye(size(Cyy)));

w_reg = Cyy_inv * Cxy';
w_reg = w_reg / sqrt(w_reg' * Cyy * w_reg);

[dx,dy] = meshgrid(-75:74,-75:74);
mask = sqrt(dx.^2 + dy.^2)<50;
im = reshape(w_reg(:,1),150,150);
st = std(im(mask));
c = redblue(256);

figure(3)
imagesc(im./st)
title("cell3")
colormap(c)
caxis([-10,10])
daspect([1 1 1])

%% CRM
clear("raw_image")

m_long_c = reshape(m_cone(33:182,33:182,:), 150*150, 9750)'; % make a bit smaller to aid with speed
l_long_c = reshape(l_cone(33:182,33:182,:), 150*150, 9750)';

cell_activity1 = tdfread("../../data/File110Ron.txt").x0x2D2;
cell_activity2 = tdfread("../../data/File167Ron.txt").x0x2D2; 
cell_activity3 = tdfread("../../data/File194Ron.txt").x0x2D2;
cell_activity4 = tdfread("../../data/File270+L-Moff.txt").x0;

cell_activity5 = tdfread("../../data/File130Gn.txt").x0x2D2;
cell_activity6 = tdfread("../../data/File278Gon.txt").x0x2D2;
cell_activity7 = tdfread("../../data/File284Gon.txt").x0;
cell_activity8 = tdfread("../../data/File287Gon.txt").x0x2D2;

cell_activity9 = tdfread("../../data/File078Moff.txt").x0x2D2;
cell_activity10 = tdfread("../../data/File256Moff.txt").x0x2D2;
cell_activity11 = tdfread("../../data/File258Mon.txt").x0x2D2;
cell_activity12 = tdfread("../../data/File299Mon.txt").x0x2D2;

% X = [cell_activity1, cell_activity2, cell_activity3, cell_activity4, cell_activity5, cell_activity6, cell_activity7];

X = [cell_activity1, cell_activity2, cell_activity3, cell_activity4, cell_activity5, cell_activity6, cell_activity7, cell_activity8, cell_activity9, cell_activity10, cell_activity11, cell_activity12];

X = circshift(X, -3)';
X = zscore(X')';

Y = m_long_c(751:9750,:);
Y = zscore(Y)';

S = X;
T = l_long_c(751:9750,:) + m_long_c(751:9750,:);
T = zscore(T)';

%
Cxx = X*X';
Cyy = (Y * Y')./9750;
Cxy = (X * Y')./9750; 
Dxy = (S * T')./9750;

tic
gamma = 5*1e-3 * trace(Cyy);
[w_xCRM, w_yCRM, lambda3, Wxs, Wys, lambdas, corrs] = crm(Cxx, Cyy, Cxy, Dxy, gamma=gamma);
toc

%%
[dx,dy] = meshgrid(-75:74,-75:74);
mask = sqrt(dx.^2 + dy.^2)<50;
im = reshape(w_yCRM(:,1),150,150);
me = mean(im(mask));
st = std(im(mask));
colorloading = (Cyy * w_yCRM);
c = redblue(256);

%%
figure(1)
imagesc((im - me)./st)
title("0.005 Parvo CRM M in luminance null space")
colormap(c)
clim([-6,6])
colorbar;
daspect([1 1 1])

figure(2),clf;
bar([1,2,3],[mean(w_xCRM(1:4,1)), mean(w_xCRM(9:12,1)), mean(w_xCRM(5:8,1))])
hold on;
scatter([1,1,1,1], w_xCRM(1:4,1))
scatter([3,3,3,3], w_xCRM(5:8,1))
scatter([2,2,2,2], w_xCRM(9:12,1))
ylabel("CRM weight on cells")

figure(3)
imagesc(reshape(colorloading,150,150))
title("loading")
colorbar;
daspect([1 1 1])

[h, p] = ttest2(w_xCRM(1:4,1), w_xCRM(5:8,1))
[h, p] = ttest(w_xCRM(9:12,1))

