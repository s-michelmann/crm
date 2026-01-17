%% Get Stimulus, BASED ON OLD PAPER
% vidcap = cv2.VideoCapture('../1x10_256.mpg')
% counter = 0
% success = True
% s = np.zeros((256,256,9750))
% m = np.zeros((256,256,9750))
% l = np.zeros((256,256,9750))
% while success and (counter < 9750):
%     success, image = vidcap.read()
%     scone, mcone, lcone = image_to_cone(image)
%     s[:,:,counter] = scone
%     m[:,:,counter] = mcone
%     l[:,:,counter] = lcone
%     counter += 1
% from scipy.io import savemat
% mdic = {"m_cone": m[20:236,20:236,:]}
% savemat("matlab_flowershow_m.mat", mdic)
% mdic = {"l_cone": l[20:236,20:236,:]}
% savemat("matlab_flowershow_l.mat", mdic)
% mdic = {"s_cone": s[20:236,20:236,:]}
% savemat("matlab_flowershow_s.mat", mdic)

close all;
clear all;

load('matlab_flowershow_m.mat')
load('matlab_flowershow_l.mat')

mm   = median(m_cone(:));
stdm = std(m_cone(:));
m_signal  = (m_cone - mm) ./ stdm;

ml = median(l_cone(:));
stdl = std(l_cone(:));

l_signal  = (l_cone - ml ) ./ stdl;

clear("m_cone", "l_cone")


%% Compress with PCA and save (will take a few minutes)

m_long = reshape(m_signal,216*216,9750)';
l_long = reshape(l_signal,216*216,9750)';

[coeff,score,~] = pca([m_long; l_long]);

save("./pcaresults_ML.mat",'-v7.3')

%%
% load("./pcaresults_ML.mat")

top_latents_m = score(1:9750,1:500);
top_latents_l = score(9751:end,1:500);

m_pca = (top_latents_m*coeff(:,1:500)');
l_pca = (top_latents_l*coeff(:,1:500)');

1-var(m_pca(:)- m_long(:)) ./ var(m_long(:))

1-var(l_pca(:)- l_long(:)) ./ var(l_long(:))

%% Get a file

cell_activity = tdfread("./File110Ron.txt").x0x2D2;

dt = 1000./150; % in units ms.
shift = 0;

time_cell = (0:length(cell_activity)-1)*dt + 5000 - shift*6.66;
 
figure(1),clf;
plot(time_cell, cell_activity'./std(cell_activity), 'r')
hold on;

l_test = reshape(mean(mean(l_signal,1),2),9750,1);
m_test = reshape(mean(mean(m_signal,1),2),9750,1);

plot((0:length(l_test)-1)*dt, l_test./std(l_test), 'b');
plot((0:length(m_test)-1)*dt, m_test./std(m_test), 'g');

xlim([54000,57000])
ylim([-10, 10])

figure(77)
scatter(l_test, m_test)
xlabel("L")
ylabel("M")
corrcoef(l_test, m_test)

%% For the CRM stuff

cell_activity1 = tdfread("./File110Ron.txt").x0x2D2; %
cell_activity2 = tdfread("./File167Ron.txt").x0x2D2; % 
cell_activity3 = tdfread("./File194Ron.txt").x0x2D2; %

cell_activity4 = tdfread("./File130Gn.txt").x0x2D2; %
cell_activity5 = tdfread("./File278Gon.txt").x0x2D2; %
cell_activity6 = tdfread("./File284Gon.txt").x01; %
cell_activity7 = tdfread("./File287Gon.txt").x0x2D2; %

%%

X = [cell_activity1, cell_activity2, cell_activity3, cell_activity4, cell_activity5, cell_activity6, cell_activity7];
X = circshift(X, -3)';

X = (X - mean(X(:))) ./ std(X(:));
Y = top_latents_l(751:9750,:)'; % chop off first 5s frames

S = X;
T = top_latents_l(751:9750,:)' + top_latents_m(751:9750,:)';

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';
D_xy = S*T';

gamma = 5e5;

I = eye(size(C_xx,1));
C_xx = C_xx + gamma*I;
I = eye(size(C_yy,1));
C_yy = C_yy + gamma*I;
%%
l_signal_center = reshape(mean(mean(l_signal(88:128,88:128,:),1),2),9750,1);

tic

[w_x1, w_y1, lambda3] = compute_weights(C_xx, C_yy, C_xy, 0*D_xy);
cc = corrcoef(w_y1'*Y, l_signal_center(751:9750));
if cc(2,1)<0
    w_x1 = -w_x1; w_y1 = -w_y1;
end

spatial = reshape(coeff(:,1:500),216,216,500);
RFall = zeros(216,216);
for idx = 1:500
     RFall = RFall + w_y1(idx)*spatial(:,:,idx);
end

[w_x2, w_y2, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy);
cc = corrcoef(w_y2'*Y, l_signal_center(751:9750));
if cc(2,1)<0
    w_x2 = -w_x2; w_y2 = -w_y2;
end

spatial = reshape(coeff(:,1:500),216,216,500);
RFcol = zeros(216,216);
for idx = 1:500
     RFcol = RFcol + w_y2(idx)*spatial(:,:,idx);
end

toc

%%
c = redblue(256);
figure(1)
subplot(1,3,1)
imagesc(RFall)
clim([-1.1, 1.1]*max(abs(RFall(:))))
colormap(c)
colorbar;
daspect([1 1 1])
title("Response to L")

subplot(1,3,2)
imagesc(RFcol)
clim([-1.5, 1.5]*max(abs(RFcol(:))))
colormap(c)
colorbar;
daspect([1 1 1])
title("Response to L in luminance nullspace")

subplot(1,3,3)
imagesc(-RFcol)
clim([-1.5, 1.5]*max(abs(RFcol(:))))
colormap(c)
colorbar;
daspect([1 1 1])
title("Response to L inv.")


figure(2),clf;

scatter(w_x1, w_x2)
hold on;
plot([1e-4,7e-4],[0,0], 'k--')
xlabel("weights CCA")
ylabel("weights CRM")