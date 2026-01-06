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

luminance_raw = m_cone + l_cone;
mlr   = mean(luminance_raw(:));
stdlr = std(luminance_raw(:));

luminance  = (luminance_raw - mlr) ./ stdlr;

chromatic_raw = m_cone - l_cone;
mcr = mean(chromatic_raw(:));
stdcr = std(chromatic_raw(:));

chromatic  = (chromatic_raw - mcr ) ./ stdcr;

clear("chromatic_raw", "luminance_raw", "m_cone", "l_cone")
mean(mean(chromatic(200:226,200:226,:),1),2)
%% Compress with PCA and save (will take a few minutes)

l = reshape(luminance,216*216,9750)';
lc = reshape(chromatic,216*216,9750)';

[coeff,score,~] = pca([l; lc]);

save("./pcaresults.mat",'-v7.3')

%%
%combined = [l; lc];
%top_latents_all = score(:,1:500) *  coeff(:,1:500)';
%corrcoef(top_latents_all(:), combined(:))

top_latents_lum = score(1:9750,1:500);
top_latents_col = score(9751:end,1:500);

lum_pca = (top_latents_lum*coeff(:,1:500)');
col_pca = (top_latents_col*coeff(:,1:500)');

1-var(lum_pca(:)- l(:)) / var(l(:))

1-var(col_pca(:)- lc(:)) / var(lc(:))


%%
%load("./pcaresults.mat")
%% Get a file

cell_activity = tdfread("./File110Ron.txt").x0x2D2;

dt = 1000./150; % in units ms.
shift = 0;

time_cell = (0:length(cell_activity)-1)*dt + 5000 - shift*6.66;
 
figure(1),clf;
plot(time_cell, cell_activity'./std(cell_activity), 'r')
hold on;

lumtest = reshape(mean(mean(luminance,1),2),9750,1);
coltest = reshape(mean(mean(chromatic,1),2),9750,1);

plot((0:length(lumtest)-1)*dt, lumtest./std(lumtest), 'b');
plot((0:length(lumtest)-1)*dt, coltest./std(lumtest), 'g');

xlim([54000,57000])
ylim([-10, 10])

% myFiles = dir('*.txt');
% for k = 1:length(myFiles)
%   FileName = myFiles(k).name;
%   % fullFileName = fullfile('./', baseFileName);
%   fprintf(1, 'Now reading %s\n', FileName);
% 
%   data = tdfread(FileName);
% 
% end


%% For the CRM stuff

cell_activity1 = tdfread("./File110Ron.txt").x0x2D2; % -0.24 - -0.21
cell_activity2 = tdfread("./File167Ron.txt").x0x2D2; % -0.19 - -0.05
cell_activity3 = tdfread("./File194Ron.txt").x0x2D2; %-0.16 - -0.05

cell_activity4 = tdfread("./File130Gn.txt").x0x2D2; %0.707
cell_activity5 = tdfread("./File278Gon.txt").x0x2D2; % 0.87
cell_activity6 = tdfread("./File284Gon.txt").x01; % 0.47 - 0.16
cell_activity7 = tdfread("./File287Gon.txt").x0x2D2; %0.77 - 0.45

%%
% Xraw = [cell_activity1, cell_activity2, cell_activity3]';
% Xraw = [cell_activity4, cell_activity5, cell_activity6, cell_activity7]';
% Xraw = [cell_activity1, cell_activity2, cell_activity3, cell_activity4, cell_activity5, cell_activity6]';

%Xraw = [cell_activity1, cell_activity4]';
%X = Xraw - mean(Xraw(:));

X = circshift(cell_activity1, -3)';
X = X - mean(X);

Y = top_latents_col(751:9750,:)'; % chop off first 5s frames
S = X;
T = top_latents_lum(751:9750,:)';

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';
D_xy = S*T';

gamma = 1e7;

I = eye(size(C_xx,1));
C_xx = C_xx + gamma*I; %Tuzhilina, Tozzi, Hastie (2021)
I = eye(size(C_yy,1));
C_yy = C_yy + gamma*I;
%%
color_signal = reshape(mean(mean(chromatic(88:128,88:128,:),1),2),9750,1);
lumin_signal = reshape(mean(mean(luminance(88:128,88:128,:),1),2),9750,1);

tic

[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, 0*D_xy);
cc = corrcoef(w_y'*Y, lumin_signal(751:9750));
if cc(2,1)<0
    w_x = -w_x; w_y = -w_y;
end
% corrcoef(w_x'*X, w_y'*Y)
% corrcoef(w_x'*S, w_y'*T)
spatial = reshape(coeff(:,1:500),216,216,500);
RFall = zeros(216,216);
for idx = 1:500
     av = mean(score(:,idx));
     RFall = RFall + (w_y(idx)-av)*spatial(:,:,idx);
end
% corrcoef(w_y'*Y, mean(l(751:9750,:),2))
% corrcoef(w_y'*Y, mean(lc(751:9750,:),2))

[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy);
cc = corrcoef(w_y'*Y, color_signal(751:9750));
if cc(2,1)<0
    w_x = -w_x; w_y = -w_y;
end
% corrcoef(w_x'*X, w_y'*Y)
% corrcoef(w_x'*S, w_y'*T)
spatial = reshape(coeff(:,1:500),216,216,500);
RFcol = zeros(216,216);
for idx = 1:500
     av = mean(score(:,idx));
     RFcol = RFcol + (w_y(idx)-av)*spatial(:,:,idx);
end
% corrcoef(w_y'*Y, mean(l(751:9750,:),2))
% corrcoef(w_y'*Y, mean(lc(751:9750,:),2))

% RFref = zeros(216,216);
% for idx = 1:100
%     RFref = RFref + av*spatial(:,:,idx);
% end

toc

c = redblue(256);
figure(1)
subplot(1,2,1)
imagesc(RFall)
hold on;
plot(108,108,'k.')
clim([-1.1, 1.1]*max(abs(RFall(:))))
colormap(c)
colorbar;
daspect([1 1 1])
title("No constraint")

subplot(1,2,2)
imagesc(RFcol)
hold on;
plot(108,108,'k.')
clim([-1.1, 1.1]*max(abs(RFcol(:))))
colormap(c)
colorbar;
daspect([1 1 1])
title("M-L | M+L = 0")


% figure(2)
% [ymds,~,disparities] = mdscale(pdist(X),1);
% % scatter(ymds(:,1), ymds(:,2), [], w_x)
% scatter(ymds, w_x)
% colormap(c)
% clim([-0.01, 0.01])

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Movie Stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cell_activity = tdfread("./File110Ron.txt").x0x2D2;
cell_activity = tdfread("./File130Gn.txt").x0x2D2;

Y = chromatic(:,:,751:9750); % chop off first 5s frames

results = zeros(216,216,201);
shiftcounter = 1;
for shift = [-3]
    disp(shiftcounter);

    X = circshift(cell_activity, shift); % neural data
    Mdata = tensorprod(Y,X,3,1);
    Mshuffles = zeros(216,216,100);

    for i = 1:200
        Xrand = circshift(cell_activity, randi(6000)+3000); % neural data
        Mshuffles(:,:,i) = tensorprod(Y,Xrand,3,1);
    end
    
    mu = mean(Mshuffles,3);
    sd = sqrt(mean(Mshuffles.^2,3) - mu.^2);
    
    results(:,:,shiftcounter) = (Mdata - mu) ./ sd;

    shiftcounter = shiftcounter + 1;
end

%%
c = redblue(256);
colormap(c)
v = VideoWriter("RF0.avi");
open(v)

for i = 1:201
   imagesc((0:216)*(4/256), (0:216)*(4/256), results(:,:,i) )
   clim([-40,40])
   colormap(c)
   xlabel("Deg")
   ylabel("Deg")
   daspect([1 1 1])
   frame = getframe(gcf);
   writeVideo(v,frame)
end

close(v)