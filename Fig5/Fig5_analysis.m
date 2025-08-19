% Data citation: https://www.jneurosci.org/content/45/10/e1241242025 Rosenblum et al. 2025

close all;
clear all;

tic % 511 seconds

%% Load data + Make spectra

nff = 1024;
fs = 2000;
nsc = 2000; % 1s chunks for spectrum
nov = floor(nsc/2);

load('PFC_black.mat')
y = Samples(:);
y = (y-mean(y))./std(y);
[sy,f,t] = spectrogram(y,hamming(nsc, 'periodic'),nov,nff,fs);
Y = abs(sy).^2;

load('HPC_blue.mat')
x = Samples(:);
x = (x-mean(x))./std(x);
[sx,f,t] = spectrogram(x,hamming(nsc, 'periodic'),nov,nff,fs);
X = abs(sx).^2;

X = log(X(f<100,:));
Y = log(Y(f<100,:));
f = f(f<100);

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

[r_wxCCA, r_wyCCA, r_lamCCA, wxcxywyCCA, wxdxywyCCA, wxcxxwxCCA, wycyywyCCA] = compute_weights_full(C_xx, C_yy, C_xy, 0*C_xy);

% use CRM to denoise, see spectral_denoising.m for example on simulation.
sfilt = 10;
ffilt = 60;

filter = exp( - (f-ffilt).^2 ./ (2*sfilt*sfilt));

S = filter.*Y;
T = filter.*X;  

D_xy = S*T';
a = norm(C_xy,'fro')/norm(D_xy,'fro');
D_xy = a*D_xy;
m
[r_wxCRM, r_wyCRM, r_lamCRM, wxcxywyCRM, wxdxywyCRM, wxcxxwxCRM, wycyywyCRM] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);


load('VT1.mat')
tt = (Timestamps' - min(TimeStamps))*1e-6; % From behavior, measured in mus.
positionX = [];
positionY = [];
for timepoint = t % t comes from the spectrum. 4813 time points in seconds
    [~, idx] = min( (tt - timepoint).^2 );
    positionX = [positionX, ExtractedX(median(idx))];
    positionY = [positionY, ExtractedY(median(idx))];
end

%%

w_xCCA = r_wxCCA(:,1);
w_xCRM = r_wxCRM(:,1);

w_yCCA = r_wyCCA(:,1);
w_yCRM = r_wyCRM(:,1);

col1 = "#7678ed";
col2 = "#f35b04";

figure(1),clf;
subplot(2,3,1)
imagesc(t, f, X)
axis xy
a=colorbar;
caxis([-10,20])
a.Label.String = 'log(Power)'
title("Spectrogram of HPC")
ylabel("Frequency [Hz]")
xlabel("Time [s]")
set(gca, 'tickdir','out');
text(-500,110, "a", 'FontSize', 16)

subplot(2,3,2)
imagesc(t, f, Y)
axis xy
b=colorbar;
caxis([-10,20])
b.Label.String = 'log(Power)'
title("Spectrogram of mPFC")
xlabel("Time [s]")
ylabel("Frequency [Hz]")
set(gca, 'tickdir','out');
text(-500,110, "b", 'FontSize', 16)

subplot(2,3,3)
plot(f, w_xCRM*1000, 'o-', 'color', col1)
hold on
plot(f, w_yCRM*1000, 'o-', 'color', col2)
plot([0,110], [0,0],'k--')
ylim([-8, 7])
xlim([0,100])
xlabel("Frequency [Hz]")
ylabel("CRM weight")
set(gca, 'tickdir','out');
legend('HPC', 'mPFC', 'Location','northwest')
ax = gca
box(ax,'off')
text(-20,8.3, "c", 'FontSize', 16)
axes('Position',[0.785 0.63 0.1, 0.09])
box on
plot(f,w_xCCA*10000, 'o-', 'color', col1)
hold on;
plot(f,w_yCCA*10000, 'o-', 'color', col2)
ylabel({'CCA'; 'weight'})
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,4)
plot(100*X'*w_xCRM, 'o-','color', col1)
hold on
plot(100*Y'*w_yCRM, 'o-','color', col2)
plot([2100,2200],[0,0],['k--'])
ylabel("CRM components")
xlabel("Time [s]")
set(gca, 'tickdir','out');
xlim([2100,2160])
text(2087, 5.5, "d", 'FontSize', 16)
ylim([-15, 4])
ax = gca
box(ax,'off')
axes('Position',[0.22 0.2 0.1, 0.1])
box on
plot(f, Y*(Y'*w_yCRM) ./ std(Y*(Y'*w_yCRM)),'o-', 'color', col2)
hold on
plot(f, X*(X'*w_xCRM) ./ std(X*(X'*w_xCRM)),'o-', 'color', col1)
plot([0,100],[0,0],'k--')
xlabel("Frequency [Hz]")
ylabel({'CRM'; 'Loading'})
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,5)
crmresult = (X'*w_xCRM) + (Y'*w_yCRM);
crmresult = smoothSpatial([positionX; positionY]', crmresult, 10);
crmresult = (crmresult - mean(crmresult)) ./ std(crmresult);
scatter(positionX, positionY, 100, crmresult, '.')
caxis([-2.5,2.5])
xlim([180,680])
ylim([40,450])
xlabel("X position in the maze")
ylabel("Y position in the maze")
set(gca, 'tickdir','out');
b=colorbar;
b.Label.String = 'CRM Component (z-scored)';
text(50, 480, "e", 'FontSize', 16)

subplot(2,3,6)
ccaresult = X'*w_xCCA;
ccaresult = smoothSpatial([positionX; positionY]', ccaresult, 10);
ccaresult = (ccaresult - mean(ccaresult)) ./ std(ccaresult);
scatter(positionX, positionY, 100, ccaresult, '.')
caxis([-2.5,2.5])
xlim([180,680])
ylim([40,450])
xlabel("X position in the maze")
ylabel("Y position in the maze")
set(gca, 'tickdir','out');
b=colorbar;
b.Label.String = 'CCA Component (z-scored)';
text(50, 490, "f", 'FontSize', 16)

annotation('arrow', [0.52, 0.48], [0.43, 0.4]);

%exportgraphics(figure(1), 'fig5.pdf');


%%
toc

%% Helpers
function newData = smoothSpatial(oldData, behaviorData, numSmooth)

    for i=1:size(oldData,1)
        distances = sqrt(sum(bsxfun(@minus, oldData, oldData(i,:)).^2,2));
        [~, indmin] = sort(distances,'ascend');
        closeData = behaviorData(indmin(1:numSmooth));
        newData(i) = sum(closeData)/length(closeData);
    end
end



%% Test for statistcs

figure(2),clf;

N_btstrps = 10000;
pos = sqrt( (positionX - 600).^2 + (positionY-250).^2 ); % distance from home cage
result = crmresult;

cc = corrcoef(pos, result);
bestc = cc(2,1);

for btstrp_idx = 1:N_btstrps
    cc = corrcoef(circshift(pos, randi(length(pos))), result);
    nulldistro(btstrp_idx) = cc(2,1);
end

p = (sum(bestc < nulldistro)+1) / N_btstrps;

hist(nulldistro,-0.2:0.01:0.2)
xlabel("corcoef(CRM; x-position in maze)")
hold on;
plot([bestc, bestc], [0,1000], 'r')
title(p)