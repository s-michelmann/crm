% Data citation: https://www.jneurosci.org/content/45/10/e1241242025 Rosenblum et al. 2025

close all;
clear all;

nff = 2048;
fs = 2000;
nsc = 2000; % 1s chunks for spectrum
nov = floor(nsc/2);

load('../../data/PFC_black.mat')
y = Samples(:);
y = (y-mean(y))./std(y);
[sy,f,t] = spectrogram(y,hamming(nsc, 'periodic'),nov,nff,fs);
Y = abs(sy).^2;

load('../../data/HPC_blue.mat')
x = Samples(:);
x = (x-mean(x))./std(x);
[sx,f,t] = spectrogram(x,hamming(nsc, 'periodic'),nov,nff,fs);
X = abs(sx).^2;

X = log(X(f<100,:));
Y = log(Y(f<100,:));
f = f(f<100);



%%
% use CRM to denoise, see spectral_denoising.m for example on simulation.

Cxy = X*Y';
Cxx = X*X';
Cyy = Y*Y';

sfilt = 10;
ffilt = 60;

filter = exp( - (f-ffilt).^2 ./ (2*sfilt*sfilt));

S = filter.*Y;
T = filter.*X;  

Dxy = S*T';
a = norm(Cxy,'fro')/norm(Dxy,'fro');
Dxy = a*Dxy;

gamma = 0.01 * trace(Cyy);

[w_xCRM, w_yCRM, lambda3, Wxs, Wys, lambdas, corrs] = crm(Cxx, Cyy, Cxy, Dxy, gamma=gamma);


%% 
% Residualize and use CCA
fnoise = 2*3.14*60;
t = (1:length(x))/fs;

Xcleaned = zeros(size(x));
Xfit = zeros(size(x));
b0 = sin(fnoise*t)' \ x;
b1 = cos(fnoise*t)' \ x;
xfit =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
xcleaned = x -  xfit';

Ycleaned = zeros(size(y));
Yfit = zeros(size(y));
b0 = sin(fnoise*t)' \ y;
b1 = cos(fnoise*t)' \ y;
yfit =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
ycleaned = y -  yfit';

[sy,f,t] = spectrogram(ycleaned,hamming(nsc, 'periodic'),nov,nff,fs);
Ycleaned = abs(sy).^2;

[sx,f,t] = spectrogram(xcleaned,hamming(nsc, 'periodic'),nov,nff,fs);
Xcleaned = abs(sx).^2;

Xcleaned = log(Xcleaned(f<100,:));
Ycleaned = log(Ycleaned(f<100,:));
f = f(f<100);

% "Clean" Covariance Matrices
C_xy = Xcleaned*Ycleaned';
C_xx = Xcleaned*Xcleaned';
C_yy = Ycleaned*Ycleaned';

[w_xCCA, w_yCCA, lambda3, Wxs, Wys, lambdas, corrs] = crm(C_xx, C_yy, C_xy, 0*Dxy);

imagesc(t, f, Ycleaned)


%%
load('../../data/VT1.mat')
tt = (Timestamps' - min(TimeStamps))*1e-6; % From behavior, measured in mus.
positionX = [];
positionY = [];
for timepoint = t % t comes from the spectrum. 4813 time points in seconds
    [~, idx] = min( (tt - timepoint).^2 );
    positionX = [positionX, ExtractedX(median(idx))];
    positionY = [positionY, ExtractedY(median(idx))];
end

%%


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
crmresult = X'*w_xCRM;
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
b.Label.String = 'HPC CRM Component (z-scored)';
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
b.Label.String = 'HPC CCA Component (z-scored)';
text(50, 490, "f", 'FontSize', 16)

annotation('arrow', [0.52, 0.48], [0.43, 0.4]);

%exportgraphics(figure(1), 'fig5.pdf');


%
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
result = X'*w_xCRM;

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