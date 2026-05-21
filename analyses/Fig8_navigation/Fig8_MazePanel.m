% Data citation: https://www.jneurosci.org/content/45/10/e1241242025 Rosenblum et al. 2025


% close all;
% clear all;

%% Set spectral parameters and load behavior data
nff = 1024;
fs = 2000;
nsc = 2000; % 1s chunks for spectrum
nov = floor(nsc/2);

load('../../data/VT1.mat')
tt = (TimeStamps' - min(TimeStamps))*1e-6; % From behavior, measured in mus.

% CRM: Compute covariance matrices, and the Dxy via band-pass filtering
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

% Now that 't' exists from the spectrogram, we can extract behavior
positionX = zeros(1, length(t));
positionY = zeros(1, length(t));
for i = 1:length(t)
    [~, idx] = min( (tt - t(i)).^2 );
    positionX(i) = ExtractedX(idx);
    positionY(i) = ExtractedY(idx);
end

% Format, Center, and Normalize Data for CRM
X = log(X(f<100,:));
Y = log(Y(f<100,:));
f = f(f<100);

% Z-score across the time dimension (dim 2)
X = (X - mean(X, 2)) ./ std(X, 0, 2);
Y = (Y - mean(Y, 2)) ./ std(Y, 0, 2);

% Compute proper covariance matrices
N = size(X, 2);
Cxy = (X*Y') / (N-1);
Cxx = (X*X') / (N-1);
Cyy = (Y*Y') / (N-1);

sfilt = 2;
ffilt = 60;
filter = exp( - (f-ffilt).^2 ./ (2*sfilt*sfilt));

% S and T inherit the correct relative amplitude directly from Y and X
S = filter.*Y;
T = filter.*X;  
Dxy = (S*T') / (N-1); 

[w_xCRM, w_yCRM, lambda3, Wxs, Wys, lambdas, corrs] = crm(Cxx, Cyy, Cxy, Dxy, gamma=0.01, sparsity=0.4);

% Partial CCA: subtract noise signal, and perform CCA on residuals.
f_noise = 60;
t_raw = (0:length(y)-1)' / fs; % Create high-resolution time vector for the raw samples
M = [cos(2*pi*f_noise*t_raw), sin(2*pi*f_noise*t_raw)]; % Design matrix for 60Hz signal

load('../../data/PFC_black.mat')
y = Samples(:);
y = (y-mean(y))./std(y);

% Fit and subtract for PFC
beta_y = M \ y;         % Ordinary least squares fit
y_noise = M * beta_y;   % Reconstruct the 60Hz noise
y_clean = y - y_noise;  % Residual signal
[sy,f_cca,t_cca] = spectrogram(y_clean, hamming(nsc, 'periodic'), nov, nff, fs);
Ycleaned = abs(sy).^2;

load('../../data/HPC_blue.mat')
x = Samples(:);
x = (x-mean(x))./std(x);
beta_x = M \ x; 
x_noise = M * beta_x;
x_clean = x - x_noise;

[sx,f_cca,t_cca] = spectrogram(x_clean, hamming(nsc, 'periodic'), nov, nff, fs);
Xcleaned = abs(sx).^2;

% Format, Center, and Normalize Data for CCA
Xcleaned = log(Xcleaned(f_cca<100,:));
Ycleaned = log(Ycleaned(f_cca<100,:));

% Z-score across the time dimension
Xcleaned = (Xcleaned - mean(Xcleaned, 2)) ./ std(Xcleaned, 0, 2);
Ycleaned = (Ycleaned - mean(Ycleaned, 2)) ./ std(Ycleaned, 0, 2);

Cxy_cca = (Xcleaned*Ycleaned') / (N-1);
Cxx_cca = (Xcleaned*Xcleaned') / (N-1);
Cyy_cca = (Ycleaned*Ycleaned') / (N-1);

[w_xCCA, w_yCCA, lambda3_cca, Wxs_cca, Wys_cca, lambdas_cca, corrs_cca] = crm(Cxx_cca, Cyy_cca, Cxy_cca, 0*Dxy);

% Make plot
col1 = "#7678ed";
col2 = "#f35b04";
figure(1),clf;

subplot(2,4,1)
imagesc(t, f, X)
axis xy
a=colorbar;
clim([-10,20])
a.Label.String = 'log(Power)';
title("Frequency-centered Spectrogram of HPC")
ylabel("Frequency [Hz]")
xlabel("Time [s]")
set(gca, 'tickdir','out');

subplot(2,4,2)
imagesc(t, f, Y)
axis xy
b=colorbar;
clim([-10,20])
b.Label.String = 'log(Power)';
title("Frequency-centered Spectrogram of mPFC")
xlabel("Time [s]")
ylabel("Frequency [Hz]")
set(gca, 'tickdir','out');

subplot(2,4,3)
plot(f, w_xCRM./std(w_xCRM), '-', 'color', col1)
hold on
plot(f, w_yCRM./std(w_yCRM), '-', 'color', col2)
plot([0,110], [0,0],'k--')
ylim([-8, 7])
xlim([0,100])
xlabel("Frequency [Hz]")
ylabel("CRM weight")
set(gca, 'tickdir','out');
legend('HPC', 'mPFC', 'Location','northwest')
ax = gca;
box(ax,'off')

subplot(2,4,4)
plot(f,w_xCCA./std(w_xCCA), '-', 'color', col1)
hold on;
plot(f,w_yCCA./std(w_yCCA), '-', 'color', col2)
ylabel({'pCCA'; 'weight'})
set(gca, 'tickdir','out');

subplot(2,4,5)
plot(X'*w_xCRM./std(w_xCRM), '-','color', col1)
hold on
plot(Y'*w_yCRM./std(w_yCRM), '-','color', col2)
plot([2100,2200],[0,0],'k--')
ylabel("CRM components")
xlabel("Time [s]")
set(gca, 'tickdir','out');
xlim([2100,2160])
ax = gca;
box(ax,'off')

subplot(2,4,6)
plot(f, Y*(Y'*w_yCRM) ./ std(Y*(Y'*w_yCRM)),'-', 'color', col2)
hold on
plot(f, X*(X'*w_xCRM) ./ std(X*(X'*w_xCRM)),'-', 'color', col1)
plot([0,100],[0,0],'k--')
xlabel("Frequency [Hz]")
ylabel({'CRM'; 'Loading'})
set(gca, 'tickdir','out');
box(ax,'off')

subplot(2,4,7)
crmresult = X'*w_xCRM;
crmresult = smoothSpatial([positionX; positionY]', crmresult, 10);
crmresult = (crmresult - mean(crmresult)) ./ std(crmresult);
scatter(positionX, positionY, 100, crmresult, '.')
clim([-2.5,2.5])
xlim([180,680])
ylim([40,450])
xlabel("X position in the maze")
ylabel("Y position in the maze")
set(gca, 'tickdir','out');
b=colorbar;
b.Label.String = 'HPC CRM Component (z-scored)';

subplot(2,4,8)
ccaresult = Xcleaned'*w_xCCA; % Fixed to use Xcleaned instead of X
ccaresult = smoothSpatial([positionX; positionY]', ccaresult, 10);
ccaresult = (ccaresult - mean(ccaresult)) ./ std(ccaresult);
scatter(positionX, positionY, 100, ccaresult, '.')
clim([-2.5,2.5])
xlim([180,680])
ylim([40,450])
xlabel("X position in the maze")
ylabel("Y position in the maze")
set(gca, 'tickdir','out');
b=colorbar;
b.Label.String = 'HPC CCA Component (z-scored)';

figure(2),clf;

load('../../data/PFC_black.mat')
y = Samples(:);
y = (y-mean(y))./std(y);

load('../../data/HPC_blue.mat')
x = Samples(:);
x = (x-mean(x))./std(x);

plot(x(38000:44000))
hold on;
plot(y(38000:44000))
xlabel("3 seconds")



%% Helpers
function newData = smoothSpatial(oldData, behaviorData, numSmooth)

    for i=1:size(oldData,1)
        distances = sqrt(sum(bsxfun(@minus, oldData, oldData(i,:)).^2,2));
        [~, indmin] = sort(distances,'ascend');
        closeData = behaviorData(indmin(1:numSmooth));
        newData(i) = sum(closeData)/length(closeData);
    end
end