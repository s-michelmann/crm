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

[r_wxCRM, r_wyCRM, r_lamCRM, wxcxywyCRM, wxdxywyCRM, wxcxxwxCRM, wycyywyCRM] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);

%%

w_xCCA = r_wxCCA(:,1);
w_xCRM = r_wxCRM(:,1);

w_yCCA = r_wyCCA(:,1);
w_yCRM = r_wyCRM(:,1);

figure(1),clf;
subplot(2,3,1)
imagesc(t, f, X)
a=colorbar;
caxis([-10,20])
a.Label.String = 'log(Power)'
title("Spectrogram of HPC")
ylabel("Frequency [Hz]")
xlabel("Time [s]")
set(gca, 'tickdir','out');
text(-500,-10, "A", 'FontSize', 16)

subplot(2,3,4)
imagesc(t, f, Y)
b=colorbar;
caxis([-10,20])
b.Label.String = 'log(Power)'
title("Spectrogram of mPFC")
xlabel("Time [s]")
ylabel("Frequency [Hz]")
set(gca, 'tickdir','out');
text(-500,-10, "B", 'FontSize', 16)

subplot(2,3,2)
plot(f, w_xCCA, 'o-')
hold on
plot(f, w_yCCA, 'o-')
plot([0,110], [0,0],'k--')
title("Canonical Vectors")
xlabel("Frequency [Hz]")
ylabel("Weight")
set(gca, 'tickdir','out');
legend('HPC', 'mPFC', 'Location','northwest')
text(-20,6.4e-4, "C", 'FontSize', 16)

subplot(2,3,3)
plot(X'*w_xCCA)
hold on
plot(Y'*w_yCCA)
title("CCA Components")
xlim([2100,2200])
xlabel("Time [s]")
set(gca, 'tickdir','out');
legend('HPC', 'mPFC', 'Location','southwest')
text(2075, 0.0205, "D", 'FontSize', 16)

subplot(2,3,5)
plot(f,w_xCRM, 'o-')
hold on
plot(f,w_yCRM, 'o-')
plot([0,110],[0,0],'k--')
title("CRM Vectors")
xlabel("Frequency [Hz]")
ylabel("Weight")
set(gca, 'tickdir','out');
legend('HPC', 'mPFC', 'Location','southeast')
text(-20,2.8e-3, "E", 'FontSize', 16)

subplot(2,3,6)
plot(X'*w_xCRM)
hold on
plot(Y'*w_yCRM)
title("CRM Components")
xlabel("Time [s]")
set(gca, 'tickdir','out');
xlim([2100,2200])
legend('HPC', 'mPFC', 'Location','southwest')
text(2075, 0.05, "F", 'FontSize', 16)
%exportgraphics(figure(1), 'HPC_mPFC_spec.pdf');

toc
