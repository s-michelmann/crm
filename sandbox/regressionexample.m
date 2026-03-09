clear all; close all;

%% Simulate data
fs = 500;               % Sampling frequency (Hz)
T = 1000;               % Duration (seconds)
t = (0:1/fs:T-1/fs)';   % Time vector
f_noise = 50;           % The "contaminant" frequency
n_channels = 2;         % Number of signals
spectral_width = 4;     % Approximate bandwidth spread (Hz)

raw_noise = randn(length(t), n_channels);
[b, a] = butter(2, 100/(fs/2)); 
background = filter(b, a, raw_noise);
background = background ./ std(background);

% Shared amplitude envelope
modulation = randn(length(t), 1);
[b, a] = butter(4, 1/(fs/2)); 
shared_signal_amplitude = abs(filter(b, a, modulation));
shared_signal_amplitude = shared_signal_amplitude ./ std(shared_signal_amplitude);

% Shared non-stationary hum
hum = sin(2*pi*f_noise*t); 
hum = hum ./ std(hum); 

for i = 1:n_channels
    f_signal = 10 + 60*(i-1); % Yields 10 Hz and 70 Hz
    
    % Generate smooth, slow random frequency deviations
    fm_noise = randn(length(t), 1);
    [b_fm, a_fm] = butter(2, 2/(fs/2)); % 2 Hz lowpass for FM noise
    fm_noise = filter(b_fm, a_fm, fm_noise);
    fm_noise = fm_noise ./ std(fm_noise); 
    phase_drift = cumsum(fm_noise) * (1/fs) * (spectral_width * 2 * pi); % Use phase drift to control the spread in Hz
    carrier = sin(2*pi*f_signal*t + phase_drift); % Create carrier with constant amplitude but broad spectrum
    signal = shared_signal_amplitude .* carrier;     % Apply the shared envelope
    
    simulated_signals(:,i) = background(:,i) + signal + 3*hum.*(t < T/2);
end

%% Compute Spectrograms
nff = 512;
nsc = 200; 
nov = floor(nsc/2);  

x = simulated_signals(:,1);
x = (x - mean(x)) ./ std(x);
[sx,f,t_spec] = spectrogram(x, hamming(nsc, 'periodic'), nov, nff, fs);
X = abs(sx).^2;
X = X(f<100, :);

y = simulated_signals(:,2);
y = (y - mean(y)) ./ std(y);
[sy,f,t_spec] = spectrogram(y, hamming(nsc, 'periodic'), nov, nff, fs);
Y = abs(sy).^2;
Y = Y(f<100, :);
f = f(f<100);

figure(1); clf;
subplot(2,1,1)
imagesc(t_spec,f,X)

subplot(2,1,2)
imagesc(t_spec,f,Y)

%% Perform CRM/CCA
% X = X - mean(X, 2);
X = X ./ std(X(:));

% Y = Y - mean(Y, 2);
Y = Y ./ std(Y(:));

N = size(X, 2);
C_xy = (X * Y') / (N - 1);
C_xx = (X * X') / (N - 1);
C_yy = (Y * Y') / (N - 1);

Dxy = diag(ones(length(f), 1));
[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, 0*Dxy, 'f', 1, 'gamma', 0.05, 'chlsky', true);

figure(2); clf;
plot(f, w_x, 'r', 'LineWidth', 1.5)
hold on;
plot(f, w_y, 'b', 'LineWidth', 1.5)
title(sprintf("CCA Weights (Spectral Width: %d Hz)", spectral_width))
xlabel('Frequency (Hz)')
ylabel('Weight')
legend('Channel 1 (w_x)', 'Channel 2 (w_y)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% "remove" noise with residualized CCA v. CRM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Isolate the 60 Hz noise to construct the penalty matrix Dxy
[b_60, a_60] = butter(2, [48 52]/(fs/2), 'bandpass');
noise_x = filter(b_60, a_60, simulated_signals(:,1));
noise_y = filter(b_60, a_60, simulated_signals(:,2));

nx = (noise_x - mean(noise_x)) ./ std(noise_x);
[snx, f_full, ~] = spectrogram(nx, hamming(nsc, 'periodic'), nov, nff, fs);
X_noise = abs(snx).^2;
X_noise = X_noise(f_full < 100, :);
% % X_noise = X_noise - mean(X_noise, 2);
% X_noise = X_noise ./ std(X_noise, 0, 2);
X_noise = X_noise ./ std(X_noise(:));

ny = (noise_y - mean(noise_y)) ./ std(noise_y);
[sny, f_full, ~]  = spectrogram(ny, hamming(nsc, 'periodic'), nov, nff, fs);
Y_noise = abs(sny).^2;
Y_noise = Y_noise(f_full < 100, :);
% Y_noise = Y_noise - mean(Y_noise, 2);
% Y_noise = Y_noise ./ std(Y_noise, 0, 2);
Y_noise = Y_noise ./ std(Y_noise(:));

% The CRM penalization matrix (Cross-covariance of the confound)
N = size(X_noise, 2);
Dxy = (X_noise * Y_noise') / (N - 1);
[w_x_crm, w_y_crm, lambda3] = compute_weights(C_xx, C_yy, C_xy, Dxy, 'f', 1, 'gamma', 0.05, 'chlsky', true);

% Global regression
X_reg = [sin(2*pi*f_noise*t), cos(2*pi*f_noise*t), ones(size(t))];
beta = X_reg \ simulated_signals;
fit_reg = X_reg * beta;
cleaned_signals = simulated_signals - fit_reg;

% Spectrogram of residualized X
cx = cleaned_signals(:,1);
cx = (cx - mean(cx)) ./ std(cx);
[scx, f_res, ~] = spectrogram(cx, hamming(nsc, 'periodic'), nov, nff, fs);
X_clean = abs(scx).^2;
X_clean = X_clean(f_res < 100, :);
% X_clean = X_clean - mean(X_clean, 2);
% X_clean = X_clean ./ std(X_clean, 0, 2);
X_clean = X_clean ./ std(X_clean(:));

% Spectrogram of residualized Y
cy = cleaned_signals(:,2);
cy = (cy - mean(cy)) ./ std(cy);
[scy,f_res ~, ~] = spectrogram(cy, hamming(nsc, 'periodic'), nov, nff, fs);
Y_clean = abs(scy).^2;
Y_clean = Y_clean(f_full < 100, :);
% Y_clean = Y_clean - mean(Y_clean, 2);
% Y_clean = Y_clean ./ std(Y_clean, 0, 2);
Y_clean = Y_clean ./ std(Y_clean(:));
% Clean Covariance Matrices
C_xx_clean = (X_clean * X_clean') / (N - 1);
C_yy_clean = (Y_clean * Y_clean') / (N - 1);
C_xy_clean = (X_clean * Y_clean') / (N - 1);

figure(3); clf;
subplot(2,1,1)
imagesc(t_spec,f,X_clean)
title("Resizualized")

subplot(2,1,2)
imagesc(t_spec,f,Y_clean)

% Normal CCA on Cleaned Data (Dxy = 0)
[w_x_cca, w_y_cca, ~] = compute_weights(C_xx_clean, C_yy_clean, C_xy_clean, 0*Dxy, 'f', 1, 'gamma', 0.05, 'chlsky', true);

% Plot the Comparison
figure(4); clf;

subplot(2,1,1)
plot(f, w_x_crm, 'r', 'LineWidth', 2); hold on;
plot(f, w_x_cca, 'k--', 'LineWidth', 1.5);
title('Channel 1 Weights (w_x): CRM vs CCA post-Regression');
ylabel('Weight');
legend('CRM (Raw Data + Dxy)', 'CCA (Cleaned Data)');
grid on;

subplot(2,1,2)
plot(f, w_y_crm, 'b', 'LineWidth', 2); hold on;
plot(f, w_y_cca, 'k--', 'LineWidth', 1.5);
title('Channel 2 Weights (w_y): CRM vs CCA post-Regression');
xlabel('Frequency (Hz)');
ylabel('Weight');
legend('CRM (Raw Data + Dxy)', 'CCA (Cleaned Data)');
grid on;