% clear all; close all;
rng(1);

%% Simulate data
fs = 500;               % Sampling frequency (Hz)
T = 1000;               % Duration (seconds)
t = (0:1/fs:T-1/fs)';   % Time vector
f_noise = 50;           % The "contaminant" frequency
n_channels = 2;         % Number of signals
spectral_width = 4;     % Approximate bandwidth  (Hz)

raw_noise = randn(length(t), n_channels);
[b, a] = butter(2, 100/(fs/2)); 
background = filter(b, a, raw_noise);
background = background ./ std(background);

% Shared amplitude envelope
modulation = randn(length(t), 1);
[b, a] = butter(4, 0.5/(fs/2)); 
shared_signal_amplitude = abs(filter(b, a, modulation));
shared_signal_amplitude = shared_signal_amplitude ./ std(shared_signal_amplitude);

% Shared non-stationary hum
hum = sin(2*pi*f_noise*t); 
hum = hum ./ std(hum); 

for i = 1:n_channels
    f_signal = 10 + 70*(i-1); % 10 Hz and 80 Hz center frequency
    
    fm_noise = randn(length(t), 1);     % Generate smooth, slow random frequency deviations
    [b_fm, a_fm] = butter(2, 2/(fs/2)); % 2 Hz lowpass for FM noise
    fm_noise = filter(b_fm, a_fm, fm_noise);
    fm_noise = fm_noise ./ std(fm_noise); 
    phase_drift = cumsum(fm_noise) * (1/fs) * (spectral_width * pi); % Use phase drift to control the spread
    carrier = sin(2*pi*f_signal*t + phase_drift); % Create carrier with constant amplitude but broad spectrum
    signal = shared_signal_amplitude .* carrier;     % Apply the shared envelope
    
    simulated_signals(:,i) = background(:,i) + signal + 4*hum.*(cos(t*0.02).^2).*(t<(T/2));
end

%% Compute Spectrograms

[X,f,t_spec] = computer_spectrum(simulated_signals(:,1), fs);
[Y,f,t_spec] = computer_spectrum(simulated_signals(:,2), fs);

figure(1); clf;
subplot(2,2,1)
plot(t,simulated_signals(:,1))
xlim([499.5,501])
ylim([-6,6])
subplot(2,2,2)
plot(t,simulated_signals(:,2))
xlim([499.5,501])
ylim([-6,6])
subplot(2,2,3)
imagesc(t_spec,f,log(X))
xlabel("time")
ylabel("frequency")
caxis([-10,3])
subplot(2,2,4)
imagesc(t_spec,f,log(Y))
caxis([-10,3])
xlabel("time")
ylabel("frequency")

%% Perform CRM/CCA

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
xlabel('Frequency (Hz)')
ylabel('CCA weight')
legend('Channel 1 (w_x)', 'Channel 2 (w_y)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% "remove" noise with residualized CCA v. CRM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Isolate the 60 Hz noise to construct the penalty matrix Dxy
[b_60, a_60] = butter(2, [48, 52]/(fs/2), 'bandpass');
noise_x = filter(b_60, a_60, simulated_signals(:,1));
noise_y = filter(b_60, a_60, simulated_signals(:,2));

[X_noise, f, t_spec] = computer_spectrum(noise_x, fs);
[Y_noise, f, t_spec] = computer_spectrum(noise_y, fs);

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
[X_clean, f, t_spec] = computer_spectrum(cleaned_signals(:,1), fs);
[Y_clean, f, t_spec] = computer_spectrum(cleaned_signals(:,2), fs);

% Clean Covariance Matrices
C_xx_clean = (X_clean * X_clean') / (N - 1);
C_yy_clean = (Y_clean * Y_clean') / (N - 1);
C_xy_clean = (X_clean * Y_clean') / (N - 1);

figure(3); clf;
subplot(2,2,1)
plot(t,cleaned_signals(:,1))
xlim([499.5,501])
ylim([-6,6])
subplot(2,2,2)
plot(t,cleaned_signals(:,2))
xlim([499.5,501])
ylim([-6,6])
subplot(2,2,3)
imagesc(t_spec,f,log(Y_clean))
xlabel("time")
ylabel("frequency")
caxis([-10,3])
subplot(2,2,4)
imagesc(t_spec,f,log(X_clean))
caxis([-10,3])
xlabel("time")
ylabel("frequency")

% Normal CCA on Cleaned Data (Dxy = 0)
[w_x_cca, w_y_cca, ~] = compute_weights(C_xx_clean, C_yy_clean, C_xy_clean, 0*Dxy, 'f', 1, 'gamma', 0.05, 'chlsky', true);

% Plot the Comparison
figure(4); clf;
subplot(2,1,1)
plot(f, w_x_crm, 'r', 'LineWidth', 2); hold on;
plot(f, w_x_cca, 'k--', 'LineWidth', 1.5);
ylabel('CRM and reg.CCA weights');
legend('CRM', 'CCA (Cleaned Data)');
subplot(2,1,2)
plot(f, w_y_crm, 'b', 'LineWidth', 2); hold on;
plot(f, w_y_cca, 'k--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('CRM and reg.CCA weights');
legend('CRM', 'CCA (Cleaned Data)');

function [X,f,t] = computer_spectrum(x, fs)
    nff = 512;
    nsc = 200; 
    nov = floor(nsc/2);  
    x = (x - mean(x)) ./ std(x);
    [sx, f, t] = spectrogram(x, hamming(nsc, 'periodic'), nov, nff, fs);
    X = abs(sx).^2;
    X = X(f<100, :);
    X = X ./ std(X(:)); % Variance normalization for easier numerics.
    f = f(f<100);
end