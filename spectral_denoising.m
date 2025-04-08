
rng(1)

T = 1000; % duration in time bins.
fsx = 20;
fsy = 80;
sa = 6;   % spectral width
As = -4;

fn = 60;
sn = 2;
An = 10;

A0 = 1; % background noise

% Make slowly varying signal time seris
f = As*make_timeseries(T,30); % signal time series.
g = An*make_timeseries(T,30)+3; % noise time series at 60 Hz

X = zeros(100,T);
Y = zeros(100,T);

for fr = 1:100        % Frequency bins
    for t = 1:T      % recordings
        X(fr,t) = A0*randn(1);
        X(fr,t) = X(fr,t) + f(t) * exp( - (fr-fsx).^2 ./ (2*sa*sa));
        X(fr,t) = X(fr,t) + g(t) * exp( - (fr-fn).^2 ./ (2*sn*sn));

        Y(fr,t) = A0*randn(1);
        Y(fr,t) = Y(fr,t) + f(t) * exp( - (fr-fsy).^2 ./ (2*sa*sa));
        Y(fr,t) = Y(fr,t) + g(t) * exp( - (fr-fn).^2 ./ (2*sn*sn));
    end
end


%% Perform the analysis
C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

[w_xCCA, w_yCCA, ~] = compute_weights(C_xx,C_yy,C_xy, 0*C_xy,1);

% use CRM to denoise
filt=zeros(100,1);
sfilt=2;
ffilt=60;
for fr = 1:100
    filt(fr) = exp( - (fr-ffilt).^2 ./ (2*sfilt*ffilt));
end
S = filt.*X;  % Same brain areas in different experiment. Just noise
T = filt.*Y;
D_xy = S*T';
[w_xCRM, w_yCRM, ~] = compute_weights(C_xx,C_yy,C_xy, D_xy,1);

%%
figure(1),clf;
subplot(2,3,1)
imagesc(X)
title("Spectrogram of area A")
ylabel("Frequency")
xlabel("Time")
set(gca, 'tickdir','out');
text(-100,-10, "A", 'FontSize', 16)

subplot(2,3,4)
imagesc(Y)
title("Spectrogram of area B")
xlabel("Time")
ylabel("Frequency")
set(gca, 'tickdir','out');
text(-100,-10, "B", 'FontSize', 16)

subplot(2,3,2)
plot(w_xCCA, 'o-')
hold on
plot(w_yCCA, 'o-')
plot([0,100],[0,0],'k--')
title("Canonical Vectors")
xlabel("Frequency")
ylabel("Weight")
set(gca, 'tickdir','out');
ylim([-5e-3, 5e-3])
text(-10,6e-3, "C", 'FontSize', 16)

subplot(2,3,3)
plot(X'*w_xCCA)
hold on
plot(Y'*w_yCCA)
title("CCA Components")
xlabel("Time")
set(gca, 'tickdir','out');
plot(f*std(X'*w_xCRM),'k.', 'LineWidth', 1)
ylim([-0.1, 0.1])
text(-200,0.12, "D", 'FontSize', 16)

subplot(2,3,5)
plot(w_xCRM, 'o-')
hold on
plot(w_yCRM, 'o-')
plot([0,100],[0,0],'k--')
title("CRM Vectors")
xlabel("Frequency")
ylabel("Weight")
set(gca, 'tickdir','out');
ylim([-5e-3, 5e-3])
text(-10,6e-3, "E", 'FontSize', 16)

subplot(2,3,6)
plot(X'*w_xCRM)
hold on
plot(Y'*w_yCRM)
title("CRM Components")
xlabel("Time")
set(gca, 'tickdir','out');
plot(f*std(X'*w_xCRM), 'k.', 'LineWidth', 1)
ylim([-0.1, 0.1])
text(-200, 0.12, "F", 'FontSize', 16)
exportgraphics(figure(1), 'simulation2.pdf');

function y = make_timeseries(T,w)
    data = randn(T, 1);
    y = smoothdata(data, "gaussian", w);
end