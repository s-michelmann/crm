
% close all;
% clear all;

load('../../data/E65.mat') % a struct with data & behavior


behavioralVariables = nic_output.behavioralVariables;
trialn = behavioralVariables.Trial;
%trials_use = unique(trialn); % use all trials
trials_use = [9,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,53,55,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210];

ROIactivities = nic_output.ROIactivities;
[T,N] = size(ROIactivities);
neural_data_all = double(ROIactivities( ismember(trialn, trials_use), :));

% Set rng() and load data

Datarange    = nansum(neural_data_all,2)>0;
Neurons      = nansum(neural_data_all,1)>1;
neural_data  = neural_data_all(Datarange, Neurons)';

behavioral_data = [behavioralVariables.Position(Datarange), ...
    behavioralVariables.Position_X(Datarange), ...
    behavioralVariables.Evidence(Datarange), ...
    behavioralVariables.Velocity(Datarange), ...
    behavioralVariables.Yvelocity(Datarange), ...
    behavioralVariables.Xvelocity(Datarange), ...
    behavioralVariables.Time(Datarange), ...
    behavioralVariables.ViewAngle(Datarange),...
    behavioralVariables.Choice(Datarange),...
    behavioralVariables.EvidenceL(Datarange),...
    behavioralVariables.TrialType(Datarange),...
    behavioralVariables.Collision(Datarange),...
    behavioralVariables.NumberOfCues(Datarange),...
    behavioralVariables.NearestCueType(Datarange),...
    behavioralVariables.EvidenceR(Datarange)]';

behavioral_data = [behavioralVariables.Position(Datarange), ...
    behavioralVariables.Position_X(Datarange), ...
    behavioralVariables.Evidence(Datarange), ...
    behavioralVariables.EvidenceR(Datarange), ...
    behavioralVariables.EvidenceL(Datarange), ...
    behavioralVariables.NumberOfCues(Datarange)]';

corr = behavioralVariables.Choice(Datarange) == behavioralVariables.TrialType(Datarange);
error = behavioralVariables.Choice(Datarange) ~= behavioralVariables.TrialType(Datarange);


flag1 = corr;
flag2 = error;

X = neural_data(:,flag1);
X = zscore(X')';

Y = behavioral_data(:, flag1);
Y = zscore(Y')';

S = neural_data(:, flag2);
S = zscore(S')';

T = behavioral_data(:, flag2);
T = zscore(T')';

Cxy = X*Y';
Cxx = X*X';
Cyy = Y*Y';
Dxy = S*T';

tic
gamma = 0.01 * trace(Cyy);
[w_xCRM, w_yCRM, lambda3, Wxs, Wys, lambdas, corrs] = crm(Cxx, Cyy, Cxy, Dxy, gamma=gamma);
[w_xCCA, w_yCCA, lambda3, Wxs, Wys, lambdas, corrs] = crm(Cxx, Cyy, Cxy, 0*Dxy, gamma=gamma);
toc

% Analysis

posy = behavioral_data(1,:);
posx = behavioral_data(2,:);
evi  = behavioral_data(3,:);

corrcoef(w_xCRM'*X, w_yCRM'*Y)
corrcoef(w_xCRM'*S, w_yCRM'*T)
corrcoef(w_xCCA'*X, w_yCCA'*Y)
corrcoef(w_xCCA'*S, w_yCCA'*T)

c = redblue(256);

figure(1),clf;

subplot(2,2,1)
X = zscore(neural_data')';
Y = zscore(behavioral_data')';
s1 = (w_xCCA'*X./sqrt(sum((w_xCCA'*X).^2)));
s2 = (w_yCCA'*Y./sqrt(sum((w_yCCA'*Y).^2)));
cca_dotproduct = s1.*s2;
%sum(s1.*s2) % is corrcoef(w_xCCA'*X, w_yCCA'*Y)
scatter(posy, abs(evi), [], cca_dotproduct, 'filled', 'MarkerFaceAlpha',0.3)
caxis([-5e-4,5e-4])
daspect([10 1 1])
ylim([0,20])
xlabel("Position [cm]")
ylabel("Total accumulated evidence")
colorbar;
title("CCA")
hold on;
plot([200, 200], [-20, 20], 'k--')

subplot(2,2,2)
t1 = (w_xCRM'*X./sqrt(sum((w_xCRM'*X).^2)));
t2 = (w_yCRM'*Y./sqrt(sum((w_yCRM'*Y).^2)));
%sum(t1.*t2) % is corrcoef(w_xCRM'*X, w_yCRM'*Y)
crm_dotproduct = t1.*t2;
scatter(posy, abs(evi), [], crm_dotproduct, 'filled', 'MarkerFaceAlpha',0.3)
ylim([0,20])
colormap(c)
caxis([-5e-4,5e-4])
daspect([10 1 1])
xlabel("Position [cm]")
ylabel("Total accumulated evidence")
colorbar;
title("CRM (unique to accumulation)")
hold on;
plot([200, 200], [-20, 20], 'k--')


[counts, edges, bin_indices] = histcounts(posy,20);

subplot(2,2,3)
% scatter(posy, cca_dotproduct, '.')
hold on;
num_bins = length(edges);
cca_medians = zeros(num_bins, 1);
err_lower = zeros(num_bins, 1);
err_upper = zeros(num_bins, 1);
for i = 1:num_bins
    bin_data = cca_dotproduct(bin_indices == i);
    if ~isempty(bin_data)
        cca_medians(i) = median(bin_data);
        ci = bootci(1000, {@median, bin_data}, 'Alpha', 0.1);
        err_lower(i) = cca_medians(i) - ci(1);
        err_upper(i) = ci(2) - cca_medians(i);
    else
        cca_medians(i) = NaN;
        err_lower(i) = NaN;
        err_upper(i) = NaN;
    end
end
errorbar(edges, cca_medians, err_lower, err_upper, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r')
xlim([0,300])
ylim([-1e-4, 4e-4])
plot([0,300],[0,0],'k--')
title("cca")
colorbar;


subplot(2,2,4)
% scatter(posy, crm_dotproduct,'.')
hold on;
num_bins = length(edges);
crm_medians = zeros(num_bins, 1);
err_lower = zeros(num_bins, 1);
err_upper = zeros(num_bins, 1);
for i = 1:num_bins
    bin_data = crm_dotproduct(bin_indices == i);
    if ~isempty(bin_data)
        crm_medians(i) = median(bin_data);
        ci = bootci(1000, {@median, bin_data}, 'Alpha', 0.1);
        err_lower(i) = crm_medians(i) - ci(1);
        err_upper(i) = ci(2) - crm_medians(i);
    else
        crm_medians(i) = NaN;
        err_lower(i) = NaN;
        err_upper(i) = NaN;
    end
end
errorbar(edges, crm_medians, err_lower, err_upper, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r')
xlim([0,300])
ylim([-1e-5, 1e-4])
plot([0,300],[0,0],'k--')
title("crm")
colorbar;


%%

figure(2)
imagesc(neural_data(1:50,1010:1100))
caxis([0,2.5])
colorbar;

%% Stats


corrcoef(w_xCRM, w_xCCA)

corrcoef(Cxx * w_xCRM, Cxx * w_xCCA)

corrcoef(w_xCRM'*neural_data, w_xCCA'*neural_data)