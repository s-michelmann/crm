close all
clear; 
clc;
seed = 1234; % You can choose any integer value
rng(seed);
addpath("toolbox/crm/");
n_components = 10;
item_scaling = 0.2;

% Define parameters
numItems = 200; % Number of items 
num_itemFeatures = 40;
numContextFeatures = 40; % Number of features unique to the context
num_globalFeatures = 16;
numFeatures = num_itemFeatures + numContextFeatures +num_globalFeatures; % Total dimensionality of the multivariate pattern
unique_noise_scaling = 0.05;
context_fidelity = 3.5; %what is the variance of the context patterns
% projection scaling
distortion_scaling = 25;
covariance_regularization = 0.005;
% Define global mean patterns for each class and context
mu_A_context = 1*randn(1, numContextFeatures); % Global mean pattern for Class A
mu_B_context = 1*randn(1, numContextFeatures); % Global mean pattern for Class B
mu_global = randn(1, num_globalFeatures); % Global mean pattern for Class B

% generate item patterns: 
itemsX = item_scaling.*rand(num_itemFeatures, numItems);

%  covariance matrices for each class-context 
% here identity-matrix with scaling of variance only
SigmaContext = eye(numContextFeatures)./context_fidelity; 
SigmaGlobal = eye(num_globalFeatures)./context_fidelity; 

% Generate multivariate context patterns
pattern_context_A = mvnrnd(mu_A_context, SigmaContext, numItems/2);
pattern_context_B = mvnrnd(mu_B_context, SigmaContext, numItems/2);
pattern_context = cat(1, pattern_context_A, pattern_context_B)';
pattern_global =  mvnrnd(mu_global, SigmaGlobal, numItems)';

itemsXinContext = cat(1, itemsX, pattern_context, pattern_global);
% create a random but systematic distortion
% projection_vector = randn(1, numContextFeatures);
u = rand(1, numFeatures) - 0.5;
b = 1 / sqrt(2);
projection_vector = -b * sign(u) .* log(1 - 2 * abs(u));

% Normalize and scalae the projection vector
projection_vector = projection_vector / norm(projection_vector); 
projection_vector = ones(size(projection_vector)) + projection_vector.*distortion_scaling;

% now project all items on the distortion direction.
itemsYinContext = itemsXinContext .* projection_vector';

% add some noise
itemsXinContext_n = itemsXinContext + ...
    unique_noise_scaling.*rand(size(itemsXinContext));
itemsYinContext_n = itemsYinContext + ...
    unique_noise_scaling.*rand(size(itemsYinContext));

% center
itemsXinContext_n = itemsXinContext_n-mean(itemsXinContext_n,2);
itemsYinContext_n = itemsYinContext_n-mean(itemsYinContext_n,2);

% Calculate the item-by-item correlation matrix
C_items = corr(itemsXinContext_n, itemsYinContext_n);

% Visualize the correlation matrix with labeled axes
figure;
imagesc(1-C_items);
caxis([0 1.5]);
colormap(parula(256));
colorbar; % Add a color bar to indicate correlation values
axis on; % Turn on the axis
title('(1- Correlation) - Multivariate dissimilarity matrix');
% Label the axes
xticks([numItems/4, 3*numItems/4]);
xticklabels({'Class A (context 2)', 'Class B  (context 2)'});
yticks([numItems/4, 3*numItems/4]);
yticklabels({'Class A  (context 1)', 'Class B (context 1)'});

% computing averages
D = 1-C_items;

% Calculate averages
diag_vals = diag(D);
off_diag_vals = D(~eye(size(D)));
top_left_vals = D(1:numItems/2, 1:numItems/2);
top_right_vals = D(numItems/2+1:end, 1:numItems/2);
bottom_right_vals = D(numItems/2+1:end, numItems/2+1:end);
bottom_left_vals = D(1:numItems/2, numItems/2+1:end);

% remove local diagonals
top_left_vals_od = top_left_vals(~eye(size(top_left_vals)));
bottom_right_vals_od = bottom_right_vals(~eye(size(bottom_right_vals)));

class_vals = cat(1, top_left_vals_od, bottom_right_vals_od);
baseline_vals = cat(1, bottom_left_vals, top_right_vals);

% %% PLOTTING Rain Clouds
% Create raincloud plots
figure;
hold on;
h1 = raincloud_plot(diag_vals(:), 'box_on', 1, 'color', hex2rgb('#F35B04'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.1, 'dot_dodge_amount', 0.1, 'box_col_match', 0);
h2 = raincloud_plot(class_vals(:), 'box_on', 1, 'color', hex2rgb('#F7B801'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.3, 'dot_dodge_amount', 0.3, 'box_col_match', 0);
h3 = raincloud_plot(baseline_vals(:), 'box_on', 1, 'color', hex2rgb('#3D348B'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.5, 'dot_dodge_amount', 0.5, 'box_col_match', 0);
% h4 = raincloud_plot(D(:), 'box_on', 1, 'color', hex2rgb('#7678ED'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.7, 'dot_dodge_amount', 0.7, 'box_col_match', 0);

% Set the legend
legend([h1{1}, h2{1}, h3{1}], {'Item', 'Class', 'Baseline'}, ...
       'FontName', 'Helvetica', 'FontSize', 16, 'Box', 'off', 'Location', 'northwest');

% Customize the plot
% ylim([-3 6]);
xlabel('Multivariate distance (1-correlation)', 'FontName', 'Helvetica', 'FontSize', 16);
set(gca, 'YColor', 'none');
set(gca, 'XTickLabel', get(gca, 'XTickLabel'), 'FontName', 'Helvetica', 'FontSize', 16);
numel(find(class_vals < mean(diag_vals)))/numel(find(class_vals))
%% maximize on item similarity

% Feature-by-feature Cxy computations
Cxy =  (itemsXinContext_n*itemsYinContext_n')./numItems;

% Cxx and Cyy computations
Cxx = cov(itemsXinContext_n');
Cyy = cov(itemsYinContext_n');

itemsX_contextA = itemsXinContext_n(:,1:numItems/2);
itemsX_contextB = itemsXinContext_n(:,numItems/2+1:end);

itemsY_contextA = itemsYinContext_n(:,1:numItems/2);
itemsY_contextB = itemsYinContext_n(:,numItems/2+1:end);

Dxy = zeros(size(Cxy));

for item_idx1 = 1: numItems/2
        Dxy = Dxy + repmat(itemsX_contextA(:, item_idx1), [1, numItems/2-1])*...
            itemsY_contextA(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
end

for item_idx1 = 1: numItems/2
        Dxy = Dxy + repmat(itemsX_contextB(:, item_idx1), [1, numItems/2-1])*...
            itemsY_contextB(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
end
Dxy = Dxy./((numItems/2-1)*(numItems/2-1));

% use parCCA
itemsXctxFree = [];
itemsYctxFree = [];
for ff = 1 : n_components
    [w_x, w_y, lbd3i] = compute_weights(Cxx, Cyy, Cxy, Dxy, ff, covariance_regularization);
    itemsXctxFree = cat(1, itemsXctxFree, w_x'*itemsXinContext_n);
    itemsYctxFree =  cat(1,itemsYctxFree, (itemsYinContext_n' * w_y)');

end

C_items = corr((itemsXctxFree), (itemsYctxFree));
D = 1-C_items;

% Visualize the correlation matrix with labeled axes
figure;
subplot(211)
imagesc((D)); colorbar;
% Label the axes
xticks([numItems/4, 3*numItems/4]);
xticklabels({'Class A (context 2)', 'Class B  (context 2)'});
yticks([numItems/4, 3*numItems/4]);
yticklabels({'Class A  (context 1)', 'Class B (context 1)'});

caxis([0 1.5]);
subplot(212)
plot(zscore(itemsXctxFree(1,:)))
hold on;
plot(zscore(itemsYctxFree(1,:)))
% plotting averages 
% Calculate averages
diag_vals = diag(D);
off_diag_vals = D(~eye(size(D)));
top_left_vals = D(1:numItems/2, 1:numItems/2);
top_right_vals = D(numItems/2+1:end, 1:numItems/2);
bottom_right_vals = D(numItems/2+1:end, numItems/2+1:end);
bottom_left_vals = D(1:numItems/2, numItems/2+1:end);
ylim([-5 5]);


% Get the size of the figure
fig_position = get(gcf, 'Position');
set(gcf, 'Position', [fig_position(1), fig_position(2), 350, 550]);

% remove local diagonals
top_left_vals_od = top_left_vals(~eye(size(top_left_vals)));
bottom_right_vals_od = bottom_right_vals(~eye(size(bottom_right_vals)));

class_vals = cat(1, top_left_vals_od, bottom_right_vals_od);
baseline_vals = cat(1, bottom_left_vals, top_right_vals);

% PLOTTING Rain Clouds
% Create raincloud plots
figure;
hold on;
subplot(211);
h1 = raincloud_plot(diag_vals(:), 'box_on', 1, 'color', hex2rgb('#F35B04'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.1, 'dot_dodge_amount', 0.1, 'box_col_match', 0);
h2 = raincloud_plot(class_vals(:), 'box_on', 1, 'color', hex2rgb('#F7B801'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.3, 'dot_dodge_amount', 0.3, 'box_col_match', 0);
h3 = raincloud_plot(baseline_vals(:), 'box_on', 1, 'color', hex2rgb('#3D348B'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.5, 'dot_dodge_amount', 0.5, 'box_col_match', 0);
xlim([-0.02 0.09]);
subplot(212);
% Set the legend
legend([h1{1}], {'Item'}, ...
       'FontName', 'Helvetica', 'FontSize', 16, 'Box', 'off', 'Location', 'northeast');
xlabel('Distance', 'FontName', 'Helvetica', 'FontSize', 16);
set(gca, 'YColor', 'none');


%
subplot(212);
yyaxis left
h1 = raincloud_plot(diag_vals(:), 'box_on', 1, 'color', hex2rgb('#F35B04'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.1, 'dot_dodge_amount', 0.1, 'box_col_match', 0);
yyaxis right
h2 = raincloud_plot(class_vals(:), 'box_on', 1, 'color', hex2rgb('#F7B801'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.3, 'dot_dodge_amount', 0.3, 'box_col_match', 0);
h3 = raincloud_plot(baseline_vals(:), 'box_on', 1, 'color', hex2rgb('#3D348B'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.5, 'dot_dodge_amount', 0.5, 'box_col_match', 0);
% h4 = raincloud_plot(D(:), 'box_on', 1, 'color', hex2rgb('#7678ED'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.7, 'dot_dodge_amount', 0.7, 'box_col_match', 0);

% Set the legend
legend([h1{1}, h2{1}, h3{1}], {'Item', 'Class', 'Baseline'}, ...
       'FontName', 'Helvetica', 'FontSize', 16, 'Box', 'off', 'Location', 'northeast');

% Customize the plot
% ylim([-3 6]);
xlabel('Distance', 'FontName', 'Helvetica', 'FontSize', 16);
set(gca, 'YColor', 'none');
% set(gca, 'XTickLabel', get(gca, 'XTickLabel'), 'FontName', 'Helvetica', 'FontSize', 16);
% xlim([-1 3])

% p-value calculation
numel(find(class_vals < mean(diag_vals)))/numel(find(class_vals));
numel(find(off_diag_vals < mean(diag_vals)))/numel(find(class_vals))

%% now maximize on context similarity

% Cxy becomes the context covariance matrix
Cxy2 = Dxy;
Dxy2 = zeros(size(Cxy));

% use parCCA
itemsXctxMax = [];
itemsYctxMax = [];
for ff = 1 : n_components
    [w_x, w_y, lbd3i] = compute_weights(Cxy2, Cyy, Cxy, Dxy2, ff, covariance_regularization);

    itemsXctxMax = cat(1, itemsXctxMax, real(w_x)'*itemsXinContext_n);
    itemsYctxMax = cat(1, itemsYctxMax, (itemsYinContext_n' * real(w_y))');

end


%%
C_items = corr((itemsXctxMax), (itemsYctxMax));
D = 1-C_items;


%
figure;
subplot(211)
imagesc((D));
caxis([0 1.5]);
subplot(212)
plot(zscore(itemsXctxMax(1,:)))
hold on;
plot(zscore(itemsYctxMax(1,:)));

% Get the size of the figure
fig_position = get(gcf, 'Position');
set(gcf, 'Position', [fig_position(1), fig_position(2), 350, 550]);
%
figure;
diag_vals = diag(D);
off_diag_vals = D(~eye(size(D)));
top_left_vals = D(1:numItems/2, 1:numItems/2);
top_right_vals = D(numItems/2+1:end, 1:numItems/2);
bottom_right_vals = D(numItems/2+1:end, numItems/2+1:end);
bottom_left_vals = D(1:numItems/2, numItems/2+1:end);

% remove local diagonals
top_left_vals_od = top_left_vals(~eye(size(top_left_vals)));
bottom_right_vals_od = bottom_right_vals(~eye(size(bottom_right_vals)));

class_vals = cat(1, top_left_vals_od, bottom_right_vals_od);
baseline_vals = cat(1, bottom_left_vals, top_right_vals);
%
% Visualize the correlation matrix with labeled axes

h1 = raincloud_plot(diag_vals(:), 'box_on', 1, 'color', hex2rgb('#F35B04'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.1, 'dot_dodge_amount', 0.1, 'box_col_match', 0);
yyaxis right
h2 = raincloud_plot(class_vals(:), 'box_on', 1, 'color', hex2rgb('#F7B801'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.3, 'dot_dodge_amount', 0.3, 'box_col_match', 0);
h3 = raincloud_plot(baseline_vals(:), 'box_on', 1, 'color', hex2rgb('#3D348B'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.5, 'dot_dodge_amount', 0.5, 'box_col_match', 0);
% h4 = raincloud_plot(D(:), 'box_on', 1, 'color', hex2rgb('#7678ED'), 'alpha', 0.5, 'box_dodge', 1, 'box_dodge_amount', 0.7, 'dot_dodge_amount', 0.7, 'box_col_match', 0);

% Set the legend
legend([h1{1}, h2{1}, h3{1}], {'Item', 'Class', 'Baseline'}, ...
       'FontName', 'Helvetica', 'FontSize', 16, 'Box', 'off', 'Location', 'northeast');

xlabel('Distance', 'FontName', 'Helvetica', 'FontSize', 16);
set(gca, 'YColor', 'none');
