close all
clear; 
clc;
seed = 1234; % You can choose any integer value
rng(seed);
addpath(genpath(pwd));
num_components = 10;

% Define parameters
numItems = 200; % Number of items 
num_itemFeatures = 40;
numContextFeatures = 40; % Number of features unique to the context
num_globalFeatures = 40;
num_Features = num_itemFeatures + numContextFeatures +num_globalFeatures; % Total dimensionality of the multivariate pattern
item_scaling = 0.2;
context_scaling = 2;
context_fidelity = 1;
global_scaling = 1.5;
unique_noise_scaling = 0.7;%0.1;%0.05;
covariance_regularization = 0.005;
projection_scaling = 3.5;

% Generate multivariate context patterns for context 1
A = randn(numContextFeatures/2);
[U,~] = eig((A+A')/2);
covMatA = U*diag(abs(rand(numContextFeatures/2,1)))*U';
mu_A_context = context_scaling.*rand(numContextFeatures/2,1);
% Generate multivariate context patterns for context 2
A = randn(numContextFeatures/2);
[U,~] = eig((A+A')/2);
covMatB = U*diag(abs(rand(numContextFeatures/2,1)))*U';
mu_B_context = context_scaling.*rand(numContextFeatures/2,1);

% sample and concatenate two context vectors
pattern_context_A = cat(2, zeros(numItems/2 , ...
    numContextFeatures/2), mvnrnd(mu_A_context, ...
    covMatA./context_fidelity, numItems/2));
pattern_context_B = cat(2, mvnrnd(mu_B_context, ...
    covMatB./context_fidelity, numItems/2), ...
    zeros(numItems/2 , ...
    numContextFeatures/2));

% Generate multivariate context patterns for global
A = randn(num_globalFeatures);
[U,~] = eig((A+A')/2);
covMatC = U*diag(abs(rand(num_globalFeatures,1)))*U';
mu_global = global_scaling.*rand(num_globalFeatures,1);
pattern_global =  mvnrnd(mu_global, ...
    covMatC, numItems)';

% generate item patterns: 
itemsX = item_scaling.*rand(num_itemFeatures, numItems);



A = randn(num_itemFeatures);
[U,~] = eig((A+A')/2);
Mi = U.*projection_scaling + eye(num_itemFeatures);
% now project all items on the distortion direction.
itemsY = Mi*itemsX ;

A = randn(numContextFeatures);
[U,~] = eig((A+A')/2);
Mc1 = U.*projection_scaling + eye(numContextFeatures);
% now project all items on the distortion direction.
Mc1(:, 1:numContextFeatures/2)= 0;

patternY_context_A = pattern_context_A *Mc1;

A = randn(numContextFeatures);
[U,~] = eig((A+A')/2);
Mc2 = U.*projection_scaling + eye(numContextFeatures);
Mc2(:, numContextFeatures/2+1:end)= 0;
% now project all items on the distortion direction.
patternY_context_B = pattern_context_B *Mc2;


A = randn(num_globalFeatures);
[U,~] = eig((A+A')/2);
Mg = U.*projection_scaling + eye(num_globalFeatures);
% now project all items on the distortion direction.
pattern_global_Y = Mg* pattern_global;



pattern_context = cat(1, pattern_context_A, pattern_context_B)';
pattern_contextY = cat(1, patternY_context_A, patternY_context_B)';


% bring everything together
itemsXinContext = cat(1, itemsX, pattern_context, pattern_global);
itemsYinContext = cat(1, itemsY, pattern_contextY, pattern_global_Y);

% add some noise
itemsXinContext_n = itemsXinContext + ...
    unique_noise_scaling.*rand(size(itemsXinContext));
itemsYinContext_n = itemsYinContext + ...
    unique_noise_scaling.*rand(size(itemsYinContext));

% Calculate the item-by-item correlation matrix
C_items = corr(itemsXinContext_n, itemsYinContext_n);

% Visualize the correlation matrix with labeled axes
figure;
imagesc(1-C_items);
clim([0 1.5]);
colormap(parula(256));
colorbar; % Add a color bar to indicate correlation values
axis on; % Turn on the axis
title('(1 - Correlation) - Multivariate dissimilarity matrix');
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
xlim([0 2]);
% Set the legend
legend([h1{1}, h2{1}, h3{1}], {'Item', 'Class', 'Baseline'}, ...
       'FontName', 'Helvetica', 'FontSize', 16, 'Box', 'off', 'Location', 'northwest');

% Customize the plot
% ylim([-3 6]);
xlabel('Multivariate distance (1-correlation)', 'FontName', 'Helvetica', 'FontSize', 16);
set(gca, 'YColor', 'none');
set(gca, 'FontName', 'Helvetica', 'FontSize', 16);

disp(['p-class: ' num2str(numel(find(baseline_vals < mean(class_vals)))/numel(find(baseline_vals))) ])

disp(['p-item: ' num2str(numel(find(class_vals < mean(diag_vals)))/numel(find(class_vals))) ])
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
% 
% Dxy = zeros(size(Cxy));
% 
% for item_idx1 = 1: numItems/2
%         Dxy = Dxy + repmat(itemsX_contextA(:, item_idx1), [1, numItems/2-1])*...
%             itemsY_contextA(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
% end
% 
% for item_idx1 = 1: numItems/2
%         Dxy = Dxy + repmat(itemsX_contextB(:, item_idx1), [1, numItems/2-1])*...
%             itemsY_contextB(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
% end
% Dxy = Dxy./((numItems/2-1)*(numItems/2-1));
% 
% % use parCCA
% itemsXctxFree = [];
% itemsYctxFree = [];
% for ff = 1 : num_components
%     [w_x, w_y, lbd3i] = compute_weights(Cxx, Cyy, Cxy, Dxy, ff, covariance_regularization);
%     itemsXctxFree = cat(1, itemsXctxFree, real(w_x'*itemsXinContext_n));
%     itemsYctxFree =  cat(1,itemsYctxFree, real(itemsYinContext_n' * w_y)');
% 
% end

%


Dxy = zeros(size(Cxy));

for item_idx1 = 1: numItems/2
        Dxy = Dxy + repmat(itemsX_contextA(:, item_idx1), [1, numItems/2-1])*...
            itemsY_contextA(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
        % new: average patterns is better than average of correlations!
        % Dxy = Dxy + itemsX_contextA(:, item_idx1)*...
        %     mean(itemsY_contextA(:, [1:item_idx1-1,item_idx1+1:numItems/2])');
      
end

for item_idx1 = 1: numItems/2
        Dxy = Dxy + repmat(itemsX_contextB(:, item_idx1), [1, numItems/2-1])*...
            itemsY_contextB(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
         % Dxy = Dxy + itemsX_contextB(:, item_idx1)*...
         %    mean(itemsY_contextB(:, [1:item_idx1-1,item_idx1+1:numItems/2])');
end
Dxy = Dxy./((numItems/2-1)*(numItems/2-1));
% Dxy = Dxy./numItems;
% use parCCA
itemsXctxFree = [];
itemsYctxFree = [];
for ff = 1 : num_components
    [w_x, w_y, lbd3i] = compute_weights(Cxx, Cyy, Cxy, Dxy, ff, covariance_regularization);
    itemsXctxFree = cat(1, itemsXctxFree, real(w_x'*itemsXinContext_n));
    itemsYctxFree =  cat(1,itemsYctxFree, real(itemsYinContext_n' * w_y)');

end
%



C_items = corr((itemsXctxFree), (itemsYctxFree));
D = 1-C_items;

% Visualize the correlation matrix with labeled axes
figure;
subplot(211)
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

xlim([-0.02 0.2])
subplot(212);
% Set the legend
legend([h1{1}], {'Item'}, ...
       'FontName', 'Helvetica', 'FontSize', 16, 'Box', 'off', 'Location', 'northeast');
xlabel('Distance', 'FontName', 'Helvetica', 'FontSize', 16);
set(gca, 'YColor', 'none');

% xlim([-0.0005 0.002]);
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
set(gca, 'FontName', 'Helvetica', 'FontSize', 16);

% set(gca, 'XTickLabel', get(gca, 'XTickLabel'), 'FontName', 'Helvetica', 'FontSize', 16);
% xlim([-1 3])

% p-value calculation

disp(['p-class: ' num2str(numel(find(baseline_vals < mean(class_vals)))/numel(find(baseline_vals))) ])

disp(['p-item: ' num2str(numel(find(class_vals < mean(diag_vals)))/numel(find(class_vals))) ])
%% now maximize on context similarity


% 
% Cxy = zeros(size(Cxy));
% 
% for item_idx1 = 1: numItems/2
%         Cxy = Cxy + repmat(itemsX_contextA(:, item_idx1), [1, numItems/2-1])*...
%             itemsY_contextA(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
% end
% 
% for item_idx1 = 1: numItems/2
%         Cxy = Cxy + repmat(itemsX_contextB(:, item_idx1), [1, numItems/2-1])*...
%             itemsY_contextB(:, [1:item_idx1-1,item_idx1+1:numItems/2])';
% end
% Cxy = Cxy./((numItems/2-1)*(numItems/2-1));
% 

Cxy = ((mean(itemsX_contextA,2)*mean(itemsY_contextA,2)')...
    + (mean(itemsX_contextB,2)*mean(itemsY_contextB,2)'))./2

% Dxy = zeros(size(Cxy));
% 
% for item_idx1 = 1: numItems/2
%         Dxy = Dxy + repmat(itemsX_contextA(:, item_idx1), [1, numItems/2])*...
%             itemsY_contextB';
% end
% 
% for item_idx1 = 1: numItems/2
%          Dxy = Dxy + repmat(itemsX_contextB(:, item_idx1), [1, numItems/2])*...
%             itemsY_contextA';
% end
% Dxy = Dxy./((numItems/2)*(numItems/2));

Dxy = ((mean(itemsX_contextA,2)*mean(itemsY_contextB,2)')...
    + (mean(itemsX_contextB,2)*mean(itemsY_contextA,2)'))./2;

% % use parCCA
itemsXctxMax = [];
itemsYctxMax = [];
for ff = 1 : num_components
    [w_x, w_y, lbd3i] = compute_weights(Cxx, Cyy, Cxy, Dxy, ff, covariance_regularization);
    itemsXctxMax = cat(1, itemsXctxMax, real(w_x'*itemsXinContext_n));
    itemsYctxMax =  cat(1,itemsYctxMax, real(itemsYinContext_n' * w_y)');

end


%

% C_items = corr((itemsXctxMax-mean(itemsXctxMax,1)), (itemsYctxMax-mean(itemsYctxMax,1)));
C_items = corr((itemsXctxMax), (itemsYctxMax));

D = 1-C_items;


%
figure;
subplot(211)
imagesc((D));
caxis([0 1.5]);
colorbar;
subplot(212)
plot(zscore(itemsXctxMax(1,:)))
hold on;
plot(zscore(itemsYctxMax(1,:)));
ylim([-3 3]);
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
% set(gca, 'XTickLabel', get(gca, 'XTickLabel'), 'FontName', 'Helvetica', 'FontSize', 16);
% xlim([-1 3])
disp(['p-class: ' num2str(numel(find(baseline_vals < mean(class_vals)))/numel(find(baseline_vals))) ])

disp(['p-item: ' num2str(numel(find(class_vals < mean(diag_vals)))/numel(find(class_vals))) ])