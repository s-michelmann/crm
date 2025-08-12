close all
% clear;
clc;
seed = 1234; % You can choose any integer value
rng(seed);
restoredefaultpath;
num_components = 20;
addpath(genpath('crm/'));


dir_code = "";
%% paths...

restoredefaultpath;
homeDir  = getenv('HOME');

cwd_path = pwd;
addpath(genpath(fileparts(cwd_path)));

% Get the project
idx = strfind(cwd_path, 'CRM_project');

data_path = fullfile( cwd_path(1 : idx + numel('CRM_project') - 1), ...
    "reagh_data");

% 

%% load in the data

mat_schem = [];
mat_ep = [];
mat_ctx = [];
mat_raw = [];
mat_avg = [];
for ss = 1 : 22
    if ss == 15; continue; end
    if ss == 18; continue; end

    sj_path = fullfile(data_path, ['sub-1' sprintf('%02d', ss)]);
        load(fullfile(sj_path, 'correlationMatrix_avg'));

    load(fullfile(sj_path, 'correlationMatrix_raw'));
    % load(fullfile(sj_path, 'correlationMatrix_schema'));
    load(fullfile(sj_path, 'correlationMatrix_episode'));
    load(fullfile(sj_path, 'correlationMatrix_context'));


    mat_avg= cat(3, mat_avg, avgCorrMatrix);

    mat_raw = cat(3, mat_raw, correlationMatrix_raw);
    mat_ep = cat(3, mat_ep, correlationMatrix_episode);
    mat_ctx = cat(3, mat_ctx, correlationMatrix_context);
    % mat_schem = cat(3, mat_schem, correlationMatrix_schema);

end

%% plot
% Define parameters
people = {'tommy', 'lisa'};
% mishkas cafe delta cafe nugget grocery coop grocery
places = {'delta', 'mishkas', 'coop', 'nugget'};
videos = {'video1', 'video2', 'video3'};

conditions = {};
for pl = 1:length(places)
    for p = 1:length(people)
        conditions{end+1} = sprintf('%s_%s', people{p}, places{pl});
    end
end
%% get all the labels
allLabels = {};
for i = 1:length(conditions)
    cond = conditions{i};
    for j = 1:3
                            

        allLabels{end+1} = [cond '_' num2str(j)];
    end
end
%% plot
new_labels = cellfun(@(s) strjoin(...
    cellfun(@(w) [upper(w(1)), lower(w(2:end))], ...
strsplit(strrep(s, '_', ' ')), 'UniformOutput', false), ' '), ...
allLabels, 'UniformOutput', false);

correlationMatrix = mean(mat_raw(:,:,:),3);
figure;
subplot(131);

nanmask = eye(size(correlationMatrix));
nanmask(nanmask == 1) = nan;
imagesc(correlationMatrix+nanmask);
colorbar;
axis equal tight;
xticks(1:length(new_labels));
yticks(1:length(new_labels));
xticklabels(new_labels);
yticklabels(new_labels);
xtickangle(90);
title('Correlation Matrix of All Video Vectors (raw)');
caxis([-0.6 0.6]);

set(gcf, 'color', 'white');
t_axis  = [1:1:25];
bound_samples = [0;12; 24];
adj = 0.5;
lw = 1;
for bb = 2: numel(bound_samples)
    x = [t_axis(bound_samples(bb-1)+1)-adj t_axis(bound_samples(bb))+adj ...
         t_axis(bound_samples(bb))+adj  t_axis(bound_samples(bb-1)+1)-adj];
    
    y = [t_axis(bound_samples(bb-1)+1)-adj t_axis(bound_samples(bb-1)+1)-adj...
         t_axis(bound_samples(bb))+adj  t_axis(bound_samples(bb))+adj];
    patch(x,y,'w','LineWidth', lw,'EdgeColor','white', 'FaceColor', 'none')
end

correlationMatrix = mean(mat_ctx(:,:,:),3);
subplot(132);
nanmask = eye(size(correlationMatrix));
nanmask(nanmask == 1) = nan;
imagesc(correlationMatrix+nanmask);
colorbar;
axis equal tight;
xticks(1:length(new_labels));
yticks(1:length(new_labels));
xticklabels(new_labels);
yticklabels(new_labels);
xtickangle(90);
title('Correlation Matrix of All Video Vectors (context components)');
caxis([-0.6 0.6]);

set(gcf, 'color', 'white');
for bb = 2: numel(bound_samples)
    x = [t_axis(bound_samples(bb-1)+1)-adj t_axis(bound_samples(bb))+adj ...
         t_axis(bound_samples(bb))+adj  t_axis(bound_samples(bb-1)+1)-adj];
    
    y = [t_axis(bound_samples(bb-1)+1)-adj t_axis(bound_samples(bb-1)+1)-adj...
         t_axis(bound_samples(bb))+adj  t_axis(bound_samples(bb))+adj];
    patch(x,y,'w','LineWidth', lw,'EdgeColor','white', 'FaceColor', 'none')
end



correlationMatrix = mean(mat_ep(:,:,:),3);
subplot(133);
nanmask = eye(size(correlationMatrix));
nanmask(nanmask == 1) = nan;
imagesc(correlationMatrix+nanmask);
colorbar;
axis equal tight;
xticks(1:length(new_labels));
yticks(1:length(new_labels));
xticklabels(new_labels);
yticklabels(new_labels);
xtickangle(90);
title('Correlation Matrix of All Video Vectors (episode components)');
caxis([-0.6 0.6]);
for bb = 2: numel(bound_samples)
    x = [t_axis(bound_samples(bb-1)+1)-adj t_axis(bound_samples(bb))+adj ...
         t_axis(bound_samples(bb))+adj  t_axis(bound_samples(bb-1)+1)-adj];
    
    y = [t_axis(bound_samples(bb-1)+1)-adj t_axis(bound_samples(bb-1)+1)-adj...
         t_axis(bound_samples(bb))+adj  t_axis(bound_samples(bb))+adj];
    patch(x,y,'w','LineWidth', lw,'EdgeColor','white', 'FaceColor', 'none')
end

set(gcf, 'color', 'white');

exportgraphics(figure(1), 'fig4_left.pdf');

%% schema matrix
schemaModel = zeros(size(correlationMatrix));
n = size(schemaModel, 1);
schemaModel(1:floor(n/2), 1:floor(n/2)) = 1; % Top left block
schemaModel(ceil(n/2)+1:end, ceil(n/2)+1:end) = 1; % Bottom right block
%% context matrix
contextModel = zeros(size(correlationMatrix));
n = size(contextModel, 1);
contextModel(1:floor(n/4), 1:floor(n/4)) = 1; % Top left block
contextModel(ceil(n/4)+1:floor(n/2), ceil(n/4)+1:floor(n/2)) = 1;  %middle block 1
contextModel(floor(n/2)+1:floor(n/2)+ceil(n/4), floor(n/2)+1:floor(n/2)+ceil(n/4)) = 1; %middle block 2
contextModel(floor(n/2)+ceil(n/4)+1:end, floor(n/2)+ceil(n/4)+1:end) = 1; % Bottom right block
%% episode matrix
episodeModel = zeros(size(correlationMatrix));
blockSize = 3; % Size of each block
for i = 1:blockSize:n
    episodeModel(i:min(i+blockSize-1, n), i:min(i+blockSize-1, n)) = 1;
end
%%
schema_corrs = zeros(size(mat_raw,3),3);
context_corrs = zeros(size(mat_raw,3),3);
episode_corrs = zeros(size(mat_raw,3),3);

for ss = 1: size(mat_raw,3)
    raw = mat_raw(:,:,ss);
    ctx = mat_ctx(:,:,ss);
    ep = mat_ep(:,:,ss);

    % get the schema model correlations
    schema_corrs(ss,1) = corr(...
        raw(eye(size(raw))==0),...
        schemaModel(eye(size(raw))==0));
    schema_corrs(ss,2) = corr(...
        ctx(eye(size(ctx))==0),...
        schemaModel(eye(size(ctx))==0));
    schema_corrs(ss,3) = corr(...
        ep(eye(size(ep))==0),...
        schemaModel(eye(size(ep))==0));

    % get the context model correlations

    context_corrs(ss,1) = corr(...
        raw(eye(size(raw))==0),...
        contextModel(eye(size(raw))==0));
    context_corrs(ss,2) = corr(...
        ctx(eye(size(ctx))==0),...
        contextModel(eye(size(ctx))==0));
    context_corrs(ss,3) = corr(...
        ep(eye(size(ep))==0),...
        contextModel(eye(size(ep))==0));

    % get the episode model correlations

    episode_corrs(ss,1) = corr(...
        raw(eye(size(raw))==0),...
        episodeModel(eye(size(raw))==0));
    episode_corrs(ss,2) = corr(...
        ctx(eye(size(ctx))==0),...
        episodeModel(eye(size(ctx))==0));
    episode_corrs(ss,3) = corr(...
        ep(eye(size(ep))==0),...
        episodeModel(eye(size(ep))==0));
end

%% Plot the raw data 
nGroups = 3;
groupLabels = {'Schema model', 'Context model', 'Episode model'};

% 1. Create figure
figure('Color', 'w');
subplot(131);
hold on;
plot_vec = [schema_corrs(:,1),context_corrs(:,1) ,episode_corrs(:,1)];
% 2. Draw boxplots with custom styling
boxplot(plot_vec, ...
    'Colors',       lines(nGroups), ...       % distinct colors
    'Notch',        'on', ...                 % notched boxes
    'Widths',       0.5, ...                  % narrower boxes
    'Symbol',       '', ...                   % hide default outliers
    'Labels',       groupLabels, ...
    'Whisker',      1.5);                     % whisker length
% Precompute colors
colors = lines(nGroups);   
% 3. Overlay individual data points (jittered)
for i = 1:nGroups
    x = i + (rand(size(plot_vec,1),1)-0.5)*0.2;  % slight horizontal jitter
    scatter(x, plot_vec(:,i), ...
        36, ...                  % marker size
        colors(i,:), ...% match box color
        'filled', ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor', 'none');
end

% 4. Polish axes and remove top/right border
set(gca, ...
    'Box',         'off', ...    % turn off the full box
    'TickDir',     'out', ...    % move ticks outside
    'FontSize',    12, ...
    'LineWidth',   1, ...
    'XLim',        [0.5 nGroups+0.5]);

ylabel('Correlation Value','FontSize',14);
title('Model correlations (raw data)','FontSize',16);

% run two‐sample t‐tests
[~, p12] = ttest(plot_vec(:,1), plot_vec(:,2), 'tail', 'right');
[~, p23] = ttest(plot_vec(:,2), plot_vec(:,3), 'tail', 'right');

[~, p13] = ttest(plot_vec(:,1), plot_vec(:,3), 'tail', 'right');

mean(plot_vec,1)
std(plot_vec,0,1)

% compute a baseline y‐position just above your highest point
ymax = max(plot_vec(:));
offset = range(plot_vec(:)) * 0.05;    % 5% of data range
lineWidth = 1;
col = [0.5 0.5 0.5];                  % gray

% annotate 1 vs 2
if p12 < 0.05
    y1 = ymax + offset;
    y2 = y1 + offset;
    plot([1 1 2 2], [y1 y2 y2 y1], 'Color', col, 'LineWidth', lineWidth)
    text(1.5, y2 + offset*0.2, '*', ...
         'Color', col, 'FontSize', 14, ...
         'HorizontalAlignment','center')
end

% annotate 1 vs 3
if p13 < 0.05
    y1 = ymax + 3*offset;  % stack above the first line
    y2 = y1 + offset;
    plot([1 1 3 3], [y1 y2 y2 y1], 'Color', col, 'LineWidth', lineWidth)
    text(2, y2 + offset*0.2, '*', ...
         'Color', col, 'FontSize', 14, ...
         'HorizontalAlignment','center')
end

ylim([-0.2 y2+2*offset])
% Plot context  data 
nGroups = 3;
groupLabels = {'Schema model', 'Context model', 'Episode model'};

% 1. Create figure
subplot(132);
hold on;
plot_vec = [schema_corrs(:,2),context_corrs(:,2) ,episode_corrs(:,2)];
% 2. Draw boxplots with custom styling
boxplot(plot_vec, ...
    'Colors',       lines(nGroups), ...       % distinct colors
    'Notch',        'on', ...                 % notched boxes
    'Widths',       0.5, ...                  % narrower boxes
    'Symbol',       '', ...                   % hide default outliers
    'Labels',       groupLabels, ...
    'Whisker',      1.5);                     % whisker length
% Precompute colors
colors = lines(nGroups);   
% 3. Overlay individual data points (jittered)
for i = 1:nGroups
    x = i + (rand(size(plot_vec,1),1)-0.5)*0.2;  % slight horizontal jitter
    scatter(x, plot_vec(:,i), ...
        36, ...                  % marker size
        colors(i,:), ...% match box color
        'filled', ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor', 'none');
end

% 4. Polish axes and remove top/right border
set(gca, ...
    'Box',         'off', ...    % turn off the full box
    'TickDir',     'out', ...    % move ticks outside
    'FontSize',    12, ...
    'LineWidth',   1, ...
    'XLim',        [0.5 nGroups+0.5]);

ylabel('Correlation Value','FontSize',14);
title('Model correlations (context data)','FontSize',16);

% run two‐sample t‐tests
[~, p12] = ttest(plot_vec(:,1), plot_vec(:,2), 'tail', 'right');
[~, p23] = ttest(plot_vec(:,2), plot_vec(:,3), 'tail', 'right');

[~, p13] = ttest(plot_vec(:,1), plot_vec(:,3), 'tail', 'right');
mean(plot_vec,1)
std(plot_vec,0,1)

% compute a baseline y‐position just above your highest point
ymax = max(plot_vec(:));
offset = range(plot_vec(:)) * 0.05;    % 5% of data range
lineWidth = 1;
col = [0.5 0.5 0.5];                  % gray

% annotate 1 vs 2
if p12 < 0.05
    y1 = ymax + offset;
    y2 = y1 + offset;
    plot([1 1 2 2], [y1 y2 y2 y1], 'Color', col, 'LineWidth', lineWidth)
    text(1.5, y2 + offset*0.2, '*', ...
         'Color', col, 'FontSize', 14, ...
         'HorizontalAlignment','center')
end

% annotate 1 vs 3
if p13 < 0.05
    y1 = ymax + 3*offset;  % stack above the first line
    y2 = y1 + offset;
    plot([1 1 3 3], [y1 y2 y2 y1], 'Color', col, 'LineWidth', lineWidth)
    text(2, y2 + offset*0.2, '*', ...
         'Color', col, 'FontSize', 14, ...
         'HorizontalAlignment','center')
end

ylim([-0.2 y2+2*offset])
% Plot episode  data 
nGroups = 3;
groupLabels = {'Schema model', 'Context model', 'Episode model'};

% 1. Create figure
subplot(133); hold on;
plot_vec = [schema_corrs(:,3),context_corrs(:,3) ,episode_corrs(:,3)];
% 2. Draw boxplots with custom styling
boxplot(plot_vec, ...
    'Colors',       lines(nGroups), ...       % distinct colors
    'Notch',        'on', ...                 % notched boxes
    'Widths',       0.5, ...                  % narrower boxes
    'Symbol',       '', ...                   % hide default outliers
    'Labels',       groupLabels, ...
    'Whisker',      1.5);                     % whisker length
% Precompute colors
colors = lines(nGroups);   
% 3. Overlay individual data points (jittered)
for i = 1:nGroups
    x = i + (rand(size(plot_vec,1),1)-0.5)*0.2;  % slight horizontal jitter
    scatter(x, plot_vec(:,i), ...
        36, ...                  % marker size
        colors(i,:), ...% match box color
        'filled', ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor', 'none');
end

% 4. Polish axes and remove top/right border
set(gca, ...
    'Box',         'off', ...    % turn off the full box
    'TickDir',     'out', ...    % move ticks outside
    'FontSize',    12, ...
    'LineWidth',   1, ...
    'XLim',        [0.5 nGroups+0.5]);

ylabel('Correlation Value','FontSize',14);
title('Model correlations (episode data)','FontSize',16);

% run two‐sample t‐tests
[~, p12] = ttest(plot_vec(:,1), plot_vec(:,2), 'tail', 'right');
[~, p23] = ttest(plot_vec(:,2), plot_vec(:,3), 'tail', 'right');

[~, p13] = ttest(plot_vec(:,1), plot_vec(:,3), 'tail', 'right');
mean(plot_vec,1)
std(plot_vec,0,1)

% compute a baseline y‐position just above your highest point
ymax = max(plot_vec(:));
offset = range(plot_vec(:)) * 0.05;    % 5% of data range
lineWidth = 1;
col = [0.5 0.5 0.5];                  % gray

% annotate 1 vs 2
if p12 < 0.05
    y1 = ymax + offset;
    y2 = y1 + offset;
    plot([1 1 2 2], [y1 y2 y2 y1], 'Color', col, 'LineWidth', lineWidth)
    text(1.5, y2 + offset*0.2, '*', ...
         'Color', col, 'FontSize', 14, ...
         'HorizontalAlignment','center')
end

% annotate 1 vs 3
if p13 < 0.05
    y1 = ymax + 3*offset;  % stack above the first line
    y2 = y1 + offset;
    plot([1 1 3 3], [y1 y2 y2 y1], 'Color', col, 'LineWidth', lineWidth)
    text(2, y2 + offset*0.2, '*', ...
         'Color', col, 'FontSize', 14, ...
         'HorizontalAlignment','center')
end

ylim([-0.2 y2+2*offset])  
exportgraphics(figure(1), 'fig4_right.pdf');

%% 
figure;
subplot(131);
imagesc(schemaModel-eye(size(schemaModel))); 
axis off;
set(gcf,'color', 'white');
caxis([0 1]);
colorbar off;
subplot(132);

imagesc(contextModel-eye(size(contextModel))); 
axis off;
set(gcf,'color', 'white');
caxis([0 1]);
colorbar off;

subplot(133);
imagesc(episodeModel-eye(size(episodeModel))); 
axis off;
set(gcf,'color', 'white');
caxis([0 1]);
colorbar off;

exportgraphics(figure(1), 'fig4_insets.pdf');
