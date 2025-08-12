close all
clear;
clc;
seed = 1234; % You can choose any integer value
rng(seed);
restoredefaultpath;
num_components = 20;




%% Paths
homeDir  = getenv('HOME');

cwd_path = pwd;
addpath(genpath(fileparts(cwd_path)));

% Get the project
idx = strfind(cwd_path, 'CRM_project');

data_path = fullfile( cwd_path(1 : idx + numel('CRM_project') - 1), ...
    "reagh_data");


%% Load and process data

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

videoLabels = {'v1', 'v2', 'v3'};


for ss = 1:22
    if ss == 15 || ss == 18
        continue;
    end

    sj_path = fullfile(data_path, ['sub-1' sprintf('%02d', ss)]);


    % Initialize storage
    allVectors = [];
    allLabels = {};
    % Load data into structure
    dataStruct = struct();
    for p = 1:length(people)
        for pl = 1:length(places)
            person = people{p};
            place = places{pl};
            fieldName = sprintf('%s_%s', person, place);
            videoData = [];

            for v = 1:length(videos)
                video = videos{v};
                filename = fullfile(sj_path, sprintf('mPFC_%s_%s_%s_betas.csv', video, person, place));
                if isfile(filename)
                    tempData = readmatrix(filename);
                    videoData = [videoData, tempData];
                else
                    warning('Missing file: %s', filename);
                end
            end
            dataStruct.(fieldName).video = videoData;
        end
    end

    % Collect all vectors and labels
    for i = 1:length(conditions)
        cond = conditions{i};
        videoData = dataStruct.(cond).video;
        for j = 1:size(videoData, 2)
            allVectors = [allVectors, videoData(:, j)];
            allLabels{end+1} = sprintf('%s_%s', cond, videoLabels{j});
        end
    end


    %% Compute full correlation matrix
    correlationMatrix = corr(allVectors, 'Rows', 'pairwise');
    mask = eye(size(correlationMatrix));
    correlationMatrix(mask == 1) = nan;
    %% Average correlations by condition
    nConds = length(conditions);
    avgCorrMatrix = zeros(nConds);

    % Group indices by condition
    groupIndices = containers.Map;
    for i = 1:nConds
        cond = conditions{i};
        idx = find(contains(allLabels, cond));
        groupIndices(cond) = idx;
    end

    % Compute average correlation between condition pairs
    for i = 1:nConds
        for j = 1:nConds
            % if i == j; continue; end
            idx_i = groupIndices(conditions{i});
            idx_j = groupIndices(conditions{j});
            subCorr = correlationMatrix(idx_i, idx_j);
            avgCorrMatrix(i, j) = mean(subCorr(:), 'omitnan');
        end
    end
    %% Plot averaged correlation matrix
    % figure;
    % imagesc(avgCorrMatrix);
    % colorbar;
    % axis equal tight;
    % xticks(1:nConds);
    % yticks(1:nConds);
    % xticklabels(conditions);
    % yticklabels(conditions);
    % xtickangle(90);
    % title('Averaged Correlation Matrix by Condition (Sorted by Place)');
    % ax = gca;
    % ax.TickLabelInterpreter = 'none';
    % set(gcf, 'color', 'white');


    filename = fullfile(sj_path, 'correlationMatrix_avg');
    save (filename, 'avgCorrMatrix');
end



