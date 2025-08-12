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



%% load in the data
for ss = 1 : 22
    if ss == 15; continue; end
    if ss == 18; continue; end

    sj_path = fullfile(data_path, ['sub-1' sprintf('%02d', ss)]);

    files = dir(fullfile(sj_path, 'mPFC*'));
    fileNames = {files.name}';
    


    % Define parameters
    people = {'tommy', 'lisa'};
    % mishkas cafe delta cafe nugget grocery coop grocery
    places = {'delta', 'mishkas', 'coop', 'nugget'};
    videos = {'video1', 'video2', 'video3'};

    % Initialize structure to hold data
    dataStruct = struct();

    % Loop through each person and place
    for p = 1:length(people)
        for pl = 1:length(places)
            person1 = people{p};
            place = places{pl};

            % Initialize an empty array to concatenate video data
            videoData = [];

            % Loop through each video and concatenate
            for v = 1:length(videos)
                video = videos{v};
                filename = fullfile(sj_path,...
                    sprintf('mPFC_%s_%s_%s_betas.csv', video, person1, place));
                if isfile(filename)
                    tempData = readmatrix(filename);
                    videoData = [videoData, tempData];
                else
                    warning('File not found: %s', filename);
                end
            end



            % Store in structure with dynamic field names
            fieldName = sprintf('%s_%s', person1, place);
            dataStruct.(fieldName).video = videoData;
        end
    end
    %% compute all correlations


% # create model matrices for our effects
% x_lab = ['Tommy Cafe1', 'Lisa Cafe1', 'Tommy Cafe2', 'Lisa Cafe2',
%      'Tommy Grocery1', 'Lisa Grocery1', 'Tommy Grocery2', 'Lisa Grocery2']
% y_lab = ['Tommy Cafe1', 'Lisa Cafe1', 'Tommy Cafe2', 'Lisa Cafe2',
%      'Tommy Grocery1', 'Lisa Grocery1', 'Tommy Grocery2', 'Lisa Grocery2']
% 
% #x_lab = ['Tommy Delta', 'Lisa Delta', 'Tommy Mishkas', 'Lisa Mishkas',
% #     'Tommy CoOp', 'Lisa CoOp', 'Tommy Nugget', 'Lisa Nugget']
% #y_lab = ['Tommy Delta', 'Lisa Delta', 'Tommy Mishkas', 'Lisa Mishkas',
% #     'Tommy CoOp', 'Lisa CoOp', 'Tommy Nugget', 'Lisa Nugget']


    % Define the order of conditions and video labels
    conditions = {
        'tommy_delta', 'lisa_delta', ...
        'tommy_mishkas', 'lisa_mishkas', ...
        'tommy_coop', 'lisa_coop', ...
        'tommy_nugget', 'lisa_nugget'
        };
    videoLabels = {'v1', 'v2', 'v3'};

    % Initialize storage
    allVectors = [];
    allLabels = {};

    % Extract each video vector and label it
    for i = 1:length(conditions)
        cond = conditions{i};
        videoData = dataStruct.(cond).video;  % size: features x 3
        for j = 1:size(videoData, 2)
            allVectors = [allVectors, videoData(:, j)];
            allLabels{end+1} = sprintf('%s_%s', cond, videoLabels{j});
        end
    end

    % Compute full correlation matrix
    correlationMatrix = corr(allVectors);
    correlationMatrix_raw = correlationMatrix;
    filename = fullfile(sj_path, 'correlationMatrix_raw');
    save (filename, 'correlationMatrix_raw');
    figure;
    nanmask = eye(size(correlationMatrix));
    nanmask(nanmask == 1) = nan;
    imagesc(correlationMatrix+nanmask);
    colorbar;
    axis equal tight;
    xticks(1:length(allLabels));
    yticks(1:length(allLabels));
    xticklabels(allLabels);
    yticklabels(allLabels);
    xtickangle(90);
    title('Full Correlation Matrix of All Video Vectors');

    % Fix underscore interpretation in labels
    ax = gca;
    ax.TickLabelInterpreter = 'none';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %% Episode, vs sampe place different person patterns
    % Initialize arrays to collect paired vectors
    sameEpisode1 = [];
    sameEpisode2 = [];

    % Loop through each condition
    for p = 1:length(people)
        for pl = 1:length(places)
            key = sprintf('%s_%s', people{p}, places{pl});
            videoData = dataStruct.(key).video;  % features x 3

            % All pairwise comparisons between different videos
            for i = 1:3
                for j = 1:3
                    if i == j
                        continue;  % skip self-comparisons
                    end
                    Vec1 = videoData(:, i);
                    Vec2 = videoData(:, j);
                    sameEpisode1 = [sameEpisode1, Vec1];
                    sameEpisode2 = [sameEpisode2, Vec2];
                end
            end
        end
    end

    % Mean-center the vectors
    sameEpisode1 = sameEpisode1 - mean(sameEpisode1, 2);
    sameEpisode2 = sameEpisode2 - mean(sameEpisode2, 2);

    % Estimate the feature-by-feature covariance matrix
    Cov_sameEpisode = (sameEpisode1 * sameEpisode2') / size(sameEpisode1, 2);

    Cov_xx_sameEpisode = (sameEpisode1 * sameEpisode1') / size(sameEpisode1, 2);
    Cov_yy_sameEpisode = (sameEpisode2 * sameEpisode2') / size(sameEpisode2, 2);

    % Initialize arrays to collect paired vectors
    samePlaceDifferenPerson1 = [];
    samePlaceDifferenPerson2 = [];

    % Loop through each place and compare Tommy vs Lisa
    for pl = 1:length(places)
        keyTommy = sprintf('tommy_%s', places{pl});
        keyLisa = sprintf('lisa_%s', places{pl});

        dataTommy = dataStruct.(keyTommy).video;  % features x 3
        dataLisa = dataStruct.(keyLisa).video;    % features x 3

        % All pairwise comparisons between Tommy and Lisa videos
        for i = 1:3
            for j = 1:3
                Vec1 = dataTommy(:, i);
                Vec2 = dataLisa(:, j);
                samePlaceDifferenPerson1 = [samePlaceDifferenPerson1, Vec1];
                samePlaceDifferenPerson2 = [samePlaceDifferenPerson2, Vec2];
            end
        end
    end

    % Mean-center the vectors
    samePlaceDifferenPerson1 = samePlaceDifferenPerson1 - mean(samePlaceDifferenPerson1, 2);
    samePlaceDifferenPerson2 = samePlaceDifferenPerson2 - mean(samePlaceDifferenPerson2, 2);

    % Estimate the feature-by-feature covariance matrix
    Cov_samePlaceDifferenPerson = (samePlaceDifferenPerson1 * samePlaceDifferenPerson2') / size(samePlaceDifferenPerson1, 2);

    % Optional: also compute auto-covariances
    Cov_xx_PlaceDifferenPerson = (samePlaceDifferenPerson1 * samePlaceDifferenPerson1') / size(samePlaceDifferenPerson1, 2);
    Cov_yy_PlaceDifferenPerson = (samePlaceDifferenPerson2 * samePlaceDifferenPerson2') / size(samePlaceDifferenPerson2, 2);


    %% CRM for contrast

    WX = [];
    WY = [];

    for ff = 1 : num_components
        ff
        [w_x, w_y, lbd3i] = compute_weights(Cov_xx_sameEpisode, ...
            Cov_yy_sameEpisode, Cov_sameEpisode, Cov_samePlaceDifferenPerson, ff, 0.001);
        if ~ isreal(w_x); continue; end
        if ~ isreal(w_y); continue; end

        WX = cat(2, WX, w_x);
        WY = cat(2, WY, w_y);

    end


    allVectorsX = WX'*allVectors;
    allVectorsY = WY'*allVectors;


    % Normalize columns (zero mean, unit variance)
    A = (allVectorsX - mean(allVectorsX, 1)) ./ std(allVectorsX, 0, 1);
    B = (allVectorsY - mean(allVectorsY, 1)) ./ std(allVectorsY, 0, 1);

    % Compute correlation matrix: each entry is correlation between a column of A and a column of B
    correlationMatrix_crm = A' * B / size(A, 1);
    
    correlationMatrix_episode = correlationMatrix_crm;
    filename = fullfile(sj_path, 'correlationMatrix_episode');
    save (filename, 'correlationMatrix_episode');
    %% plot
    figure;
    nanmask = eye(size(correlationMatrix));
    nanmask(nanmask == 1) = nan;
    imagesc(correlationMatrix+nanmask);
    colorbar;
    axis equal tight;
    xticks(1:length(allLabels));
    yticks(1:length(allLabels));
    xticklabels(allLabels);
    yticklabels(allLabels);
    xtickangle(90);
    title('Correlation Matrix of All Video Vectors (raw)');

    % Fix underscore interpretation in labels
    ax = gca;
    ax.TickLabelInterpreter = 'none';
    set(gcf, 'color', 'white');
    figure;
    nanmask = eye(size(correlationMatrix_crm));
    nanmask(nanmask == 1) = nan;
    imagesc(correlationMatrix_crm+nanmask);
    colorbar;
    axis equal tight;
    xticks(1:length(allLabels));
    yticks(1:length(allLabels));
    xticklabels(allLabels);
    yticklabels(allLabels);
    xtickangle(90);
    title('Correlation Matrix of All Video Vectors (components)');

    % Fix underscore interpretation in labels
    ax = gca;
    ax.TickLabelInterpreter = 'none';
    set(gcf, 'color', 'white');
end
