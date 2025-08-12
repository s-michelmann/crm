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

            % Read recall data
            recallFile = sprintf('mPFC_recall_%s_%s_betas.csv', person1, place);
            if isfile(recallFile)
                recallData = readmatrix(recallFile);
            else
                warning('Recall file not found: %s', recallFile);
                recallData = [];
            end

            % Store in structure with dynamic field names
            fieldName = sprintf('%s_%s', person1, place);
            dataStruct.(fieldName).video = videoData;
            dataStruct.(fieldName).recall = recallData;
        end
    end
    %% compute all correlations
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

    % figure;
    % nanmask = eye(size(correlationMatrix));
    % nanmask(nanmask == 1) = nan;
    % imagesc(correlationMatrix+nanmask);
    % colorbar;
    % axis equal tight;
    % xticks(1:length(allLabels));
    % yticks(1:length(allLabels));
    % xticklabels(allLabels);
    % yticklabels(allLabels);
    % xtickangle(90);
    % title('Full Correlation Matrix of All Video Vectors');
    % 
    % % Fix underscore interpretation in labels
    % ax = gca;
    % ax.TickLabelInterpreter = 'none';

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define schemas
    schema1 = {'delta', 'mishkas'};  % Schema A
    schema2 = {'coop', 'nugget'};      % Schema B
    people = {'tommy', 'lisa'};

    % ---------- Context-Level Covariance ----------
    context1 = [];
    context2 = [];

    all_places = [schema1, schema2];

    for i = 1:length(all_places)
        place = all_places{i};

        % Collect all video vectors for this context
        contextVectors = [];
        for p = 1:length(people)
            key = sprintf('%s_%s', people{p}, place);
            contextVectors = [contextVectors, dataStruct.(key).video];  % features x 3
        end

        % All pairwise comparisons within this context (excluding self-pairs)
        for a = 1:size(contextVectors, 2)
            for b = 1:size(contextVectors, 2)
                if a == b
                    continue;
                end
                context1 = [context1, contextVectors(:, a)];
                context2 = [context2, contextVectors(:, b)];
            end
        end
    end

    % Mean-center
    context1 = context1 - mean(context1, 2);
    context2 = context2 - mean(context2, 2);

    % Covariance matrix for context
    Cov_context = (context1 * context2') / size(context1, 2);

    Cov_xx_context = (context1 * context1') / size(context1, 2);
    Cov_yy_context = (context2 * context2') / size(context2, 2);
    % ---------- Schema-Level Covariance ----------
    schema1_data = [];
    schema2_data = [];


    place1 = schema1{1};
    place2 = schema1{2};

    % Collect all videos from schema1 and schema2
    schema1Vectors = [];
    schema2Vectors = [];

    for p = 1:length(people)
        schema1Vectors = [schema1Vectors, dataStruct.(sprintf('%s_%s', people{p}, place1)).video];
        schema2Vectors = [schema2Vectors, dataStruct.(sprintf('%s_%s', people{p}, place2)).video];
    end

    % All pairwise comparisons between schema1 and schema2
    for i = 1:size(schema1Vectors, 2)
        for j = 1:size(schema2Vectors, 2)
            schema1_data = [schema1_data, schema1Vectors(:, i)];
            schema2_data = [schema2_data, schema2Vectors(:, j)];
        end
    end

    place1 = schema2{1};
    place2 = schema2{2};

    % Collect all videos from schema1 and schema2
    schema1Vectors = [];
    schema2Vectors = [];

    for p = 1:length(people)
        schema1Vectors = [schema1Vectors, dataStruct.(sprintf('%s_%s', people{p}, place1)).video];
        schema2Vectors = [schema2Vectors, dataStruct.(sprintf('%s_%s', people{p}, place2)).video];
    end

    % All pairwise comparisons between schema1 and schema2
    for i = 1:size(schema1Vectors, 2)
        for j = 1:size(schema2Vectors, 2)
            schema1_data = [schema1_data, schema1Vectors(:, i)];
            schema2_data = [schema2_data, schema2Vectors(:, j)];
        end
    end

    % Mean-center
    schema1_data = schema1_data - mean(schema1_data, 2);
    schema2_data = schema2_data - mean(schema2_data, 2);

    % Covariance matrix for schema
    Cov_schema = (schema1_data * schema2_data') / size(schema1_data, 2);

    % % Display results
    % figure;
    % title('Covariance matrix for context-level comparisons:');
    % imagesc(Cov_context);
    % figure;
    % title('Covariance matrix for schema-level comparisons:');
    % imagesc(Cov_schema);
    %%
    %% CRM for contrast

    WX = [];
    WY = [];

    for ff = 1 : num_components
        ff
        [w_x, w_y, lbd3i] = compute_weights(Cov_xx_context, ...
            Cov_yy_context, Cov_context, Cov_schema, ff, 0.001);
        if ~ isreal(w_x); continue; end
        if ~ isreal(w_y); continue; end
        WX = cat(2, WX, (w_x));
        WY = cat(2, WY, (w_y));

    end


    allVectorsX = WX'*allVectors;
    allVectorsY = WY'*allVectors;


    % Normalize columns (zero mean, unit variance)
    A = (allVectorsX - mean(allVectorsX, 1)) ./ std(allVectorsX, 0, 1);
    B = (allVectorsY - mean(allVectorsY, 1)) ./ std(allVectorsY, 0, 1);

    % Compute correlation matrix: each entry is correlation between a column of A and a column of B
    correlationMatrix_crm = A' * B / size(A, 1);

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

    correlationMatrix_context = correlationMatrix_crm;
    filename = fullfile(sj_path, 'correlationMatrix_context');
    save (filename, 'correlationMatrix_context');
end