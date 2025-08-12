clear;
clc;
close all
restoredefaultpath;
homeDir  = getenv('HOME');
ft_pth = fullfile(homeDir, 'Documents', 'fieldtrip'); % Construct the path to the Documents folder

addpath(ft_pth);
ft_defaults;

dir_code = "crm_canon_prev_word_10fold";
%% define paths
cwd_path = pwd;
addpath(genpath(fileparts(cwd_path)));

% Get the project
idx = strfind(cwd_path, 'CRM_project');
parentDir = fileparts(cwd_path);

data_path = fullfile( cwd_path(1 : idx + numel('CRM_project') - 1), ...
    "Podcast", "ds005574");

stim_path = fullfile(data_path, "stimuli");
transcript_tsv  = fullfile(stim_path,...
    "whisper-medium","transcript.tsv" );
features_file  = fullfile(stim_path,...
    "whisper-medium", "features.hdf5");

ecog_path = fullfile(data_path,...
    "derivatives", "ecogprep");
%% load in the stim data
feature_data = h5read(features_file, '/vectors');
dataTable = readtable(transcript_tsv,...
    'FileType', 'text', 'Delimiter', '\t');
word_onsets = dataTable.start;
word_offsets = dataTable.xEnd;
words = dataTable.word;
embeddings = feature_data';
num_components = 50; % 50 components of embeddings

%% load in a subject
for ss = 1: 9
    sub_code = ['sub-0' num2str(ss)];
    deriv_path = fullfile(data_path,  "derivatives",...
        dir_code, sub_code);
    if ~exist(deriv_path, 'dir'); mkdir(deriv_path); end
    % Read the .fif file
    filename = fullfile(ecog_path,sub_code , 'ieeg',...
        [sub_code '_task-podcast_desc-highgamma_ieeg.fif']);
    data = ft_read_data(filename);

    chan_tsv = fullfile(data_path, sub_code, ...
        'ieeg', [sub_code '_task-podcast_channels.tsv']);

    elec_tsv = fullfile(data_path, sub_code, ...
        'ieeg', ...
        [sub_code ...
        '_space-MNI152NLin2009aSym_electrodes.tsv']);

    % Read the .tsv file to get channel labels
    channel_info = readtable(chan_tsv, ...
        'FileType', 'text', 'Delimiter', '\t');
    % Read the .tsv file to get channel labels
    elec_info = readtable(elec_tsv, ...
        'FileType', 'text', 'Delimiter', '\t');


    ecog_indices = find(...
        strcmp(channel_info.type, 'ECOG') & ...
        strcmp(channel_info.status, 'good'));

    channel_labels = channel_info.name(ecog_indices);
    % match the x, y, z
    % Initialize an array to store coordinates
    xyz_coords = nan(size(channel_labels,1), 3);

    % Loop through each label and find the corresponding coordinates
    for ii = 1: size(channel_labels,1)
        % Find the index in elec_info where the name matches the label
        idx = find(strcmp(elec_info.name, channel_labels{ii}));

        % If a match is found, store the coordinates
        if ~isempty(idx)
            xyz_coords(ii, 1) = elec_info.x(idx);
            xyz_coords(ii, 2) = elec_info.y(idx);
            xyz_coords(ii, 3) = elec_info.z(idx);
        end
    end
    %% convert to fieldtrip
    eeg =[];
    eeg.trial{1} = data;
    eeg.label = channel_labels;
    eeg.time{1} = [1/512:1/512:length(data)/512];
    eeg.elec.label = channel_labels;
    eeg.elec.elecpos = xyz_coords;
    eeg.elec.chanpos = xyz_coords;
    eeg.fsample = 512;
    eeg.sampleinfo = [1 length(eeg.time{1})];
    %% downsample to 100Hz
    cfg = [];
    cfg.resamplefs = 100;
    eeg = ft_resampledata(cfg, eeg);
    eeg.sampleinfo = [1 length(eeg.time{1})];

    % Validate the EEG structure using ft_datatype_raw
    %% pca embeddings down to 50/100;
    [coeff, score, latent] = pca(embeddings);
    reduced_array = score(:, 1:num_components);

    % start at word 21 (we want 10 seconds of padding +)
    n_word_padding = 20;
    word_start_index = n_word_padding+1;

    % these are the current words
    emb_words = reduced_array(word_start_index:end-n_word_padding, :);
    emb_words_circ = reduced_array(...
        word_start_index-1:end-1-n_word_padding, :);
    % center:
    emb_words = emb_words-mean(emb_words,1);
    emb_words_circ = emb_words_circ-mean(emb_words_circ,1);

    onsets = word_onsets(word_start_index:end-n_word_padding);
    onsets_circ = word_onsets(word_start_index-1:end-1-n_word_padding);

    %% cut EEG to word onset times
    % word samples
    w_samples1 = arrayfun(@(x) nearest(eeg.time{1},x), onsets)...
        +eeg.sampleinfo(1)-1;
    w_samples2 = arrayfun(@(x) nearest(eeg.time{1},x), onsets_circ)...
        +eeg.sampleinfo(1)-1;
    % padding = 5 seonds
    pad_s = 4;
    cfg = [];
    cfg.trl = [w_samples1-pad_s*eeg.fsample, ...
        w_samples1+pad_s*eeg.fsample, ...
        -pad_s.*ones(size(w_samples1)).*eeg.fsample];
    eeg_sig = ft_redefinetrial(cfg, eeg);

    %% for each channel: concatenate the vector and compute canonical correlation

    results = [];
    cfg = [];
    cfg.keeptrials = 'yes';
    eeg_tl = ft_timelockanalysis(cfg, eeg_sig);
    eeg_dat = eeg_tl.trial;
    clear eeg_tl;

    num_iterations = 10;
    results.num_iterations = num_iterations;
    lag_axis = [-4:1/eeg_sig.fsample:4];
    results.lag_axis = lag_axis;
    n_steps = length(lag_axis);
    n_words = size(emb_words, 1);
    results.n_words = n_words;
    num_channels = length(eeg_sig.label);
    results.num_channels = num_channels;
    Ws = nan(num_channels, num_iterations, n_steps);
    Vs = nan(num_channels, num_iterations, n_steps);
    %%

    r1_canon = nan(num_iterations, n_steps) ;
    r1_crm   = nan(num_iterations, n_steps) ;

    r1_canon_control  = nan(num_iterations, n_steps) ;
    r1_crm_control    = nan(num_iterations, n_steps) ;

    for ll = 1 : n_steps
        fprintf('at %d percent\r', round(100 * ll / n_steps));
       
        cv = cvpartition(n_words, 'KFold', num_iterations);

        for idx = 1:num_iterations
            % Get train and test indices for the current iteration
            train_indices{idx} = cv.training(idx);
            test_indices{idx} = cv.test(idx);

            % Split the data
            eeg_train = eeg_dat(train_indices{idx}, :, ll);
            eeg_test = eeg_dat(test_indices{idx}, :, ll);
            embeddings_train = emb_words(train_indices{idx}, :);
            embeddings_test = emb_words(test_indices{idx}, :);

            embeddings_train_ctrl = emb_words_circ(train_indices{idx}, :);
            embeddings_test_ctrl = emb_words_circ(test_indices{idx}, :);


            Cxx = eeg_train' * eeg_train;
            Cyy = embeddings_train' * embeddings_train;
            Cxy = eeg_train' * embeddings_train;
            Dxy = eeg_train' * embeddings_train_ctrl;

            [Wx, Wy, ~]  = compute_weights(Cxx,Cyy,Cxy, zeros(size(Cxy)),1, 0.001);

            [Vx, Vy, lbd3i] = compute_weights(Cxx,Cyy,Cxy, Dxy,1, 0.001);


            r1_canon(idx, ll) = corr(eeg_test*Wx, embeddings_test* Wy);
            r1_crm(idx, ll)    = corr(eeg_test*Vx, embeddings_test* Vy);

            r1_canon_control(idx, ll)  = corr(eeg_test*Wx, embeddings_test_ctrl* Wy);
            r1_crm_control(idx, ll)  = corr(eeg_test*Vx, embeddings_test_ctrl* Vy);
            Ws(:, idx, ll) = Wx;
            Vs(:, idx, ll) = Vx;


        end


    end
    results.Ws = Ws;
    results.Vs = Vs;
    results.r1_canon = r1_canon;
    results.r1_crm = r1_crm;
    results.r1_canon_control = r1_canon_control;
    results.r1_crm_control = r1_crm_control;
    save_file = fullfile(deriv_path, "results_gamma50Fin.mat");
    save(save_file, "results");
end

%% results and plotting
%% Initialize variables to store results from all subjects
fsample = 100;
all_r1_canon = [];
all_r1_crm = [];
all_r1_canon_control = [];
all_r1_crm_control = [];

% Load results from each subject
for sub = 1:9
    sub_code = sprintf('sub-%02d', sub);
    deriv_path = fullfile(data_path,  "derivatives",...
        dir_code, sub_code);
    load(fullfile(deriv_path, ['results_gamma50Fin.mat']), 'results');
    
    % Concatenate results across subjects
    % Average across folds
    avg_r1_canon = mean(results.r1_canon, 1);
    avg_r1_crm = mean(results.r1_crm, 1);
    avg_r1_canon_control = mean(results.r1_canon_control, 1);
    avg_r1_crm_control = mean(results.r1_crm_control, 1);
    
    % Concatenate results across subjects
    all_r1_canon = cat(1, all_r1_canon, avg_r1_canon);
    all_r1_crm = cat(1, all_r1_crm, avg_r1_crm);
    all_r1_canon_control = cat(1, all_r1_canon_control, avg_r1_canon_control);
    all_r1_crm_control = cat(1, all_r1_crm_control, avg_r1_crm_control);
end

% Calculate mean and standard error across subjects
mean_r1_canon = mean(all_r1_canon, 1);
std_r1_canon = std(all_r1_canon, [], 1) / sqrt(size(all_r1_canon, 1));

mean_r1_crm = mean(all_r1_crm, 1);
std_r1_crm = std(all_r1_crm, [], 1) / sqrt(size(all_r1_crm, 1));

mean_r1_canon_control = mean(all_r1_canon_control, 1);
std_r1_canon_control = std(all_r1_canon_control, [], 1) / sqrt(size(all_r1_canon_control, 1));

mean_r1_crm_control = mean(all_r1_crm_control, 1);
std_r1_crm_control = std(all_r1_crm_control, [], 1) / sqrt(size(all_r1_crm_control, 1));

%% Plot results with shaded error bars
figure;
hold on;
[-4:1/fsample:4]
% Set background color to white
set(gcf, 'Color', 'w');

% Plot canonical correlation results
shadedErrorBar([-4:1/fsample:4], mean_r1_canon, std_r1_canon, 'lineprops', {'-','Color','#3D348B'}, 'patchSaturation', 0.1);
shadedErrorBar([-4:1/fsample:4], mean_r1_crm, std_r1_crm, 'lineprops', {'-','Color','#F35B04'}, 'patchSaturation', 0.1);

% % Plot control results
% shadedErrorBar([-4:1/fsample:4], mean_r1_canon_control, std_r1_canon_control, 'lineprops', {'--','Color','#7678ED'}, 'patchSaturation', 0.1);
% shadedErrorBar([-4:1/fsample:4], mean_r1_crm_control, std_r1_crm_control, 'lineprops', {'--','Color','#F18701'}, 'patchSaturation', 0.1);

% Set font size and type
set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

legend({'CCA', 'CRM', 'CCA Control', 'CRM Control'});
xlabel('Time around word onset');
ylabel('Correlation');
title('Contextual vs. context free LLM-alignment');
hold off;

exportgraphics(figure(1), 'fig3b.pdf');
%% compute topographies at peak

all_epos = [];
all_topos_cca =[];
all_topos_crm = [];
for ss = 1: 9
    sub_code = ['sub-0' num2str(ss)];
    deriv_path = fullfile(data_path,  "derivatives",...
        dir_code, sub_code);
    if ~exist(deriv_path, 'dir'); mkdir(deriv_path); end
    % Read the .fif file
    filename = fullfile(ecog_path,sub_code , 'ieeg',...
        [sub_code '_task-podcast_desc-highgamma_ieeg.fif']);
    data = ft_read_data(filename);

    chan_tsv = fullfile(data_path, sub_code, ...
        'ieeg', [sub_code '_task-podcast_channels.tsv']);

    elec_tsv = fullfile(data_path, sub_code, ...
        'ieeg', ...
        [sub_code ...
        '_space-MNI152NLin2009aSym_electrodes.tsv']);

    % Read the .tsv file to get channel labels
    channel_info = readtable(chan_tsv, ...
        'FileType', 'text', 'Delimiter', '\t');
    % Read the .tsv file to get channel labels
    elec_info = readtable(elec_tsv, ...
        'FileType', 'text', 'Delimiter', '\t');


    ecog_indices = find(...
        strcmp(channel_info.type, 'ECOG') & ...
        strcmp(channel_info.status, 'good'));

    channel_labels = channel_info.name(ecog_indices);
    % match the x, y, z
    % Initialize an array to store coordinates
    xyz_coords = nan(size(channel_labels,1), 3);

    % Loop through each label and find the corresponding coordinates
    for ii = 1: size(channel_labels,1)
        % Find the index in elec_info where the name matches the label
        idx = find(strcmp(elec_info.name, channel_labels{ii}));

        % If a match is found, store the coordinates
        if ~isempty(idx)
            xyz_coords(ii, 1) = elec_info.x(idx);
            xyz_coords(ii, 2) = elec_info.y(idx);
            xyz_coords(ii, 3) = elec_info.z(idx);
        end
    end
    %% convert to fieldtrip
    eeg =[];
    eeg.trial{1} = data;
    eeg.label = channel_labels;
    eeg.time{1} = [1/512:1/512:length(data)/512];
    eeg.elec.label = channel_labels;
    eeg.elec.elecpos = xyz_coords;
    eeg.elec.chanpos = xyz_coords;
    eeg.fsample = 512;
    eeg.sampleinfo = [1 length(eeg.time{1})];
    %% downsample to 100Hz
    cfg = [];
    cfg.resamplefs = 100;
    eeg = ft_resampledata(cfg, eeg);
    eeg.sampleinfo = [1 length(eeg.time{1})];
    
    %% pca embeddings down to 100;
    [coeff, score, latent] = pca(embeddings);
    reduced_array = score(:, 1:num_components);

    % start at word 21 (we want 10 seconds of padding +)
    n_word_padding = 20;
    word_start_index = n_word_padding+1;

    % these are the current words
    emb_words = reduced_array(word_start_index:end-n_word_padding, :);
    emb_words_circ = reduced_array(...
        word_start_index-1:end-1-n_word_padding, :);
    % center:
    emb_words = emb_words-mean(emb_words,1);
    emb_words_circ = emb_words_circ-mean(emb_words_circ,1);

    onsets = word_onsets(word_start_index:end-n_word_padding);
    onsets_circ = word_onsets(word_start_index-1:end-1-n_word_padding);

    %% cut EEG to word onset times
    % word samples
    w_samples1 = arrayfun(@(x) nearest(eeg.time{1},x), onsets)...
        +eeg.sampleinfo(1)-1;
    w_samples2 = arrayfun(@(x) nearest(eeg.time{1},x), onsets_circ)...
        +eeg.sampleinfo(1)-1;
    % padding = 5 seonds
    pad_s = 4;
    cfg = [];
    cfg.trl = [w_samples1-pad_s*eeg.fsample, ...
        w_samples1+pad_s*eeg.fsample, ...
        -pad_s.*ones(size(w_samples1)).*eeg.fsample];
    eeg_sig = ft_redefinetrial(cfg, eeg);
    cfg = [];
    cfg.keeptrials = 'yes';
    eeg_tl = ft_timelockanalysis(cfg, eeg_sig);
    % pick the zero-lag EEG
    eeg_dat = eeg_tl.trial(:,:,nearest(eeg_tl.time,0));
    %%
    %load the result
    load(fullfile(deriv_path,...
        ['results_gamma50Fin.mat']), 'results');
    all_epos = cat(1, all_epos, eeg_sig.elec.elecpos);
    cca_topos = [];
    crm_topos = [];
    for foldidx = 1: 5
        tmp_dat = eeg_dat* ...
            results.Ws(:,...
            foldidx,nearest(eeg_tl.time,0));
        cca_topos = cat(1, cca_topos, ...
            abs(corr(tmp_dat, eeg_dat)));
        tmp_dat = eeg_dat* ...
            results.Vs(:,...
            foldidx,nearest(eeg_tl.time,0));
        crm_topos = cat(1, crm_topos, ...
            abs(corr(tmp_dat, eeg_dat)));
    end
    cca_topo = mean(cca_topos);
    crm_topo = mean(crm_topos);

    all_topos_cca = cat(1, all_topos_cca, cca_topo');
    all_topos_crm = cat(1, all_topos_crm, crm_topo');

    
end
%%
topos_delta = (all_topos_crm - all_topos_cca);
selection = all_topos_cca> 0.1;
mesh_pth =  [ft_pth '/template/anatomy/'];
close all;
plot_ecog((all_topos_cca(selection)),mesh_pth, all_epos(selection,:),[0.1 1], 0.9, [-90 0], 20, 1, 2, flipud(hot(256)) ,1)
colorbar;
exportgraphics(figure(1), 'fig3c_top.pdf');

selection = all_topos_crm> 0.1;
close all;
plot_ecog((all_topos_crm(selection)),mesh_pth, all_epos(selection,:),[0.1 1], 0.9, [-90 0], 20, 1, 2, flipud(hot(256)) ,1)

colorbar;
exportgraphics(figure(1), 'fig3c_mid.pdf');
selection = abs(topos_delta)> 0.05;

close all;
plot_ecog((topos_delta(selection)),mesh_pth, all_epos(selection,:),[-0.4 0.4], 0.9, [-90 0], 20, 1, 2, [] ,1)
colorbar;
exportgraphics(figure(1), 'fig3c_bot.pdf');

%% export for plotting in python


% save_path = fullfile(data_path,  "derivatives",...
%     dir_code);
% save(fullfile(save_path, "all_topos_crm50Fin.mat"), 'all_topos_crm');
% save(fullfile(save_path, "all_topos_cca50Fin.mat"), 'all_topos_cca');
% save(fullfile(save_path, "all_epos.mat"), 'all_epos');