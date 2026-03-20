%% result 1: removing 1-word overlap removes all word-onset effects and pre-stim alingment

clear;
clc;
close all
restoredefaultpath;
homeDir  = getenv('HOME');
ft_pth = fullfile(homeDir, 'Documents', 'fieldtrip'); % Construct the path to the Documents folder
addpath(fullfile(ft_pth, 'external', 'mne'));

addpath(ft_pth);
ft_defaults;

dir_code = "crm_canon_prev_word_10fold_parCCA";

%%
audio_result.num_iterations = 6;
step_size = 1;

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
audioFile  = fullfile(stim_path,...
    "podcast.wav");
transcriptFile = fullfile(stim_path,...
    "spectral",'transcript.tsv');
ecog_path = fullfile(data_path,...
    "derivatives", "ecogprep");

%% load in the stim data
feature_data = h5read(features_file, '/vectors');
%% use gpt-2 instead
load(fullfile(stim_path,'gpt-2', 'gpt2_embeddings.mat'))
dataTable = readtable(transcript_tsv,...
    'FileType', 'text', 'Delimiter', '\t');
word_onsets = dataTable.start;
word_offsets = dataTable.xEnd;
words = dataTable.word;
% embeddings = feature_data';
num_components = 50; % 50 components of embeddings

% %% print out the words:
%
% outputFile = "words.txt";
% fid = fopen(outputFile, 'w');
% for i = 1:numel(words)
%     fprintf(fid, '%s\n', string(words(i)));
% end
% fclose(fid);

%% prepare the audio
% Load audio
[x, fs] = audioread(audioFile);
x = mean(x, 2); % convert to mono if stereo

% Parameters
winLength   = round(0.025 * fs);   % 25 ms window
hopLength   = round(0.010 * fs);   % 10 ms hop
fftLength   = 2^nextpow2(winLength);
num_bands = 8;
% Compute 8‑Mel‑band spectrogram
[S, f, t] = melSpectrogram(x, fs, ...
    'Window', hann(winLength, "periodic"), ...
    'OverlapLength', winLength - hopLength, ...
    'FFTLength', fftLength, ...
    'NumBands', num_bands);

% Convert to dB for visualization
SdB = 10 * log10(S + eps);
%
% % Plot
% figure;
% surf(t, f, SdB, 'EdgeColor', 'none');
% axis tight;
% view(0, 90);
% xlabel('Time (s)');
% ylabel('Mel Frequency (Hz)');
% title('8‑Mel‑Band Spectrogram');
% colorbar;
%% --- ENVELOPE aligned to spectrogram frames ---
env = envelope(x, winLength, 'rms');
env_ds = env(1:hopLength:end);
env_ds = env_ds(1:length(t)); % match time frames

% concatenate
features = [SdB; env_ds.']; % [9 × T]


%% --- EPOCHING ±4 seconds around each onset ---

epochDur = 8; % total seconds
epochPre = 4;
epochPost = 4;
fsFeature = 100;
nTime = round(epochDur * fsFeature)+1; % should be ~800 at 100 Hz
taxis = linspace(-epochPre, epochPost, nTime); % relative time axis
halfDur = 4; % seconds before/after
framesPerSec = 1 / (hopLength/fs);
halfFrames = round(halfDur * framesPerSec);
epochFrames = 2*halfFrames + 1;
numWords = numel(word_onsets);
epoched_features = nan(numWords,num_bands+1,length(taxis));
for i = 1:numWords % Convert onset time to spectrogram frame index
    [~, onsetFrame] = min(abs(t - word_onsets(i)));
    startFrame = onsetFrame - halfFrames;
    endFrame = onsetFrame + halfFrames; % Boundary handling
    padFront = []; padBack = [];
    if startFrame < 1
        padFront = zeros(num_bands + 1, 1 - startFrame);
        startFrame = 1;
    end
    if endFrame > size(features,2)
        padBack = zeros(num_bands + 1, endFrame - size(features,2));
        endFrame = size(features,2);
    end
    epoch = [padFront, features(:, startFrame:endFrame), padBack];
    epoched_features(i,:,:) = epoch; % each is [9 × epochFrames]
end
%%  audio analysis

%% select every other word!!
words  = words(1:step_size:end);
word_onsets = word_onsets(1:step_size:end);
embeddings = embeddings(1:step_size:end,:);
epoched_features = epoched_features(1:step_size:end,:,:);
%% pca embeddings down to 50/100;

%
% [coeff, zcore, latent] = pca(embeddings);
% reduced_array = score(:, 1:num_components);

% start at word 21 (we want 10 seconds of padding +)
n_word_padding = 20;
word_start_index = n_word_padding+1;

[unique_words, ~, word_ids] = unique(words);
emb_unique = embeddings(accumarray(word_ids, 1:numel(word_ids), [], @min), :);

[coeff, score_unique, latent, ~, explained, mu] = pca(emb_unique);
embeddings_centered = embeddings - mean(embeddings);
reduced_array = embeddings_centered * coeff(:, 1:num_components);

% %% try with random embeddings:
% random_unique_emb = randn(numel(unique_words), 50); % each row = random 50‑dim vector
% reduced_array = random_unique_emb(word_ids, :);


% these are the current words
emb_words = reduced_array(word_start_index:end-n_word_padding, :);
emb_words_circ = reduced_array(...
    word_start_index-1:end-1-n_word_padding, :);
% center:
emb_words = emb_words-mean(emb_words,1);
emb_words_circ = emb_words_circ-mean(emb_words_circ,1);

onsets = word_onsets(word_start_index:end-n_word_padding);
onsets_circ = word_onsets(word_start_index-1:end-1-n_word_padding);

epoched_audio = epoched_features(word_start_index:end-n_word_padding,:,:);
epoched_audio = epoched_audio-mean(epoched_audio);
n_steps = length(taxis);

audio_result.taxis = taxis;

audio_result.n_steps = length(taxis);
audio_result.n_words = size(emb_words, 1);
audio_result.Ws = nan(num_bands+1, audio_result.num_iterations , n_steps);
audio_result.Vs = nan(num_bands+1, audio_result.num_iterations , n_steps);
audio_result.Us = nan(num_bands+1, audio_result.num_iterations , n_steps);
%%
r1_par_cca = nan(audio_result.num_iterations, n_steps) ;
r1_canon = nan(audio_result.num_iterations, n_steps) ;
r1_crm   = nan(audio_result.num_iterations, n_steps) ;

r1_par_cca_control = nan(audio_result.num_iterations, n_steps) ;
r1_canon_control  = nan(audio_result.num_iterations, n_steps) ;
r1_crm_control    = nan(audio_result.num_iterations, n_steps) ;
[unique_words, ~, word_ids] = unique(words(word_start_index:end-n_word_padding, :));

probe_array = nan(size(epoched_audio,1),audio_result.num_iterations);
for ll = 1 : n_steps
    probe_ = false;
    if ll == 360; probe_ = true; end
    fprintf('at %d percent\r', round(100 * ll / n_steps));


    n = audio_result.n_words;
    K = audio_result.num_iterations;
    block_size = ceil(n / K);
    blocks = repelem(1:K, block_size);
    blocks = blocks(1:n);

    % groups = blocks * 10000 + word_ids';
    % 
    % n = audio_result.n_words; % total number of words 
    % K = 4; % number of temporal blocks 
    % % Assign each word to one of 4 contiguous blocks 
    % block_size = ceil(n / K); 
    % blocks = repelem(1:K, block_size); 
    % blocks = blocks(1:n); 
    % % trim to exact length % Use blocks as the grouping variable 
    % cv = cvpartition(blocks, 'KFold', K, 'Stratify', true);

    % cv = cvpartition(groups, 'KFold', audio_result.num_iterations, 'Stratify', false);

    % cv = cvpartition(audio_result.n_words, 'KFold', audio_result.num_iterations );

    for idx = 1:audio_result.num_iterations
        % Get train and test indices for the current iteration
        % midpoint = floor(size(emb_words, 1)/2);

        % if idx == 1
        % 
        %     train_indices{idx} =  1:midpoint;
        %     test_indices{idx} = midpoint+1:n;
        % else
        %     train_indices{idx} =  midpoint+1:n;
        %     test_indices{idx} = 1:midpoint;
        % end
        train_indices{idx} = blocks == idx;
        choices = 1:audio_result.num_iterations;
        choices(idx) = [];
        test_indices{idx} = blocks == choices(randi(length(choices)));

        % Split the data
        audio_train = epoched_audio(train_indices{idx}, :, ll);
        audio_test = epoched_audio(test_indices{idx}, :, ll);
        embeddings_train = emb_words(train_indices{idx}, :);
        embeddings_test = emb_words(test_indices{idx}, :);

        embeddings_train_ctrl = emb_words_circ(train_indices{idx}, :);
        embeddings_train_ctrl = embeddings_train_ctrl - mean(embeddings_train_ctrl);
        embeddings_test_ctrl = emb_words_circ(test_indices{idx}, :);
        embeddings_test_ctrl = embeddings_test_ctrl - mean(embeddings_test_ctrl);

        Cxx = audio_train' * audio_train;
        Cyy = embeddings_train' * embeddings_train;
        Cxy = audio_train' * embeddings_train;
        Dxy = audio_train' * embeddings_train_ctrl;

        Czy = embeddings_train_ctrl' * embeddings_train;
        Czz = embeddings_train_ctrl' * embeddings_train_ctrl;


        Dxy_par = -Dxy*inv(Czz)*Czy;


        [Wx, Wy, ~] = canoncorr(audio_train, embeddings_train);

        % [Wx, Wy, ~]  = crm(Cxx,Cyy,Cxy, zeros(size(Cxy)),1, 0.001);

        [Vx, Vy, lbd3i] = crm(Cxx,Cyy,Cxy, Dxy, gamma=0.001);

        [Ux, Uy, ~] = crm(Cxx,Cyy,Cxy, Dxy_par, gamma=0.001);


        r1_canon(idx, ll) = corr(audio_test*Wx(:,1), embeddings_test* Wy(:,1));
        r1_crm(idx, ll)    = corr(audio_test*Vx, embeddings_test* Vy);
        r1_par_cca(idx, ll) = corr(audio_test*Ux, embeddings_test* Uy);

        r1_canon_control(idx, ll)  = corr(audio_test*Wx(:,1), embeddings_test_ctrl* Wy(:,1));
        r1_crm_control(idx, ll)  = corr(audio_test*Vx, embeddings_test_ctrl* Vy);
        r1_par_cca_control(idx, ll) = corr(audio_test*Ux, embeddings_test_ctrl* Uy);

        if probe_
            probe_array(test_indices{idx},idx) = ...
                zscore(audio_test*Vx) .* zscore(embeddings_test* Vy);
        end

        Ws(:, idx, ll) = Wx(:,1);
        Vs(:, idx, ll) = Vx;
        Us(:, idx, ll) = Ux;


    end

end

%% results and plotting

% Calculate mean and standard error across subjects
mean_r1_canon = mean(r1_canon, 1);
std_r1_canon = std(r1_canon, [], 1) / sqrt(size(r1_canon, 1));

mean_r1_crm = mean(r1_crm, 1);
std_r1_crm = std(r1_crm, [], 1) / sqrt(size(r1_crm, 1));

mean_r1_par_cca = mean(r1_par_cca, 1);
std_r1_par_cca = std(r1_par_cca, [], 1) / sqrt(size(r1_par_cca, 1));

mean_r1_canon_control = mean(r1_canon_control, 1);
std_r1_canon_control = std(r1_canon_control, [], 1) / sqrt(size(r1_canon_control, 1));

mean_r1_crm_control = mean(r1_crm_control, 1);
std_r1_crm_control = std(r1_crm_control, [], 1) / sqrt(size(r1_crm_control, 1));

mean_r1_par_cca_control = mean(r1_par_cca_control, 1);
std_r1_par_cca_control = std(r1_par_cca_control, [], 1) / sqrt(size(r1_par_cca_control, 1));

%% Plot results with shaded error bars
figure;
hold on;
% Set background color to white
set(gcf, 'Color', 'w');

% Plot canonical correlation results
shadedErrorBar(taxis, mean_r1_canon, std_r1_canon, 'lineprops', {'-','Color','#1565C0 '}, 'patchSaturation', 0.1);
shadedErrorBar(taxis, mean_r1_crm, std_r1_crm, 'lineprops', {'-','Color','#C21807'}, 'patchSaturation', 0.1);
% shadedErrorBar(taxis, mean_r1_par_cca, std_r1_par_cca, 'lineprops', {'-','Color','#F4B400'}, 'patchSaturation', 0.1);

% % Plot control results #244F4B
shadedErrorBar(taxis, mean_r1_canon_control, std_r1_canon_control, 'lineprops', {'--','Color','#00897B '}, 'patchSaturation', 0.1);
shadedErrorBar(taxis, mean_r1_crm_control, std_r1_crm_control, 'lineprops', {'--','Color','#7B1FA2'}, 'patchSaturation', 0.1);
% shadedErrorBar(taxis, mean_r1_par_cca_control, std_r1_par_cca_control, 'lineprops', {'-','Color','#E65100'}, 'patchSaturation', 0.1);

% Set font size and type
set(gca, 'FontSize', 16, 'FontName', 'Helvetica');
legend({'CCA', 'CRM',  'CCA Control', 'CRM Control'});

% legend({'CCA', 'CRM', 'PAR CCA', 'CCA Control', 'CRM Control', 'PAR CCA Control'});
xlabel('Time around word onset');
ylabel('Correlation');
title('Contextual vs. context free LLM-alignment');
hold off;


%% find the words that contribute to high prestim correlation

plot(nanmean(probe_array, 2))

hist(nanmean(probe_array, 2), 100)
probe_vals = nanmean(probe_array, 2);
probe_words= words(word_start_index:end-n_word_padding);
probe_onsets = word_onsets(word_start_index:end-n_word_padding);
thresh = 6;
array_test = [ probe_words(find(probe_vals>thresh) -2), probe_words(find(probe_vals>thresh) -1),  probe_words(probe_vals>thresh), ...
    probe_words(find(probe_vals>thresh) +1), ...
    probe_words(find(probe_vals>thresh) +2), ];

cell2table(array_test)