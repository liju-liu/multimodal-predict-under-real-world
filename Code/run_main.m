clc; 
clear;


% Set random seed for reproducibility
Rand_seed = 189; %random seed;

% Define parameter search ranges for SVR (Support Vector Regression)
C_range = logspace(-4, 4, 10);       % C values: 10 values from 10^-4 to 10^4(PANSS-T and PANSS-G),7 values from 10^-3 to 10^3(PANSS-N)
gamma_range = logspace(-4, 4, 10);   % gamma values: 10 values from 10^-4 to 10^4(PANSS-T and PANSS-G),7 values from 10^-3 to 10^3(PANSS-N)

%% === Multimodal model: WPE + EEG ===
% Load z-scored feature matrix (combined WPE and EEG features)
X_WPEEEG_zscore = importdata('H:\multimodal predict under real world\Data\X_WPEEEG_zscore.mat'); 

% Load target variable: change in PANSS total score
Y_PANSS_T = importdata('H:\multimodal predict under real world\Data\Y_PANSS_T.mat');

% Load feature names for the combined WPE and EEG features
feature_names_WPEEEG = importdata('H:\multimodal predict under real world\Data\feature_names_AALEEG.mat');

% Define save path for results
save_path = 'H:\WPEandEEG_predict\result\WPEEEG';

% Run nested SVR with specified parameters and save results
res_WPEEEG = run_nested_svr_v3(X_WPEEEG_zscore, Y_PANSS_T, feature_names_WPEEEG, ...
    Rand_seed, C_range, gamma_range, save_path, 'doplot', true);


%% === Unimodal model: WPE only ===
% Load z-scored WPE-only feature matrix
X_WPE_zscore = importdata('H:\multimodal predict under real world\Data\X_WPE_zscore.mat'); 

% Load feature names for WPE features
feature_names_WPE = importdata('H:\multimodal predict under real world\Data\feature_names_AAL.mat');

% Define save path for results
save_path = 'H:\WPEandEEG_predict\result\WPE';

% Run nested SVR using only WPE features
res_WPE = run_nested_svr_v3(X_WPE_zscore, Y_PANSS_T, feature_names_WPE, ...
    Rand_seed, C_range, gamma_range, save_path, 'doplot', true);


%% === Unimodal model: EEG only ===
% Load z-scored EEG-only feature matrix
X_EEG_zscore = importdata('H:\multimodal predict under real world\Data\X_EEG_zscore.mat'); 

% Load feature names for EEG features
feature_names_EEG = importdata('H:\multimodal predict under real world\Data\feature_names_EEG.mat');

% Define save path for results
save_path = 'H:\WPEandEEG_predict\result\EEG';

% Run nested SVR using only EEG features
res_EEG = run_nested_svr_v3(X_EEG_zscore, Y_PANSS_T, feature_names_EEG, ...
    Rand_seed, C_range, gamma_range, save_path, 'doplot', true);
