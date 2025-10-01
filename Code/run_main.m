clc; 
clear;


% Set random seed for reproducibility
Rand_seed = 189; %

% Define parameter search ranges for SVR (Support Vector Regression)
C_range = logspace(-4, 4, 10);       % C values: 10 values from 10^-4 to 10^4(PANSS-T and PANSS-G),7 values from 10^-3 to 10^3(PANSS-N)
gamma_range = logspace(-4, 4, 10);   % gamma values: 10 values from 10^-4 to 10^4(PANSS-T and PANSS-G),7 values from 10^-3 to 10^3(PANSS-N)

%% === Multimodal model: WPE + EEG ===
% Load feature matrix (combined WPE and EEG features)
X_WPEEEG_zscore = importdata('I:\sz_preprocessed\fMRI\pre_treatment\WPE_EEG_feature_NEW.mat'); 

% Load target variable: change in PANSS total score
Y_PANSS_T = importdata('I:\sz_preprocessed\fMRI\pre_treatment\Y_PANSS_T.mat');
% Load covariates
covariates = importdata('I:\sz_preprocessed\fMRI\pre_treatment\covariance.mat'); 

% Load feature names for the combined WPE and EEG features
feature_names_WPEEEG = importdata('I:\sz_preprocessed\fMRI\pre_treatment\feature_names_AALEEG.mat');

% Define save path for results
save_path = 'I:\multimodal-predict-under-real-world\Result\PANSS-T\WPEEEG2';

% Run nested SVR with specified parameters and save results
res_WPEEEG = run_nested_svr_v3(X_WPEEEG_zscore, Y_PANSS_T, feature_names_WPEEEG, ...
   covariates, Rand_seed, C_range, gamma_range, save_path, 'doplot', true);

%% === Unimodal model: WPE only ===
% Load z-scored WPE-only feature matrix
X_WPE_zscore = importdata('I:\sz_preprocessed\fMRI\pre_treatment\WPE_feature_90.mat'); 

% Load feature names for WPE features
feature_names_WPE = importdata('I:\sz_preprocessed\fMRI\pre_treatment\feature_names_AAL.mat');

% Define save path for results
save_path = 'I:\multimodal-predict-under-real-world\Result\PANSS-T\WPE2';

% Run nested SVR using only WPE features
res_WPE = run_nested_svr_v3(X_WPE_zscore, Y_PANSS_T, feature_names_WPE, ...
    covariates,Rand_seed, C_range, gamma_range, save_path, 'doplot', true);


%% === Unimodal model: EEG only ===
% Load z-scored EEG-only feature matrix
X_EEG_zscore = importdata('I:\sz_preprocessed\fMRI\pre_treatment\EEG_feature_NEW.mat'); 

% Load feature names for EEG features
feature_names_EEG = importdata('I:\sz_preprocessed\fMRI\pre_treatment\feature_names_EEG.mat');

% Define save path for results
save_path = 'I:\multimodal-predict-under-real-world\Result\PANSS-T\EEG2';

% Run nested SVR using only EEG features
res_EEG = run_nested_svr_v3(X_EEG_zscore, Y_PANSS_T, feature_names_EEG, ...
    covariates,Rand_seed, C_range, gamma_range, save_path, 'doplot', true);
