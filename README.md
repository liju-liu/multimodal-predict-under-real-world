Enhancing Prediction of Individualized Antipsychotic Outcome with fMRI-EEG Feature Integration: A Real-World Study
This repository provides data and code for data analysis in the article “Enhancing Prediction 
of Individualized Antipsychotic Outcome with fMRI-EEG Feature Integration in First-Episode Schizophrenia: A Real-World Study”.
Overview:
All code and processed data to reproduce our results have been made publicly available at  https://github.com/Liu.

Code:
The code folder contains all the code used to run the analyses and generate the figures. This folder contains the following files:
1.run_nested_svr_v3:Performs nested cross-validation with SVR for multimodal and single modal prediction.
2.run_main:Main script to execute the complete prediction pipeline.
3.PLS_v3:Partial least squares analysis to relate brain features with PANSS symptom reduction.

Data:
The data folder contains all the data used to run the analyses. All feature data (i.e., X_WPEEEG_zscore, X_WPE_zscore, 
X_EEG_zscore) have been adjusted for sex, age, and illness duration, and subsequently z-scored. This folder contains the following files:
1.X_WPEEEG_zscore:Combined fMRI (WPE) and EEG features (z-scored).
2.X_WPE_zscore:fMRI complexity features only.
3.X_EEG_zscore:EEG features only.
4.Y_PANSS_T: PANSS total score changes.
5.Y_PANSS_N:PANSS negative subscale changes.
6.Y_PANSS_G:PANSS general subscale changes.
7.feature_names_AALEEG:Names of multimodal features.
8.feature_names_AAL:Names of fMRI (AAL atlas) features.
9.feature_names_EEG:Names of EEG features.

