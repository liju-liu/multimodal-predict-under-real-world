Enhancing Prediction of Individualized Antipsychotic Outcome with fMRI-EEG Feature Integration: A Real-World Study
This repository provides data and code for data analysis in the article “Enhancing Prediction
of Individualized Antipsychotic Outcome with fMRI-EEG Feature Integration in First-Episode Schizophrenia: A Real-World Study”.
Overview:
All code and processed data to reproduce our results have been made publicly available at  https://github.com/liju-liu/multimodal-predict-under-real-world.



Code:
The code folder contains all the code used to run the analyses and generate the figures. This folder contains the following files:
1.run\_nested\_svr\_v3:Performs nested cross-validation with SVR for multimodal and single modal prediction.
2.run\_main:Main script to execute the complete prediction pipeline.
3.PLS\_v3:Partial least squares analysis to relate brain features with PANSS symptom reduction.



Data:
The data folder contains all the data used to run the analyses. All feature data (i.e., X\_WPEEEG\_zscore, X\_WPE\_zscore,
X\_EEG\_zscore) have been adjusted for sex, age, and illness duration, and subsequently z-scored. This folder contains the following files:
1.X\_WPEEEG\_zscore:Combined fMRI (WPE) and EEG features (z-scored).
2.X\_WPE\_zscore:fMRI complexity features only.
3.X\_EEG\_zscore:EEG features only.
4.Y\_PANSS\_T: PANSS total score changes.
5.Y\_PANSS\_N:PANSS negative subscale changes.
6.Y\_PANSS\_G:PANSS general subscale changes.
7.feature\_names\_AALEEG:Names of multimodal features.
8.feature\_names\_AAL:Names of fMRI (AAL atlas) features.
9.feature\_names\_EEG:Names of EEG features.

