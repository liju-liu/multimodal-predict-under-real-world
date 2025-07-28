clc; 
clear;

%% === Check for Required Function: brewermap ===
if ~exist('brewermap', 'file')
    error('Please install the brewermap function (search in File Exchange or Add-On Explorer).');
end

%% === Load Input Data and Define Save Path ===
% EEG: 78 × 15, WPE: 78 × 90, Y: 78 × 1
X_EEG = importdata('H:\multimodal predict under real world\Data\X_EEG_zscore.mat');
X_WPE = importdata('H:\multimodal predict under real world\Data\X_WPE_zscore.mat');
Y = importdata('H:\multimodal predict under real world\Data\Y_PANSS_N.mat');

EEG_names = importdata('H:\multimodal predict under real world\Data\feature_names_EEG.mat');
WPE_names = importdata('H:\multimodal predict under real world\Data\feature_names_AAL.mat');

% Create save directory if it doesn't exist
save_dir = 'H:\sz_preprocessed\fMRI\result\picture\PLSNEW\PANSS-T\';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% === Standardize Data and Concatenate Modalities ===
% EEG and WPE features are already z-scored if necessary
X_all = [X_EEG, X_WPE];  % Final feature matrix: 78 samples × 105 features
feature_names = [EEG_names, WPE_names];
feature_names = feature_names(:);  % Ensure it's a column vector

%% === Run PLS Regression ===
nComponents = 10;  % Number of PLS components to compute
[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X_all, Y, nComponents);

%% === Select Best Component Based on Variance Explained in Y ===
[~, best_component_idx] = max(PCTVAR(2,:));
best_XS = XS(:, best_component_idx);  % Scores for best component
best_weights = stats.W(:, best_component_idx);  % Corresponding feature weights

%% === Permutation Test on Variance Explained by Best Component ===
rng(1);
nPerm = 10000;
null_pctvar = zeros(nPerm,1);

% Generate null distribution by shuffling Y
for i = 1:nPerm
    Y_perm = Y(randperm(length(Y)));
    [~, ~, ~, ~, ~, PCTVAR_perm] = plsregress(X_all, Y_perm, nComponents);
    null_pctvar(i) = max(PCTVAR_perm(2,:));  % Max variance explained by permuted Y
end

% Two-tailed p-value
p_expalvariance = mean(null_pctvar >= best_component_idx);

%% === Bootstrap Resampling to Estimate Stability of Weights ===
nBootstrap = 1000;
rng(1);  % For reproducibility
[nSample, nFeature] = size(X_all);
boot_weights = zeros(nFeature, nBootstrap);

fprintf('Running %d bootstrap iterations...\n', nBootstrap);

for i = 1:nBootstrap
    idx = ceil(rand(length(Y),1) * length(Y));  % Sample with replacement
    X_boot = X_all(idx, :);
    Y_boot = Y(idx);
    
    try
        [~, ~, ~, ~, ~, ~, ~, stats_boot] = plsregress(X_boot, Y_boot, nComponents);
        boot_weights(:, i) = stats_boot.W(:, best_component_idx);
    catch
        warning('Bootstrap iteration %d failed. Skipping.', i);
        boot_weights(:, i) = NaN;
    end
end

% Remove failed bootstrap iterations
valid_boot_idx = all(~isnan(boot_weights), 1);
boot_weights = boot_weights(:, valid_boot_idx);

% Compute bootstrap z-scores
mean_weights = mean(boot_weights, 2);
std_weights = std(boot_weights, 0, 2);
boot_z = mean_weights ./ std_weights;

%% === Plot Top 20 Features by Bootstrap Z-score (Separated by Sign) ===
[~, sorted_idx_abs] = sort(abs(boot_z), 'descend');
top_idx_20 = sorted_idx_abs(1:20);

top_z_20 = boot_z(top_idx_20);
top_names_20 = feature_names(top_idx_20);

% Split into positive and negative
pos_mask = top_z_20 > 0;
neg_mask = top_z_20 < 0;

neg_idx = flip(find(neg_mask));
pos_idx = find(pos_mask);

% Reorder for plotting (negative on top, positive on bottom)
top_idx_final = [neg_idx; pos_idx];
top_weights_final = top_z_20(top_idx_final);
top_names_final = top_names_20(top_idx_final);

% Color gradients
n_neg = sum(neg_mask);
n_pos = sum(pos_mask);
cmap_neg = brewermap(n_neg, 'GnBu');
cmap_pos = flipud(brewermap(n_pos, 'YlOrRd'));
bar_colors_final = [cmap_neg; cmap_pos];

% Create bar plot
fig2_boot_v2 = figure('Color', 'w', 'Position', [100, 100, 400, 600]);
b = barh(1:length(top_weights_final), top_weights_final, ...
    'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.5);
b.CData = bar_colors_final;

set(gca, 'YTick', 1:length(top_weights_final), ...
         'YTickLabel', top_names_final, ...
         'YDir', 'reverse', ...
         'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 10, ...
         'LineWidth', 0.8, 'TickLabelInterpreter', 'none');
xlabel('Bootstrap Weight Z-score', 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Feature Name', 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 11);
title('Top 20 Features by Bootstrap Z-score', 'FontName', 'Arial', 'FontWeight', 'bold');
box on; grid off;

%% === Figure 1: Variance Explained by PLS Components ===
fig1 = figure('Color', 'w', 'Position', [100, 100, 400, 300]);
bar(PCTVAR(2,:)*100, 'FaceColor', [0 0.36 0.62]);
xlabel('PLS Component','FontSize', 11);
ylabel('Variance Explained (%)','FontSize', 11);
title('Variance Explained in Y by PLS Components');
grid off;
set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', ...
    'TickLabelInterpreter', 'none', 'LineWidth', 0.8, 'FontSize', 10);

%% === Figure 3: Scatter Plot of PLS Score vs Y with CI and Permutation p ===
nPerm = 10000;
rng(1);
observed_r = corr(best_XS, Y);
null_r = zeros(nPerm,1);
for i = 1:nPerm
    Y_perm = Y(randperm(length(Y)));
    null_r(i) = corr(best_XS, Y_perm);
end
p_value = mean(abs(null_r) >= abs(observed_r));

% Fit linear model
mdl = fitlm(best_XS, Y);

% Predict Y and 95% CI
xq = linspace(min(best_XS), max(best_XS), 100)';
[yhat, yCI] = predict(mdl, xq);

% Plot scatter and regression line
fig3 = figure('Color', 'w', 'Position', [100, 100, 400, 350]);
scatter(best_XS, Y, 60, 'filled', 'MarkerFaceColor', [0 0.36 0.62]); hold on;
plot(xq, yhat, 'k-', 'LineWidth', 2);
fill([xq; flipud(xq)], [yCI(:,1); flipud(yCI(:,2))], ...
     [0.45 0.45 0.45], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

xlabel(sprintf('PLS Component %d Score', best_component_idx), 'FontName', 'Arial', 'FontWeight', 'bold','FontSize', 11);
ylabel('PANSS Reduction', 'FontName', 'Arial', 'FontWeight', 'bold','FontSize', 11);
title(sprintf('r = %.2f, p = %.4f', observed_r, p_value), 'FontName', 'Arial', 'FontWeight', 'bold');
set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'LineWidth', 0.8,'FontSize', 9);
box on; grid off;

%% === Figure 4: Null Distribution of Permutation r ===
fig4 = figure('Color', 'w', 'Position', [100, 100, 150, 120]);
histogram(null_r, 'Normalization','probability','FaceColor',[0.7 0.7 0.7]);
xline(observed_r, 'r--', 'LineWidth', 2);
xlabel('Permutation r', 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('Probability', 'FontName', 'Arial', 'FontWeight', 'bold');
title(sprintf('Null Distribution (Component %d)', best_component_idx), 'FontName', 'Arial', 'FontWeight', 'bold');
set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'LineWidth', 0.8);
box on; grid off;

%% === Save Figures as PDF ===
print(fig1, fullfile(save_dir, 'PLS1_variance.pdf'), '-dpdf', '-painters');
print(fig2_boot_v2, fullfile(save_dir, 'Bootstrap_Top20_PLS_weights.pdf'), '-dpdf', '-painters');
print(fig3, fullfile(save_dir, 'PLS1_scatter.pdf'), '-dpdf', '-painters');
print(fig4, fullfile(save_dir, 'PLS1_permutation.pdf'), '-dpdf', '-painters');
