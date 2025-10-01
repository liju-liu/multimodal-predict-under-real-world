clc; 
clear;

%% === Check for Required Function: brewermap ===
if ~exist('brewermap', 'file')
    error('Please install the brewermap function (search in File Exchange or Add-On Explorer).');
end

%% === Load Input Data and Define Save Path ===
% EEG: 78 × 15, WPE: 78 × 90, Y: 78 × 1
X_EEG = importdata('I:\sz_preprocessed\fMRI\pre_treatment\X_EEG_zscore.mat');
X_WPE = importdata('I:\sz_preprocessed\fMRI\pre_treatment\X_WPE_zscore.mat');
Y = importdata('I:\sz_preprocessed\fMRI\pre_treatment\Y_PANSS_G.mat');

EEG_names = importdata('I:\sz_preprocessed\fMRI\pre_treatment\feature_names_EEG.mat');
WPE_names = importdata('I:\sz_preprocessed\fMRI\pre_treatment\feature_names_AAL.mat');

% Create save directory if it doesn't exist
save_dir = 'I:\multimodal-predict-under-real-world\Result\PLS\PANSS-G\';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% === Standardize Data and Concatenate Modalities ===
% EEG and WPE features are already z-scored if necessary
X_EEG = zscore(X_EEG);
X_WPE = zscore(X_WPE);
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
observed_explY = PCTVAR(2, best_component_idx);
%% === Permutation Test on Variance Explained by Best Component ===
rng(52);
nPerm = 10000;
null_pctvar = zeros(nPerm,1);

% Generate null distribution by shuffling Y
for i = 1:nPerm
    Y_perm = Y(randperm(length(Y)));
    [~, ~, ~, ~, ~, PCTVAR_perm] = plsregress(X_all, Y_perm, nComponents);
    null_pctvar(i) = PCTVAR_perm(2,best_component_idx);  % Max variance explained by permuted Y
end

% Two-tailed p-value
p_expalvariance = mean(null_pctvar > observed_explY);

%% === Bootstrap Resampling with Component Matching & Sign Alignment ===
nBootstrap = 1000;
rng(1);
[nSample, nFeature] = size(X_all);
boot_weights = nan(nFeature, nBootstrap);

fprintf('Running %d bootstrap iterations...\n', nBootstrap);

for i = 1:nBootstrap
    idx = ceil(rand(nSample,1) * nSample);    % 有放回抽样
    X_boot = X_all(idx, :);
    Y_boot = Y(idx);

    try
        [XL_b, YL_b, XS_b, YS_b, BETA_b, PCTVAR_b, MSE_b, stats_b] = ...
            plsregress(X_boot, Y_boot, nComponents);

        % === 成分匹配：找与原始 best_XS 最相似的成分 ===
        corrs = zeros(1, nComponents);
        for k = 1:nComponents
            c = corr(XS_b(:,k), best_XS(idx), 'rows','complete');
            if isnan(c); c = 0; end
            corrs(k) = c;
        end
        [~, k_star] = max(abs(corrs));
        sgn = sign(corrs(k_star)); if sgn == 0, sgn = 1; end

        boot_weights(:, i) = stats_b.W(:, k_star) * sgn;

    catch
        warning('Bootstrap iteration %d failed. Skipping.', i);
    end
end

% 去掉失败列
valid_idx = all(~isnan(boot_weights),1);
boot_weights = boot_weights(:, valid_idx);

%% === Compute Bootstrap Ratio (BSR) ===
std_weights = std(boot_weights, 0, 2);
std_weights(std_weights==0) = eps;   % 防除零
boot_z = best_weights ./ std_weights;   % Bootstrap Ratio

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
