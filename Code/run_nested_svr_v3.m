function results = run_nested_svr_v2(X, Y, feature_names, Random_seed, C_range, gamma_range, save_path, varargin)
% Function: Nested cross-validation + SVR + optional permutation test + plotting
% Inputs:
%   - X: feature matrix (nSamples x nFeatures)
%   - Y: target variable (nSamples x 1)
%   - feature_names: names of features (cell array)
%   - Random_seed: seed for reproducibility
%   - C_range, gamma_range: hyperparameter ranges for SVR
%   - save_path: path to save figures and results
% Optional name-value arguments:
%   - 'nPerm': number of permutations (default = 0)
%   - 'doPlot': whether to generate plots (default = true)
%   - 'outerFold': number of outer folds in nested CV (default = 10)

%% === Parse optional inputs ===
% Set up parser and extract optional parameters
p = inputParser;
addParameter(p, 'nPerm', 0);
addParameter(p, 'doPlot', true);
addParameter(p, 'outerFold', 10);
parse(p, varargin{:});
nPerm = p.Results.nPerm;
doPlot = p.Results.doPlot;
nFolds_outer = p.Results.outerFold;
nFolds_inner = 5;  % Fixed number of inner folds
rng(Random_seed);  % Set random seed

% Create directory for saving results if it doesn't exist
save_dir = save_path;
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

[nSamples, nFeatures] = size(X);
feature_names = feature_names(:);  % Ensure column format
%% === Step 1: Covariate Regression (Auto-detect single/dual modality) ===
% covariates = zscore(covariates);  % Z-score normalize covariates
% X_cov = [ones(size(covariates,1),1), covariates];  % Add intercept term to design matrix
% 
% % Automatically detect dual modality based on feature count
% is_dual_modality = false;
% if size(X,2) == 105  % Assuming first 90 are WPE, last 15 are EEG
%     X_WPE = X(:,1:90);
%     X_EEG = X(:,91:end);
%     is_dual_modality = true;
% elseif size(X,2) == 90  % Only WPE features
%     X_WPE = X;
%     X_EEG = [];
% elseif size(X,2) == 15  % Only EEG features
%     X_WPE = [];
%     X_EEG = X;
% else
%     % Unknown structure – apply covariate regression to the whole matrix
%     warning('Unknown feature dimension structure, performing global covariate regression.');
%     for i = 1:size(X,2)
%         b = X_cov \ X(:,i);  % Regress out covariates
%         X(:,i) = X(:,i) - X_cov * b;
%     end
%     X = zscore(X);  % Normalize after regression
% end
% 
% % Perform separate covariate regression for each modality if present
% if is_dual_modality || ~isempty(X_WPE) || ~isempty(X_EEG)
%     if ~isempty(X_WPE)
%         for i = 1:size(X_WPE,2)
%             b = X_cov \ X_WPE(:,i);
%             X_WPE(:,i) = X_WPE(:,i) - X_cov * b;
%         end
%         X_WPE = zscore(X_WPE);
%     end
% 
%     if ~isempty(X_EEG)
%         for i = 1:size(X_EEG,2)
%             b = X_cov \ X_EEG(:,i);
%             X_EEG(:,i) = X_EEG(:,i) - X_cov * b;
%         end
%         X_EEG = zscore(X_EEG);
%     end
% 
%     % Concatenate the two modalities after preprocessing
%     X = [X_WPE, X_EEG];
% end
%% === Step 2: Nested CV + SVR model training ===
% Outer CV to estimate generalization performance
outerCV = cvpartition(nSamples, 'KFold', nFolds_outer);
y_pred_all = zeros(nSamples, 1);                 % Store all predictions
feature_weights_sum = zeros(nFeatures, 1);       % Accumulate feature sensitivities
mse_all = zeros(nFolds_outer, 1);                % Store MSE per fold
r_all = zeros(nFolds_outer, 1);                  % Store correlation per fold

for foldIdx = 1:nFolds_outer
    % Split training and test sets
    trainIdx = outerCV.training(foldIdx);
    testIdx = outerCV.test(foldIdx);
    X_train = X(trainIdx,:);
    Y_train = Y(trainIdx);
    X_test = X(testIdx,:);

    % --- Inner CV to select best hyperparameters ---
    best_mse = inf;
    for C = C_range
        for gamma = gamma_range
            mse_cv = 0;
            innerCV = cvpartition(length(Y_train), 'KFold', nFolds_inner);
            for innerFold = 1:nFolds_inner
                tr = innerCV.training(innerFold);
                val = innerCV.test(innerFold);
                model = fitrsvm(X_train(tr,:), Y_train(tr), ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, ...
                    'KernelScale', gamma, ...
                    'Standardize', false);
                y_pred = predict(model, X_train(val,:));
                mse_cv = mse_cv + mean((y_pred - Y_train(val)).^2);
            end
            mse_cv = mse_cv / nFolds_inner;
            if mse_cv < best_mse
                best_mse = mse_cv;
                best_C = C;
                best_gamma = gamma;
            end
        end
    end

    % --- Train final model on full training set with best params ---
    model = fitrsvm(X_train, Y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', best_C, ...
        'KernelScale', best_gamma, ...
        'Standardize', false);
    y_pred = predict(model, X_test);
    y_pred_all(testIdx) = y_pred;
    mse_all(foldIdx) = mean((y_pred - Y(testIdx)).^2);
    r_all(foldIdx) = corr(y_pred, Y(testIdx));

    % --- Estimate feature importance using sensitivity method ---
    delta = 1e-4;
    feature_weights = zeros(nFeatures,1);
    for f = 1:nFeatures
        X_test_plus = X_test; X_test_plus(:,f) = X_test(:,f) + delta;
        X_test_minus = X_test; X_test_minus(:,f) = X_test(:,f) - delta;
        y_plus = predict(model, X_test_plus);
        y_minus = predict(model, X_test_minus);
        feature_weights(f) = mean(abs(y_plus - y_minus));
    end
    feature_weights_sum = feature_weights_sum + feature_weights;
end

%% === Step 3: Report main results ===
r_final = corr(y_pred_all, Y);                            % Overall correlation
mse_final = mean((Y - y_pred_all).^2);                    % Overall MSE
fprintf('\nNested CV Performance: r = %.3f, MSE = %.3f\n', r_final, mse_final);

%% === Step 4: Permutation test (optional) ===
% Permute target Y and repeat nested CV to get null distribution
p_value = NaN;
if nPerm > 0
    fprintf('Running strict permutation test with %d permutations...\n', nPerm);
    r_null = zeros(nPerm, 1);
    for i = 1:nPerm
        fprintf('Permutation %d / %d\n', i, nPerm);
        Y_perm = Y(randperm(nSamples));
        y_perm_all = zeros(nSamples,1);

        for foldIdx = 1:nFolds_outer
            trainIdx = outerCV.training(foldIdx);
            testIdx = outerCV.test(foldIdx);
            X_train = X(trainIdx,:);
            Y_train_perm = Y_perm(trainIdx);
            X_test = X(testIdx,:);

            % Re-tune hyperparameters on permuted labels
            innerCV = cvpartition(length(Y_train_perm), 'KFold', nFolds_inner);
            best_mse = inf;
            for C = C_range
                for gamma = gamma_range
                    mse_inner_total = 0;
                    for innerFold = 1:nFolds_inner
                        innerTrainIdx = innerCV.training(innerFold);
                        innerValIdx = innerCV.test(innerFold);
                        model = fitrsvm(X_train(innerTrainIdx,:), Y_train_perm(innerTrainIdx), ...
                            'KernelFunction', 'rbf', ...
                            'BoxConstraint', C, ...
                            'KernelScale', gamma, ...
                            'Standardize', false);
                        Y_val_pred = predict(model, X_train(innerValIdx,:));
                        mse_inner_total = mse_inner_total + mean((Y_val_pred - Y_train_perm(innerValIdx)).^2);
                    end
                    avg_mse = mse_inner_total / nFolds_inner;
                    if avg_mse < best_mse
                        best_mse = avg_mse;
                        best_C_perm = C;
                        best_gamma_perm = gamma;
                    end
                end
            end

            % Train permuted model and predict
            model_perm = fitrsvm(X_train, Y_train_perm, ...
                'KernelFunction', 'rbf', ...
                'BoxConstraint', best_C_perm, ...
                'KernelScale', best_gamma_perm, ...
                'Standardize', false);
            y_perm_all(testIdx) = predict(model_perm, X_test);
        end

        r_null(i) = corr(y_perm_all, Y);  % Compare with original Y
    end

    % Compute two-tailed p-value
    p_value = mean(abs(r_null) >= abs(r_final));
    fprintf('Permutation test (strict) p = %.4f\n', p_value);

    % Plot null distribution
    if doPlot
        fig3 = figure('Color', 'w', 'Position', [100, 100, 150, 120]);
        histogram(r_null, 'Normalization','probability','FaceColor',[0.7 0.7 0.7]);
        xline(r_final, 'r--', 'LineWidth', 2);
        title(sprintf('Permutation Null (strict, p = %.4f)', p_value));
        xlabel('Permutation R'); ylabel('Probability');
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'LineWidth', 0.8); box on;
    end
end

%% === Step 5: Visualization ===
if doPlot
    % --- Scatter plot of true vs. predicted ---
    mdl = fitlm(Y, y_pred_all);             % Fit linear model for plotting
    xq = linspace(min(Y), max(Y), 100)';
    [yhat, yCI] = predict(mdl, xq);         % Predict CI for regression line

    fig1 = figure('Color', 'w','Position', [100, 100, 400, 350]);
    hold on;
    scatter(Y, y_pred_all, 60, [0 0.36 0.62], 'filled', 'MarkerFaceAlpha', 0.7);
    fill([xq; flipud(xq)], [yCI(:,1); flipud(yCI(:,2))], ...
        [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    plot(xq, yhat, 'k-', 'LineWidth', 2);
    xlabel('True PANSS-N Reduction');
    ylabel('Predicted PANSS-N Reduction');
    title(sprintf('r = %.2f, p = %.4f', r_final, p_value));
    set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'LineWidth', 0.8, 'FontSize', 11);
    axis tight; box on;

    % --- Bar plot of top N feature importances ---
    mean_weights = feature_weights_sum / nFolds_outer;
    [~, sorted_idx] = sort(mean_weights, 'descend');
    topN = min(20, nFeatures);
    top_idx = sorted_idx(1:topN);
    top_weights = mean_weights(top_idx);
    top_names = feature_names(top_idx);

    % Set colors (use ColorBrewer if available)
    try
        colors = brewermap(topN, 'YlGnBu');
    catch
        colors = repmat([0.2 0.4 0.6], topN, 1);
    end

    fig2 = figure('Color', 'w', 'Position', [100, 100, 500, 350]);
    b = bar(top_weights, 'FaceColor', 'flat');
    b.CData = colors;
    set(gca, 'XTick', 1:topN, 'XTickLabel', top_names, ...
        'XTickLabelRotation', 45, ...
        'FontName', 'Arial', 'FontWeight', 'bold', ...
        'FontSize', 10, 'LineWidth', 0.8, ...
        'TickLabelInterpreter', 'none');
    xlabel('Feature Names');
    ylabel('Average Importance (Sensitivity)');
    title('Top 20 Most Important Features from SVR');
    box on;
end

%% === Step 6: Output results ===
results.r = r_final;
results.mse = mse_final;
results.r_all = r_all;
results.mse_all = mse_all;
results.p_value = p_value;
results.y_true = Y;
results.y_pred = y_pred_all;
results.feature_weights = feature_weights_sum / nFolds_outer;
results.feature_names = feature_names;

%% === Step 7: Save plots ===
if doPlot
    print(fig1, fullfile(save_dir, 'scatter.pdf'), '-dpdf', '-painters');
    print(fig2, fullfile(save_dir, 'feature_weights.pdf'), '-dpdf', '-painters');
end
if doPlot && nPerm > 0
    print(fig3, fullfile(save_dir, 'null_permutation.pdf'), '-dpdf', '-painters');
end
end
