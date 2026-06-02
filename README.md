# Anomaly Detection - Stroke Risk

Cameron Ela, `cameronela13@gmail.com`

Compares supervised and unsupervised anomaly detection methods for stroke risk prediction on a class-imbalanced dataset. All pipelines are cross-validated on **F2 score** to weight recall twice as heavily as precision, important for a screening context where missing a stroke case is more costly than a false alarm.

---

## Dataset

[Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

4,422 records after filtering (adults, binary gender). 80/20 stratified train/test split:
- **Train**: 3,537 samples (198 stroke, 3,339 no-stroke)
- **Test**: 885 samples (49 stroke, 836 no-stroke)

---

## Pipeline

Every model shares the same preprocessing steps:

| Step | Description |
|------|-------------|
| `CleanData` | Encode/drop raw columns, dummy-encode `work_type` |
| `CategoricalMedianBMIImputer` | Impute missing BMI as group median by categorical combination |
| `SkippableUnderSampler` | Optional random under-sampling (10/90 or 15/85 ratio); disabled for unsupervised models |
| `FeatureEngineer` | Mean-center numerics; add `age²`, `log_glucose`, `bmi_underweight`, and senior interaction terms |
| `FeatureSelector` | Three passes: rare-category pruning (10%), high-correlation pruning (0.4), feature-to-sample ratio cap (0.05) |
| `ImportancePruner` | Drop features with logistic regression coefficient magnitude below 1e-4 (supervised models only) |
| `StandardScaler` | Z-score normalization |

All hyperparameters including under-sampling ratio are tuned jointly via 5-fold stratified cross-validation.

---

## Models

### Supervised

`LogisticRegression`
| penalty: L1/L2/ElasticNet; solvers: lbfgs, liblinear, saga; class_weight='balanced'

`XGBClassifier`
| booster: DART; `scale_pos_weight` balances classes; tunes depth, eta, gamma, regularization

`SVC`
| kernel: linear/poly/RBF; class_weight='balanced'; probability=True for Platt scaling |

### Unsupervised (anomaly detection specific models)

Custom 5-fold CV fits on stroke-negative fold samples (or all fold samples for Isolation Forest) and scores F2 on the full validation fold.

`OneClassSVM`
| Trained on stroke-negative samples; linear/poly/RBF kernels

`IsolationForest`
| Trained on all samples; tunes contamination, n_estimators, max_samples, max_features

`Autoencoder`
| PyTorch encoder-decoder; cross-validates input dimension, latent dimension, learning rate, and contamination threshold; less conservative feature selection to give the network more input signal

---

## Evaluation

Models are compared on the held-out test set using accuracy, F2 score, precision, recall, confusion matrices, ROC curves (AUC), and precision-recall curves. Anomaly model scores are derived from the negated decision function so that higher values correspond to greater stroke likelihood, keeping the ROC/PR convention consistent across all six models.

---

## Selected Model: XGBClassifier

XGBoost is the recommended model for this task. Among the six models evaluated, it achieves the highest recall on the test set while remaining interpretable as a tree model. The DART booster with `scale_pos_weight` handles the severe class imbalance directly, and joint CV over under-sampling ratio and regularization parameters ensures the model generalizes rather than overfitting to the resampled training folds.

Key advantages over alternatives:
- **vs. Logistic Regression**: captures non-linear feature interactions without requiring engineered polynomial terms, heuristics makes it potentially easier to interpret than Logistic Regression
- **vs. SVC**: produces calibrated probability scores and directly interpretable feature importances
- **vs. anomaly detection models**: likely benefits from stroke labels during training, giving it a decisive recall advantage

---

## Project Structure

```
stroke_prediction.ipynb
healthcare-dataset-stroke-data.csv
logistic_regression.joblib
xgbclassifier.joblib
svc.joblib
oneclasssvm.joblib
isolationforest.joblib
autoencoder.joblib
requirements.txt
```

---

## Citations

- American Heart Association — blood glucose management
- NIH — bimodality in blood glucose distribution (PMID 12453963)
- Cleveland Clinic — BMI classification
- Medline — blood glucose reference ranges
- NIH — demographic disparities in stroke occurrence (PMC12321967)
