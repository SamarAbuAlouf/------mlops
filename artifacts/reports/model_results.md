#  Final Model Results Summary
**Project:** Olist E-Commerce Delivery Delay Prediction (MLOps Task 2)
**Champion Model:** HistGradientBoostingClassifier (Balanced)
**Optimal Decision Threshold:** `0.50`

---
###  Evaluation on Untouched Test Set (Holdout 15%):
- **ROC-AUC:** `0.7226` (Significantly outperforms random chance of 0.50)
- **PR-AUC (Average Precision):** `0.1217` (Double the natural prevalence of the minority class)
- **F1-Score:** `0.1825`
- **Recall (Late Orders):** `29.68%` of all actual late orders were successfully captured.
- **Balanced Accuracy:** `57.91%`

---
###  Confusion Matrix:
- **True Negatives:** 11,642 on-time orders correctly classified.
- **True Positives:** 284 delayed orders successfully identified for early proactive intervention.
- **False Positives:** 1,872 on-time orders incorrectly flagged as delayed (false alarms).
- **False Negatives:** 673 delayed orders missed by the model.

---
###  Saved Artifacts:
- **Champion Model:** `artifacts/models/final_model.joblib`
- **Baseline Model:** `artifacts/models/baseline_model.joblib`
- **Fitted Preprocessor:** `artifacts/models/preprocessor.joblib`
- **Feature Names List:** `artifacts/models/feature_names.json`
