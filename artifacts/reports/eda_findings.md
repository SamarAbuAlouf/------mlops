#  Exploratory Data Analysis (EDA) Findings Summary
**Project:** Olist Delivery Late Prediction (MLOps Task 2)
**Population Analyzed:** Training Split Only (67,529 orders)
---
###  Key Statistical Insights:

1. **Class Imbalance:**
   - Late delivery rate in the training set: **9.03%** vs. **90.97%** on-time deliveries.
   - Requires training models with balanced class weights (`class_weight='balanced'`) and prioritizing metrics tailored for minority classes (PR-AUC, ROC-AUC, F1-Score, Recall).

2. **Geographic Impact & State Disparities:**
   - **Intrastate delivery (`is_same_state == 1`):** The late delivery rate drops to **4.5%**.
   - **Interstate delivery (`is_same_state == 0`):** The late delivery rate surges to **12.1%**!
   - Northern and Northeastern states (such as AL, MA, AP, PA) experience the highest delay rates (>20%) due to seller concentration in the Southeast (São Paulo).

3. **Geospatial Distance & Freight Costs:**
   - Delayed orders exhibit significantly higher average transit distances and freight values compared to on-time deliveries.
   - The freight-to-price ratio (`freight_ratio`) serves as a strong proxy for logistical complexity and heavy/bulky freight.

4. **Promised Delivery Windows (Lead Time):**
   - The platform allocates longer estimated delivery windows for distant routes, yet shipments with high weight or complex multi-regional paths still show a significantly higher risk of missing the deadline.

---
###  Modeling & Feature Engineering Decisions:
- **Strict Anti-Leakage Policy:** Exclude all post-purchase timestamps and events (`order_delivered_carrier_date`, `order_delivered_customer_date`, `review_score`).
- **Feature Interactions:** Engineer cross-feature interactions combining macro-regions, distance, freight cost ratios, and package physical density.
- **Pipeline Preservation:** Fit all transformers (imputers, scalers, one-hot encoders) strictly on the training set and persist the serialized pipeline object in `artifacts/models/preprocessor.joblib`.
