# Breast Cancer Classification -- Machine Learning Assignment 2

**Course:** Machine Learning
**Program:** M.Tech (AIML/DSE)
**Assignment:** Assignment -- 2
**BITS-ID:** 2025AB05070
------------------------------------------------------------------------

## 1️⃣ Problem Statement

The objective of this project is to build and deploy multiple machine
learning classification models to predict whether a breast tumor is
**Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on
diagnostic features.

The project demonstrates a complete end-to-end machine learning workflow
including:

-   Data preprocessing
-   Model implementation
-   Performance evaluation
-   Streamlit web application development
-   Deployment on Streamlit Community Cloud

Six classification models are implemented and compared using multiple
evaluation metrics.

------------------------------------------------------------------------

## 2️⃣ Dataset Description

**Dataset Name:** Breast Cancer Dataset

This dataset contains diagnostic measurements of breast mass cells
extracted from digitized images of fine needle aspirate (FNA) of breast
tissue.

### Dataset Characteristics:

-   Total Instances: 569
-   Total Features: 30 numerical features
-   Target Variable: Diagnosis
    -   M → Malignant (1)
    -   B → Benign (0)
-   Classification Type: Binary Classification

This dataset satisfies the assignment requirements:

-   ✔ Minimum 12 features
-   ✔ Minimum 500 instances

------------------------------------------------------------------------

## 3️⃣ Models Implemented

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Gaussian Naive Bayes
5.  Random Forest (Ensemble Model)
6.  XGBoost (Ensemble Boosting Model)

All models were trained and evaluated using an 80-20 train-test split.

------------------------------------------------------------------------

## 4️⃣ Evaluation Metrics

Each model was evaluated using:

-   Accuracy
-   AUC Score (Area Under ROC Curve)
-   Precision
-   Recall
-   F1 Score
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## 5️⃣ Model Performance Comparison

  ML Model              Accuracy   AUC    Precision   Recall   F1 Score   MCC
  --------------------- ---------- ------ ----------- -------- ---------- ------
  Logistic Regression   0.97       0.99   0.97        0.95     0.96       0.94
  Decision Tree         0.94       0.94   0.93        0.93     0.93       0.88
  KNN                   0.94       0.98   0.93        0.93     0.93       0.88
  Naive Bayes           0.96       0.99   0.97        0.93     0.95       0.92
  Random Forest         0.95       0.99   0.95        0.93     0.94       0.90
  XGBoost               0.95       0.99   0.95        0.93     0.94       0.90


------------------------------------------------------------------------

## 6️⃣ Observations

-   Logistic Regression - Achieved the highest overall performance with the best Accuracy (0.97) and MCC (0.94). High AUC (0.99) indicates excellent class  separability. The strong performance suggests the dataset is nearly linearly separable.
-   Decision Tree - Performed slightly lower than Logistic Regression. Although interpretable, it may slightly overfit or underfit compared to ensemble models. Moderate MCC (0.88) indicates good but not optimal correlation.
-   KNN - Performed similarly to Decision Tree. High AUC (0.98) shows good probability ranking. However, performance is slightly lower than Logistic Regression due to sensitivity to feature scaling and local decision boundaries.
-   Naive Bayes - Performed very well with high AUC (0.99). Slightly lower Recall compared to Logistic Regression suggests some misclassification of positive cases. Independence assumption may limit performance.
-   Random Forest - Strong and stable performance due to ensemble averaging. Good balance between bias and variance. However, performance is slightly below Logistic Regression on this dataset.
-   XGBoost - Comparable to Random Forest with strong AUC (0.99). Boosting improves generalization, but in this dataset, it does not significantly outperform Logistic Regression.

------------------------------------------------------------------------

## 9️⃣ Conclusion

Based on the evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, and MCC):

All models perform very well on the breast cancer dataset.

The dataset appears to be well-structured and reasonably separable.

Linear and ensemble methods perform slightly better than tree-based standalone models.

Among all models:

🏆 Logistic Regression performed the best overall

Highest Accuracy: 0.97

Highest MCC: 0.94

Very high AUC: 0.99

Strong Precision & Recall balance

This indicates the dataset is likely close to linearly separable, which explains why Logistic Regression performs exceptionally well.

------------------------------------------------------------------------
