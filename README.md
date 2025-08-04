# Predict Income in the Census Income Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-0D1117?style=for-the-badge&logo=seaborn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>  

---

## Overview  

**View my project [here](https://github.com/aditi-dheer/adult-income-prediction/blob/main/CensusDataPrediction.ipynb)!**  

This project predicts whether an individual earns **above or below \$50K per year** using demographic and employment-related data from the **U.S. Census Income dataset**. It is a **supervised binary classification** problem, implemented using a full **machine learning pipeline** including data preprocessing, bias mitigation, and model selection.

Key applications of such predictions include:

- **Targeted marketing**
- **Credit risk assessment**
- **Public policy and socio-economic analysis**

---

## Key Features  

- **Data Cleaning & Preprocessing**
  - Dropped irrelevant or redundant columns
  - Winsorized outliers in numeric columns
  - Replaced missing values with group-specific strategies
  - Consolidated categorical features into fewer, more meaningful classes

- **Bias Mitigation**
  - Detected demographic imbalances based on **gender** and **race**
  - Applied **upsampling techniques** to ensure fair representation of underrepresented high-income groups

- **Feature Engineering**
  - Converted categorical variables using **one-hot encoding**
  - Simplified multi-class categories like marital status, occupation, and workclass

- **Model Training & Evaluation**
  - Trained and evaluated **six models**:
    - Logistic Regression  
    - K-Nearest Neighbors  
    - Decision Tree  
    - Random Forest  
    - Gradient Boosting  
    - Stacking Ensemble  
  - Used **GridSearchCV** for hyperparameter tuning  
  - Compared models based on **accuracy, precision, and recall**

- **Fairness Analysis**
  - Built **confusion matrices per demographic group** (gender and race)
  - Verified that final model did not disproportionately misclassify any group

---

## Evaluation Metrics

Each model was evaluated on the **test set** using the following metrics:

| Model                      | Accuracy | Precision | Recall |
|---------------------------|----------|-----------|--------|
| Stacking                  | 0.834     | 0.677    | 0.559  |
| Logistic Regression       | 0.809    | 0.580     | 0.680  |
| Decision Tree             | 0.823    | 0.674     | 0.480  |
| **Gradient Boosting (GBDT)** | **0.837** | **0.680** | **0.578** |
| Random Forest             | 0.830    | 0.665     | 0.558  |
| KNN                       | 0.823    | 0.674     | 0.480  |


> **Gradient Boosted Decision Trees (GBDT)** were selected as the final model due to a strong balance of all three metrics.

---

## Confusion Matrix by Demographics

| Group         | Precision Preserved? | Notes |
|---------------|----------------------|-------|
| **Female**    | ✅                    | Upsampled to match male high-income group size |
| **Male**      | ✅                    | No overfitting observed |
| **Minority**  | ✅                    | Upsampled to match White high-income group size |
| **White**     | ✅                    | No preference detected |

> The model was tested for **fairness across demographic subgroups**, and no significant bias was observed post-augmentation.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aditi-dheer/adult-income-prediction.git
cd adult-income-prediction
pip install pandas numpy scikit-learn matplotlib seaborn
```
---
## Usage  

Run the notebook from Jupyter:

```bash
jupyter notebook CensusDataPrediction.ipynb
```
## Dataset  

- **Source**: UCI Machine Learning Repository — [Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
- **Format**: CSV file (`censusData.csv`)  
- **Target**: `income_binary`  
  - 1 if income > \$50K  
  - 0 if income <= \$50K  

---

## Techniques Used

- **Winsorization**  
  Capped extreme values in `capital-gain`, `capital-loss`, and `hours-per-week` to limit outlier influence.  

- **Feature Consolidation**  
  Simplified categorical variables:
  - `workclass`: mapped to self-employed vs. not
  - `marital-status`: mapped to married vs. single
  - `race`: mapped to White vs. minority
  - `occupation`: grouped into Management, Skilled, Support, Service  

- **Missing Value Imputation**  
  - Filled numeric missing values (e.g., `age`) with the column mean  
  - Categorical missing values (e.g., `workclass`, `occupation`) filled with 'Unknown' or appropriate default  

- **Bias Mitigation**  
  - Upsampled underrepresented high-income groups for women and minorities  
  - Evaluated fairness using confusion matrices by group  

- **Encoding**  
  - One-hot encoded all remaining categorical variables  
  - Binary mapped consolidated features  

- **Modeling**  
  - Applied and tuned Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, and Stacking using `GridSearchCV`  
  - Evaluated each model on accuracy, precision, and recall  

- **Fairness Verification**  
  - Manually inspected performance across subgroups (e.g., gender and race) using confusion matrices to ensure no systemic bias  


