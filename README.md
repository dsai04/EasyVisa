# 🛂 EasyVisa: Predicting US Employment Visa Certification

A classification project that predicts whether a US employment-based visa application will be Certified or Denied. It uses 25,480 applications and 10 predictors, and compares five algorithms across three training setups: original data, oversampled data and undersampled data.

The dataset and business case come from the Great Learning Postgraduate Program in Data Science.

## Results at a glance

- **Final model:** a Random Forest tuned with RandomizedSearchCV, tested on a held-out 20% (5,096 applications).
- **Test performance:** accuracy 73.7%, precision 79.1%, recall 82.3%, F1 0.807, F1 macro 0.697, ROC-AUC 0.768. Precision, recall and F1 are for the Certified class.
- **Education:** certification rate is 87.2% for Doctorate holders, 78.6% for Master's, 62.2% for Bachelor's and 34.0% for High School.
- **Job experience:** 74.5% certified with experience, 56.1% without.
- **Ensembles beat single trees:** the best single decision tree reached 65.1% accuracy before tuning, against 74.2% for the best ensemble.
- **Resampling did not help:** SMOTE and undersampling gave no gain over the original data. See the comparison below.
- **Caveat:** the final pick is a judgment call and the wage column mixes units. See Limitations.

## Business question

The Office of Foreign Labor Certification (OFLC) receives a growing number of visa applications, and reviewing them by hand is slow. The goal is to find which applicant and employer attributes relate to certification, and to build a model that predicts the outcome to support screening.

## Data

25,480 rows, 12 columns. No missing values, no duplicate rows. 17,018 applications (66.8%) were certified and 8,462 (33.2%) denied.

| Variable | Type | Description |
|---|---|---|
| `case_id` | ID | Application ID (dropped before modeling) |
| `continent` | Categorical (6) | Continent of the employee |
| `education_of_employee` | Categorical (4) | Highest education level |
| `has_job_experience` | Binary | Prior job experience, Y or N |
| `requires_job_training` | Binary | Whether the employee needs job training |
| `no_of_employees` | Numeric | Number of employees at the employer |
| `yr_of_estab` | Numeric | Year the employer was established |
| `region_of_employment` | Categorical (5) | Intended US region of work |
| `prevailing_wage` | Numeric | Average wage for similar roles in the area |
| `unit_of_wage` | Categorical (4) | Hour, Week, Month or Year |
| `full_time_position` | Binary | Full-time or part-time position |
| `case_status` | Binary | Certified or Denied (target) |

## Method

1. Checked shape, data types, missing values and duplicates.
2. Univariate and bivariate analysis against `case_status`.
3. Dropped `case_id`. Created `company_age` as 2026 minus `yr_of_estab`.
4. Capped `no_of_employees` and `prevailing_wage` at the 1st and 99th percentiles.
5. Encoded education as ordered levels and one-hot encoded the other categorical variables. Target: Denied = 0, Certified = 1.
6. Split 80/20 with stratification (20,384 train, 5,096 test).
7. Trained five models on three versions of the training data: original, SMOTE-oversampled and randomly undersampled. Resampling was applied to the training set only.
8. Tuned Decision Tree, Random Forest and Gradient Boosting with RandomizedSearchCV (20 combinations, 5-fold stratified cross-validation, F1 macro). SMOTE ran inside the cross-validation pipeline, so no synthetic rows leaked into validation folds.

## Findings

### Exploratory analysis

- Most applications are from Asia (16,861), with Bachelor's (10,234) and Master's (9,634) the most common education levels.
- Certification rises with education: High School 34.0%, Bachelor's 62.2%, Master's 78.6%, Doctorate 87.2%.
- Applicants with prior experience are certified more often (74.5% vs 56.1%).
- Median prevailing wage is about 72,486 for certified and 65,431 for denied cases, but the two distributions overlap heavily, and the wage column mixes units (see Limitations).
- Requiring job training shows no visible link to the outcome: 67.9% certified with training, 66.6% without.

### Models on the original data

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.647 | 0.738 | 0.732 | 0.735 | 0.604 |
| Random Forest | 0.715 | 0.761 | 0.835 | 0.796 | 0.744 |
| Bagging | 0.685 | 0.762 | 0.768 | 0.765 | 0.706 |
| AdaBoost | 0.732 | 0.757 | 0.881 | 0.814 | 0.759 |
| Gradient Boosting | 0.742 | 0.771 | 0.874 | 0.819 | 0.767 |

### Best model in each setup

| Setup | Best model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Original data | Gradient Boosting | 74.2% | 77.1% | 87.4% | 0.819 | 0.767 |
| SMOTE oversampling | Gradient Boosting | 73.4% | 78.6% | 82.7% | 0.806 | 0.763 |
| Random undersampling | Gradient Boosting | 70.2% | 81.8% | 71.3% | 0.762 | 0.766 |
| Tuned (on SMOTE data) | Random Forest | 73.7% | 79.1% | 82.3% | 0.807 | 0.768 |

### Effect of tuning (test set)

| Model | | Accuracy | F1 | F1 macro | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | before | 0.651 | 0.736 | 0.612 | 0.613 |
| Decision Tree | after | 0.730 | 0.800 | 0.691 | 0.755 |
| Random Forest | before | 0.713 | 0.791 | 0.667 | 0.744 |
| Random Forest | after | 0.737 | 0.807 | 0.697 | 0.768 |
| Gradient Boosting | before | 0.734 | 0.806 | 0.692 | 0.763 |
| Gradient Boosting | after | 0.737 | 0.807 | 0.696 | 0.767 |

Tuning helped the Decision Tree most. Random Forest finished with the highest F1 macro and ROC-AUC. Its best settings were 300 trees, max depth 10, min samples split 5, min samples leaf 2, `max_features="sqrt"` and bootstrap on.

## Limitations

- **Model choice is close.** The three tuned models are within 0.001 to 0.002 on F1 and ROC-AUC. Untuned Gradient Boosting on the original data scores higher on accuracy (74.2%), recall (87.4%) and F1 (0.819). Random Forest was picked on F1 macro and ROC-AUC, and F1 macro was not computed for the original-data models.
- **Mixed wage units.** `prevailing_wage` mixes hourly, weekly, monthly and yearly figures and was not converted to a single unit. Hourly cases (2,157 rows) have a median "wage" of about 373, against about 76,000 for yearly cases, and only 34.6% of hourly cases are certified. Wage comparisons should be read with this in mind.
- **Bad values in `no_of_employees`.** 33 rows have negative values (minimum −26). They were capped at the 1st percentile, not corrected.
- **Fixed reference year.** `company_age` uses 2026 as a fixed year.
- **Associations only.** The model finds patterns in past decisions. It should not be used to make or automate real certification decisions.

## Repository contents

| File | Description |
|---|---|
| `EasyVisa.ipynb` | Full analysis: EDA, five models, resampling, tuning |
| `EasyVisa.csv` | Data File |
| `README.md` | This file |

## How to run

The dataset (`EasyVisa.csv`) is not included in this repository. The notebook reads it from a Google Drive path, so change the loading line to `pd.read_csv("EasyVisa.csv")` and place the file next to the notebook.

```
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
jupyter notebook EasyVisa.ipynb
```

## Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn

## Author

Dhruv Sai Gedda | [LinkedIn](https://www.linkedin.com/in/dhruv-sai) | [Portfolio](https://dsai04.github.io/)
