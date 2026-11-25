# Heart Disease Classifier

A machine learning project that predicts whether a patient has heart disease from clinical measurements. It explores the data, then compares three classification models in Scikit-Learn.

> For educational purposes only. Not for clinical use.

## Dataset

`heart-disease.csv` has 303 patient records, 13 clinical features and a binary target (1 = heart disease, 0 = no heart disease). There are no missing values. The classes are fairly balanced, with 165 disease and 138 no-disease records.

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0 to 3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar above 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (0 to 2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0 to 2) |
| `ca` | Number of major vessels (0 to 3) colored by fluoroscopy |
| `thal` | Thallium stress test result (1, 3, 6 or 7) |

## Approach

1. **Exploratory data analysis:** class balance, missing values, summary statistics, a correlation heatmap, and plots of how chest pain type, sex, age and maximum heart rate relate to heart disease.
2. **Modeling:** an 80/20 train/test split (random seed 42), then three classifiers trained and scored on the test set: logistic regression, K-nearest neighbors and random forest.

## Results

| Model | Test accuracy |
|---|---|
| Logistic Regression | 88.5% |
| Random Forest | 83.6% |
| K-Nearest Neighbors | 68.9% |

Logistic regression performed best. The test set has only 61 records, so these scores can shift with a different split.

## Next steps

- Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`
- Cross-validation for more reliable estimates
- Precision, recall and F1 alongside accuracy
- Feature scaling

## Files

- `end-to-end-heart-disease-classification.ipynb`: data exploration, modeling and evaluation
- `heart-disease.csv`: dataset

## Running the notebook

Install Python 3 with Jupyter, then install the dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn`. Open `end-to-end-heart-disease-classification.ipynb` and run the cells in order, keeping `heart-disease.csv` in the same folder.
