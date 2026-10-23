# Stroke Prediction — Machine Learning Model Comparison

This project explores the classification of stroke cases using demographic, lifestyle, and health-related information. Developed as a machine-learning analysis in Google Colab, it compares **five classification algorithms** and examines how their performance changes when working with an imbalanced dataset.

The main aim is to compare approaches and understand why **accuracy alone is not enough** when identifying the minority class. The notebook is an educational project, not a clinical tool for assessing individual stroke risk.

## Dataset

The project uses the [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/aouatifcherdid/healthcare-dataset-stroke-data), which contains **5,110 records** and information such as age, hypertension, heart disease, glucose level, BMI, smoking status, and other demographic factors. The target variable, `stroke`, indicates whether a stroke was recorded.

Only **249 records (about 4.9%)** belong to the stroke class, making class imbalance a central challenge.

**The dataset is downloaded directly from Kaggle using `kagglehub` in the notebook.** It is not stored in this GitHub repository, and users do not need to upload a CSV manually.

## Methodology

1. **Exploratory data analysis:** examine the variables, missing values, distributions, and relationships with stroke status.
2. **Data preprocessing:** remove the ID column, handle missing values, encode categorical variables, and standardize numerical features.
3. **Class imbalance:** apply SMOTE to the training data to generate synthetic minority-class examples.
4. **Model development:** train and tune five algorithms using cross-validation and hyperparameter searches:
   - Logistic Regression
   - Random Forest
   - Support Vector Classifier (SVC)
   - Multilayer Perceptron (MLP)
   - XGBoost
5. **Evaluation:** compare classification reports, confusion matrices, accuracy, precision, recall, F1 scores, and learning curves.

## Results

The results saved in the original notebook show that **XGBoost achieved the highest overall test accuracy (91.59%)**, but correctly identified only **11 out of 50** stroke cases in the test set. **Logistic Regression identified 39 out of 50** stroke cases, although its overall accuracy was lower (**73.09%**).

This difference is the main takeaway: with an imbalanced dataset, high accuracy can hide poor detection of the minority class. Recall and the number of false negatives are important when comparing the models.

## Limitations

The results are based on a single educational dataset and should not be used for medical decisions. In the original analysis, some preprocessing takes place before the train/test split, and SMOTE is applied before cross-validation; a more rigorous evaluation would perform those steps within each training fold to reduce potential data leakage. The notebook also calculates ROC AUC from predicted class labels rather than probability scores.

## How to run

1. Open [`stroke_prediction.ipynb`](stroke_prediction.ipynb) in **Google Colab**.
2. Run the notebook cells in order. The notebook uses **KaggleHub to download the dataset automatically** and reads the CSV from the downloaded directory.
3. Continue running the preprocessing, training, and evaluation cells. Hyperparameter tuning may take longer than the other steps.

If `kagglehub` is not available in the environment, install it with `%pip install kagglehub`. Other libraries used include scikit-learn, imbalanced-learn, XGBoost, pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, and missingno.

## Repository contents

- `stroke_prediction.ipynb` — complete analysis, models, and saved results
- `README.md` — project overview and instructions

**Dataset source:** [Kaggle — Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/aouatifcherdid/healthcare-dataset-stroke-data)
