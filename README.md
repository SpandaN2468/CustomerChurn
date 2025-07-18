# Customer Churn Prediction with Machine Learning

The objective of this project is to predict whether a customer is likely to churn (leave a service) based on historical behavioral and demographic data. We use various data preprocessing techniques and machine learning models (such as Logistic Regression, Random Forests, and XGBoost) to build an accurate churn prediction system.

---

## Project Structure

The work has been organized into the following notebooks:

1. **Data Exploration and Analysis**

   * Understanding the dataset (structure, columns, missing values).
   * Statistical analysis and correlation study.
   * Visualizations of churn patterns, class distribution, and key factors influencing churn.

2. **Data Pre-processing and Feature Engineering**

   * Handling missing values, outliers, and class imbalance.
   * Encoding categorical variables (e.g., One-Hot Encoding, Label Encoding).
   * Feature scaling and normalization.
   * Splitting data into training, validation, and test sets.

3. **Model Building**

   * Baseline models: Logistic Regression, Decision Trees, Random Forests.
   * Advanced models: Gradient Boosting (XGBoost/LightGBM), Support Vector Machines.
   * Hyperparameter tuning using GridSearchCV/RandomizedSearchCV.

4. **Model Evaluation**

   * Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUC-ROC.
   * Confusion matrices and feature importance plots.
   * Model comparison and selection.

5. **Deployment (Optional)**

   * Exporting the best model using `joblib` or `pickle`.
   * Sample prediction script or API endpoint.

---

## Dataset

The dataset used for churn prediction contains customer-related attributes such as:

* **Customer demographics:** gender, age, tenure.
* **Service information:** contract type, internet services, payment methods.
* **Usage metrics:** monthly charges, total charges.
* **Target variable:** `Churn` (Yes/No).

*(If your dataset is different, update this section accordingly.)*

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd CustomerChurn
   ```

2. **Install dependencies**
   Ensure you have Python 3.8+ and install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies include:

   * `pandas`, `numpy`
   * `scikit-learn`
   * `matplotlib`, `seaborn`
   * `xgboost`

3. **Run notebooks**
   Open Jupyter and run the notebooks in order:

   ```bash
   jupyter notebook
   ```

---

## Key Features

* **Exploratory Data Analysis (EDA):** Identify patterns and trends in churn.
* **Machine Learning Pipeline:** From feature engineering to model evaluation.
* **Performance Optimization:** Hyperparameter tuning to improve results.
* **Visualizations:** ROC curves, feature importances, and churn probability distributions.

---

## References

* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Churn Prediction Overview – IBM Telco dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
