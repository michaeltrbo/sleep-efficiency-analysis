# Sleep Efficiency Analysis and Prediction

**Course:** DS 3000 / ECE 9611  
**Group 9:** Tyler Lafond, Michael Trbovic, Murede Adetiba, Jakub Drotlef

## 📖 Project Overview
Sleep quality is a crucial aspect of overall health. This project applies machine learning techniques to predict sleep efficiency and identify the key lifestyle drivers that impact deep sleep. 

Our goal was to move beyond simple prediction and provide **actionable insights** into how individuals can improve their sleep quality through controllable behavioral changes.

## 📊 Dataset
We utilized the **Sleep Efficiency Dataset** (originally from Kaggle), containing data for 452 individuals.
* **Target Variable:** `Sleep efficiency` (0.0 to 1.0)
* **Key Features:** * **Physiological:** REM sleep %, Deep sleep %, Awakenings.
    * **Lifestyle:** Alcohol consumption, Caffeine intake, Exercise frequency, Smoking status.
    * **Demographic:** Age, Gender.

## 🛠️ Methodology & Results

The project is divided into three analytical tracks:

### Track 1: Regression Analysis
**Goal:** Predict the exact Sleep Efficiency score.
* **Models Used:** Linear Regression, Lasso, Random Forest, Neural Network (MLP), XGBoost.
* **Best Model:** **Random Forest Regressor**
    * *RMSE:* ~0.05
    * *R² Score:* ~0.86
* **Insight:** Tree-based models significantly outperformed linear models, suggesting non-linear relationships between sleep stages and efficiency.

### Track 2: Classification Analysis
**Goal:** Classify subjects into "High Efficiency" vs. "Low Efficiency" (Threshold > 0.85).
* **Models Used:** Logistic Regression, KNN, SVM, Random Forest Classifier.
* **Best Model:** **Random Forest Classifier**
    * *Accuracy:* ~93%
    * *Confusion Matrix:* showed minimal false negatives, making it robust for identifying poor sleepers.

### Track 3: Lifestyle Impact (Actionable Insights)
**Goal:** Isolate controllable habits to answer *"How can I get better sleep?"*
We stripped away physiological features (like REM sleep) to focus solely on behavioral inputs.

**Key Drivers of Deep Sleep (Ranked by Importance):**
1.  🍷 **Alcohol Consumption (34.2%)** - The #1 suppressor of deep sleep.
2.  ⏰ **Bedtime Consistency (21.1%)** - Earlier, consistent bedtimes correlate with higher efficiency.
3.  🏃 **Exercise Frequency (18.7%)** - Regular exercise (3+ times/week) is a strong positive predictor.
4.  🚬 **Smoking Status (14.3%)** - Smokers consistently showed lower efficiency scores.
5.  ☕ **Caffeine Consumption (11.6%)** - Surprisingly the least impactful factor in this specific dataset.

## 💻 Installation & Usage

### 1. Requirements
Ensure you have Python 3.x and the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost