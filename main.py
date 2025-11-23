# ==========================================
# PHASE 4: CODE DEMO
# Project: Sleep Efficiency Analysis and Prediction
# Group Members: Tyler Lafond, Michael Trbovic, Murede Adetiba, Jakub Drotlef
# Course: DS 3000 / ECE 9611
# ==========================================

# ---------------------------------------------------------
# 1. LIBRARIES & SETUP
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    !pip install xgboost
    import xgboost as xgb

# Visual Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries loaded successfully.")

# ---------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
# Load Dataset
try:
    df = pd.read_csv('Sleep_Efficiency.csv')
    print("Dataset Loaded. Shape:", df.shape)
except FileNotFoundError:
    print("ERROR: Please upload 'Sleep_Efficiency.csv' to the Colab environment.")

# --- Data Cleaning ---
# Handling Missing Values
# Fill numerical NaNs with mean/median, categorical with mode
df['Awakenings'] = df['Awakenings'].fillna(df['Awakenings'].median())
df['Caffeine consumption'] = df['Caffeine consumption'].fillna(0) # Assuming NaN means 0
df['Alcohol consumption'] = df['Alcohol consumption'].fillna(0)
df['Exercise frequency'] = df['Exercise frequency'].fillna(0)

# Drop ID if present (irrelevant feature)
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# --- Feature Engineering ---
# Encoding Categorical Variables
# 'Smoking status' and 'Gender' need to be numeric
df = pd.get_dummies(df, columns=['Gender', 'Smoking status'], drop_first=True)

# Convert 'Bedtime' and 'Wakeup time' to a duration or drop them
# For simplicity in this demo, we will drop specific timestamps and rely on 'Sleep duration'
cols_to_drop = ['Bedtime', 'Wakeup time']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print("\nData Preprocessing Complete.")
display(df.head())

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# ---------------------------------------------------------
# 4. TRACK 1: REGRESSION ANALYSIS
# Goal: Predict exact 'Sleep efficiency' (Continuous)
# ---------------------------------------------------------
print("\n" + "="*30)
print("TRACK 1: REGRESSION (Predicting Efficiency Score)")
print("="*30)

# Defining X and y
target = 'Sleep efficiency'
X = df.drop(target, axis=1)
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Models dictionary
reg_models = {
    "Linear Regression": LinearRegression(),
    "Lasso (L1 Regularization)": Lasso(alpha=0.001),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# Training and Evaluation
reg_results = []

print(f"{'Model':<25} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
print("-" * 65)

for name, model in reg_models.items():
    # Use scaled data for Linear/Lasso/MLP, raw for Trees (though Trees handle unscaled fine)
    if name in ["Linear Regression", "Lasso (L1 Regularization)", "Neural Network (MLP)"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    reg_results.append([name, rmse, mae, r2])
    print(f"{name:<25} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f}")

# ---------------------------------------------------------
# 5. TRACK 2: CLASSIFICATION ANALYSIS
# Goal: Predict 'High' vs 'Low' Efficiency (Categorical)
# ---------------------------------------------------------
print("\n" + "="*30)
print("TRACK 2: CLASSIFICATION (High vs Low Efficiency)")
print("="*30)

# Discretizing Target
# Threshold: 0.85 (High Efficiency = 1, Low = 0)
threshold = 0.85
y_class = (y > threshold).astype(int)

# Split for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Scale features (re-using scaler)
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

class_models = {
    "Logistic Regression": LogisticRegression(),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF Kernel)": SVC(kernel='rbf'),
    "Random Forest Clf": RandomForestClassifier(n_estimators=100, random_state=42)
}

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Model':<25} | {'Accuracy':<10} | {'CV Mean Acc':<10}")
print("-" * 60)

for name, model in class_models.items():
    # Train on full training set
    model.fit(X_train_c_scaled, y_train_c)
    preds_c = model.predict(X_test_c_scaled)

    # Metrics
    acc = accuracy_score(y_test_c, preds_c)

    # Cross Validation
    cv_results = cross_val_score(model, X_train_c_scaled, y_train_c, cv=kfold, scoring='accuracy')

    print(f"{name:<25} | {acc:<10.4f} | {cv_results.mean():<10.4f}")

# Confusion Matrix for Random Forest
rf_clf = class_models["Random Forest Clf"]
y_pred_rf = rf_clf.predict(X_test_c_scaled)
cm = confusion_matrix(y_test_c, y_pred_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Random Forest Classifier')
plt.show()

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------
print("\n" + "="*30)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*30)

# Using the Random Forest Regressor from Track 1
rf_reg = reg_models["Random Forest"]
importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# Print top 3 features
print("Top 3 Predictive Features:")
for i in range(3):
    print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")