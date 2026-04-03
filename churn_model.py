# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:31:09 2026

@author: CP24
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:29:15 2026

@author: CP24
"""
#data preparation
import pandas as pd

df= pd.read_excel('E:\python\Telecom Churn\Customer Churn.xlsx')

df.set_index(['CustomerID'],inplace=True)

# Use errors='coerce' to turn unparseable strings into NaN
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df = df.dropna(subset=['Total Charges'])

df.drop(columns=['Gender','Total Charges','Country','State','Zip Code','City','Churn Reason', 'Latitude','Longitude','Count','Lat Long'],inplace=True)

# Binary mapping for df
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

binary_features = ['Senior Citizen', 'Partner', 'Dependents', 
                   'Phone Service', 'Paperless Billing','Churn Label']

for col in binary_features:
    df[col] = df[col].map(binary_map)
    
#defining input and output
X = df.drop(['Churn Label'], axis=1)
y = df['Churn Label']

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#OneHotEncoding for mulit categorical columns
from sklearn.preprocessing import OneHotEncoder

# List of multi-class categorical columns
multi_class_cols = ['Multiple Lines', 'Internet Service', 'Online Security',
                    'Online Backup', 'Device Protection', 'Tech Support',
                    'Streaming TV', 'Streaming Movies', 'Contract', 'Payment Method']

# Create OneHotEncoder
ohe = OneHotEncoder(sparse=False, drop='first')

# Fit on training data
ohe.fit(X_train[multi_class_cols])

# Transform training and test data
X_train_ohe = ohe.transform(X_train[multi_class_cols])
X_test_ohe = ohe.transform(X_test[multi_class_cols])

# Convert back to DataFrame and keep column names
ohe_columns = ohe.get_feature_names_out(multi_class_cols)

X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe_columns, index=X_train.index)
X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe_columns, index=X_test.index)

# Drop original multi-class columns and add encoded columns
X_train_final = pd.concat([X_train.drop(multi_class_cols, axis=1), X_train_ohe], axis=1)
X_test_final = pd.concat([X_test.drop(multi_class_cols, axis=1), X_test_ohe], axis=1)

#scaling numerical columns
num_cols = ['Tenure Months', 'Monthly Charges']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data
scaler.fit(X_train_final[num_cols])

# Transform both sets
X_train_final[num_cols] = scaler.transform(X_train_final[num_cols])
X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])

#modeling-logittic regression##################################################
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Train the model
model.fit(X_train_final, y_train)

#predictin
y_pred = model.predict(X_test_final)


#evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

LR_conf_matrix = confusion_matrix(y_test, y_pred)
LR_classification_report = classification_report(y_test, y_pred)

# ROC-AUC (important for churn problems)
y_prob = model.predict_proba(X_test_final)[:, 1]
LR_roc_auc=roc_auc_score(y_test, y_prob)

#modeling-random forest##################################################

from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=250,           # more trees for stability
    max_depth=10,               # prevent overfitting
    min_samples_split=5,        # node must have 5 samples to split
    min_samples_leaf=3,         # each leaf must have at least 3 samples
    max_features='sqrt',        # reduce correlation between trees
    class_weight='balanced',    # handle class imbalance
    random_state=42,
    n_jobs=-1                   # use all CPU cores
)

rf_model.fit(X_train_final, y_train)

# Predicted class labels
y_pred_rf = rf_model.predict(X_test_final)

# Predicted probabilities for ROC-AUC
y_prob_rf = rf_model.predict_proba(X_test_final)[:, 1]

# Confusion Matrix
RF_conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Classification Report
RF_classification_report = classification_report(y_test, y_pred_rf)

# ROC-AUC Score
RF_roc_auc = roc_auc_score(y_test, y_prob_rf)

#XGBoost Churn Prediction###################################################

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Step 1: Calculate scale_pos_weight to handle class imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Step 2: Initialize XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,          # number of trees
    max_depth=5,               # control overfitting
    learning_rate=0.1,         # step size shrinkage
    subsample=0.8,             # fraction of samples per tree
    colsample_bytree=0.8,      # fraction of features per tree
    scale_pos_weight=scale_pos_weight,  # balance the churn class
    random_state=42,
    eval_metric='logloss',     # evaluation metric
    use_label_encoder=False
)

# Step 3: Train the model
xgb_model.fit(X_train_final, y_train)

# Step 4: Make predictions
y_pred_xgb = xgb_model.predict(X_test_final)
y_prob_xgb = xgb_model.predict_proba(X_test_final)[:, 1]

# Step 5: Evaluate the model
xgb_confusion_matrix = confusion_matrix(y_test, y_pred_xgb)

xgb_classification_report = classification_report(y_test, y_pred_xgb)

xgb_roc_auc = roc_auc_score(y_test, y_prob_xgb)

# ================= FEATURE IMPORTANCE ================= #

# --- 1. Logistic Regression (use absolute coefficients) ---
lr_importance = pd.Series(
    abs(model.coef_[0]), 
    index=X_train_final.columns
).sort_values(ascending=False)

lr_top10 = lr_importance.head(10)


# --- 2. Random Forest ---
rf_importance = pd.Series(
    rf_model.feature_importances_, 
    index=X_train_final.columns
).sort_values(ascending=False)

rf_top10 = rf_importance.head(10)


# --- 3. XGBoost ---
xgb_importance = pd.Series(
    xgb_model.feature_importances_, 
    index=X_train_final.columns
).sort_values(ascending=False)

xgb_top10 = xgb_importance.head(10)

