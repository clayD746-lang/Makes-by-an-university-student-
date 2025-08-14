
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataframe = pd.read_csv(r"D:\BMW_Car_Sales_Classification.csv")
new_dataframe = dataframe.drop(['Sales_Volume'], axis=1)

X = new_dataframe.drop('Sales_Classification', axis=1)
y = new_dataframe['Sales_Classification'].map({'High': 1, 'Low': 0})

categorical_cols = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
numerical_cols = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD']


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X[numerical_cols]), columns=poly.get_feature_names_out(numerical_cols))


X_engineered = X.copy()
X_engineered['Price_per_Engine_Size'] = X['Price_USD'] / X['Engine_Size_L']
X_engineered['Mileage_per_Year'] = X['Mileage_KM'] / (2025 - X['Year'])
X_engineered['Is_Newer'] = (X['Year'] >= 2020).astype(int)

X_engineered = pd.concat([
    X_engineered.drop(columns=numerical_cols),
    X_poly,
    X_engineered[['Price_per_Engine_Size', 'Mileage_per_Year', 'Is_Newer']] # 添加新工程特征
], axis=1)

X_engineered = X_engineered.loc[:,~X_engineered.columns.duplicated()]

engineered_numerical_cols = [col for col in X_engineered.columns if col not in categorical_cols]
engineered_numerical_cols.sort()

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), engineered_numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

feature_selector = SelectFromModel(XGBClassifier(random_state=42, scale_pos_weight=2.28, n_estimators=100))

# 预处理， 特征选择 ，XGBoost模型
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', feature_selector),
    ('classifier', XGBClassifier(
        random_state=42,
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1], # 动态设置不平衡权重
        n_estimators=300
    ))
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("XGBoost Pipeline Report (after feature selection):\n", classification_report(y_test, y_pred))
print("\nXGBoost Pipeline Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nXGBoost Pipeline ROC AUC: {roc_auc_score(y_test, y_pred_proba)}")

param_dist = {
    'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
}

tuning_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42,
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1]
    ))
])

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

best_pipeline = random_search.best_estimator_
y_pred_best = best_pipeline.predict(X_test)
y_pred_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

print("\nBest Parameters found by RandomizedSearchCV:", random_search.best_params_)
print("\nXGBoost Tuned Report:\n", classification_report(y_test, y_pred_best))
print("\nXGBoost Tuned Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print(f"\nXGBoost Tuned ROC AUC: {roc_auc_score(y_test, y_pred_proba_best)}")


temp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, n_estimators=100))
])
temp_pipeline.fit(X_train, y_train)


encoded_cols = temp_pipeline.named_steps['preprocessor'].get_feature_names_out()

# 取特征重要性数值
importances = temp_pipeline.named_steps['classifier'].feature_importances_
feature_importances = pd.Series(importances, index=encoded_cols).sort_values(ascending=False)

print("\nTop 20 Features by Importance:")
print(feature_importances.head(20))

