import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from scipy.stats import uniform, randint

# Učitavanje podataka
data = pd.read_csv('cleaned_property_data_no_outliers.csv')

# Uklanjanje duplikata
data = data.drop_duplicates()

# Pretprocesiranje podataka
data = data.dropna(subset=['kvadratura', 'cena'])
label_encoders = {}
for column in ['grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'stanje', 'grejanje', 'lift', 'podrum']:
    label_encoders[column] = LabelEncoder()
    data[column] = data[column].fillna('missing')
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Definisanje target kolone
target = data['cena']

# Detekcija i uklanjanje outlier-a
model = xgb.XGBRegressor(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
outliers = []

for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    residuals = np.abs(y_test - y_pred)
    threshold = np.percentile(residuals, 98)
    outliers.extend(X_test[residuals > threshold].index)

# Identifikacija outlier-a
outlier_indices = np.unique(outliers)
print(f'Number of outliers: {len(outlier_indices)}')

# Uklanjanje outlier-a iz podataka
data_cleaned = data.drop(outlier_indices)
target_cleaned = data_cleaned['cena']

# Pretprocesiranje očišćenih podataka
best_features = ['kvadratura', 'grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'grejanje', 'lift', 'podrum']
scaler = StandardScaler()
data_cleaned[best_features] = scaler.fit_transform(data_cleaned[best_features])

# Randomized Search za optimizaciju hiperparametara
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 1),
    'colsample_bytree': uniform(0.5, 1),
    'gamma': uniform(0, 0.5)
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=50, cv=5, verbose=2, n_jobs=-1, scoring='neg_mean_absolute_error', random_state=42)
random_search.fit(data_cleaned[best_features], target_cleaned)

# Najbolji parametri
best_params = random_search.best_params_
print(f'Best parameters found: {best_params}')
print(f'Best score found: {random_search.best_score_}')

# Sačuvaj najbolje parametre u fajl
joblib.dump(best_params, 'best_params_xgboost_cleaned.pkl')

# Kreiranje modela sa najboljim parametrima
best_model = xgb.XGBRegressor(**best_params, random_state=42)

# K-fold cross-validation sa najboljim modelom na očišćenim podacima
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []
y_test_all_optimized = []
y_pred_all_optimized = []

for train_index, test_index in kf.split(data_cleaned):
    X_train, X_test = data_cleaned.iloc[train_index][best_features], data_cleaned.iloc[test_index][best_features]
    y_train, y_test = target_cleaned.iloc[train_index], target_cleaned.iloc[test_index]
    
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    y_test_all_optimized.extend(y_test)
    y_pred_all_optimized.extend(y_pred)
    
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Prikazivanje rezultata sa optimizovanim modelom na očišćenim podacima
print(f'Mean MAE with optimized XGBoost on cleaned data: {np.mean(mae_scores)}')
print(f'Mean MSE with optimized XGBoost on cleaned data: {np.mean(mse_scores)}')

# Važnost karakteristika
feature_importance = best_model.feature_importances_

# Kreiranje dijagrama važnosti karakteristika
plt.figure(figsize=(10, 6))
plt.bar(best_features, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Predicting Apartment Prices')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vizualizacija predikcija i stvarnih vrednosti sa optimizovanim modelom na očišćenim podacima
plt.figure(figsize=(10, 6))
plt.scatter(y_test_all_optimized, y_pred_all_optimized, alpha=0.3)
plt.plot([min(y_test_all_optimized), max(y_test_all_optimized)], [min(y_test_all_optimized), max(y_test_all_optimized)], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices with Optimized XGBoost on Cleaned Data')
plt.show()

# Distribucija grešaka sa optimizovanim modelom na očišćenim podacima
errors_optimized = np.array(y_test_all_optimized) - np.array(y_pred_all_optimized)
plt.figure(figsize=(10, 6))
sns.histplot(errors_optimized, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors with Optimized XGBoost on Cleaned Data')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = data_cleaned[best_features + ['cena']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
