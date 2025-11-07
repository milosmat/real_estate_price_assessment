import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Učitavanje podataka
data = pd.read_csv('cleaned_property_data_no_outliers.csv')

# Uklanjanje duplikata
data = data.drop_duplicates()

# Pretprocesiranje podataka
data = data.dropna(subset=['kvadratura', 'cena'])
label_encoders = {}
for column in ['grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'stanje', 'grejanje', 'lift', 'podrum', 'terasa']:
    label_encoders[column] = LabelEncoder()
    data[column] = data[column].fillna('missing')
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Odabir relevantnih kolona
features = data[['kvadratura', 'grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'stanje', 'grejanje', 'lift', 'podrum', 'terasa']]
target = data['cena']

# Skaliranje kvadrature
scaler = StandardScaler()
features.loc[:, 'kvadratura'] = scaler.fit_transform(features[['kvadratura']])

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []
y_test_all = []
y_pred_all = []

for train_index, test_index in kf.split(features):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    y_test_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Prikazivanje rezultata
print(f'Mean MAE: {np.mean(mae_scores)}')
print(f'Mean MSE: {np.mean(mse_scores)}')

# Važnost karakteristika
feature_importance = model.feature_importances_

# Kreiranje dijagrama važnosti karakteristika
plt.figure(figsize=(10, 6))
plt.bar(features.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Predicting Apartment Prices')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vizualizacija predikcija i stvarnih vrednosti
plt.figure(figsize=(10, 6))
plt.scatter(y_test_all, y_pred_all, alpha=0.3)
plt.plot([min(y_test_all), max(y_test_all)], [min(y_test_all), max(y_test_all)], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices')
plt.show()

# Distribucija grešaka
errors = np.array(y_test_all) - np.array(y_pred_all)
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()

# Optimizacija hiperparametara korišćenjem GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

#grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
#grid_search.fit(features, target)

#print(f'Best parameters found: {grid_search.best_params_}')
#print(f'Best score found: {grid_search.best_score_}')

# Sačuvaj najbolje parametre u fajl
#joblib.dump(grid_search.best_params_, 'best_params.pkl')

# Ponovno treniranje modela sa najboljim parametrima
#best_model = grid_search.best_estimator_
# Učitavanje sačuvanih parametara
best_params = joblib.load('best_params.pkl')

# Kreiranje modela sa učitanim parametrima
best_model = RandomForestRegressor(**best_params, random_state=42)
# K-fold cross-validation sa najboljim modelom
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []
y_test_all_optimized = []
y_pred_all_optimized = []

for train_index, test_index in kf.split(features):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    y_test_all_optimized.extend(y_test)
    y_pred_all_optimized.extend(y_pred)
    
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Prikazivanje rezultata sa optimizovanim modelom
print(f'Mean MAE with optimized Random Forest: {np.mean(mae_scores)}')
print(f'Mean MSE with optimized Random Forest: {np.mean(mse_scores)}')

# Vizualizacija predikcija i stvarnih vrednosti sa optimizovanim modelom
plt.figure(figsize=(10, 6))
plt.scatter(y_test_all_optimized, y_pred_all_optimized, alpha=0.3)
plt.plot([min(y_test_all_optimized), max(y_test_all_optimized)], [min(y_test_all_optimized), max(y_test_all_optimized)], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices with Optimized Random Forest')
plt.show()

# Distribucija grešaka sa optimizovanim modelom
errors_optimized = np.array(y_test_all_optimized) - np.array(y_pred_all_optimized)
plt.figure(figsize=(10, 6))
sns.histplot(errors_optimized, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors with Optimized Random Forest')
plt.show()
