import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

# Učitavanje podataka
data = pd.read_csv('filtered_property_data.csv')

# Osnovno čišćenje i poravnanje ID-jeva PRE uklanjanja kolone 'id'
data = data.drop_duplicates()
data = data.dropna(subset=['kvadratura', 'cena'])

# Sačuvaj ID poravnat sa očišćenim podacima
ids_aligned = data['id'].astype(str).reset_index(drop=True)
label_encoders = {}
for column in ['grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'stanje', 'grejanje', 'lift', 'podrum']:
    label_encoders[column] = LabelEncoder()
    data[column] = data[column].fillna('missing')
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Ukloni kolonu 'id' nakon što smo izdvojili ids_aligned
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Odabir relevantnih kolona (najbolji rezultat)
best_features = ['kvadratura', 'grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'grejanje', 'lift', 'podrum']
target = data['cena']

# Skaliranje kvadrature
scaler = StandardScaler()
data[best_features] = scaler.fit_transform(data[best_features])

# Učitavanje sačuvanih najboljih parametara
best_params = joblib.load('best_params_xgboost_random_search.pkl')

# Kreiranje modela sa najboljim parametrima
best_model = xgb.XGBRegressor(**best_params, random_state=42)

# K-fold cross-validation sa najboljim modelom
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []
y_test_all_optimized = []
y_pred_all_optimized = []
test_indices = []

for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index][best_features], data.iloc[test_index][best_features]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    y_test_all_optimized.extend(y_test)
    y_pred_all_optimized.extend(y_pred)
    test_indices.extend(test_index)
    
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Prikazivanje rezultata sa optimizovanim modelom
print(f'Mean MAE with optimized XGBoost: {np.mean(mae_scores)}')
print(f'Mean MSE with optimized XGBoost: {np.mean(mse_scores)}')

average_actual_price = np.mean(y_test_all_optimized)

# Izračunavanje MAE u procentima
mae_percentage = (np.mean(mae_scores) / average_actual_price) * 100
print(f'Mean MAE as percentage of average actual price: {mae_percentage:.2f}%')

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

# Vizualizacija predikcija i stvarnih vrednosti sa optimizovanim modelom
plt.figure(figsize=(10, 6))
plt.scatter(y_test_all_optimized, y_pred_all_optimized, alpha=0.3)
plt.plot([min(y_test_all_optimized), max(y_test_all_optimized)], [min(y_test_all_optimized), max(y_test_all_optimized)], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices with Optimized XGBoost')
plt.show()

# Distribucija grešaka sa optimizovanim modelom
errors_optimized = np.array(y_test_all_optimized) - np.array(y_pred_all_optimized)
plt.figure(figsize=(10, 6))
sns.histplot(errors_optimized, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors with Optimized XGBoost')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = data[best_features + ['cena']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Save predictions to CSV (poravnato sa indeksima KFold podela nad očišćenim podacima)
ids_aligned_indexed = ids_aligned.iloc[test_indices].reset_index(drop=True)
predictions_df = pd.DataFrame({
    'id': ids_aligned_indexed,
    'actual_prices': pd.Series(y_test_all_optimized).reset_index(drop=True),
    'predicted_prices': pd.Series(y_pred_all_optimized).reset_index(drop=True)
})
predictions_df.to_csv('xgb_predictions.csv', index=False)

# Izračunavanje grešaka
predictions_df['error_percentage'] = (np.abs(predictions_df['actual_prices'] - predictions_df['predicted_prices']) / predictions_df['actual_prices']) * 100

# Identifikacija stanova sa greškom većom od 50%
large_errors = predictions_df[predictions_df['error_percentage'] > 100]

# Print outliers
print("Stanovi sa greškom većom od 98%:")
print(large_errors)

# Visualize all data points and outliers on a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(predictions_df['actual_prices'], predictions_df['predicted_prices'], alpha=0.3, label='All Data', color='blue')
plt.scatter(large_errors['actual_prices'], large_errors['predicted_prices'], color='r', label='Error > 98%')
plt.plot([min(predictions_df['actual_prices']), max(predictions_df['actual_prices'])], 
         [min(predictions_df['actual_prices']), max(predictions_df['actual_prices'])], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices with XGBoost')
plt.legend()
plt.show()

# Izračunavanje apsolutne greške
predictions_df['absolute_error'] = np.abs(predictions_df['actual_prices'] - predictions_df['predicted_prices'])

# Definisanje praga za udaljenost od crvene linije (možeš prilagoditi ovaj prag)
error_threshold = 300000  # Primer praga, prilagodi prema potrebi

# Identifikacija stanova sa apsolutnom greškom većom od praga
large_errors = predictions_df[predictions_df['absolute_error'] > error_threshold]

# Print outliers
print("Stanovi sa apsolutnom greškom većom od praga:")
print(large_errors)

# Visualize all data points and outliers on a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(predictions_df['actual_prices'], predictions_df['predicted_prices'], alpha=0.3, label='All Data', color='blue')
plt.scatter(large_errors['actual_prices'], large_errors['predicted_prices'], color='r', label=f'Error > {error_threshold}')
plt.plot([min(predictions_df['actual_prices']), max(predictions_df['actual_prices'])], 
         [min(predictions_df['actual_prices']), max(predictions_df['actual_prices'])], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices with XGBoost')
plt.legend()
plt.show()
