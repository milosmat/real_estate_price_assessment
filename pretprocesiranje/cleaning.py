import pandas as pd

# Učitavanje podataka iz CSV datoteke
data = pd.read_csv('cleaned_property_data_no_outliers.csv')

# Definisanje tolerancije za cene blizu nule (npr. ±100 EUR)
tolerance = 15000

# Filtriranje stanova gde je cena 0 ili blizu 0
near_zero_price_properties = data[data['cena'].abs() <= tolerance]

# Prikazivanje stanova u terminalu
print(near_zero_price_properties)

# Uklanjanje stanova sa cenom 0 ili blizu 0 iz originalnog dataframe-a
data_without_near_zero_prices = data[data['cena'].abs() > tolerance]

# Sačuvavanje prečišćenih podataka u novu CSV datoteku
data_without_near_zero_prices.to_csv('cleaned_property_data_no_outliers.csv', index=False)

print("Data without near zero price properties has been saved to 'cleaned_property_data_no_outliers.csv'")
