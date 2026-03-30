import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#a)
data = pd.read_csv("data_C02_emission.csv")

features = ["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]

X = data[features]
y = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#b)
plt.scatter(X_train["Engine Size (L)"], y_train, color='blue', label='Train')
plt.scatter(X_test["Engine Size (L)"], y_test, color='red', label='Test')

plt.xlabel("Engine size")
plt.ylabel("CO2 emissions")
plt.legend()
plt.show()

#c)
sc = StandardScaler()

plt.hist(X_train["Engine Size (L)"], bins=30)
plt.title("Prije skliranja")
plt.show()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

plt.hist(X_train_scaled[:,0], bins=30)
plt.title("Nakon skaliranja")
plt.show()


#d)
audi_data = data[data['Make'] == 'Audi']
print(f"Broj mjerenja za Audi: {len(audi_data)}") 
audi_4_cyl = audi_data[audi_data['Cylinders'] == 4]
print(f"Prosječna CO2 emisija Audi (4 cilindra): {audi_4_cyl['CO2 Emissions (g/km)'].mean():.2f} g/km") 

#e)
print("Broj vozila po broju cilindara:")
print(data.groupby('Cylinders').size()) 
print("Prosječna emisija CO2 s obzirom na broj cilindara:")
print(data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean())

#f) 
fuel_stats = data[data['Fuel Type'].isin(['D', 'X'])].groupby('Fuel Type')['Fuel Consumption City (L/100km)'].agg(['mean', 'median'])
print("Statistika potrošnje za Dizel (D) i Regularni benzin (X):")
print(fuel_stats)

#g)
max_diesel_4 = data[(df['Fuel Type'] == 'D') & (data['Cylinders'] == 4)].sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(1)
print("Vozilo s 4 cilindra na dizel i najvećom gradskom potrošnjom:")
print(max_diesel_4[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#h)
manual_vehicles = data[data['Transmission'].str.startswith('M')]
print(f"Broj vozila s ručnim mjenjačem: {len(manual_vehicles)}")

#i)
print("Matrica korelacije:")
print(data.corr(numeric_only=True))
