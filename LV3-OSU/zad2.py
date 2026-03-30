import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

#a)
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=30, edgecolor='black')
plt.title('Raspodjela emisije CO2')
plt.xlabel('CO2 Emissions (g/km)')
plt.show()

#b)
data['Fuel Type'] = data['Fuel Type'].astype('category')
plt.figure()
plt.scatter(data['Fuel Consumption City (L/100km)'], 
            data['CO2 Emissions (g/km)'], 
            c=data['Fuel Type'].cat.codes, 
            cmap='viridis', 
            alpha=0.6)
plt.title('Odnos gradske potrošnje i emisije CO2')
plt.xlabel('Gradska potrošnja (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

#c)
plt.figure()
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.title('Izvangradska potrošnja s obzirom na tip goriva')
plt.suptitle('')
plt.show()

#d)
plt.figure()
data.groupby('Fuel Type').size().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Broj vozila po tipu goriva')
plt.ylabel('Broj vozila')
plt.xlabel('Tip goriva')
plt.show()

#e)
plt.figure()
data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Prosječna CO2 emisija po broju cilindara')
plt.ylabel('Prosječna CO2 emisija (g/km)')
plt.xlabel('Broj cilindara')
plt.show()