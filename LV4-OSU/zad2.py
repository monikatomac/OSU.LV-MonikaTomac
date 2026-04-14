import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv("data_C02_emission.csv")

features_num = ["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)"]
features_cat = ["Fuel Type"]

df_encoded = pd.get_dummies(df, columns=features_cat)

X = df_encoded.drop("CO2 Emissions (g/km)", axis=1)
y = df["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)

errors = np.abs(y_test - y_pred)
max_error = np.max(errors)

print("Maksimalna pogreška:", max_error)

max_error_index = errors.idxmax()

print("Redak s najvećom pogreškom:")
print(df.loc[max_error_index])