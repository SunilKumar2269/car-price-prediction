import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

print(df.head())
df["car_age"] = 2024 - df["year"]
df.drop("year", axis=1, inplace=True)
df = df[df["selling_price"] < 2000000]

# Drop car name column
df = df.drop('name', axis=1)

# Convert categorical columns
le = LabelEncoder()

df['fuel'] = le.fit_transform(df['fuel'])
df['seller_type'] = le.fit_transform(df['seller_type'])
df['transmission'] = le.fit_transform(df['transmission'])
df['owner'] = le.fit_transform(df['owner'])

# Features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('car_price_model.pkl', 'wb'))

print("Model trained successfully")


# Predict on test data
y_pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Evaluate model
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance for Car Price Prediction")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
