import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

print(df.head())

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
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('car_price_model.pkl', 'wb'))

print("Model trained successfully")