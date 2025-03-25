import pandas as pd
import numpy as np
import joblib  # Use joblib instead of pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
file_path = "house_price_dataset.csv"  # Ensure this file exists in the same directory
df = pd.read_csv(file_path)

# Check if the dataset loaded correctly
print("Dataset Columns:", df.columns)

# Identify the correct target column
target_column = None
for col in df.columns:
    if "price" in col.lower():  # Automatically detect the correct "Price" column
        target_column = col
        break

if target_column is None:
    raise ValueError("Error: No column found with 'price' in its name. Check your dataset!")

# Define Features (X) and Target (y)
X = df.drop(columns=[target_column])  # Drop target column
y = df[target_column]

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLR model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate R² Score (Model Accuracy)
r2 = r2_score(y_test, y_pred)
print(f"Model R² Score: {r2:.4f}")

# Save trained model using joblib
joblib.dump(model, "house_price_model.joblib")

# Save accuracy score
with open("accuracy_score.txt", "w") as file:
    file.write(f"Model R² Score: {r2:.4f}")

print("\n✅ Model trained and saved successfully!")
print(f"📊 Accuracy (R² Score): {r2:.4f}")
