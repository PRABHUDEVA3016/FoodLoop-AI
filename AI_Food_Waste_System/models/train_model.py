import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("data/food_demand.csv")

# Convert day into numbers
day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

data["day_number"] = data["day"].map(day_mapping)

# Input features
X = data[
    [
        "day_number",
        "students_expected",
        "meals_prepared",
        "holiday"
    ]
]

# Target
y = data["meals_consumed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

# Calculate error
error = mean_absolute_error(y_test, predictions)

print("AI model trained successfully!")
print("Mean Absolute Error:", round(error, 2))

# Save model
joblib.dump(model, "models/food_demand_model.pkl")

print("Improved model saved successfully!")