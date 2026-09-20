import pandas as pd

# Load the food demand dataset
data = pd.read_csv("data/food_demand.csv")

print("Dataset loaded successfully!")
print()

# Display the dataset
print(data)

print()
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

print()
print("Column names:")
print(data.columns.tolist())