import joblib

# Load trained AI model
model = joblib.load("models/food_demand_model.pkl")

print("🍱 SMART FOOD DEMAND PREDICTOR")
print("--------------------------------")

# Day mapping
day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

# Get user input
day = input("Enter day (Monday-Sunday): ").strip().capitalize()

if day not in day_mapping:
    print("❌ Invalid day entered.")
    print("Please enter a day from Monday to Sunday.")
    exit()

students = int(input("Enter expected number of students: "))
meals_prepared = int(input("Enter planned number of meals: "))

holiday = int(input("Is it a holiday? (1 = Yes, 0 = No): "))

# Convert day to number
day_number = day_mapping[day]

# Make prediction
prediction = model.predict([
    [day_number, students, meals_prepared, holiday]
])

predicted_meals = round(prediction[0])

# Calculate possible surplus
surplus = meals_prepared - predicted_meals

print()
print("🤖 AI PREDICTION")
print("----------------------------")
print("Day:", day)
print("Expected students:", students)
print("Planned meals:", meals_prepared)
print("Holiday:", "Yes" if holiday == 1 else "No")
print("Predicted meals consumed:", predicted_meals)

if surplus > 0:
    print("⚠️ Possible surplus:", surplus, "meals")
elif surplus == 0:
    print("✅ Production matches predicted demand.")
else:
    print("⚠️ Possible shortage:", abs(surplus), "meals")