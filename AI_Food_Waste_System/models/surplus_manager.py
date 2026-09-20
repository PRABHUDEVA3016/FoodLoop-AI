def check_surplus(meals_prepared, predicted_consumption):
    """
    Calculate possible surplus food.
    """

    surplus = meals_prepared - predicted_consumption

    print()
    print("🍱 SURPLUS FOOD MANAGEMENT")
    print("----------------------------")

    print("Meals prepared:", meals_prepared)
    print("Predicted consumption:", predicted_consumption)

    if surplus > 0:
        print("⚠️ Surplus detected:", surplus, "meals")
        print()
        print("Next action: Check food condition")
        
    elif surplus == 0:
        print("✅ Production matches predicted demand")
        print("No surplus detected.")

    else:
        shortage = abs(surplus)
        print("⚠️ Possible shortage:", shortage, "meals")
        print("Next action: Consider additional production")

    return surplus


# Test the system
if __name__ == "__main__":

    meals_prepared = 950
    predicted_consumption = 872

    surplus = check_surplus(
        meals_prepared,
        predicted_consumption
    )