import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONSUMABILITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "consumability_pipeline.joblib"
)

QUANTITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "quantity_pipeline.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

MODELS_LOADED = False
MODEL_ERROR = None

consumability_model = None
quantity_model = None


try:

    consumability_model = joblib.load(
        CONSUMABILITY_MODEL_PATH
    )

    quantity_model = joblib.load(
        QUANTITY_MODEL_PATH
    )

    MODELS_LOADED = True

except Exception as e:

    MODEL_ERROR = str(e)
    MODELS_LOADED = False


# ============================================================
# HELPER — GET FEATURES EXPECTED BY A PIPELINE
# ============================================================

def get_expected_features(model):

    """
    Get the feature names expected by the saved sklearn pipeline.
    """

    if hasattr(model, "feature_names_in_"):

        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):

                return list(step.feature_names_in_)

    return None


# ============================================================
# FSSAI REQUIREMENT CALCULATIONS
# ============================================================

def calculate_fssai_flags(
    storage_mode,
    storage_temperature_c,
    food_age_hours,
    cooking_core_temperature_c,
    cooking_time_minutes,
    cooling_60_to_21_hours,
    cooling_21_to_5_hours,
    reheated,
    reheating_core_temperature_c,
    reheating_time_minutes
):

    """
    Calculate the four FSSAI-related boolean features
    used by the consumability model.

    Foods in this project:
        Rice
        Roti
        Dal

    All three are vegetarian foods.
    """

    # --------------------------------------------------------
    # 1. COOKING REQUIREMENT
    #
    # Vegetarian food:
    #   60°C for 10 minutes
    #       OR
    #   65°C for 2 minutes
    # --------------------------------------------------------

    cooking_requirement_met = (
        (
            float(cooking_core_temperature_c) >= 60
            and
            float(cooking_time_minutes) >= 10
        )
        or
        (
            float(cooking_core_temperature_c) >= 65
            and
            float(cooking_time_minutes) >= 2
        )
    )


    # --------------------------------------------------------
    # 2. STORAGE REQUIREMENT
    #
    # Hot food:
    #   65°C or above
    #
    # Cold/refrigerated food:
    #   5°C or below
    #
    # Room/ambient food:
    #   kept within the food-safety time limit.
    # --------------------------------------------------------

    storage_mode_normalized = str(
        storage_mode
    ).strip().lower()

    storage_temperature = float(
        storage_temperature_c
    )

    food_age = float(
        food_age_hours
    )

    if (
        "refriger" in storage_mode_normalized
        or
        "cold" in storage_mode_normalized
    ):

        storage_requirement_met = (
            storage_temperature <= 5
        )

    elif (
        "hot" in storage_mode_normalized
        or
        "holding" in storage_mode_normalized
    ):

        storage_requirement_met = (
            storage_temperature >= 65
        )

    else:

        # Ambient / room-temperature handling.
        # FSSAI guidance limits ready-to-eat food
        # in the 5°C–60°C danger zone to a short period.
        storage_requirement_met = (
            food_age <= 4
        )


    # --------------------------------------------------------
    # 3. COOLING REQUIREMENT
    #
    # 60°C → 21°C:
    #   maximum 2 hours
    #
    # 21°C → 5°C:
    #   further maximum 4 hours
    # --------------------------------------------------------

    cooling_requirement_met = (
        float(cooling_60_to_21_hours) <= 2
        and
        float(cooling_21_to_5_hours) <= 4
    )


    # --------------------------------------------------------
    # 4. REHEATING REQUIREMENT
    #
    # If food was not reheated:
    #   requirement is treated as satisfied / not applicable.
    #
    # If reheated:
    #   core temperature >= 75°C
    #   and at least 2 minutes.
    # --------------------------------------------------------

    if not bool(reheated):

        reheating_requirement_met = True

    else:

        reheating_requirement_met = (
            float(reheating_core_temperature_c) >= 75
            and
            float(reheating_time_minutes) >= 2
        )


    return (
        bool(cooking_requirement_met),
        bool(storage_requirement_met),
        bool(cooling_requirement_met),
        bool(reheating_requirement_met)
    )


# ============================================================
# BUILD CONSUMABILITY INPUT
# ============================================================

def build_consumability_input(
    food_name,
    initial_quantity_kg,
    storage_mode,
    storage_temperature_c,
    food_age_hours,
    cooking_core_temperature_c,
    cooking_time_minutes,
    cooling_60_to_21_hours,
    cooling_21_to_5_hours,
    reheated,
    reheating_core_temperature_c,
    reheating_time_minutes
):

    """
    Build the exact 16-feature input required by the
    corrected consumability model.
    """

    # --------------------------------------------------------
    # Calculate FSSAI flags
    # --------------------------------------------------------

    (
        fssai_cooking_requirement_met,
        fssai_storage_requirement_met,
        fssai_cooling_requirement_met,
        fssai_reheating_requirement_met
    ) = calculate_fssai_flags(

        storage_mode=storage_mode,

        storage_temperature_c=
            storage_temperature_c,

        food_age_hours=
            food_age_hours,

        cooking_core_temperature_c=
            cooking_core_temperature_c,

        cooking_time_minutes=
            cooking_time_minutes,

        cooling_60_to_21_hours=
            cooling_60_to_21_hours,

        cooling_21_to_5_hours=
            cooling_21_to_5_hours,

        reheated=
            reheated,

        reheating_core_temperature_c=
            reheating_core_temperature_c,

        reheating_time_minutes=
            reheating_time_minutes
    )


    # --------------------------------------------------------
    # Create complete 16-feature DataFrame
    # --------------------------------------------------------

    data = {

        "food_name":
            food_name,

        "initial_quantity_kg":
            float(initial_quantity_kg),

        "storage_mode":
            storage_mode,

        "storage_temperature_c":
            float(storage_temperature_c),

        "food_age_hours":
            float(food_age_hours),

        "cooking_core_temperature_c":
            float(cooking_core_temperature_c),

        "cooking_time_minutes":
            float(cooking_time_minutes),

        "cooling_60_to_21_hours":
            float(cooling_60_to_21_hours),

        "cooling_21_to_5_hours":
            float(cooling_21_to_5_hours),

        "reheated":
            reheated,

        "reheating_core_temperature_c":
         (
           float(reheating_core_temperature_c)
           if reheating_core_temperature_c is not None
           else None
         ),

        "reheating_time_minutes":
         (
           float(reheating_time_minutes)
           if reheating_time_minutes is not None
           else None
         ),

        "fssai_cooking_requirement_met":
            fssai_cooking_requirement_met,

        "fssai_storage_requirement_met":
            fssai_storage_requirement_met,

        "fssai_cooling_requirement_met":
            fssai_cooling_requirement_met,

        "fssai_reheating_requirement_met":
            fssai_reheating_requirement_met
    }


    return pd.DataFrame([data])


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_food_quality(
    food_name,
    storage_mode,
    storage_temperature_c,
    food_age_hours,
    cooking_core_temperature_c,
    cooking_time_minutes,
    cooling_60_to_21_hours,
    cooling_21_to_5_hours,
    reheated,
    reheating_core_temperature_c,
    reheating_time_minutes,
    initial_quantity_kg
):

    """
    Complete food-quality prediction.

    Stage 1:
        Consumability classification

    Stage 2:
        Consumable quantity regression

    Returns:
        status
        consumable_quantity_kg
        initial_quantity_kg
    """

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not MODELS_LOADED:

        raise RuntimeError(
            f"Food quality models could not be loaded: "
            f"{MODEL_ERROR}"
        )


    # ========================================================
    # STAGE 1 — CONSUMABILITY CLASSIFICATION
    # ========================================================

    consumability_input = build_consumability_input(

        food_name=food_name,

        initial_quantity_kg=
            initial_quantity_kg,

        storage_mode=storage_mode,

        storage_temperature_c=
            storage_temperature_c,

        food_age_hours=
            food_age_hours,

        cooking_core_temperature_c=
            cooking_core_temperature_c,

        cooking_time_minutes=
            cooking_time_minutes,

        cooling_60_to_21_hours=
            cooling_60_to_21_hours,

        cooling_21_to_5_hours=
            cooling_21_to_5_hours,

        reheated=
            reheated,

        reheating_core_temperature_c=
            reheating_core_temperature_c,

        reheating_time_minutes=
            reheating_time_minutes
    )


    # --------------------------------------------------------
    # Match exact feature order expected by model
    # --------------------------------------------------------

    consumability_features = (
        get_expected_features(
            consumability_model
        )
    )


    if consumability_features is not None:

        missing_features = [
            feature
            for feature in consumability_features
            if feature not in consumability_input.columns
        ]

        if missing_features:

            raise ValueError(
                "Consumability model requires columns "
                f"that were not provided: {missing_features}"
            )

        consumability_input = (
            consumability_input[
                consumability_features
            ]
        )


    # --------------------------------------------------------
    # Predict consumability
    # --------------------------------------------------------

    status_prediction = (
        consumability_model.predict(
            consumability_input
        )
    )


    predicted_status = str(
        status_prediction[0]
    )


    # ========================================================
    # NORMALIZE STATUS
    # ========================================================

    predicted_status = (
        predicted_status
        .strip()
        .upper()
    )


    if predicted_status in [
        "1",
        "TRUE",
        "CONSUMABLE"
    ]:

        predicted_status = "CONSUMABLE"

    elif predicted_status in [
        "0",
        "FALSE",
        "NOT_CONSUMABLE"
    ]:

        predicted_status = "NOT_CONSUMABLE"


    # ========================================================
    # STAGE 2 — QUANTITY PREDICTION
    # ========================================================

    quantity_input = pd.DataFrame([
        {
            "food_name":
                food_name,

            "initial_quantity_kg":
                float(initial_quantity_kg),

            "consumable_status":
                predicted_status
        }
    ])


    # --------------------------------------------------------
    # Match exact quantity-model feature order
    # --------------------------------------------------------

    quantity_features = (
        get_expected_features(
            quantity_model
        )
    )


    if quantity_features is not None:

        missing_quantity_features = [
            feature
            for feature in quantity_features
            if feature not in quantity_input.columns
        ]

        if missing_quantity_features:

            raise ValueError(
                "Quantity model requires columns "
                f"that were not provided: "
                f"{missing_quantity_features}"
            )

        quantity_input = (
            quantity_input[
                quantity_features
            ]
        )


    # --------------------------------------------------------
    # Predict quantity
    # --------------------------------------------------------

    quantity_prediction = (
        quantity_model.predict(
            quantity_input
        )
    )


    estimated_quantity = float(
        quantity_prediction[0]
    )


    # ========================================================
    # PHYSICAL CONSISTENCY CHECK
    # ========================================================

    estimated_quantity = max(
        0.0,
        estimated_quantity
    )

    estimated_quantity = min(
        estimated_quantity,
        float(initial_quantity_kg)
    )


    # --------------------------------------------------------
    # NOT CONSUMABLE = ZERO REDISTRIBUTABLE QUANTITY
    # --------------------------------------------------------

    if predicted_status == "NOT_CONSUMABLE":

        estimated_quantity = 0.0


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "status":
            predicted_status,

        "consumable_quantity_kg":
            estimated_quantity,

        "initial_quantity_kg":
            float(initial_quantity_kg)
    }