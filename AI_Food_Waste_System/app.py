import os
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_SDK_AVAILABLE = True
except ImportError:
    firebase_admin = None
    credentials = None
    db = None
    FIREBASE_SDK_AVAILABLE = False

from models.food_quality import predict_food_quality, MODELS_LOADED


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="FoodLoop AI",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOGIN / ROLE SESSION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None


# ============================================================
# LOGIN / ROLE SELECTION PAGE
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none; }
        .login-container { max-width: 900px; margin: 0 auto; padding-top: 4rem; }
        .login-title { text-align: center; font-size: 2.4rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem; }
        .login-subtitle { text-align: center; color: #6b7280; font-size: 1rem; margin-bottom: 2.5rem; }
        .role-card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 2rem; min-height: 245px; text-align: center; }
        .role-icon { font-size: 2.5rem; margin-bottom: 1rem; }
        .role-title { font-size: 1.25rem; font-weight: 650; color: #111827; margin-bottom: 0.6rem; }
        .role-description { color: #6b7280; font-size: 0.9rem; line-height: 1.5; margin-bottom: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🍱 FoodLoop AI</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="login-subtitle">Smart Food Waste Management & Redistribution Platform</div>""",
        unsafe_allow_html=True
    )
    st.markdown(
        """<h3 style="text-align:center; color:#111827;">Select your role</h3>""",
        unsafe_allow_html=True
    )
    st.write("")

    provider_col, receiver_col = st.columns(2)

    with provider_col:
        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">🍱</div>
                <div class="role-title">Food Provider / Donor</div>
                <div class="role-description">
                    For restaurants, hotels, institutions, kitchens and organizations
                    that have surplus food available for donation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(
            "Enter as Food Provider",
            use_container_width=True,
            type="primary",
            key="provider_login"
        ):
            st.session_state.logged_in = True
            st.session_state.user_role = "provider"
            st.session_state.user_id = "provider_demo_001"
            st.rerun()

    with receiver_col:
        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">🤝</div>
                <div class="role-title">Food Receiver / NGO</div>
                <div class="role-description">
                    For NGOs, charitable organizations, community groups and others
                    who are in need of surplus food.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(
            "Enter as Food Receiver",
            use_container_width=True,
            key="receiver_login"
        ):
            st.session_state.logged_in = True
            st.session_state.user_role = "receiver"
            st.session_state.user_id = "receiver_demo_001"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        background-color: #f8fafb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    /* Sidebar brand */
    .sidebar-brand {
        padding: 0.5rem 0 1.5rem 0;
    }

    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.15rem;
    }

    .sidebar-brand-subtitle {
        font-size: 0.78rem;
        color: #6b7280;
    }

    /* Page headings */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Section heading */
    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Dashboard cards */
    .dashboard-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.1rem;
        min-height: 115px;
    }

    .card-label {
        color: #6b7280;
        font-size: 0.82rem;
        margin-bottom: 0.45rem;
    }

    .card-value {
        color: #111827;
        font-size: 1.55rem;
        font-weight: 700;
    }

    .card-description {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    /* Info boxes */
    .info-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.5rem;
    }

    /* Remove excessive top spacing */
    div.block-container {
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEMAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "food_demand_model.pkl"
)


# ============================================================
# LOAD DEMAND FORECASTING MODEL
# ============================================================

@st.cache_resource
def load_demand_model():
    return joblib.load(DEMAND_MODEL_PATH)


try:
    demand_model = load_demand_model()
    demand_model_loaded = True
    demand_model_error = None

except Exception as e:
    demand_model = None
    demand_model_loaded = False
    demand_model_error = str(e)


# ============================================================
# SESSION STATE
# ============================================================

if "demand_result" not in st.session_state:
    st.session_state.demand_result = None

if "food_quality_result" not in st.session_state:
    st.session_state.food_quality_result = None


if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "provider_profile" not in st.session_state:
    st.session_state.provider_profile = {
        "organization_name": "Food Provider",
        "location": ""
    }

# Receiver-side state
if "receiver_profile" not in st.session_state:
    st.session_state.receiver_profile = {
        "organization_name": "",
        "people_required": 0,
        "preferred_food": "Any",
        "required_quantity_kg": 0.0
    }

if "receiver_requests" not in st.session_state:
    st.session_state.receiver_requests = []

# Temporary demo listings for the receiver interface.
# These will later be replaced by real provider/backend data.
if "demo_food_listings" not in st.session_state:
    st.session_state.demo_food_listings = [
        {
            "id": "FL-DEMO-001",
            "food": "Rice",
            "quantity_kg": 25.0,
            "provider": "Food Provider",
            "pickup_window": "Today • 6:00 PM – 8:00 PM"
        },
        {
            "id": "FL-DEMO-002",
            "food": "Roti",
            "quantity_kg": 10.0,
            "provider": "Food Provider",
            "pickup_window": "Today • 7:00 PM – 9:00 PM"
        },
        {
            "id": "FL-DEMO-003",
            "food": "Dal",
            "quantity_kg": 15.0,
            "provider": "Food Provider",
            "pickup_window": "Today • 6:30 PM – 8:30 PM"
        }
    ]


# ============================================================

# ============================================================
# FIREBASE / SHARED BACKEND
# ============================================================

@st.cache_resource
def initialize_firebase():
    """Initialize Firebase Admin SDK once per Streamlit process."""
    if not FIREBASE_SDK_AVAILABLE:
        return None, (
            "Firebase Admin SDK is not installed. "
            "Run: pip install firebase-admin"
        )

    if "firebase" not in st.secrets:
        return None, (
            "Firebase secrets are not configured. "
            "Add a [firebase] section to Streamlit secrets."
        )

    try:
        firebase_config = dict(st.secrets["firebase"])
        database_url = firebase_config.pop("database_url", None)

        if not database_url:
            return None, "Firebase database_url is missing from secrets."

        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(
                cred,
                {"databaseURL": database_url}
            )

        return db.reference("/"), None

    except Exception as e:
        return None, f"Firebase initialization error: {e}"


def firebase_ready():
    root, error = initialize_firebase()
    return root is not None, error


def firebase_get(path):
    root, error = initialize_firebase()
    if root is None:
        raise RuntimeError(error)
    return db.reference(path).get()


def firebase_push(path, data):
    root, error = initialize_firebase()
    if root is None:
        raise RuntimeError(error)
    return db.reference(path).push(data).key


def firebase_update(path, data):
    root, error = initialize_firebase()
    if root is None:
        raise RuntimeError(error)
    db.reference(path).update(data)


def firebase_delete(path):
    root, error = initialize_firebase()
    if root is None:
        raise RuntimeError(error)
    db.reference(path).delete()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def create_notification(user_id, title, message, notification_type="info"):
    """Create a notification for a provider or receiver."""
    if not user_id:
        return

    firebase_push(
        f"foodloop/notifications/{user_id}",
        {
            "title": title,
            "message": message,
            "type": notification_type,
            "created_at": now_iso(),
            "read": False
        }
    )


def get_food_listings():
    data = firebase_get("foodloop/food_listings") or {}
    if isinstance(data, dict):
        return [
            {"id": listing_id, **listing}
            for listing_id, listing in data.items()
            if isinstance(listing, dict)
        ]
    return []


def get_food_requests():
    data = firebase_get("foodloop/food_requests") or {}
    if isinstance(data, dict):
        return [
            {"id": request_id, **request}
            for request_id, request in data.items()
            if isinstance(request, dict)
        ]
    return []


def get_notifications(user_id):
    data = firebase_get(f"foodloop/notifications/{user_id}") or {}
    if isinstance(data, dict):
        return [
            {"id": notification_id, **notification}
            for notification_id, notification in data.items()
            if isinstance(notification, dict)
        ]
    return []


def format_request_status(status):
    if status == "pending":
        return "Pending"
    if status == "approved":
        return "Approved"
    if status == "rejected":
        return "Rejected"
    if status == "cancelled":
        return "Cancelled"
    return status.replace("_", " ").title()


def provider_matches_request(request, provider_id):
    """
    A request belongs to a provider when:
    - it was raised against one of that provider's listings, or
    - it is a general request without a provider/listing.
    """
    request_provider = request.get("provider_id")
    return request_provider in (None, "", provider_id)


def render_firebase_setup_warning(error):
    st.warning(
        "The shared FoodLoop database is not connected yet."
    )
    st.code(error, language="text")
    st.info(
        "Install firebase-admin and configure the Firebase service-account "
        "credentials plus Realtime Database URL in Streamlit secrets."
    )


def render_live_refresh_hint():
    st.caption(
        "Live data is read from Firebase. This page refreshes automatically "
        "when Streamlit fragment refresh is available; otherwise use Refresh."
    )


def refresh_button(key):
    if st.button("↻ Refresh", key=key):
        st.rerun()


# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🍱 FoodLoop AI</div>
            <div class="sidebar-brand-subtitle">Smart Food Waste Management</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if st.session_state.user_role == "provider":
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Demand Prediction",
                "Food Quality AI",
                "Food Requests",
                "My Donations",
                "About"
            ],
            label_visibility="collapsed"
        )
    else:
        page = st.radio(
            "Navigation",
            [
                "Receiver Dashboard",
                "Live Food",
                "Create Request",
                "My Requests",
                "About"
            ],
            label_visibility="collapsed"
        )

    st.divider()

    st.caption(
        "Logged in as: " +
        ("Food Provider / Donor" if st.session_state.user_role == "provider" else "Food Receiver / NGO")
    )

    if st.button("Log out", use_container_width=True, key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.rerun()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def show_page_header(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if st.session_state.user_role == "provider" and page == "Dashboard":

    show_page_header(
        "Dashboard",
        "Overview of your food demand and food-quality analysis."
    )

    demand_result = st.session_state.demand_result
    food_quality_result = st.session_state.food_quality_result


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    if demand_result is not None:

        predicted_consumption = demand_result["predicted_consumption"]
        meals_prepared = demand_result["meals_prepared"]
        surplus = demand_result["surplus"]

        if surplus >= 0:
            surplus_display = f"{surplus} meals"
        else:
            surplus_display = f"{abs(surplus)} meals shortage"

    else:

        predicted_consumption = 0
        meals_prepared = 0
        surplus_display = "Not available"


    if food_quality_result is not None:

        food_status = food_quality_result["status"]
        estimated_quantity = food_quality_result[
            "estimated_quantity_kg"
        ]
        estimated_people = food_quality_result[
            "estimated_people"
        ]

    else:

        food_status = "Not analyzed"
        estimated_quantity = 0
        estimated_people = 0


    card1, card2, card3, card4 = st.columns(4)


    with card1:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-label">Predicted Consumption</div>
                <div class="card-value">
                    {predicted_consumption}
                </div>
                <div class="card-description">
                    meals
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with card2:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-label">Meals Prepared</div>
                <div class="card-value">
                    {meals_prepared}
                </div>
                <div class="card-description">
                    prepared meals
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with card3:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-label">Food Status</div>
                <div class="card-value">
                    {food_status}
                </div>
                <div class="card-description">
                    latest food-quality result
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with card4:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-label">Consumable Food</div>
                <div class="card-value">
                    {estimated_quantity:.2f} kg
                </div>
                <div class="card-description">
                    estimated remaining quantity
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # CURRENT STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Current Status</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Latest prediction information from the system.'
        '</div>',
        unsafe_allow_html=True
    )


    status_col1, status_col2 = st.columns(2)


    with status_col1:

        if demand_result is not None:

            if demand_result["surplus"] > 0:

                st.success(
                    f"♻️ Surplus detected: "
                    f"{demand_result['surplus']} meals."
                )

            elif demand_result["surplus"] == 0:

                st.success(
                    "✅ Prepared meals match predicted demand."
                )

            else:

                st.warning(
                    f"⚠️ Potential shortage: "
                    f"{abs(demand_result['surplus'])} meals."
                )

        else:

            st.info(
                "No demand prediction has been generated yet."
            )


    with status_col2:

        if food_quality_result is not None:

            if food_quality_result["status"] == "CONSUMABLE":

                st.success(
                    f"♻️ {food_quality_result['food_name']} "
                    f"is classified as consumable."
                )

            else:

                st.warning(
                    f"⚠️ {food_quality_result['food_name']} "
                    f"is classified as NOT CONSUMABLE."
                )

        else:

            st.info(
                "No food-quality analysis has been generated yet."
            )


    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Quick Actions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Use the navigation panel to run an analysis.'
        '</div>',
        unsafe_allow_html=True
    )


    quick1, quick2 = st.columns(2)


    with quick1:

        st.markdown(
            """
            <div class="info-card">
                <h4>📊 Demand Prediction</h4>
                <p>
                    Predict expected meal consumption and identify
                    possible surplus or shortage.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with quick2:

        st.markdown(
            """
            <div class="info-card">
                <h4>🔬 Food Quality AI</h4>
                <p>
                    Analyze cooked food conditions and estimate
                    consumability and remaining quantity.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE 2 — DEMAND PREDICTION
# ============================================================

elif st.session_state.user_role == "provider" and page == "Demand Prediction":

    show_page_header(
        "Meal Demand & Surplus Prediction",
        "Predict expected meal consumption and identify possible surplus or shortage."
    )


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    if not demand_model_loaded:

        st.error(
            "Demand forecasting model could not be loaded."
        )

        st.code(
            demand_model_error,
            language="text"
        )

    else:

        st.markdown(
            '<div class="section-title">Input Parameters</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Enter the expected student count and meal preparation details.'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            students_expected = st.number_input(
                "👨‍🎓 Expected Students",
                min_value=0,
                value=900,
                step=10,
                key="students_expected"
            )


        with col2:

            meals_prepared = st.number_input(
                "🍱 Meals Prepared",
                min_value=0,
                value=950,
                step=10,
                key="meals_prepared"
            )


        col3, col4 = st.columns(2)


        with col3:

            day = st.selectbox(
                "📅 Day",
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ],
                key="demand_day"
            )


        with col4:

            holiday = st.selectbox(
                "🏖️ Holiday",
                ["No", "Yes"],
                key="demand_holiday"
            )


        # ----------------------------------------------------
        # CONVERT INPUTS
        # ----------------------------------------------------

        day_number = {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 7
        }[day]

        holiday_number = 1 if holiday == "Yes" else 0


        st.markdown("<br>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # PREDICTION BUTTON
        # ----------------------------------------------------

        if st.button(
            "🤖 Predict Meal Demand",
            use_container_width=True,
            type="primary"
        ):

            input_data = pd.DataFrame(
                [
                    {
                        "day_number": day_number,
                        "students_expected": students_expected,
                        "meals_prepared": meals_prepared,
                        "holiday": holiday_number
                    }
                ]
            )

            try:

                prediction = demand_model.predict(
                    input_data
                )

                predicted_consumption = round(
                    float(prediction[0])
                )

                surplus = (
                    int(meals_prepared)
                    - predicted_consumption
                )


                # Save result
                st.session_state.demand_result = {

                    "predicted_consumption":
                        predicted_consumption,

                    "surplus":
                        surplus,

                    "students_expected":
                        students_expected,

                    "meals_prepared":
                        meals_prepared,

                    "day":
                        day,

                    "holiday":
                        holiday
                }

                st.success(
                    "Meal demand prediction completed."
                )

            except Exception as e:

                st.session_state.demand_result = None

                st.error(
                    f"Demand prediction error: {e}"
                )


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        demand_result = st.session_state.demand_result


        if demand_result is not None:

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Prediction Results'
                '</div>',
                unsafe_allow_html=True
            )


            predicted_consumption = (
                demand_result["predicted_consumption"]
            )

            surplus = demand_result["surplus"]


            result1, result2, result3 = st.columns(3)


            with result1:

                st.metric(
                    "🤖 Predicted Consumption",
                    f"{predicted_consumption} meals"
                )


            with result2:

                st.metric(
                    "🍱 Meals Prepared",
                    f"{demand_result['meals_prepared']} meals"
                )


            with result3:

                if surplus >= 0:

                    st.metric(
                        "♻️ Predicted Surplus",
                        f"{surplus} meals"
                    )

                else:

                    st.metric(
                        "⚠️ Predicted Shortage",
                        f"{abs(surplus)} meals"
                    )


            st.markdown("<br>", unsafe_allow_html=True)


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if surplus > 0:

                percentage = (
                    surplus
                    / demand_result["meals_prepared"]
                    * 100
                    if demand_result["meals_prepared"] > 0
                    else 0
                )

                st.success(
                    f"♻️ SURPLUS DETECTED — "
                    f"{surplus} meals may remain unused."
                )

                st.info(
                    f"Estimated surplus: "
                    f"{percentage:.2f}% of prepared meals."
                )


            elif surplus == 0:

                st.success(
                    "✅ BALANCED — Prepared meals match "
                    "predicted demand."
                )


            else:

                st.warning(
                    f"⚠️ POTENTIAL SHORTAGE — "
                    f"{abs(surplus)} additional meals "
                    f"may be required."
                )


            # ------------------------------------------------
            # SURPLUS CALCULATION
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">'
                'Surplus Calculation'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "**Surplus = Meals Prepared − "
                "Predicted Consumption**"
            )

            st.write(
                f"**{demand_result['meals_prepared']} − "
                f"{predicted_consumption} = "
                f"{surplus} meals**"
            )


            # ------------------------------------------------
            # CHART
            # ------------------------------------------------

            chart_data = pd.DataFrame(
                {
                    "Meals": [
                        predicted_consumption,
                        demand_result["meals_prepared"]
                    ]
                },
                index=[
                    "Predicted Consumption",
                    "Meals Prepared"
                ]
            )


            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">'
                'Meal Comparison'
                '</div>',
                unsafe_allow_html=True
            )

            st.bar_chart(chart_data)


        else:

            st.info(
                "Enter the input values and click "
                "**Predict Meal Demand** to generate the result."
            )


# ============================================================
# PAGE 3 — FOOD QUALITY AI
# ============================================================

elif st.session_state.user_role == "provider" and page == "Food Quality AI":

    show_page_header(
        "Food Quality AI",
        "Analyze cooked-food conditions and estimate consumability and remaining quantity."
    )


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    if not MODELS_LOADED:

        st.error(
            "Food quality AI models are not available."
        )

    else:

        st.markdown(
            '<div class="section-title">Food Information</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Enter the storage, cooking, cooling and reheating conditions.'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # FOOD INPUTS — COLUMN 1
        # ----------------------------------------------------

        food_col1, food_col2 = st.columns(2)


        with food_col1:

            food_name = st.selectbox(
                "🍚 Food Item",
                ["Rice", "Roti", "Dal"],
                key="food_name"
            )


            storage_mode = st.selectbox(
                "🌡️ Storage Mode",
                ["cold", "hot", "ambient"],
                key="storage_mode"
            )


            storage_temperature_c = st.number_input(
                "Storage Temperature (°C)",
                value=5.0,
                step=0.1,
                key="storage_temperature"
            )


            food_age_hours = st.number_input(
                "Food Age (hours)",
                min_value=0.0,
                value=12.0,
                step=0.5,
                key="food_age"
            )


            cooking_core_temperature_c = st.number_input(
                "Cooking Core Temperature (°C)",
                value=75.0,
                step=0.1,
                key="cooking_core_temperature"
            )


            cooking_time_minutes = st.number_input(
                "Cooking Time (minutes)",
                min_value=0.0,
                value=25.0,
                step=1.0,
                key="cooking_time"
            )


        # ----------------------------------------------------
        # FOOD INPUTS — COLUMN 2
        # ----------------------------------------------------

        with food_col2:

            cooling_60_to_21_hours = st.number_input(
                "Cooling 60→21°C (hours)",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="cooling_60_21"
            )


            cooling_21_to_5_hours = st.number_input(
                "Cooling 21→5°C (hours)",
                min_value=0.0,
                value=2.0,
                step=0.1,
                key="cooling_21_5"
            )


            reheated_option = st.selectbox(
                "🔄 Reheated?",
                ["No", "Yes"],
                key="reheated_option"
            )


            reheated = (
                1 if reheated_option == "Yes"
                else 0
            )


            reheating_core_temperature_c = st.number_input(
                "Reheating Core Temperature (°C)",
                min_value=0.0,
                value=75.0,
                step=0.1,
                disabled=(reheated == 0),
                key="reheating_core_temperature"
            )


            reheating_time_minutes = st.number_input(
                "Reheating Time (minutes)",
                min_value=0.0,
                value=10.0,
                step=1.0,
                disabled=(reheated == 0),
                key="reheating_time"
            )


            initial_quantity_kg = st.number_input(
                "🍱 Initial Food Quantity (kg)",
                min_value=0.1,
                value=20.0,
                step=0.5,
                key="initial_quantity"
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # ANALYSIS BUTTON
        # ----------------------------------------------------

        analyze_food = st.button(
            "🔬 Analyze Food Quality",
            use_container_width=True,
            type="primary",
            key="analyze_food_button"
        )


        if analyze_food:

            try:

                # If food was not reheated,
                # use None for reheating parameters.
                if reheated == 0:

                    reheating_core_temperature_c = None
                    reheating_time_minutes = None


                # ------------------------------------------------
                # RUN FOOD QUALITY MODELS
                # ------------------------------------------------

                food_result = predict_food_quality(

                    food_name=food_name,

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
                        reheating_time_minutes,

                    initial_quantity_kg=
                        initial_quantity_kg
                )


                predicted_status = (
                    food_result["status"]
                )


                estimated_quantity = float(
                    food_result[
                        "consumable_quantity_kg"
                    ]
                )


                # ------------------------------------------------
                # SERVING SIZE
                # ------------------------------------------------

                serving_size_g = {
                    "Rice": 250,
                    "Roti": 50,
                    "Dal": 150
                }[food_name]


                estimated_people = int(
                    estimated_quantity * 1000
                    / serving_size_g
                )


                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                st.session_state.food_quality_result = {

                    "food_name":
                        food_name,

                    "initial_quantity_kg":
                        initial_quantity_kg,

                    "status":
                        predicted_status,

                    "estimated_quantity_kg":
                        estimated_quantity,

                    "estimated_people":
                        estimated_people
                }


                st.success(
                    "Food quality analysis completed."
                )


            except Exception as e:

                st.session_state.food_quality_result = None

                st.error(
                    f"Food quality prediction error: {e}"
                )


        # --------------------------------------------------------
        # DISPLAY FOOD QUALITY RESULT
        # --------------------------------------------------------

        food_quality_result = (
            st.session_state.food_quality_result
        )


        if food_quality_result is not None:

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Food Quality AI Result'
                '</div>',
                unsafe_allow_html=True
            )


            fq1, fq2, fq3 = st.columns(3)


            with fq1:

                if (
                    food_quality_result["status"]
                    == "CONSUMABLE"
                ):

                    st.success(
                        "✓ CONSUMABLE"
                    )

                else:

                    st.error(
                        "✗ NOT CONSUMABLE"
                    )


            with fq2:

                st.metric(
                    "Estimated Remaining Food",
                    f"{food_quality_result['estimated_quantity_kg']:.2f} kg"
                )


            with fq3:

                st.metric(
                    "Estimated People Served",
                    food_quality_result["estimated_people"]
                )


            st.markdown("<br>", unsafe_allow_html=True)


            # ------------------------------------------------
            # REDISTRIBUTION MESSAGE
            # ------------------------------------------------

            if (
                food_quality_result["status"]
                == "CONSUMABLE"
            ):

                st.success(
                    f"♻️ Approximately "
                    f"{food_quality_result['estimated_quantity_kg']:.2f} kg "
                    f"of {food_quality_result['food_name']} "
                    f"is estimated to be consumable and can "
                    f"proceed to redistribution assessment."
                )

            else:

                st.warning(
                    f"⚠️ The AI classified "
                    f"{food_quality_result['food_name']} as "
                    f"NOT CONSUMABLE. It should not be sent "
                    f"for food redistribution based on this "
                    f"model result."
                )


        else:

            st.info(
                "Enter the cooked-food conditions and click "
                "**Analyze Food Quality** to generate the result."
            )



# ============================================================
# PAGE 4 — FOOD REQUESTS
# ============================================================

elif st.session_state.user_role == "provider" and page == "Food Requests":

    show_page_header(
        "Food Requests",
        "View live requests raised by food receivers and approve suitable requests."
    )

    render_live_refresh_hint()
    refresh_button("provider_requests_refresh")

    firebase_ok, firebase_error = firebase_ready()

    if not firebase_ok:
        render_firebase_setup_warning(firebase_error)

    else:

        def render_provider_requests():
            try:
                requests = get_food_requests()
                listings = get_food_listings()

                provider_id = st.session_state.user_id

                relevant_requests = [
                    request for request in requests
                    if request.get("status") == "pending"
                    and provider_matches_request(request, provider_id)
                ]

                st.markdown(
                    '<div class="section-title">Live Requests</div>',
                    unsafe_allow_html=True
                )

                if not relevant_requests:
                    st.success(
                        "No pending food requests right now."
                    )
                    return

                for request in relevant_requests:

                    is_general_request = not request.get("listing_id")

                    if is_general_request:
                        provider_listings = [
                            listing for listing in listings
                            if listing.get("provider_id") == provider_id
                            and listing.get("status") == "available"
                            and float(listing.get("quantity_kg", 0)) > 0
                            and listing.get("food") == request.get("food")
                        ]
                    else:
                        provider_listings = [
                            listing for listing in listings
                            if listing.get("id") == request.get("listing_id")
                            and listing.get("provider_id") == provider_id
                            and listing.get("status") == "available"
                        ]

                    with st.container(border=True):

                        request_col1, request_col2 = st.columns([2.2, 1])

                        with request_col1:

                            st.markdown(
                                f"### {request.get('food', 'Food')}"
                            )

                            st.write(
                                f"**Receiver:** "
                                f"{request.get('receiver_name', 'Receiver')}"
                            )

                            st.write(
                                f"**Requested quantity:** "
                                f"{float(request.get('quantity_kg', 0)):.1f} kg"
                            )

                            st.write(
                                f"**People to serve:** "
                                f"{int(request.get('people_to_serve', 0))}"
                            )

                            st.caption(
                                f"Request ID: {request['id']}"
                            )

                            if request.get("requirement"):
                                st.write(
                                    f"**Requirement:** {request['requirement']}"
                                )

                        with request_col2:

                            selected_listing_id = request.get("listing_id")

                            if is_general_request:

                                if provider_listings:
                                    options = {
                                        f"{listing['food']} — "
                                        f"{float(listing.get('quantity_kg', 0)):.1f} kg "
                                        f"({listing['id']})":
                                        listing["id"]
                                        for listing in provider_listings
                                    }

                                    selected_label = st.selectbox(
                                        "Use your available donation",
                                        list(options.keys()),
                                        key=f"listing_for_{request['id']}"
                                    )

                                    selected_listing_id = options[selected_label]

                                else:
                                    st.warning(
                                        "No matching available donation "
                                        "from your account."
                                    )

                            approve_col, reject_col = st.columns(2)

                            with approve_col:

                                if st.button(
                                    "Accept",
                                    key=f"accept_{request['id']}",
                                    use_container_width=True,
                                    type="primary"
                                ):

                                    if not selected_listing_id:
                                        st.warning(
                                            "You need a matching available "
                                            "food listing before accepting "
                                            "this request."
                                        )
                                    else:

                                        listing = next(
                                            (
                                                item for item in listings
                                                if item["id"] == selected_listing_id
                                            ),
                                            None
                                        )

                                        if not listing:
                                            st.error(
                                                "The selected food listing "
                                                "could not be found."
                                            )
                                        else:

                                            available = float(
                                                listing.get(
                                                    "quantity_kg", 0
                                                )
                                            )

                                            requested = float(
                                                request.get(
                                                    "quantity_kg", 0
                                                )
                                            )

                                            if requested <= 0:
                                                st.error(
                                                    "Requested quantity "
                                                    "must be greater than zero."
                                                )

                                            elif requested > available:
                                                st.error(
                                                    f"Only {available:.1f} kg "
                                                    f"is currently available "
                                                    f"for this donation."
                                                )

                                            else:

                                                remaining = round(
                                                    available - requested,
                                                    2
                                                )

                                                firebase_update(
                                                    f"foodloop/food_requests/{request['id']}",
                                                    {
                                                        "status": "approved",
                                                        "provider_id": provider_id,
                                                        "provider_name":
                                                            st.session_state.provider_profile[
                                                                "organization_name"
                                                            ],
                                                        "listing_id":
                                                            selected_listing_id,
                                                        "approved_at":
                                                            now_iso(),
                                                        "updated_at":
                                                            now_iso()
                                                    }
                                                )

                                                listing_updates = {
                                                    "quantity_kg": remaining,
                                                    "updated_at": now_iso()
                                                }

                                                if remaining <= 0:
                                                    listing_updates["quantity_kg"] = 0
                                                    listing_updates["status"] = "claimed"

                                                firebase_update(
                                                    f"foodloop/food_listings/{selected_listing_id}",
                                                    listing_updates
                                                )

                                                create_notification(
                                                    request.get("receiver_id"),
                                                    "Food request approved",
                                                    (
                                                        f"Your request for "
                                                        f"{requested:.1f} kg of "
                                                        f"{request.get('food')} "
                                                        f"has been approved by "
                                                        f"{st.session_state.provider_profile['organization_name']}."
                                                    ),
                                                    "success"
                                                )

                                                st.success(
                                                    "Request approved successfully."
                                                )
                                                st.rerun()

                            with reject_col:

                                if st.button(
                                    "Reject",
                                    key=f"reject_{request['id']}",
                                    use_container_width=True
                                ):

                                    firebase_update(
                                        f"foodloop/food_requests/{request['id']}",
                                        {
                                            "status": "rejected",
                                            "provider_id": provider_id,
                                            "provider_name":
                                                st.session_state.provider_profile[
                                                    "organization_name"
                                                ],
                                            "updated_at": now_iso()
                                        }
                                    )

                                    create_notification(
                                        request.get("receiver_id"),
                                        "Food request rejected",
                                        (
                                            f"Your request for "
                                            f"{float(request.get('quantity_kg', 0)):.1f} kg "
                                            f"of {request.get('food')} was rejected."
                                        ),
                                        "warning"
                                    )

                                    st.success(
                                        "Request rejected."
                                    )
                                    st.rerun()

            except Exception as e:
                st.error(f"Could not load food requests: {e}")

        if hasattr(st, "fragment"):
            live_requests = st.fragment(run_every="5s")(
                render_provider_requests
            )
            live_requests()
        else:
            render_provider_requests()


# ============================================================
# PAGE 5 — MY DONATIONS
# ============================================================

elif st.session_state.user_role == "provider" and page == "My Donations":

    show_page_header(
        "My Donations",
        "Publish surplus food so receivers can see it and submit requests."
    )

    firebase_ok, firebase_error = firebase_ready()

    if not firebase_ok:
        render_firebase_setup_warning(firebase_error)

    else:

        latest_quality = st.session_state.food_quality_result

        st.markdown(
            '<div class="section-title">Publish Food Donation</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Only food you intend to redistribute should be published. "
            "If you have run Food Quality AI, you can use its latest "
            "consumable result as the donation quantity."
        )

        use_ai_result = st.checkbox(
            "Use latest Food Quality AI result",
            value=(
                latest_quality is not None
                and latest_quality.get("status") == "CONSUMABLE"
            ),
            key="use_latest_quality_result"
        )

        with st.form("publish_food_form"):

            donation_col1, donation_col2 = st.columns(2)

            with donation_col1:

                donation_food = st.selectbox(
                    "Food Item",
                    ["Rice", "Roti", "Dal"],
                    key="donation_food"
                )

                donation_quantity = st.number_input(
                    "Available Quantity (kg)",
                    min_value=0.1,
                    value=10.0,
                    step=0.5,
                    key="donation_quantity"
                )

                provider_name = st.text_input(
                    "Provider / Institution Name",
                    value=st.session_state.provider_profile[
                        "organization_name"
                    ],
                    key="provider_name"
                )

            with donation_col2:

                pickup_window = st.text_input(
                    "Pickup Window",
                    value="Today • 6:00 PM – 8:00 PM",
                    key="pickup_window"
                )

                pickup_location = st.text_input(
                    "Pickup Location",
                    value=st.session_state.provider_profile["location"],
                    key="pickup_location"
                )

                donation_notes = st.text_area(
                    "Notes",
                    placeholder="Optional details for the receiver.",
                    key="donation_notes"
                )

            publish = st.form_submit_button(
                "Publish Food Donation",
                use_container_width=True,
                type="primary"
            )

        if publish:

            if not provider_name.strip():
                st.error("Enter the provider/institution name.")

            elif not pickup_location.strip():
                st.error("Enter the pickup location.")

            elif use_ai_result and (
                latest_quality is None
                or latest_quality.get("status") != "CONSUMABLE"
            ):
                st.error(
                    "The latest Food Quality AI result is not "
                    "classified as consumable."
                )

            else:

                if use_ai_result:

                    final_food = latest_quality["food_name"]
                    final_quantity = float(
                        latest_quality["estimated_quantity_kg"]
                    )
                    quality_status = latest_quality["status"]

                else:

                    final_food = donation_food
                    final_quantity = float(donation_quantity)
                    quality_status = "MANUALLY_PUBLISHED"

                if final_quantity <= 0:
                    st.error(
                        "Donation quantity must be greater than zero."
                    )

                else:

                    provider_id = st.session_state.user_id

                    st.session_state.provider_profile = {
                        "organization_name":
                            provider_name.strip(),
                        "location":
                            pickup_location.strip()
                    }

                    firebase_update(
                        f"foodloop/users/{st.session_state.user_id}",
                        {
                            "role": "provider",
                            "organization_name":
                                provider_name.strip(),
                            "location":
                                pickup_location.strip(),
                            "updated_at": now_iso()
                        }
                    )

                    listing_id = firebase_push(
                        "foodloop/food_listings",
                        {
                            "provider_id": provider_id,
                            "provider_name": provider_name.strip(),
                            "food": final_food,
                            "quantity_kg": round(final_quantity, 2),
                            "unit": "kg",
                            "pickup_window":
                                pickup_window.strip(),
                            "pickup_location":
                                pickup_location.strip(),
                            "notes":
                                donation_notes.strip(),
                            "quality_status":
                                quality_status,
                            "status": "available",
                            "created_at": now_iso(),
                            "updated_at": now_iso()
                        }
                    )

                    st.success(
                        f"Donation published successfully. "
                        f"Listing ID: {listing_id}"
                    )

        st.divider()

        st.markdown(
            '<div class="section-title">My Live Donations</div>',
            unsafe_allow_html=True
        )

        refresh_button("provider_donations_refresh")

        try:

            my_listings = [
                listing for listing in get_food_listings()
                if listing.get("provider_id") == st.session_state.user_id
            ]

            if not my_listings:

                st.info(
                    "You have not published any food donations yet."
                )

            else:

                for listing in reversed(my_listings):

                    with st.container(border=True):

                        col1, col2, col3 = st.columns([2, 1, 1.5])

                        with col1:

                            st.markdown(
                                f"### {listing.get('food', 'Food')}"
                            )

                            st.write(
                                f"**Listing ID:** {listing['id']}"
                            )

                            st.write(
                                f"**Pickup:** "
                                f"{listing.get('pickup_window', 'Not specified')}"
                            )

                        with col2:

                            st.metric(
                                "Available",
                                f"{float(listing.get('quantity_kg', 0)):.1f} kg"
                            )

                        with col3:

                            status = listing.get("status", "available")

                            if status == "available":
                                st.success("Available")
                            elif status == "claimed":
                                st.info("Fully allocated")
                            else:
                                st.warning(status.title())

        except Exception as e:
            st.error(f"Could not load your donations: {e}")


# ============================================================
# PAGE 6 — ABOUT
# ============================================================


elif st.session_state.user_role == "provider" and page == "About":

    show_page_header(
        "About FoodLoop AI",
        "AI-powered food waste reduction and sustainable redistribution system."
    )


    st.markdown(
        """
        <div class="info-card">

        <h3>🍱 FoodLoop AI</h3>

        <p>
        FoodLoop AI is designed to help institutional kitchens
        reduce avoidable food waste by combining meal-demand
        prediction with AI-based food quality assessment.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div class="info-card">

            <h4>📊 Meal Demand Prediction</h4>

            <p>
            Uses the trained demand forecasting model to estimate
            expected meal consumption and identify potential
            surplus or shortage.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="info-card">

            <h4>🔬 Food Quality AI</h4>

            <p>
            Uses food-quality models to classify cooked food as
            consumable or not consumable and estimate the
            remaining quantity suitable for consumption.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">Supported Food Items</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The current food-quality system supports:"
    )

    st.write(
        "• Rice\n\n"
        "• Roti\n\n"
        "• Dal"
    )


    st.markdown("<br>", unsafe_allow_html=True)


    st.caption(
        "FoodLoop AI — Smart Food Waste Management System"
    )


# ============================================================
# RECEIVER PAGES
# ============================================================

elif st.session_state.user_role == "receiver":

    # ========================================================
    # RECEIVER DASHBOARD
    # ========================================================

    if page == "Receiver Dashboard":

        show_page_header(
            "Receiver Dashboard",
            "Find suitable surplus food and manage your organization's requests."
        )

        firebase_ok, firebase_error = firebase_ready()

        if not firebase_ok:
            render_firebase_setup_warning(firebase_error)

        else:

            try:
                listings = get_food_listings()
                requests = [
                    request for request in get_food_requests()
                    if request.get("receiver_id") == st.session_state.user_id
                ]

                available_quantity = sum(
                    float(listing.get("quantity_kg", 0))
                    for listing in listings
                    if listing.get("status") == "available"
                )

                pending_count = sum(
                    1 for request in requests
                    if request.get("status") == "pending"
                )

                approved_count = sum(
                    1 for request in requests
                    if request.get("status") == "approved"
                )

                requested_quantity = sum(
                    float(request.get("quantity_kg", 0))
                    for request in requests
                    if request.get("status") != "cancelled"
                )

                card1, card2, card3, card4 = st.columns(4)

                with card1:
                    st.metric(
                        "Live Food",
                        f"{available_quantity:.1f} kg"
                    )

                with card2:
                    st.metric(
                        "Pending Requests",
                        pending_count
                    )

                with card3:
                    st.metric(
                        "Approved Requests",
                        approved_count
                    )

                with card4:
                    st.metric(
                        "Food Requested",
                        f"{requested_quantity:.1f} kg"
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-title">Organization Profile</div>',
                    unsafe_allow_html=True
                )

                profile = st.session_state.receiver_profile

                with st.form("receiver_requirement_form"):

                    col1, col2 = st.columns(2)

                    with col1:

                        organization_name = st.text_input(
                            "Organization / NGO Name",
                            value=profile["organization_name"],
                            placeholder="Enter organization name"
                        )

                        people_required = st.number_input(
                            "People to Serve",
                            min_value=0,
                            value=int(profile["people_required"]),
                            step=1
                        )

                    with col2:

                        preferred_food = st.selectbox(
                            "Preferred Food",
                            ["Any", "Rice", "Roti", "Dal"],
                            index=[
                                "Any",
                                "Rice",
                                "Roti",
                                "Dal"
                            ].index(profile["preferred_food"])
                        )

                        required_quantity_kg = st.number_input(
                            "Required Quantity (kg)",
                            min_value=0.0,
                            value=float(profile["required_quantity_kg"]),
                            step=0.5
                        )

                    save_requirement = st.form_submit_button(
                        "Save Organization Profile",
                        use_container_width=True,
                        type="primary"
                    )

                if save_requirement:

                    if not organization_name.strip():
                        st.error(
                            "Enter the organization / NGO name."
                        )

                    else:

                        st.session_state.receiver_profile = {
                            "organization_name":
                                organization_name.strip(),
                            "people_required":
                                people_required,
                            "preferred_food":
                                preferred_food,
                            "required_quantity_kg":
                                required_quantity_kg
                        }

                        firebase_update(
                            f"foodloop/users/{st.session_state.user_id}",
                            {
                                "role": "receiver",
                                "organization_name":
                                    organization_name.strip(),
                                "people_required":
                                    people_required,
                                "preferred_food":
                                    preferred_food,
                                "required_quantity_kg":
                                    required_quantity_kg,
                                "updated_at": now_iso()
                            }
                        )

                        st.success(
                            "Organization profile saved."
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                notifications = get_notifications(
                    st.session_state.user_id
                )

                unread = [
                    notification for notification in notifications
                    if not notification.get("read", False)
                ]

                if unread:

                    st.markdown(
                        '<div class="section-title">Notifications</div>',
                        unsafe_allow_html=True
                    )

                    for notification in reversed(unread[-5:]):

                        if notification.get("type") == "success":
                            st.success(
                                f"**{notification.get('title')}** — "
                                f"{notification.get('message')}"
                            )
                        elif notification.get("type") == "warning":
                            st.warning(
                                f"**{notification.get('title')}** — "
                                f"{notification.get('message')}"
                            )
                        else:
                            st.info(
                                f"**{notification.get('title')}** — "
                                f"{notification.get('message')}"
                            )

                st.info(
                    "Use **Live Food** to choose an available donation, "
                    "or use **Create Request** when the food you need "
                    "is not currently listed."
                )

            except Exception as e:
                st.error(f"Could not load receiver data: {e}")


    # ========================================================
    # LIVE FOOD
    # ========================================================

    elif page == "Live Food":

        show_page_header(
            "Live Food",
            "View food donations currently available from providers."
        )

        render_live_refresh_hint()
        refresh_button("receiver_live_food_refresh")

        firebase_ok, firebase_error = firebase_ready()

        if not firebase_ok:
            render_firebase_setup_warning(firebase_error)

        else:

            def render_live_food():

                try:

                    listings = [
                        listing for listing in get_food_listings()
                        if listing.get("status") == "available"
                        and float(listing.get("quantity_kg", 0)) > 0
                    ]

                    food_filter = st.selectbox(
                        "Food Type",
                        ["All", "Rice", "Roti", "Dal"],
                        key="live_food_filter"
                    )

                    if food_filter != "All":
                        listings = [
                            listing for listing in listings
                            if listing.get("food") == food_filter
                        ]

                    if not listings:

                        st.info(
                            "No live food donations are available right now. "
                            "You can create a new request from the sidebar."
                        )
                        return

                    for listing in listings:

                        with st.container(border=True):

                            col1, col2, col3 = st.columns(
                                [2.2, 1.2, 1.5]
                            )

                            with col1:

                                st.markdown(
                                    f"### {listing.get('food', 'Food')}"
                                )

                                st.write(
                                    f"**Provider:** "
                                    f"{listing.get('provider_name', 'Provider')}"
                                )

                                st.write(
                                    f"**Pickup location:** "
                                    f"{listing.get('pickup_location', 'Not specified')}"
                                )

                                if listing.get("notes"):
                                    st.caption(
                                        listing["notes"]
                                    )

                            with col2:

                                st.metric(
                                    "Available",
                                    f"{float(listing.get('quantity_kg', 0)):.1f} kg"
                                )

                            with col3:

                                st.write("**Pickup window**")
                                st.write(
                                    listing.get(
                                        "pickup_window",
                                        "Not specified"
                                    )
                                )

                            request_col1, request_col2 = st.columns(
                                [2, 1]
                            )

                            with request_col1:

                                max_quantity = float(
                                    listing.get("quantity_kg", 0)
                                )

                                requested_quantity = st.number_input(
                                    "Quantity to request (kg)",
                                    min_value=0.1,
                                    max_value=max_quantity,
                                    value=min(1.0, max_quantity),
                                    step=0.5,
                                    key=f"live_request_qty_{listing['id']}"
                                )

                            with request_col2:

                                st.write("")
                                st.write("")

                                if st.button(
                                    "Request Food",
                                    key=f"live_request_{listing['id']}",
                                    use_container_width=True,
                                    type="primary"
                                ):

                                    organization_name = (
                                        st.session_state.receiver_profile[
                                            "organization_name"
                                        ]
                                    )

                                    if not organization_name.strip():

                                        st.warning(
                                            "Save your organization name "
                                            "in Receiver Dashboard first."
                                        )

                                    elif listing.get("provider_id") == st.session_state.user_id:

                                        st.warning(
                                            "A receiver cannot request "
                                            "its own listing."
                                        )

                                    else:

                                        existing_requests = get_food_requests()

                                        duplicate = any(
                                            request.get("receiver_id")
                                            == st.session_state.user_id
                                            and request.get("listing_id")
                                            == listing["id"]
                                            and request.get("status")
                                            == "pending"
                                            for request in existing_requests
                                        )

                                        if duplicate:

                                            st.warning(
                                                "You already have a pending "
                                                "request for this donation."
                                            )

                                        else:

                                            request_id = firebase_push(
                                                "foodloop/food_requests",
                                                {
                                                    "receiver_id":
                                                        st.session_state.user_id,
                                                    "receiver_name":
                                                        organization_name,
                                                    "food":
                                                        listing.get("food"),
                                                    "quantity_kg":
                                                        float(requested_quantity),
                                                    "people_to_serve":
                                                        int(
                                                            st.session_state.receiver_profile[
                                                                "people_required"
                                                            ]
                                                        ),
                                                    "listing_id":
                                                        listing["id"],
                                                    "provider_id":
                                                        listing.get("provider_id"),
                                                    "provider_name":
                                                        listing.get("provider_name"),
                                                    "status": "pending",
                                                    "requirement":
                                                        "",
                                                    "created_at":
                                                        now_iso(),
                                                    "updated_at":
                                                        now_iso()
                                                }
                                            )

                                            create_notification(
                                                listing.get("provider_id"),
                                                "New food request",
                                                (
                                                    f"{organization_name} requested "
                                                    f"{float(requested_quantity):.1f} kg "
                                                    f"of {listing.get('food')}."
                                                ),
                                                "info"
                                            )

                                            st.success(
                                                f"Request submitted successfully. "
                                                f"Request ID: {request_id}"
                                            )
                                            st.rerun()

                except Exception as e:
                    st.error(
                        f"Could not load live food: {e}"
                    )

            if hasattr(st, "fragment"):
                live_food_fragment = st.fragment(run_every="5s")(
                    render_live_food
                )
                live_food_fragment()
            else:
                render_live_food()


    # ========================================================
    # CREATE REQUEST
    # ========================================================

    elif page == "Create Request":

        show_page_header(
            "Create Food Request",
            "Request food directly when a suitable live donation is not available."
        )

        firebase_ok, firebase_error = firebase_ready()

        if not firebase_ok:
            render_firebase_setup_warning(firebase_error)

        else:

            profile = st.session_state.receiver_profile

            with st.form("create_general_request_form"):

                col1, col2 = st.columns(2)

                with col1:

                    food_required = st.selectbox(
                        "Food Required",
                        ["Rice", "Roti", "Dal"]
                    )

                    quantity_required = st.number_input(
                        "Quantity Required (kg)",
                        min_value=0.1,
                        value=10.0,
                        step=0.5
                    )

                    people_to_serve = st.number_input(
                        "People to Serve",
                        min_value=1,
                        value=max(
                            1,
                            int(profile["people_required"])
                        ),
                        step=1
                    )

                with col2:

                    organization_name = st.text_input(
                        "Organization / NGO Name",
                        value=profile["organization_name"],
                        placeholder="Enter organization name"
                    )

                    pickup_preference = st.text_input(
                        "Preferred Pickup Window",
                        value="As soon as possible"
                    )

                    requirement = st.text_area(
                        "Additional Requirement",
                        placeholder=(
                            "Example: Required for dinner service "
                            "for 40 people."
                        )
                    )

                submit_request = st.form_submit_button(
                    "Submit Food Request",
                    use_container_width=True,
                    type="primary"
                )

            if submit_request:

                if not organization_name.strip():
                    st.error(
                        "Enter the organization / NGO name."
                    )

                elif quantity_required <= 0:
                    st.error(
                        "Requested quantity must be greater than zero."
                    )

                else:

                    st.session_state.receiver_profile[
                        "organization_name"
                    ] = organization_name.strip()

                    request_id = firebase_push(
                        "foodloop/food_requests",
                        {
                            "receiver_id":
                                st.session_state.user_id,
                            "receiver_name":
                                organization_name.strip(),
                            "food":
                                food_required,
                            "quantity_kg":
                                float(quantity_required),
                            "people_to_serve":
                                int(people_to_serve),
                            "listing_id":
                                None,
                            "provider_id":
                                None,
                            "provider_name":
                                None,
                            "status":
                                "pending",
                            "requirement":
                                requirement.strip(),
                            "pickup_preference":
                                pickup_preference.strip(),
                            "created_at":
                                now_iso(),
                            "updated_at":
                                now_iso()
                        }
                    )

                    firebase_update(
                        f"foodloop/users/{st.session_state.user_id}",
                        {
                            "role": "receiver",
                            "organization_name":
                                organization_name.strip(),
                            "updated_at":
                                now_iso()
                        }
                    )

                    st.success(
                        f"Food request submitted successfully. "
                        f"Request ID: {request_id}"
                    )

                    st.info(
                        "Providers with matching food can now see "
                        "this request under Food Requests."
                    )


    # ========================================================
    # MY REQUESTS
    # ========================================================

    elif page == "My Requests":

        show_page_header(
            "My Requests",
            "Track requests raised by your organization and see provider decisions."
        )

        render_live_refresh_hint()
        refresh_button("receiver_requests_refresh")

        firebase_ok, firebase_error = firebase_ready()

        if not firebase_ok:
            render_firebase_setup_warning(firebase_error)

        else:

            def render_receiver_requests():

                try:

                    requests = [
                        request for request in get_food_requests()
                        if request.get("receiver_id")
                        == st.session_state.user_id
                    ]

                    notifications = get_notifications(
                        st.session_state.user_id
                    )

                    if notifications:

                        recent = notifications[-5:]

                        st.markdown(
                            '<div class="section-title">Recent Notifications</div>',
                            unsafe_allow_html=True
                        )

                        for notification in reversed(recent):

                            notification_type = notification.get(
                                "type",
                                "info"
                            )

                            message = (
                                f"**{notification.get('title', 'Notification')}** — "
                                f"{notification.get('message', '')}"
                            )

                            if notification_type == "success":
                                st.success(message)
                            elif notification_type == "warning":
                                st.warning(message)
                            else:
                                st.info(message)

                    st.markdown(
                        '<div class="section-title">Request History</div>',
                        unsafe_allow_html=True
                    )

                    if not requests:

                        st.info(
                            "You have not submitted any food requests yet."
                        )
                        return

                    pending_count = sum(
                        1 for request in requests
                        if request.get("status") == "pending"
                    )

                    approved_count = sum(
                        1 for request in requests
                        if request.get("status") == "approved"
                    )

                    rejected_count = sum(
                        1 for request in requests
                        if request.get("status") == "rejected"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric("Pending", pending_count)

                    with c2:
                        st.metric("Approved", approved_count)

                    with c3:
                        st.metric("Rejected", rejected_count)

                    st.markdown("<br>", unsafe_allow_html=True)

                    for request in reversed(requests):

                        with st.container(border=True):

                            col1, col2, col3 = st.columns(
                                [2, 1, 1.3]
                            )

                            with col1:

                                st.markdown(
                                    f"### {request.get('food', 'Food')}"
                                )

                                st.write(
                                    f"Request ID: **{request['id']}**"
                                )

                                st.write(
                                    f"Quantity: **"
                                    f"{float(request.get('quantity_kg', 0)):.1f} kg**"
                                )

                                if request.get("provider_name"):
                                    st.write(
                                        f"Provider: **"
                                        f"{request.get('provider_name')}**"
                                    )

                                if request.get("listing_id"):
                                    st.caption(
                                        f"Donation: {request['listing_id']}"
                                    )

                            with col2:

                                status = request.get(
                                    "status",
                                    "pending"
                                )

                                if status == "pending":
                                    st.warning("Pending")
                                elif status == "approved":
                                    st.success("Approved")
                                elif status == "rejected":
                                    st.error("Rejected")
                                elif status == "cancelled":
                                    st.info("Cancelled")
                                else:
                                    st.info(
                                        format_request_status(status)
                                    )

                            with col3:

                                if request.get("pickup_preference"):
                                    st.write(
                                        "**Pickup:** "
                                        f"{request['pickup_preference']}"
                                    )

                                if status == "approved":

                                    st.write(
                                        "Your request has been "
                                        "**approved by the provider.**"
                                    )

                                elif status == "pending":

                                    if st.button(
                                        "Cancel Request",
                                        key=f"receiver_cancel_{request['id']}",
                                        use_container_width=True
                                    ):

                                        firebase_update(
                                            f"foodloop/food_requests/{request['id']}",
                                            {
                                                "status": "cancelled",
                                                "updated_at": now_iso()
                                            }
                                        )

                                        create_notification(
                                            request.get("provider_id"),
                                            "Food request cancelled",
                                            (
                                                f"{request.get('receiver_name', 'A receiver')} "
                                                f"cancelled the request for "
                                                f"{float(request.get('quantity_kg', 0)):.1f} kg "
                                                f"of {request.get('food')}."
                                            ),
                                            "warning"
                                        )

                                        st.rerun()

                except Exception as e:
                    st.error(
                        f"Could not load your requests: {e}"
                    )

            if hasattr(st, "fragment"):
                live_receiver_requests = st.fragment(run_every="5s")(
                    render_receiver_requests
                )
                live_receiver_requests()
            else:
                render_receiver_requests()


    # ========================================================
    # RECEIVER ABOUT
    # ========================================================

    elif page == "About":

        show_page_header(
            "About FoodLoop AI",
            "AI-powered food waste reduction and sustainable redistribution platform."
        )

        st.markdown(
            """
            <div class="info-card">

            <h3>🤝 Food Receiver / NGO</h3>

            <p>
            Receivers can view live surplus food, request suitable
            donations, create new food requirements and track
            provider approval in one place.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Receiver Workflow</div>',
            unsafe_allow_html=True
        )

        st.write(
            "1. Save your organization profile.\n\n"
            "2. Check Live Food for available donations.\n\n"
            "3. Request a suitable food donation, or create a new request.\n\n"
            "4. Wait for the provider to approve or reject the request.\n\n"
            "5. Track the decision from My Requests."
        )

        st.caption(
            "FoodLoop AI — Smart Food Waste Management System"
        )
