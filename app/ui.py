# app/ui.py
#
# Streamlit UI for:
#  - uploading an image
#  - predicting Azerbaijani dish
#  - showing nutritional values
#  - multiple-food selection & portion-size estimation (grams)
#  - AI-based personalised nutrition explanation

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make sure we can import from src/
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from predict import predict_dish_and_nutrition, compute_multi_nutrition
from genai_explainer import generate_explanation
from calories import NUTRITION_TABLE, get_portion_grams

st.set_page_config(page_title="FoodVisionAI - Azerbaijani Cuisine", page_icon="🍽️")

st.title("Azerbaijani Cuisine Nutrition Analyzer 🇦🇿")

st.write(
    """
Upload an image of an Azerbaijani dish and the app will:

1. Recognize the dish using a fine-tuned EfficientNet-B0 model  
2. Estimate its nutritional values (calories, fat, carbs, protein)  
3. Let you correct or refine the detection (multiple dishes, portion size in grams)  
4. Generate an AI-based, personalised nutrition explanation
    """
)

# Sidebar: user profile + AI
with st.sidebar:
    st.header("User profile (optional)")

    goal = st.selectbox(
        "Goal",
        ["General health", "Weight loss", "Muscle gain"],
        index=0,
    )
    activity = st.selectbox(
        "Activity level",
        ["Low", "Moderate", "High"],
        index=1,
    )
    diet_pref = st.selectbox(
        "Diet preference",
        ["No specific preference", "Vegetarian", "Low-carb"],
        index=0,
    )

    use_ai_explanation = st.checkbox(
        "Enable AI nutrition explanation (uses API)",
        value=True,
    )

# Build profile once so we can use it everywhere
profile = {
    "goal": goal,
    "activity": activity,
    "diet_preference": diet_pref,
}

uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# =====================================================================
# MAIN FLOW – only if we have an image
# =====================================================================
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing image with FoodVisionAI..."):
        result = predict_dish_and_nutrition(img)

    # Extract prediction results
    dish = result["dish"]
    confidence = result["confidence"]
    nutrition = result["nutrition"]
    top_candidates = result.get("top_candidates", [])
    all_classes = result.get("all_classes", [])

    # ---------------------------------------------------------
    # CASE 1: The model is NOT confident enough
    # ---------------------------------------------------------
    if dish is None:
        st.subheader("Model prediction")
        st.error("The model could not confidently identify any dish in the image.")
        st.info(f"Confidence score of the best guess: {confidence * 100:.2f}% (below threshold)")

        # Show the top candidates so the user sees what the model roughly considered
        if top_candidates:
            st.write("Top candidates (low confidence):")
            for c in top_candidates:
                st.write(f"- {c['dish']} ({c['confidence'] * 100:.1f}%)")

        st.subheader("Manual dish & portion input")

        manual_dish = st.selectbox(
            "Select the dish you actually see in the image:",
            options=sorted(NUTRITION_TABLE.keys()),
        )

        default_grams = float(get_portion_grams(manual_dish))
        manual_grams = st.number_input(
            f"Approximate grams of {manual_dish}:",
            min_value=10.0,
            max_value=2000.0,
            value=default_grams,
            step=10.0,
        )

        if st.button("Estimate nutrition for manual input"):
            # Use the same multi-nutrition helper for consistency
            dish_portions = [{"dish": manual_dish, "grams": manual_grams}]
            details, total = compute_multi_nutrition(dish_portions)

            st.subheader("Estimated nutritional information")

            d = details[0]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Calories", f"{d['calories']:.0f} kcal")
                st.metric("Fat", f"{d['fat']:.1f} g")
            with col2:
                st.metric("Carbs", f"{d['carbs']:.1f} g")
                st.metric("Protein", f"{d['protein']:.1f} g")

            st.write(
                f"Dish: **{manual_dish}**, "
                f"Amount: **{d['grams']:.0f} g** (~{d['portions']:.2f} portions)"
            )

            st.subheader("AI Nutrition Analysis")
            if use_ai_explanation:
                with st.spinner("Generating AI explanation..."):
                    explanation = generate_explanation(
                        meal_details=details,
                        total_nutrition=total,
                        profile=profile,
                    )
                st.write(explanation)
            else:
                st.info(
                    "AI explanation is disabled (speed/cost optimisation). "
                    "Enable it in the sidebar if you want a personalised analysis."
                )

        # We handled everything for low-confidence case
        st.stop()

    # ---------------------------------------------------------
    # CASE 2: Model is confident enough → normal flow
    # ---------------------------------------------------------

    # Fallbacks if keys are missing
    if not top_candidates:
        top_candidates = [{"dish": dish, "confidence": confidence}]
    if not all_classes:
        all_classes = [dish]

    # ---- Prediction summary ----
    st.subheader("Model prediction")
    st.write(f"**Top-1 predicted dish:** {dish}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # ---- Multiple-food detection (soft version via top-k candidates) ----
    st.subheader("Detected dishes (top candidates)")

    candidate_labels = [
        f"{c['dish']} ({c['confidence'] * 100:.1f}%)" for c in top_candidates
    ]
    default_selection = [candidate_labels[0]] if candidate_labels else []

    selected_labels = st.multiselect(
        "Select the dishes that are actually present in the image "
        "(you can choose multiple):",
        options=candidate_labels,
        default=default_selection,
    )

    # Map selected labels back to dish names
    selected_dishes = []
    for label in selected_labels:
        idx = candidate_labels.index(label)
        selected_dishes.append(top_candidates[idx]["dish"])

    # ---- Main dish correction (optional) ----
    st.subheader("Main dish correction (optional)")

    main_dish_default_idx = all_classes.index(dish) if dish in all_classes else 0
    corrected_main_dish = st.selectbox(
        "If the main dish is wrong, select the correct one:",
        options=all_classes,
        index=main_dish_default_idx,
    )

    # If user did not select any dishes in multiselect, fall back to main dish
    if not selected_dishes:
        selected_dishes = [corrected_main_dish]

    # ---- Portion-size estimation (per dish, in grams) ----
    st.subheader("Portion-size estimation (grams per dish)")

    st.write(
        "For each dish you selected, enter an approximate amount in grams. "
        "Defaults are typical portion sizes."
    )

    dish_portions = []
    for d_name in selected_dishes:
        default_grams = float(get_portion_grams(d_name))
        grams = st.number_input(
            f"Approximate grams of {d_name}:",
            min_value=10.0,
            max_value=2000.0,
            value=default_grams,
            step=10.0,
            key=f"grams_{d_name}",
        )
        dish_portions.append({"dish": d_name, "grams": grams})

    # Compute nutrition for all selected dishes
    details, total = compute_multi_nutrition(dish_portions)

    # ---- Per-dish nutritional information ----
    st.subheader("Estimated nutritional information (per dish)")

    for d in details:
        st.markdown(
            f"- **{d['dish']}** – {d['grams']:.0f} g "
            f"(~{d['portions']:.2f} portions): "
            f"{d['calories']:.0f} kcal, "
            f"Fat {d['fat']:.1f} g, Carbs {d['carbs']:.1f} g, "
            f"Protein {d['protein']:.1f} g"
        )

    # ---- Total nutritional information ----
    st.subheader("Estimated nutritional information (total meal)")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calories", f"{total['calories']:.0f} kcal")
        st.metric("Fat", f"{total['fat']:.1f} g")
    with col2:
        st.metric("Carbs", f"{total['carbs']:.1f} g")
        st.metric("Protein", f"{total['protein']:.1f} g")

    st.write(
        f"Total amount of food: **{total['grams']:.0f} g** "
        f"(~{total['portions']:.2f} combined portions)"
    )

    # ---- Save user feedback ----
    st.subheader("Was the prediction correct?")

    feedback = st.radio(
        "Help improve the model:",
        ["Yes, prediction was correct", "No, the model was wrong"],
        index=0,
    )

    if feedback == "No, the model was wrong":
        st.write("Select the correct main dish label:")
        correct_label = st.selectbox("Correct dish:", options=all_classes)

        if st.button("Save feedback image"):
            import uuid

            feedback_dir = Path("data/user_feedback") / correct_label
            feedback_dir.mkdir(parents=True, exist_ok=True)

            # Unique filename
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = feedback_dir / filename

            img.save(save_path)
            st.success(f"Feedback saved! Image stored as {save_path}")

    # ---- AI-based personalised explanation ----
    st.subheader("AI Nutrition Analysis")

    if use_ai_explanation:
        with st.spinner("Generating AI explanation..."):
            explanation = generate_explanation(
                meal_details=details,
                total_nutrition=total,
                profile=profile,
            )
        st.write(explanation)
    else:
        st.info(
            "AI explanation is disabled (speed/cost optimisation). "
            "Enable it in the sidebar if you want a personalised analysis."
        )
