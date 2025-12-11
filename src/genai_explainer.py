# genai_explainer.py
#
# Provides an AI-generated explanation of nutrition information
# using the OpenAI API (v1.x+ syntax).
#
# Requires environment variable:
#   export OPENAI_API_KEY="your_api_key_here"

from dotenv import load_dotenv
import os
from typing import Dict, List, Optional
from openai import OpenAI

# Load environment variables from .env (if present)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_explanation(
    meal_details: List[Dict],
    total_nutrition: Dict,
    profile: Optional[Dict] = None,
) -> str:
    """
    Generate a short, personalised nutrition analysis for a whole meal.

    Parameters
    ----------
    meal_details : list of dict
        Each element is a dict like:
        {
          "dish": str,
          "grams": float,
          "portions": float,
          "calories": float,
          "fat": float,
          "carbs": float,
          "protein": float
        }
    total_nutrition : dict
        Aggregated nutrition for the whole meal:
        {
          "calories": float,
          "fat": float,
          "carbs": float,
          "protein": float,
          "grams": float,
          "portions": float
        }
    profile : dict | None
        Optional user profile, e.g. goal, activity, diet preferences.

    Returns
    -------
    str
        AI-generated explanation.
    """

    # Build text description of the meal
    meal_lines = []
    for item in meal_details:
        dish = item.get("dish", "unknown dish")
        grams = item.get("grams", 0.0)
        portions = item.get("portions", 1.0)
        calories = item.get("calories", 0.0)
        protein = item.get("protein", 0.0)
        carbs = item.get("carbs", 0.0)
        fat = item.get("fat", 0.0)

        line = (
            f"- {dish}: {grams:.0f} g (~{portions:.2f} portions), "
            f"{calories:.0f} kcal, "
            f"protein {protein:.1f} g, carbs {carbs:.1f} g, fat {fat:.1f} g"
        )
        meal_lines.append(line)

    meal_text = "\n".join(meal_lines)

    total_kcal = total_nutrition.get("calories", 0.0)
    total_protein = total_nutrition.get("protein", 0.0)
    total_carbs = total_nutrition.get("carbs", 0.0)
    total_fat = total_nutrition.get("fat", 0.0)
    total_grams = total_nutrition.get("grams", 0.0)
    total_portions = total_nutrition.get("portions", 0.0)

    total_text = (
        f"Total meal: {total_kcal:.0f} kcal, "
        f"protein {total_protein:.1f} g, "
        f"carbohydrates {total_carbs:.1f} g, "
        f"fat {total_fat:.1f} g, "
        f"about {total_grams:.0f} g of food "
        f"(~{total_portions:.2f} combined portions)."
    )

    profile_text = ""
    if profile:
        goal = profile.get("goal", "general health")
        activity = profile.get("activity", "moderate")
        diet_pref = profile.get("diet_preference", "no specific preference")
        profile_text = (
            f"\nUser goal: {goal}.\n"
            f"Activity level: {activity}.\n"
            f"Diet preference: {diet_pref}.\n"
        )

    prompt = f"""
You are a certified nutritionist.

Analyze the following meal consisting mainly of Azerbaijani and global dishes.

Meal breakdown:
{meal_text}

{total_text}
{profile_text}

Write a clear explanation (around 5–8 short paragraphs or bullet points) covering:

1. Whether this whole meal is light / moderate / heavy in terms of energy intake.
2. How balanced it is in terms of protein, carbohydrates and fat.
3. Which dishes contribute most to calories and fat, and which are relatively lighter.
4. Practical suggestions to slightly reduce calories (portion changes, swaps, cooking methods)
   while keeping the cultural character of the meal.
5. If a user goal is provided (weight loss, muscle gain, general health), adapt your advice to it.
6. Keep the tone friendly, non-judgmental, and easy to understand.

Answer in English.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a nutrition expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content
