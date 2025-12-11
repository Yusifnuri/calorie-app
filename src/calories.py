# calories.py
#
# Approximate nutritional values per *typical portion*
# Units:
#   - calories: kcal
#   - fat: grams
#   - carbs: grams
#   - protein: grams
#
# IMPORTANT:
#  - Values below are for ONE typical portion.
#  - We will scale them linearly when the user enters grams or number of portions.

NUTRITION_TABLE = {
    # Azerbaijani dishes
    "plov": {"calories": 800, "fat": 32, "carbs": 95, "protein": 25},
    "dolma": {"calories": 350, "fat": 18, "carbs": 22, "protein": 17},
    "qutab": {"calories": 300, "fat": 12, "carbs": 32, "protein": 10},
    "lule_kabab": {"calories": 280, "fat": 20, "carbs": 4, "protein": 22},
    "shashlik": {"calories": 260, "fat": 18, "carbs": 3, "protein": 22},
    "dushbere": {"calories": 400, "fat": 14, "carbs": 48, "protein": 18},
    "piti": {"calories": 600, "fat": 28, "carbs": 40, "protein": 30},
    "xengel": {"calories": 550, "fat": 24, "carbs": 55, "protein": 20},
    "sac_kebab": {"calories": 700, "fat": 40, "carbs": 30, "protein": 35},
    "sekerbura": {"calories": 200, "fat": 11, "carbs": 22, "protein": 4},
    "paxlava": {"calories": 350, "fat": 22, "carbs": 34, "protein": 6},
    "yarpaq_xengel": {"calories": 500, "fat": 22, "carbs": 50, "protein": 18},
    "dovga": {"calories": 150, "fat": 6, "carbs": 16, "protein": 7},
    "seki_halvasi": {"calories": 250, "fat": 12, "carbs": 32, "protein": 4},
    "xash": {"calories": 350, "fat": 20, "carbs": 8, "protein": 30},
    "baki_qurabiyesi": {"calories": 180, "fat": 9, "carbs": 23, "protein": 3},

    # Global dishes
    "burger": {"calories": 550, "fat": 30, "carbs": 45, "protein": 25},
    "french_fries": {"calories": 350, "fat": 17, "carbs": 45, "protein": 4},
    "fried_chicken": {"calories": 400, "fat": 22, "carbs": 15, "protein": 30},
    "ice_cream": {"calories": 250, "fat": 14, "carbs": 28, "protein": 4},
    "pizza": {"calories": 700, "fat": 28, "carbs": 80, "protein": 28},
    "ramen": {"calories": 550, "fat": 20, "carbs": 70, "protein": 20},
    "spagetti": {"calories": 500, "fat": 12, "carbs": 80, "protein": 18},
    "sushi": {"calories": 300, "fat": 5, "carbs": 50, "protein": 15},
    "tiramisu": {"calories": 450, "fat": 28, "carbs": 45, "protein": 6},
}

# Old simple lookup (per *typical portion*)
CALORIE_TABLE = {dish: info["calories"] for dish, info in NUTRITION_TABLE.items()}

# -----------------------------
# Portion-size estimation logic
# -----------------------------

# Default assumption: 1 portion ≈ 250 g
DEFAULT_PORTION_GRAMS = 250.0

# For some dishes, portion is usually smaller/larger – override here
PORTION_GRAMS = {
    # Very dense / dessert / small pieces
    "sekerbura": 50.0,
    "baki_qurabiyesi": 40.0,
    "paxlava": 70.0,
    "seki_halvasi": 60.0,
    "tiramisu": 120.0,
    "ice_cream": 100.0,

    # Sides / snacks
    "french_fries": 150.0,
    "sushi": 200.0,

    # Soups often a bit more
    "dovga": 300.0,
    "xash": 300.0,
}


def get_portion_grams(dish_name: str) -> float:
    """
    Default grams for ONE typical portion of the dish.
    If we don't have a special value, fall back to DEFAULT_PORTION_GRAMS.
    """
    return PORTION_GRAMS.get(dish_name, DEFAULT_PORTION_GRAMS)


def nutrition_for_amount(dish_name: str, grams: float = None, portions: float = 1.0) -> dict:
    """
    Calculate nutrition for a given dish and amount.

    You can either:
      - pass grams (e.g. grams=180), or
      - pass portions (e.g. portions=0.5, 1, 1.5)

    If grams is provided, portions is ignored and computed automatically.

    Returns dict:
      {
        "dish": ...,
        "grams": ...,
        "portions": ...,
        "calories": ...,
        "fat": ...,
        "carbs": ...,
        "protein": ...
      }
    """
    base = NUTRITION_TABLE.get(dish_name)
    if base is None:
        return {
            "dish": dish_name,
            "grams": grams,
            "portions": portions,
            "calories": None,
            "fat": None,
            "carbs": None,
            "protein": None,
        }

    portion_grams = get_portion_grams(dish_name)

    # If user gave grams – convert to "how many portions?"
    if grams is not None:
        factor = grams / portion_grams
        used_grams = float(grams)
        used_portions = factor
    else:
        factor = float(portions)
        used_portions = factor
        used_grams = portion_grams * factor

    return {
        "dish": dish_name,
        "grams": used_grams,
        "portions": used_portions,
        "calories": base["calories"] * factor,
        "fat": base["fat"] * factor,
        "carbs": base["carbs"] * factor,
        "protein": base["protein"] * factor,
    }
