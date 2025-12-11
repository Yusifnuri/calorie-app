# predict.py
#
# Predicts the Azerbaijani dish from an image and returns
# nutritional information.
#
# Extended features:
# - returns top-k candidate dishes (for multiple-food / correction UI)
# - returns full class list stored in the checkpoint
# - helper to compute multi-food nutrition based on grams/portions

from pathlib import Path
from typing import Union, List, Dict, Tuple

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from calories import NUTRITION_TABLE, nutrition_for_amount

# When model confidence is below this, we say "I don't know"
CONFIDENCE_THRESHOLD = 0.10

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT_DIR / "azeri_food_model.pt"


def build_model(num_classes: int):
    """
    Build the same EfficientNet-B0 architecture used in training,
    but without loading ImageNet weights (we load fine-tuned weights).
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path):
    """
    Load fine-tuned model checkpoint and class names.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def preprocess_image(image: Union[str, Path, Image.Image]):
    """
    Preprocess an image for EfficientNet-B0:
    - Resize to 224x224
    - Convert to tensor
    - Normalize with ImageNet statistics
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    tensor = transform(img).unsqueeze(0)
    return tensor


def predict_dish_and_nutrition(
    image: Union[str, Path, Image.Image],
    checkpoint_path: Union[str, Path, None] = None,
    top_k: int = 3,
) -> Dict:
    """
    Main prediction function.

    Returns a dict:
        - dish: str | None                # top-1 prediction
        - confidence: float               # top-1 probability
        - nutrition: dict | None          # nutrition for 1 typical portion
        - top_candidates: list of {dish, confidence}
        - all_classes: list of class names
    """

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Load model and class names
    model, class_names = load_model(checkpoint_path)

    # Preprocess image
    input_tensor = preprocess_image(image).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        # Top-1 predicted class
        pred_idx = int(torch.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Top-k candidates for UI suggestions or correction
        k = min(top_k, len(class_names))
        top_values, top_indices = torch.topk(probs, k=k)

        top_candidates = []
        for score, idx in zip(top_values, top_indices):
            idx = int(idx)
            top_candidates.append(
                {
                    "dish": class_names[idx],
                    "confidence": float(score),
                }
            )

    # If confidence is too low → return unknown result
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "dish": None,
            "confidence": confidence,
            "nutrition": None,
            "top_candidates": top_candidates,
            "all_classes": class_names,
        }

    # Retrieve nutrition info for ONE TYPICAL PORTION
    if pred_class not in NUTRITION_TABLE:
        raise KeyError(f"No nutrition info available for predicted dish: {pred_class}")

    # Use helper from calories.py – 1 typical portion
    nutrition = nutrition_for_amount(pred_class, portions=1.0)

    return {
        "dish": pred_class,
        "confidence": confidence,
        "nutrition": nutrition,          # includes calories, fat, carbs, protein, grams, portions
        "top_candidates": top_candidates,
        "all_classes": class_names,
    }


def compute_multi_nutrition(
    dish_portions: List[Dict[str, float]]
) -> Tuple[List[Dict], Dict]:
    """
    Helper for UI: compute nutrition for multiple dishes.

    dish_portions: list of dicts, each like:
        {"dish": "plov", "grams": 180}
      or {"dish": "dolma", "portions": 0.5}

    Returns:
        details: list of nutrition dicts for each dish
        total:   dict with total calories / macros / grams / portions
    """
    details = []
    total = {
        "calories": 0.0,
        "fat": 0.0,
        "carbs": 0.0,
        "protein": 0.0,
        "grams": 0.0,
        "portions": 0.0,
    }

    for dp in dish_portions:
        dish_name = dp["dish"]
        grams = dp.get("grams")
        portions = dp.get("portions", 1.0)

        if grams is not None:
            n = nutrition_for_amount(dish_name, grams=grams)
        else:
            n = nutrition_for_amount(dish_name, portions=portions)

        details.append(n)

        if n["calories"] is not None:
            total["calories"] += n["calories"]
            total["fat"] += n["fat"]
            total["carbs"] += n["carbs"]
            total["protein"] += n["protein"]
            total["grams"] += n["grams"]
            total["portions"] += n["portions"]

    return details, total


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(
        description="Predict Azerbaijani dish and nutrition from an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the food image.")

    args = parser.parse_args()
    image_path = args.image_path

    result = predict_dish_and_nutrition(image_path)

    dish = result["dish"]
    conf = result["confidence"]
    nut = result["nutrition"]

    print("Image:", image_path)
    print(f"Predicted dish: {dish}")
    print(f"Confidence: {conf*100:.2f}%")

    if nut is not None:
        print("Nutrition for ~1 typical portion:")
        pprint.pprint(nut)
    else:
        print("Model is not confident enough or no nutrition info available.")
