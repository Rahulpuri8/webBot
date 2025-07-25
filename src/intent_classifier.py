
# ---------------- src/intent_classifier.py ----------------
from src.db import products_collection

def classify_intent(message: str) -> str:
    message = message.lower()

    product_keywords = [
        "price", "feature", "available", "cost", "strap", "waterproof",
        "details", "about", "digital", "analog", "display", "specs",
        "show", "battery", "automatic", "smartwatch", "leather", "steel"
    ]
    if any(word in message.lower() for word in product_keywords):
        return "product_info"

    product_names = products_collection.distinct("name")
    for name in product_names:
        if name.lower() in message.lower():
            return "product_info"

    faq_keywords = [
        "return", "warranty", "shipping", "deliver", "policy", "refund",
        "track", "cancel", "exchange", "payment", "authentic", "student discount"
    ]
    if any(word in message for word in faq_keywords):
        return "faq"

    return "unknown"



