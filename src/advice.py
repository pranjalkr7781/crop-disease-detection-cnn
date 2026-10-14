"""Generic next-step advice shown with a prediction. NOT a substitute for an agronomist."""

ADVICE = [
    ("healthy", "Leaf looks healthy. Keep monitoring and re-check in a few days."),
    ("blight", "Remove and destroy affected leaves, avoid overhead watering, improve airflow, "
               "and consider a suitable fungicide after confirming with a local agronomist."),
    ("rust", "Remove infected leaves, avoid wetting foliage, and consult a local extension "
             "service about an appropriate fungicide."),
    ("mildew", "Improve air circulation, avoid overcrowding, remove badly affected leaves, "
               "and consult a local expert about treatment."),
    ("spot", "Remove spotted leaves, keep foliage dry, rotate crops, and seek local advice on treatment."),
    ("virus", "Viral diseases cannot be cured. Remove and destroy infected plants and control insect vectors."),
]
DEFAULT = "Disease detected. Isolate affected plants and consult a local agronomist for confirmation."


def get_advice(class_name: str) -> str:
    name = class_name.lower()
    for key, text in ADVICE:
        if key in name:
            return text
    return DEFAULT
