"""Predict from one image.
Run: python -m src.predict path/to/leaf.jpg
"""
import sys
from src.inference import predict_image


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.predict path/to/leaf.jpg")
    result = predict_image(sys.argv[1])
    print(f"\nPrediction : {result['prediction']}  ({result['confidence']:.1%})")
    if result["low_confidence"]:
        print("WARNING    : low confidence. Retake the photo (good light, single leaf, in focus).")
    print(f"Advice     : {result['advice']}\n")
    print("Top predictions:")
    for r in result["top_k"]:
        print(f"  {r['label']:45s} {r['confidence']:.1%}")


if __name__ == "__main__":
    main()
