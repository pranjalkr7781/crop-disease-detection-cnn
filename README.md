# Crop Disease Detection from Leaf Images: ML Pipeline (everything before the backend)

Complete model code: data prep, training, evaluation, explainability, prediction, and mobile export.
NOT included: the backend/API and app (these just call `src/inference.py -> predict_image()`).

## Layout
```
config.py              settings
data/raw/              PUT YOUR DATASET HERE (one folder per class)
src/explore_data.py    Step 4  class counts + samples
src/prepare_data.py    Step 4  70/15/15 split
src/data.py            Step 5  loading + augmentation
src/model.py           Step 6  MobileNetV2 transfer learning
src/train.py           Step 7  head training + fine-tuning
src/evaluate.py        Step 8  report + confusion matrix on test set
src/gradcam.py         Step 9  Grad-CAM heatmaps
src/advice.py                  generic next-step advice
src/inference.py               predict_image() for CLI and backend
src/predict.py                 command-line prediction
src/export_tflite.py   Step 10 TFLite export + accuracy check
```

## Run order
```
pip install -r requirements.txt
# put images in data/raw/<ClassName>/*.jpg
python -m src.explore_data
python -m src.prepare_data
python -m src.train
python -m src.evaluate
python -m src.gradcam  path/to/leaf.jpg
python -m src.predict  path/to/leaf.jpg
python -m src.export_tflite
```
Use Google Colab (free GPU) if you have no local GPU.

## Tips
- Start with 1-3 crops. PlantVillage has plain backgrounds, so also test on real phone photos.
- Check Grad-CAM: if the heat is on the background instead of the leaf, the model is cheating.
- This is a screening aid, not a replacement for an agronomist.

## Demo app
```
pip install -r requirements.txt
# copy models/best_model.keras and models/class_names.json from Colab results.zip into ./models
streamlit run app.py
```
