import streamlit as st
from PIL import Image
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import joblib
from skimage.feature import graycomatrix, graycoprops

def preprocess_image(image_file):
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image = Image.open(image_file).convert("L")
    image = image.resize((128, 128))
    return np.asarray(image)


def extract_glcm_features(image):
    image_quantized = (image // 8).astype(np.uint8)

    glcm = graycomatrix(
        image_quantized,
        distances=[1, 2, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    return [graycoprops(glcm, prop).mean() for prop in properties]

def naive_bayes_analyze(img):
    model = joblib.load("naive_bayes/model.pkl")
    feature_names = joblib.load("naive_bayes/glcm.pkl")

    image = preprocess_image(img)
    features = extract_glcm_features(image)

    feature_data = pd.DataFrame(
        [features],
        columns=[
            "Contrast",
            "Dissimilarity",
            "Homogeneity",
            "Energy",
            "Correlation",
            "ASM"
        ]
    )
    feature_data = feature_data[feature_names]

    predict = model.predict_proba(feature_data)

    results = np.argsort(-predict[0])[:5]

    display_result([model.classes_[e] for e in results])

def knn_analyze(img):
    model = joblib.load("knn/model.pkl")
    feature_names = joblib.load("knn/glcm.pkl")

    image = preprocess_image(img)
    features = extract_glcm_features(image)

    feature_data = pd.DataFrame(
        [features],
        columns=[
            "Contrast",
            "Dissimilarity",
            "Homogeneity",
            "Energy",
            "Correlation",
            "ASM"
        ]
    )
    feature_data = feature_data[feature_names]

    predict = model.predict_proba(feature_data)

    results = np.argsort(-predict[0])[:5]

    display_result([model.classes_[e] for e in results])

def mobilenetv3_analyze(img):
    model = tf.keras.models.load_model("mobilenetv3small/model.keras")

    with open("mobilenetv3small/class_names.json", "r") as f:
        class_names = json.load(f)

    img_resized = img.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    predict = model.predict(img_batch)

    results = np.argsort(-predict)[:5][0]

    display_result([class_names[e] for e in results])

def efficientnet_analyze(img):
    model = tf.keras.models.load_model("efficientnetb0/model.keras")

    with open("efficientnetb0/class_names.json", "r") as f:
        class_names = json.load(f)

    img_resized = img.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    predict = model.predict(img_batch)

    results = np.argsort(-predict)[:5][0]

    display_result([class_names[e] for e in results])

def display_result(results):
    st.write(f"Hasil Prediksi Utama: {results[0]}")
    st.write(f"Hasil Prediksi Lainnya: {results[1]}, {results[2]}, {results[3]}, {results[4]}")

st.title("Cek Tulisan Tangan")

file_upload = st.file_uploader("Upload Tulisan Tangan Yang Ingin Dianalisis", type=["jpg", "jpeg", "png"])

if file_upload is not None:
    img = Image.open(file_upload)
    st.image(img, caption="Tulisan Tangan", width=200)

    selected_model = st.selectbox("Pilih Model", [
        "GLCM + Naive Bayes", "GLCM + KNN", "MobileNetV3Small", "EfficientNetB0"
    ])

    if st.button("Lakukan Prediksi"):
        match selected_model:
            case "GLCM + Naive Bayes": naive_bayes_analyze(file_upload)
            case "GLCM + KNN": knn_analyze(file_upload)
            case "MobileNetV3Small": mobilenetv3_analyze(img)
            case "EfficientNetB0": efficientnet_analyze(img)