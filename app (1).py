import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

MODEL_PATH = "skin_cancer_cnn_model.keras"
IMAGE_SIZE = (128, 128)

# The notebook loads folders with Keras image_dataset_from_directory.
# Keras sorts folder names alphabetically: benign, malignant.
CLASS_NAMES = ["benign", "malignant"]


@st.cache_resource
def load_trained_model():
    """Load the trained CNN once and cache it for the session."""
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, image):
    """Prepare an uploaded image and get the model prediction."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The notebook's model includes a Rescaling(1./255) layer,
    # so do not divide the image by 255 again here.
    prediction = model.predict(image_array, verbose=0)

    if prediction.shape[-1] == 1:
        malignant_score = float(prediction[0][0])
        class_index = 1 if malignant_score >= 0.5 else 0
        confidence = malignant_score if class_index == 1 else 1.0 - malignant_score
    else:
        class_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_index])

    return CLASS_NAMES[class_index], confidence


# ---------------- Website UI ----------------
st.title("🔬 Skin Cancer Image Classifier")
st.write(
    "Upload a skin image to see the prediction made by the trained "
    "Convolutional Neural Network (CNN)."
)

st.warning(
    "**Important:** This is an educational AI project, not a medical device. "
    "It cannot diagnose or rule out skin cancer. A qualified healthcare "
    "professional should assess any concerning skin change."
)

try:
    model = load_trained_model()
except Exception as error:
    st.error(
        "The model could not be loaded. Make sure "
        "`skin_cancer_cnn_model.keras` is in the same GitHub repository "
        "folder as `app.py`."
    )
    st.code(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence = predict_image(model, image)

            st.subheader("Model Prediction")
            if predicted_class == "malignant":
                st.error("Model result: Malignant class")
            else:
                st.success("Model result: Benign class")

            st.metric("Model confidence", f"{confidence * 100:.2f}%")
            st.caption(
                "Confidence is the model's output score, not the probability "
                "that a person has or does not have cancer."
            )

    except Exception as error:
        st.error("The uploaded file could not be processed.")
        st.code(str(error))

st.divider()
st.caption("BS Computer Science Project | CNN-based Image Classification")
