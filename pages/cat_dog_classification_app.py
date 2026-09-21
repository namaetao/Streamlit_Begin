import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # นำเข้าฟังก์ชัน load_model

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Cat & Dog Classification",
    page_icon=":dog:cat:",
    layout="centered"
)

# =========================================================
# Title
# =========================================================
st.header(":cat: Cat &:dog: Dog Image Classification")
st.caption("Upload an image and let the trained model predict whether it is a cat or a dog.")

st.divider()

MODEL_PATH = "model/model_cat_dog.h5"
# =========================================================
# Load Model
# =========================================================
@st.cache_resource
def load_classification_model():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_classification_model()
except Exception as e:
    st.error("❌ ไม่สามารถโหลด Model ได้")
    st.info(
        "กรุณาตรวจสอบว่าไฟล์ model อยู่ในโฟลเดอร์ "
        "`model` และชื่อไฟล์ถูกต้อง"
    )
    st.code(MODEL_PATH)
    st.stop()
# =========================================================
# Upload Image
# =========================================================
st.subheader(":outbox_tray: Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            image,
            caption="Input Image",
            width="stretch"
        )

    with col2:
        st.info(
            f"""
            **File:** {uploaded_file.name}

            **Size:** {image.size[0]} × {image.size[1]} pixels
            """
        )

    st.divider()

    # =====================================================
    # Image Preprocessing
    # =====================================================
    image_resized = image.resize((64, 64))
    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # =====================================================
    # Prediction
    # =====================================================
    if st.button("🔍 Predict", type="primary", width="stretch"):
        with st.spinner("Model is analyzing the image..."):
            prediction = model.predict(
                image_array,
                verbose=0
            )
        probability = float(prediction[0][0])
        if probability >= 0.5:
            predicted_class = "Dog"
            confidence = probability
        else:
            predicted_class = "Cat"
            confidence = 1 - probability

        confidence_percent = confidence * 100

        # =================================================
        # Prediction Result
        # =================================================
        st.subheader("🎯 Prediction Result")

        if predicted_class == "Dog":
            st.success("🐶 The model predicts: DOG")
        else:
            st.success("🐱 The model predicts: CAT")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                predicted_class
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence_percent:.2f}%"
            )

        # =================================================
        # Confidence Progress Bar
        # =================================================
        st.write("**Confidence Level**")
        st.progress(
            int(confidence_percent)
        )

        # =================================================
        # Probability Details
        # =================================================
        with st.expander("📊 Show Prediction Details"):

            cat_probability = (1 - probability) * 100
            dog_probability = probability * 100

            st.write(
                f"🐱 **Cat Probability:** {cat_probability:.2f}%"
            )

            st.write(
                f"🐶 **Dog Probability:** {dog_probability:.2f}%"
            )

            st.write(
                f"Model input size: **64 × 64 × 3**"
            )

else:
    st.info(
        "👆 Please upload an image of a cat or dog to begin."
    )


# =========================================================
# Footer Information
# =========================================================
st.divider()

with st.expander("ℹ️ About this application"):

    st.write(
        """
        This application uses a trained Deep Learning model
        to perform binary image classification.

        **Classes**
        - 🐱 Cat
        - 🐶 Dog

        **Input image**
        - Resized to 64 × 64 pixels
        - RGB color
        - Pixel values normalized to 0–1

        **Model**
        - Keras / TensorFlow
        - Binary Classification
        - Sigmoid output
        """
    )
