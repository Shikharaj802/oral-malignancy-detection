
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Oral Malignancy Detection",
    page_icon="🦷",
    layout="centered"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "oral_malignancy_best_model.keras"
    )
    return model


model = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("🦷 Oral Malignancy Detection")

st.markdown(
    """
    **Transfer Learning using MobileNetV2**

    Upload an oral image and the trained deep learning model
    will classify it as **CANCER** or **NON CANCER**.
    """
)

st.info(
    "Model: MobileNetV2 | Image Size: 224 × 224 | Classes: 2"
)

st.warning(
    "⚠️ This application is for academic/research purposes only "
    "and is not a medical diagnosis."
)

# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("📤 Upload Oral Image")

uploaded_file = st.file_uploader(
    "Choose an oral image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Oral Image",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    image_resized = image.resize((224, 224))

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    prediction = float(prediction)

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    cancer_probability = prediction * 100

    non_cancer_probability = (1 - prediction) * 100

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.subheader("🔍 Prediction Result")

    if prediction >= 0.5:

        st.error("🔴 Prediction: CANCER")

        st.metric(
            "Cancer Probability",
            f"{cancer_probability:.2f}%"
        )

    else:

        st.success("🟢 Prediction: NON CANCER")

        st.metric(
            "Non-Cancer Probability",
            f"{non_cancer_probability:.2f}%"
        )

    # --------------------------------------------------------
    # PROBABILITY DETAILS
    # --------------------------------------------------------

    st.subheader("📊 Prediction Probabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "CANCER",
            f"{cancer_probability:.2f}%"
        )

    with col2:
        st.metric(
            "NON CANCER",
            f"{non_cancer_probability:.2f}%"
        )

    # Probability bars
    st.write("Cancer probability")
    st.progress(
        max(0.0, min(1.0, prediction))
    )

    st.write("Non-Cancer probability")
    st.progress(
        max(0.0, min(1.0, 1 - prediction))
    )

    # --------------------------------------------------------
    # RAW MODEL OUTPUT
    # --------------------------------------------------------

    st.subheader("🔧 Model Diagnostic")

    st.write(
        f"Raw model output: **{prediction:.6f}**"
    )

    if prediction >= 0.5:
        st.write(
            "Threshold used: **0.50 → CANCER**"
        )
    else:
        st.write(
            "Threshold used: **0.50 → NON CANCER**"
        )

    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    with st.expander("Image information"):

        st.write(
            f"Original image size: **{image.size}**"
        )

        st.write(
            "Model input size: **224 × 224**"
        )

        st.write(
            "Color channels: **RGB**"
        )

        st.write(
            "Preprocessing: **MobileNetV2 preprocess_input**"
        )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🤖 Model Information")

st.write(
    """
    **MobileNetV2** was selected after comparing five
    transfer learning models on the test dataset.
    """
)

st.write(
    """
    Test Accuracy: **93.81%**  
    Precision: **97.22%**  
    Recall: **93.33%**  
    Specificity: **94.74%**  
    F1 Score: **95.24%**  
    ROC-AUC: **98.28%**
    """
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Oral Malignancy Detection | Transfer Learning | MobileNetV2"
)

