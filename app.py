import streamlit as st
from PIL import Image,ImageOps
import numpy as np
import cv2
import time
# Function names kept exactly as per your setup
from model_loader import load_model, run_inference
from analytics import analyze_detections
import gdown
import os

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id="10nIuL-EnkLVyolRcn9aNsNb90pFwa4o5",
        output=MODEL_PATH,
        quiet=False
    )

st.set_page_config(
    page_title="Industrial Bolt Detection & Quality Control System",
    layout="wide",
    page_icon="⚙️"
)
st.markdown(
    """
    <style>
    /* Main Area Background: Pure Black */
    .stMain {
        background-color: #0A0A0A;
    }

    /* FIX: Remove white strip at top */
    .stAppHeader, header[data-testid="stHeader"] {
        background-color: #0A0A0A !important;
        border-bottom: 1px solid #1C1917 !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        background-color: #0A0A0A !important;
    }

    /* ✅ FIX: Reduce top padding of main content block */
    .stMainBlockContainer {
        padding-top: 3rem !important;
    }

    /* ✅ FIX: Tighten spacing between all headers and next element */
    .stMain h1, .stMain h2, .stMain h3, .stMain h4 {
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* ✅ FIX: Tighten markdown paragraph gaps in main */
    .stMain [data-testid="stMarkdownContainer"] p {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* ✅ FIX: Reduce spacing between stacked blocks in main */
    .stMain [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
    .stMain [data-testid="stVerticalBlock"] > div {
        gap: 0.3rem !important;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #1C1917 !important;
    }

    /* ✅ FIX: Reduce sidebar top padding */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }

    /* ✅ FIX: Tighten sidebar header spacing */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FAF5EF !important;
        margin-top: 0rem !important;
        margin-bottom: 1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* ✅ FIX: Tighten sidebar markdown paragraph gaps */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-top: 0.15rem !important;
        margin-bottom: 0.15rem !important;
    }

    /* ✅ FIX: Reduce gap between sidebar sections/dividers */
    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Main Area Text */
    .stMain h1, .stMain h2, .stMain h3, .stMain h4, .stMain p, .stMain span {
        color: #F5F5F5 !important;
    }

    /* Sidebar Slider thumb: Amber */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #F59E0B !important;
        border-color: #F59E0B !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: #F59E0B !important;
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        color: #4ADE80 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #FB923C !important;
    }

    /* Success / Warning banners */
    .stSuccess {
        background-color: #14231A !important;
        color: #4ADE80 !important;
        border-left: 3px solid #4ADE80 !important;
    }
    .stWarning {
        background-color: #2A1C0E !important;
        color: #FB923C !important;
        border-left: 3px solid #FB923C !important;
    }

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #141414 !important;
        border: 1px dashed #3D3D3D !important;
    }

    /* Upload button */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"],
    button[kind="secondary"] {
        background-color: #1C1917 !important;
        color: #FAF5EF !important;
        border: 1px solid #F59E0B !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #292420 !important;
        color: #FBBF24 !important;
        border: 1px solid #F59E0B !important;
    }

    /* Sidebar upload button */
    [data-testid="stSidebar"] button {
        background-color: #262220 !important;
        color: #FAF5EF !important;
        border: 1px solid #F59E0B !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #2E2924 !important;
        border-color: #FBBF24 !important;
        color: #FBBF24 !important;
    }

    /* Number input +/- buttons — no orange border */
    [data-testid="stNumberInput"] button {
        background-color: #1C1917 !important;
        color: #A8A29E !important;
        border: 1px solid #2D2D2D !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background-color: #262220 !important;
        color: #FAF5EF !important;
        border: 1px solid #3D3D3D !important;
    }

    /* Number input field */
    [data-testid="stNumberInput"] input {
        background-color: #262220 !important;
        color: #A8A29E !important;
        border: 1px solid #F59E0B !important;
    }

    /* Expander */
    .streamlit-expanderContent {
        background-color: #141414 !important;
    }

    /* Horizontal rule */
    hr {
        border: 1px solid #262626;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("### ENGINEERING DASHBOARD: Industrial Bolt Detection System")
st.markdown("---")

with st.sidebar:
    st.header("🛠️ Model Configuration")
    st.markdown("---")
    st.markdown("### 1. Load Trained Model")

    uploaded_model = st.file_uploader(
        "Upload your best.pt model file",
        type=['pt'],
        help="You can upload 'best.pt' from your local PC or your Colab files."
    )

    DEFAULT_MODEL_PATH = 'best.pt'

    st.markdown("---")
    st.markdown("### 2. Detection Thresholds")

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    iou_threshold = st.slider("IoU Threshold (for NMS)", 0.0, 1.0, 0.50, 0.05)

    st.markdown("---")
    st.header("📋 Job Specifications")
    expected_input = st.number_input(
        "Expected Bolt Count for this Part",
        min_value=0,
        max_value=100,
        value=4,
        step=1,
        help="Enter the number of bolts required for this specific assembly."
    )

    st.markdown("---")
    st.header("⚙️ System Status")

    model_path_to_use = DEFAULT_MODEL_PATH

    # Check if the user uploaded a file from their PC
    if uploaded_model is not None:
        model_path_to_use = "uploaded_temp_model.pt"
        # Save the uploaded RAM file to the Colab hard drive temporarily
        with open(model_path_to_use, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("Custom model uploaded successfully from your PC!")
    else:
        # If no file is uploaded, use the one already sitting in the Colab folder
        model_path_to_use = DEFAULT_MODEL_PATH

    # Using your specified function name
    model_obj = load_model(model_path_to_use)

    if model_obj:
        bolt_model, device_name = model_obj
        st.success(f"✅ Model Loaded on {device_name.upper()}")
    else:
        st.error(f"❌ Model file NOT found at: {model_path_to_use}. Please update the path.")
        st.stop()

# --- Main Application Area ---

st.header("📸 Analysis Dashboard")
st.markdown("#### Upload Image & Analyze")

uploaded_image = st.file_uploader(
    "Choose an image file...",
    type=['jpg', 'jpeg', 'png'],
    help="Support for PNG, JPG, JPEG formats."
)

if uploaded_image and model_obj:
    image = Image.open(uploaded_image)
    image = ImageOps.exif_transpose(image) #keep the image orientation same as raw image.
    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV usage later
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Using your specified function name
    result = run_inference(bolt_model, image_bgr, conf_threshold, iou_threshold, device_name)

    total_detected, anomalies, avg_conf = analyze_detections(result, expected_input)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch', caption="Raw industrial photo")

    with col2:
        st.subheader("Detected Bolts (Annotated)")
        plotted_img = result.plot()
        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        st.image(plotted_img_rgb, width='stretch', caption=f"Detection: {avg_conf*100:.1f}% Avg. Confidence")

    st.markdown("---")
    st.header("📊 Detection Summary")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(label="Total Bolts Detected", value=f"{total_detected}")
    with metric2:
        st.metric(label="Anomalies/Missed", value=f"{anomalies}")

    if total_detected >= expected_input:
        st.success(f"✅ Detection Successful: {total_detected} bolts localized (expected: {expected_input}).")
    else:
        st.warning(f"⚠️ Warning: Found {total_detected} bolts, which is less than the expected {expected_input}.")

    with st.expander("Show Detailed Confidences"):
        if total_detected > 0:
            scores = result.boxes.conf.cpu().numpy()
            for i, score in enumerate(scores):
                st.write(f"Bolt {i+1}: {score:.4f}")
        else:
            st.write("No bolts detected.")
