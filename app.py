"""
============================================================
ML PROJECT — STREAMLIT GUI
All 5 Datasets in One App
============================================================
FOLDER STRUCTURE:
    ML_Project/
        app.py          ← this file
        models/
            mobile_dt.pkl, mobile_nb.pkl, mobile_rf.pkl, mobile_scaler.pkl
            accident_dt.pkl, accident_lr.pkl, accident_rf.pkl, accident_scaler.pkl
            isl_dt.pkl, isl_svm.pkl, isl_rf.pkl, isl_scaler.pkl
            sports_dt.pkl, sports_svm.pkl, sports_efficientnet.keras, sports_scaler.pkl
            cnn_model.h5, knn_model.pkl, rf_model.pkl, scaler.pkl, label_encoder.pkl

RUN:
    pip install streamlit scikit-learn scikit-image opencv-python joblib tensorflow librosa
    streamlit run app.py
============================================================
"""
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
matplotlib.rcParams['font.family'] = 'monospace'
import joblib
import os
import cv2
import tempfile
import warnings
warnings.filterwarnings("ignore")

os.system("git lfs pull")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Machine Learning Intelligence System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Dark base */
    .stApp {
        background-color: #0d0d0d;
        color: #e8e8e8;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #1e1e1e;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #d0d0d0 !important;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Main title */
    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #f0f0f0;
        letter-spacing: -0.03em;
        padding: 24px 0 4px 0;
        border-bottom: 2px solid #2a2a2a;
        margin-bottom: 4px;
    }

    .sub-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #555;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 24px;
        padding-top: 6px;
    }

    /* Result box */
    .result-box {
        background-color: #141414;
        border: 1px solid #2e2e2e;
        border-left: 4px solid #c8ff00;
        color: #f0f0f0;
        padding: 18px 22px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 12px 0;
        letter-spacing: -0.01em;
    }

    .result-label {
        font-size: 0.65rem;
        color: #c8ff00;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 6px;
        font-weight: 500;
    }

    .result-value {
        font-size: 1.4rem;
        color: #ffffff;
        font-weight: 700;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: #141414;
        border: 1px solid #1e1e1e;
        border-radius: 4px;
        padding: 12px 16px;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #666 !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.2rem;
        color: #e0e0e0 !important;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #0d0d0d;
        border-bottom: 1px solid #1e1e1e;
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: #555 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 10px 20px;
        border-radius: 0;
        border: none;
        border-bottom: 2px solid transparent;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: #c8ff00 !important;
        border-bottom: 2px solid #c8ff00 !important;
        background: transparent !important;
    }

    /* Inputs */
    .stSelectbox > div > div,
    .stSlider,
    .stCheckbox {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    div[data-baseweb="select"] > div {
        background-color: #141414 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 4px !important;
        color: #e0e0e0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding: 0;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #141414;
        border: 1px dashed #2a2a2a;
        border-radius: 4px;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        background-color: #c8ff00 !important;
        color: #0d0d0d !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: #d4ff33 !important;
        box-shadow: 0 0 20px rgba(200, 255, 0, 0.25) !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #f0f0f0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    h2 { font-size: 1.2rem !important; border-bottom: 1px solid #1e1e1e; padding-bottom: 10px; }
    h3 { font-size: 0.95rem !important; color: #999 !important; font-weight: 500 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e1e1e !important;
        border-radius: 4px;
    }

    /* Info / error boxes */
    .stInfo {
        background-color: #141414 !important;
        border: 1px solid #1e1e1e !important;
        color: #888 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        border-radius: 4px !important;
    }

    .stError {
        background-color: #1a0a0a !important;
        border: 1px solid #3a1515 !important;
        border-left: 3px solid #ff4444 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Spinner */
    .stSpinner {
        color: #c8ff00 !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #1e1e1e;
        margin: 24px 0;
    }

    /* Label styling */
    label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        color: #888 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    /* Checkbox */
    .stCheckbox label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        color: #ccc !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }

    /* Section tag */
    .section-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #c8ff00;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        background: rgba(200, 255, 0, 0.08);
        padding: 3px 8px;
        border-radius: 2px;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* Performance table header */
    .perf-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #555;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 8px;
        margin-top: 24px;
    }

    /* Time input container */
    .time-input-row {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    /* Plot background to match theme */
    .stPlotlyChart, .stPyplot {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Model Paths ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if filename.endswith(".pkl"):
        return joblib.load(path)
    elif filename.endswith(".keras") or filename.endswith(".h5"):
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(path)


def get_dark_fig(figsize=(8, 4)):
    """Return a matplotlib figure styled for dark theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#141414')
    ax.set_facecolor('#141414')
    ax.tick_params(colors='#888888', labelsize=8)
    ax.spines['bottom'].set_color('#2a2a2a')
    ax.spines['left'].set_color('#2a2a2a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_color('#888888')
    ax.yaxis.label.set_color('#888888')
    ax.title.set_color('#cccccc')
    return fig, ax


# ── Prediction Charts ─────────────────────────────────────────────────────────
def plot_prediction_chart(model, X, class_names, predicted_class, color="#c8ff00"):
    try:
        proba   = model.predict_proba(X)[0]
        top_n   = min(10, len(class_names))
        top_idx = np.argsort(proba)[::-1][:top_n]
        top_proba = proba[top_idx]
        top_names = [class_names[i] for i in top_idx]

        fig, ax = get_dark_fig(figsize=(8, 4))
        bar_colors = [color if n == predicted_class else '#2a2a2a' for n in top_names]
        bars = ax.barh(top_names[::-1], top_proba[::-1] * 100,
                       color=bar_colors[::-1], edgecolor='none', height=0.6)
        for bar, prob in zip(bars, top_proba[::-1]):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%", va='center', fontsize=8,
                    color='#888888', fontfamily='monospace')
        ax.set_xlabel("Confidence (%)", fontsize=8)
        ax.set_title(f"Top-{top_n} Predictions", fontsize=9, fontweight='bold', pad=12)
        ax.set_xlim(0, 120)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def plot_simple_chart(predictions_dict, color="#c8ff00"):
    fig, ax = get_dark_fig(figsize=(8, 3.5))
    names  = list(predictions_dict.keys())
    values = list(predictions_dict.values())
    bar_colors = [color if v == max(values) else '#2a2a2a' for v in values]
    bars = ax.bar(names, [v * 100 for v in values],
                  color=bar_colors, edgecolor='none', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val * 100:.1f}%", ha='center', va='bottom',
                fontsize=8, fontfamily='monospace', color='#888888')
    ax.set_ylabel("Confidence (%)", fontsize=8)
    ax.set_ylim(0, 120)
    ax.set_title("Prediction Confidence", fontsize=9, fontweight='bold', pad=12)
    plt.xticks(rotation=20, ha='right', fontsize=8)
    plt.tight_layout()
    return fig


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Multi-Modal Machine Learning Intelligence System</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ISL Video",
    "Emergency Siren",
    "Mobile Price",
    "Car Accident",
    "Sports Image"
])


# ============================================================
# TAB 1 — ISL VIDEO
# ============================================================
with tab1:
    st.markdown('<div class="section-tag">Indian Sign Language Recognition</div>', unsafe_allow_html=True)
    st.markdown("Upload a sign language video to predict the sign.")
    st.caption("Models: Decision Tree  →  SVM  →  Random Forest (HOG Features)")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_video = st.file_uploader(
            "Upload Sign Language Video",
            type=["mp4", "avi"],
            key="isl_video"
        )
        model_choice_isl = st.selectbox(
            "Select Model",
            ["Decision Tree", "SVM", "Random Forest"],
            key="isl_model"
        )
        predict_isl = st.button("Predict Sign", key="predict_isl", use_container_width=True)

    with col2:
        if uploaded_video and predict_isl:
            with st.spinner("Extracting HOG features from video..."):
                try:
                    from skimage.feature import hog

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_video.read())
                        tmp_path = tmp.name

                    cap     = cv2.VideoCapture(tmp_path)
                    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    indices = np.linspace(0, total - 1, 20, dtype=int)
                    frames  = []
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.resize(frame, (64, 64))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frames.append(frame)
                    cap.release()
                    os.unlink(tmp_path)

                    if frames:
                        all_feat = []
                        for frame in frames:
                            feat = hog(frame, orientations=9,
                                       pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2),
                                       block_norm="L2-Hys")
                            all_feat.append(feat)
                        X_input = np.mean(all_feat, axis=0).reshape(1, -1)

                        scaler = load_model("isl_scaler.pkl")

                        if "Decision Tree" in model_choice_isl:
                            model  = load_model("isl_dt.pkl")
                            color  = "#c8ff00"
                        elif "SVM" in model_choice_isl:
                            model   = load_model("isl_svm.pkl")
                            X_input = scaler.transform(X_input)
                            color   = "#00cfff"
                        else:
                            model  = load_model("isl_rf.pkl")
                            color  = "#ff6b35"

                        if model:
                            prediction = model.predict(X_input)[0]

                            isl_json = os.path.join(MODEL_DIR, "isl_class_names.json")
                            if os.path.exists(isl_json):
                                with open(isl_json) as f:
                                    isl_map = json.load(f)
                                pred_name = isl_map.get(str(int(prediction)), f"Sign {prediction}")
                            else:
                                pred_name = str(prediction)

                            st.markdown(f"""
                            <div class="result-box">
                                <div class="result-label">Predicted Sign</div>
                                <div class="result-value">{pred_name}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.caption("Sample frames from video")
                            fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
                            fig.patch.set_facecolor('#141414')
                            for i, ax in enumerate(axes):
                                frame_idx = i * 4 if i * 4 < len(frames) else -1
                                if i < len(frames):
                                    ax.imshow(frames[frame_idx], cmap="gray")
                                ax.axis("off")
                            plt.tight_layout(pad=0.2)
                            st.pyplot(fig)
                            plt.close()

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Upload a video and click Predict")

    st.markdown("---")
    st.markdown('<div class="perf-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    perf_data = pd.DataFrame({
        "Model"         : ["Decision Tree", "SVM RBF", "Random Forest"],
        "Approach"      : ["HOG + Tree Splits", "HOG + Margin Maximization", "HOG + 300 Trees"],
        "Test Accuracy" : ["~35%", "~80%", "~85%"],
        "F1 Score"      : ["~0.32", "~0.78", "~0.85"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — EMERGENCY SIREN
# ============================================================
with tab2:
    st.markdown('<div class="section-tag">Emergency Vehicle Siren Classification</div>', unsafe_allow_html=True)
    st.markdown("Upload a siren audio file to classify the emergency vehicle.")
    st.caption("Models: KNN  →  Random Forest  →  CNN (Optimized)")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_audio = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3", "flac"],
            key="siren_audio"
        )
        model_choice_siren = st.selectbox(
            "Select Model",
            ["KNN", "Random Forest", "CNN"],
            key="siren_model"
        )
        predict_siren = st.button("Classify Siren", key="predict_siren", use_container_width=True)

    with col2:
        if uploaded_audio and predict_siren:
            with st.spinner("Extracting audio features..."):
                try:
                    import librosa

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_audio.read())
                        tmp_path = tmp.name

                    y_audio, sr = librosa.load(tmp_path, sr=22050, duration=4.0, mono=True)
                    os.unlink(tmp_path)

                    mfcc     = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
                    features = mfcc.mean(axis=1).reshape(1, -1)

                    scaler_siren = load_model("scaler.pkl")
                    le           = load_model("label_encoder.pkl")

                    siren_json = os.path.join(MODEL_DIR, "siren_class_names.json")
                    if os.path.exists(siren_json):
                        with open(siren_json) as f:
                            siren_map = json.load(f)
                    else:
                        siren_map = None

                    def get_siren_name(pred_idx):
                        if siren_map:
                            return siren_map.get(str(int(pred_idx)), f"Class {pred_idx}")
                        elif le:
                            return le.inverse_transform([int(pred_idx)])[0]
                        return str(pred_idx)

                    X_scaled = scaler_siren.transform(features)

                    if "KNN" in model_choice_siren:
                        model      = load_model("knn_model.pkl")
                        color      = "#c8ff00"
                        prediction = model.predict(X_scaled)[0]
                        pred_name  = get_siren_name(prediction)
                    elif "Random Forest" in model_choice_siren:
                        model      = load_model("rf_model.pkl")
                        color      = "#ff6b35"
                        prediction = model.predict(X_scaled)[0]
                        pred_name  = get_siren_name(prediction)
                    else:
                        from tensorflow.keras.models import load_model as keras_load
                        cnn_model   = keras_load(os.path.join(MODEL_DIR, "cnn_model.h5"))
                        mfcc_mean   = mfcc.mean(axis=1).reshape(1, 40, 1)
                        pred_proba  = cnn_model.predict(mfcc_mean)[0]
                        pred_idx    = int(np.argmax(pred_proba))
                        pred_name   = get_siren_name(pred_idx)
                        color       = "#00cfff"

                        st.markdown(f"""
                        <div class="result-box">
                            <div class="result-label">Vehicle Type</div>
                            <div class="result-value">{pred_name}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        all_names = [get_siren_name(i) for i in range(len(pred_proba))]
                        pred_dict = {n: float(p) for n, p in zip(all_names, pred_proba)}
                        st.pyplot(plot_simple_chart(pred_dict, color))
                        plt.close()
                        st.stop()  # Skip duplicate output below

                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Vehicle Type</div>
                        <div class="result-value">{pred_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        proba     = model.predict_proba(X_scaled)[0]
                        all_names = [get_siren_name(i) for i in range(len(proba))]
                        pred_dict = {n: float(p) for n, p in zip(all_names, proba)}
                        st.pyplot(plot_simple_chart(pred_dict, color))
                        plt.close()
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Upload an audio file and click Classify")

    st.markdown("---")
    st.markdown('<div class="perf-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    perf_data = pd.DataFrame({
        "Model"         : ["KNN", "Random Forest", "CNN"],
        "Approach"      : ["MFCC + Distance", "MFCC + 300 Trees", "Mel Spec + Deep Learning"],
        "Test Accuracy" : ["~95%", "~95%", "~98%"],
        "F1 Score"      : ["~0.96", "~0.96", "~0.98"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 — MOBILE PRICE
# ============================================================
with tab3:
    st.markdown('<div class="section-tag">Mobile Price Range Classification</div>', unsafe_allow_html=True)
    st.markdown("Enter mobile specifications to classify price range.")
    st.caption("Models: Decision Tree  →  Naive Bayes  →  Random Forest  |  Classes: 0=Low  1=Medium  2=High  3=Very High")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Specifications")

        battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1200)
        ram           = st.slider("RAM (MB)", 256, 4000, 2000)
        int_memory    = st.slider("Internal Memory (GB)", 2, 64, 32)
        px_height     = st.slider("Pixel Height", 0, 1960, 800)
        px_width      = st.slider("Pixel Width", 500, 1998, 1200)
        clock_speed   = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
        n_cores       = st.slider("Number of Cores", 1, 8, 4)
        fc            = st.slider("Front Camera (MP)", 0, 19, 5)
        pc            = st.slider("Primary Camera (MP)", 0, 20, 10)
        sc_h          = st.slider("Screen Height (cm)", 5, 19, 12)
        sc_w          = st.slider("Screen Width (cm)", 0, 18, 6)
        talk_time     = st.slider("Talk Time (hrs)", 2, 20, 11)
        mobile_wt     = st.slider("Weight (grams)", 80, 200, 140)
        m_dep         = st.slider("Depth (cm)", 0.1, 1.0, 0.5)

        col_a, col_b = st.columns(2)
        with col_a:
            blue         = st.checkbox("Bluetooth", value=True)
            dual_sim     = st.checkbox("Dual SIM", value=True)
            four_g       = st.checkbox("4G", value=True)
        with col_b:
            three_g      = st.checkbox("3G", value=True)
            touch_screen = st.checkbox("Touch Screen", value=True)
            wifi         = st.checkbox("WiFi", value=True)

        model_choice_mobile = st.selectbox(
            "Select Model",
            ["Decision Tree", "Naive Bayes", "Random Forest"],
            key="mobile_model"
        )
        predict_mobile = st.button("Predict Price Range", key="predict_mobile", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if predict_mobile:
            with st.spinner("Predicting..."):
                try:
                    X_input = np.array([[
                        battery_power, int(blue), clock_speed, int(dual_sim),
                        fc, int(four_g), int_memory, m_dep, mobile_wt,
                        n_cores, pc, px_height, px_width, ram,
                        sc_h, sc_w, talk_time, int(three_g),
                        int(touch_screen), int(wifi)
                    ]])

                    CLASS_NAMES = ["Low (0)", "Medium (1)", "High (2)", "Very High (3)"]
                    scaler_mob  = load_model("mobile_scaler.pkl")

                    if "Decision Tree" in model_choice_mobile:
                        model  = load_model("mobile_dt.pkl")
                        color  = "#c8ff00"
                        X_pred = X_input
                    elif "Naive Bayes" in model_choice_mobile:
                        model  = load_model("mobile_nb.pkl")
                        color  = "#ffaa00"
                        X_pred = scaler_mob.transform(X_input)
                    else:
                        model  = load_model("mobile_rf.pkl")
                        color  = "#ff6b35"
                        X_pred = X_input

                    if model:
                        prediction = model.predict(X_pred)[0]
                        pred_name  = CLASS_NAMES[prediction]

                        st.markdown(f"""
                        <div class="result-box">
                            <div class="result-label">Price Range</div>
                            <div class="result-value">{pred_name}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("RAM", f"{ram} MB")
                        col_m2.metric("Battery", f"{battery_power} mAh")
                        col_m3.metric("Storage", f"{int_memory} GB")

                        fig = plot_prediction_chart(model, X_pred, CLASS_NAMES, pred_name, color)
                        if fig:
                            st.pyplot(fig)
                            plt.close()

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Set specifications and click Predict")
            st.markdown("""
            **Key features influencing price:**
            - RAM — most important predictor
            - Battery Power
            - Camera Resolution
            - Screen Resolution
            """)

    st.markdown("---")
    st.markdown('<div class="perf-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    perf_data = pd.DataFrame({
        "Model"         : ["Decision Tree", "Naive Bayes", "Random Forest"],
        "Approach"      : ["Gini Impurity Splits", "Gaussian Probability", "300 Tree Ensemble"],
        "Test Accuracy" : ["~84%", "~81%", "~90%"],
        "F1 Score"      : ["~0.83", "~0.81", "~0.90"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 — CAR ACCIDENT
# ============================================================
with tab4:
    st.markdown('<div class="section-tag">Car Accident Severity Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter accident details to predict severity.")
    st.caption("Models: Decision Tree  →  Logistic Regression  →  Random Forest  |  Classes: Fatal  /  Serious  /  Slight")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Accident Details")

        # Time field in HH:MM format
        st.markdown("**Time of Accident**")
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            accident_hour   = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=12, step=1)
        with time_col2:
            accident_minute = st.number_input("Minute (0–59)", min_value=0, max_value=59, value=0, step=1)
        accident_time_encoded = accident_hour * 60 + accident_minute  # encode as minutes since midnight

        day_of_week = st.selectbox("Day of Week",
                                    ["Monday", "Tuesday", "Wednesday",
                                     "Thursday", "Friday", "Saturday", "Sunday"])
        junction_ctrl = st.selectbox("Junction Control",
                                      ["Give way or uncontrolled",
                                       "Auto traffic signal",
                                       "Not at junction or within 20 metres",
                                       "Stop sign",
                                       "Authorised person",
                                       "Data missing or out of range"])
        junction_detail = st.selectbox("Junction Detail",
                                        ["Not at junction or within 20 metres",
                                         "T or staggered junction",
                                         "Crossroads",
                                         "Roundabout",
                                         "Mini-roundabout",
                                         "Slip road",
                                         "Other junction",
                                         "More than 4 arms (not roundabout)",
                                         "Private drive or entrance"])
        light_conditions = st.selectbox("Light Conditions",
                                         ["Daylight",
                                          "Darkness - lights lit",
                                          "Darkness - no lighting"])
        weather = st.selectbox("Weather Conditions",
                                ["Fine no high winds", "Raining no high winds",
                                 "Snowing no high winds", "Fog or mist", "Other"])
        road_surface = st.selectbox("Road Surface",
                                     ["Dry", "Wet or damp", "Snow", "Frost or ice"])
        road_type = st.selectbox("Road Type",
                                  ["Single carriageway", "Dual carriageway",
                                   "One way street", "Roundabout"])
        vehicle_type = st.selectbox("Vehicle Type",
                                     ["Car", "Motorcycle over 500cc",
                                      "Taxi/Private hire car", "Van / Goods",
                                      "Bus or coach"])
        urban_rural    = st.selectbox("Area Type", ["Urban", "Rural"])
        speed_limit    = st.selectbox("Speed Limit", [20, 30, 40, 50, 60, 70])
        num_vehicles   = st.slider("Number of Vehicles", 1, 10, 2)
        num_casualties = st.slider("Number of Casualties", 1, 20, 1)

        model_choice_acc = st.selectbox(
            "Select Model",
            ["Decision Tree", "Logistic Regression", "Random Forest"],
            key="acc_model"
        )
        predict_acc = st.button("Predict Severity", key="predict_acc", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if predict_acc:
            with st.spinner("Predicting..."):
                try:
                    encode_map = {
                        "day": {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6},
                        "junc_ctrl": {
                            "Authorised person": 0,
                            "Auto traffic sigl": 1,
                            "Auto traffic signal": 2,
                            "Data missing or out of range": 3,
                            "Give way or uncontrolled": 4,
                            "Not at junction or within 20 metres": 5,
                            "Stop sign": 6
                        },
                        "junc_det": {
                            "Crossroads": 0,
                            "Mini-roundabout": 1,
                            "More than 4 arms (not roundabout)": 2,
                            "Not at junction or within 20 metres": 3,
                            "Other junction": 4,
                            "Private drive or entrance": 5,
                            "Roundabout": 6,
                            "Slip road": 7,
                            "T or staggered junction": 8
                        },
                        "light":     {"Darkness - lights lit": 0, "Darkness - no lighting": 1, "Daylight": 2},
                        "weather":   {"Fine no high winds": 0, "Fog or mist": 1, "Other": 2,
                                      "Raining no high winds": 3, "Snowing no high winds": 4},
                        "road_surf": {"Dry": 0, "Frost or ice": 1, "Snow": 2, "Wet or damp": 3},
                        "road_type": {"Dual carriageway": 0, "One way street": 1,
                                      "Roundabout": 2, "Single carriageway": 3},
                        "vehicle":   {"Bus or coach": 0, "Car": 1, "Motorcycle over 500cc": 2,
                                      "Taxi/Private hire car": 3, "Van / Goods": 4},
                        "urban":     {"Rural": 0, "Urban": 1},
                    }

                    X_input = np.array([[
                        encode_map["day"][day_of_week],
                        encode_map["junc_ctrl"][junction_ctrl],
                        encode_map["junc_det"][junction_detail],
                        encode_map["light"][light_conditions],
                        num_casualties,
                        num_vehicles,
                        encode_map["road_surf"][road_surface],
                        encode_map["road_type"][road_type],
                        speed_limit,
                        accident_time_encoded,   # Time encoded as minutes since midnight
                        encode_map["urban"][urban_rural],
                        encode_map["weather"][weather],
                        encode_map["vehicle"][vehicle_type],
                    ]])

                    CLASS_NAMES = ["Fatal", "Serious", "Slight"]
                    scaler_acc  = load_model("accident_scaler.pkl")

                    if "Decision Tree" in model_choice_acc:
                        model  = load_model("accident_dt.pkl")
                        color  = "#c8ff00"
                        X_pred = X_input
                    elif "Logistic" in model_choice_acc:
                        model  = load_model("accident_lr.pkl")
                        color  = "#ffaa00"
                        X_pred = scaler_acc.transform(X_input)
                    else:
                        model  = load_model("accident_rf.pkl")
                        color  = "#ff6b35"
                        X_pred = X_input

                    if model:
                        prediction = model.predict(X_pred)[0]
                        if hasattr(model, 'classes_'):
                            pred_idx  = prediction
                            pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
                        else:
                            pred_name = CLASS_NAMES[prediction] if prediction < len(CLASS_NAMES) else str(prediction)

                        severity_accent = {"Fatal": "#ff4444", "Serious": "#ffaa00", "Slight": "#c8ff00"}
                        accent = severity_accent.get(pred_name, "#c8ff00")

                        st.markdown(f"""
                        <div class="result-box" style="border-left-color: {accent};">
                            <div class="result-label" style="color: {accent};">Accident Severity</div>
                            <div class="result-value">{pred_name}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Speed Limit", f"{speed_limit} mph")
                        col_m2.metric("Casualties", num_casualties)
                        col_m3.metric("Time", f"{accident_hour:02d}:{accident_minute:02d}")

                        fig = plot_prediction_chart(model, X_pred, CLASS_NAMES, pred_name, accent)
                        if fig:
                            st.pyplot(fig)
                            plt.close()

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Enter accident details and click Predict")

    st.markdown("---")
    st.markdown('<div class="perf-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    perf_data = pd.DataFrame({
        "Model"         : ["Decision Tree", "Logistic Regression", "Random Forest"],
        "Approach"      : ["Rule-based Splits", "Probabilistic Classification", "300 Tree Ensemble"],
        "Test Accuracy" : ["~87%", "~85%", "~95%"],
        "F1 Score"      : ["~0.86", "~0.84", "~0.94"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)


# ============================================================
# TAB 5 — SPORTS IMAGE
# ============================================================
with tab5:
    st.markdown('<div class="section-tag">Sports Image Classification</div>', unsafe_allow_html=True)
    st.markdown("Upload a sports image to identify the sport.")
    st.caption("Models: Decision Tree  →  SVM  →  EfficientNetB0")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_img = st.file_uploader(
            "Upload Sports Image",
            type=["jpg", "jpeg", "png"],
            key="sports_img"
        )
        model_choice_sports = st.selectbox(
            "Select Model",
            ["Decision Tree + HOG",
             "SVM RBF + HOG",
             "EfficientNetB0"],
            key="sports_model"
        )
        predict_sports = st.button("Classify Sport", key="predict_sports", use_container_width=True)

        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded Image", width=260)

    with col2:
        st.subheader("Prediction Result")

        if uploaded_img and predict_sports:
            with st.spinner("Analyzing image..."):
                try:
                    from skimage.feature import hog

                    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if "EfficientNet" in model_choice_sports:
                        from PIL import Image
                        import io

                        uploaded_img.seek(0)
                        pil_img = Image.open(io.BytesIO(uploaded_img.read())).convert("RGB")
                        pil_img = pil_img.resize((224, 224))
                        arr     = np.array(pil_img, dtype=np.float32)
                        arr     = tf.keras.applications.efficientnet.preprocess_input(arr)
                        arr     = np.expand_dims(arr, axis=0)

                        # Build model and load weights
                        def build_sports_model(num_classes=100):
                            base    = tf.keras.applications.EfficientNetB0(
                                include_top=False,
                                weights=None,
                                input_shape=(224, 224, 3),
                                pooling='avg'
                            )
                            inputs  = tf.keras.Input(shape=(224, 224, 3))
                            x       = base(inputs, training=False)
                            x       = tf.keras.layers.Dropout(0.3)(x)
                            x       = tf.keras.layers.Dense(256, activation='relu')(x)
                            x       = tf.keras.layers.Dropout(0.2)(x)
                            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                            return tf.keras.Model(inputs, outputs)

                        sports_model = build_sports_model(num_classes=100)
                        sports_model.load_weights(os.path.join(MODEL_DIR, 'sports_efficientnet.weights.h5'))

                        pred_proba = sports_model.predict(arr, verbose=0)[0]
                        pred_idx   = int(np.argmax(pred_proba))
                        color      = "#00cfff"

                        sports_json = os.path.join(MODEL_DIR, "sports_class_names.json")
                        if os.path.exists(sports_json):
                            with open(sports_json) as f:
                                sports_map = json.load(f)
                            pred_name = sports_map.get(str(pred_idx), f"Class {pred_idx}")
                        else:
                            pred_name = str(pred_idx)

                        st.markdown(f"""
                        <div class="result-box">
                            <div class="result-label">Sport Identified</div>
                            <div class="result-value">{pred_name.upper()}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        all_names = []
                        for i in range(len(pred_proba)):
                            if os.path.exists(sports_json):
                                all_names.append(sports_map.get(str(i), f"Class {i}"))
                            else:
                                all_names.append(f"Class {i}")
                        pred_dict = {n: float(p) for n, p in zip(all_names, pred_proba)}
                        top_dict  = dict(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:8])
                        st.pyplot(plot_simple_chart(top_dict, color))
                        plt.close()

                    else:
                        # HOG path
                        gray  = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2GRAY)
                        feat  = hog(gray, orientations=9,
                                    pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2),
                                    block_norm="L2-Hys")
                        X_pred = feat.reshape(1, -1)

                        if "Decision Tree" in model_choice_sports:
                            model = load_model("sports_dt.pkl")
                            color = "#c8ff00"
                        else:
                            model     = load_model("sports_svm.pkl")
                            scaler_sp = load_model("sports_scaler.pkl")
                            X_pred    = scaler_sp.transform(X_pred)
                            color     = "#00cfff"

                        if model:
                            prediction = model.predict(X_pred)[0]

                            sports_json = os.path.join(MODEL_DIR, "sports_class_names.json")
                            if os.path.exists(sports_json):
                                with open(sports_json) as f:
                                    sports_map = json.load(f)
                                pred_name = sports_map.get(str(int(prediction)), f"Class {prediction}")
                            else:
                                pred_name = str(prediction)

                            st.markdown(f"""
                            <div class="result-box">
                                <div class="result-label">Sport Identified</div>
                                <div class="result-value">{pred_name.upper()}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # HOG visualization
                            gray_small = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2GRAY)
                            from skimage import exposure as sk_exp
                            _, hog_img = hog(gray_small, orientations=9,
                                             pixels_per_cell=(8, 8),
                                             cells_per_block=(2, 2),
                                             block_norm="L2-Hys",
                                             visualize=True)
                            hog_img = sk_exp.rescale_intensity(hog_img, in_range=(0, 10))

                            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                            fig.patch.set_facecolor('#141414')
                            for ax in axes:
                                ax.set_facecolor('#141414')
                            axes[0].imshow(cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB))
                            axes[0].set_title("Input Image", color='#888', fontsize=8)
                            axes[0].axis("off")
                            axes[1].imshow(hog_img, cmap="gray")
                            axes[1].set_title("HOG Features", color='#888', fontsize=8)
                            axes[1].axis("off")
                            plt.tight_layout(pad=0.5)
                            st.pyplot(fig)
                            plt.close()

                            fig2 = plot_prediction_chart(
                                model, X_pred,
                                [str(c) for c in model.classes_] if hasattr(model, 'classes_') else [pred_name],
                                pred_name, color
                            )
                            if fig2:
                                st.pyplot(fig2)
                                plt.close()

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Upload an image and click Classify")

    st.markdown("---")
    st.markdown('<div class="perf-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    perf_data = pd.DataFrame({
        "Model"         : ["Decision Tree + HOG", "SVM RBF + HOG", "EfficientNetB0"],
        "Approach"      : ["HOG + Tree Splits", "HOG + Margin Maximization", "Deep Features + 200 Trees"],
        "Test Accuracy" : ["~8%", "~19%", "~87%"],
        "F1 Score"      : ["~0.07", "~0.18", "~0.87"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)
