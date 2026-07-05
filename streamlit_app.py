"""
Streamlit demo: same predictor as predict.py / webdemo, wrapped around
st.camera_input (snapshot-per-click -- Streamlit has no continuous live
video widget, unlike webdemo/index.html) plus a file-upload fallback.

Run locally:
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud by pointing it at this file in a
GitHub repo (see README.md).
"""
import cv2
import numpy as np
import streamlit as st

from predict import load_model, predict

st.set_page_config(page_title="Spot the Fake Photo", page_icon="\U0001f4f7")


@st.cache_resource
def get_model():
    return load_model()


def decode(uploaded_file):
    raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def show_result(bgr, bundle):
    score = predict(bgr, bundle)
    threshold = bundle["threshold"]
    st.progress(min(max(score, 0.0), 1.0))
    if score >= threshold:
        st.error(f"PHOTO OF A SCREEN ({score:.0%})")
    else:
        st.success(f"REAL PHOTO ({1 - score:.0%} real)")
    st.caption(f"raw score: {score:.4f}  |  threshold: {threshold:.2f}")


st.title("Spot the Fake Photo")
st.caption("Take a photo of something real, then a photo of a screen/printout showing a picture.")

bundle = get_model()

tab_camera, tab_upload = st.tabs(["Camera", "Upload"])

with tab_camera:
    shot = st.camera_input("Take a photo", label_visibility="collapsed")
    if shot is not None:
        show_result(decode(shot), bundle)

with tab_upload:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        show_result(decode(uploaded), bundle)
