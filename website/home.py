import streamlit as st
from PIL import Image
import os

# Page config
st.set_page_config(page_title="SolarSoundBytes", page_icon="☀️", layout="wide")

# Header
st.markdown("<h1 style='text-align: center'>Welcome to SolarSoundBytes</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center'>Mapping our global transition to renewable energy into audio-bites.</h3>", unsafe_allow_html=True)

st.write("")
st.markdown("---")

st.markdown("""<p style='text-align: center'>
        SolarSoundBytes is a data-driven machine-learning project that explores the relationship between public opinion, media coverage,
        and renewable energy development worldwide.""", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center'>
        Using NLP-based sentiment analysis of Twitter conversations and official news coverage,
        we analyze correlations with key renewable energy indicators including S&P 500 market performance and
        Ember's Monthly Wind and Solar Capacity Data across the timeframe from 2022-01-02 to 2024-12-24.</p>""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center'> Our goal is to raise awareness about the global transition towards renewable energy, by delivering valuable information through text & audio, and reach more people, in more ways.</p>", unsafe_allow_html=True)
st.write("")

# --- Image Carousel ---
st.markdown("---")

st.image('images/ren_en_Image.png', width=200, use_container_width=True)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    try:
        img_col, text_col = st.columns([1, 3])
        with img_col:
            st.image('images/LeWagonIcon.png', width=100)
        with text_col:
            st.write("Created by LeWagon Data Science")
            st.write("Barcelona Batch #2012")
    except FileNotFoundError:
        st.info("📷 Image not found")
