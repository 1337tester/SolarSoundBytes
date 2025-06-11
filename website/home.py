import streamlit as st
from PIL import Image
import os

# Import page functions
try:
    from pages.about_us import render_about_us
    #from pages.dashboard import render_dashboard
    #from pages.upcoming import render_upcoming
    from pages.behind_scenes import render_behind_scenes

except ImportError:
    pass

# Page config
st.set_page_config(page_title="SolarSoundBytes", page_icon="☀️", layout="wide")

# Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "About Us", "Dashboard", "Behind the Scenes", "Upcoming Updates"])

# Route pages
if page == "Home":
    # Header
    st.image('images/SolarSoundBytes_logo_test.png', width=1000)
    st.markdown("<h1 style='text-align: center'>Welcome to SolarSoundBytes</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center'>Mapping our global transition to solar energy into audio-bites.</h3>", unsafe_allow_html=True)

    # Add some spacing and a footer section
    st.write("")
    st.markdown("---")

    st.markdown("<h5 style='text-align: center'> SolarSoundBytes is a data-driven machine-learning project that explores the development of renewable energy worldwide, using NLP-based sentiment analysis of public opinions and official news.</h5>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'> Our goal is to raise awareness about the global transition towards renewable energy, by delivering valuable information through text & audio, and reach more people, in more ways.</p>", unsafe_allow_html=True)

    # Logos
    st.markdown("---")


    col1, col2, col3 = st.columns(3)
    with col2:
        try:
            st.image('images/LeWagon_logo.png', width=300)
            st.write("Created by LeWagon Data Science, Batch #2012")
        except FileNotFoundError:
                    st.info("📷 Image not found")


elif page == "About Us":
    render_about_us()
elif page == "Dashboard":
    st.title("📊 Dashboard")
    st.info("🚧 Coming soon!")
elif page == "Behind the Scenes":
    render_behind_scenes()
elif page == "Upcoming Updates":
    st.title("🚀 Upcoming Updates")
    st.info("💭 Future features coming soon!")
