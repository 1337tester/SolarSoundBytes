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
page_options = ["Home", "About Us", "Dashboard", "Behind the Scenes", "Upcoming Updates"]
page = st.sidebar.radio("Choose a page:", page_options)


# Route pages
if page == "Home":
    # Header
    #st.image('images/SolarSoundBytes_logo_test.png', width=1000)
    st.markdown("<h1 style='text-align: center'>Welcome to SolarSoundBytes</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center'>Mapping our global transition to renewable energy into audio-bites.</h3>", unsafe_allow_html=True)

    # Add some spacing and a footer section
    st.write("")
    st.markdown("---")

    st.markdown("""
                <p style='text-align: center'>
                SolarSoundBytes is a data-driven machine-learning project that explores the development of renewable energy worldwide,
                using NLP-based sentiment analysis of public opinions and official news. Both mapped onto data on the development of
                renewable energy and energy storage technologies in the same timeframe from 2022-01-02 to 2024-12-24.
                </p>"""
                , unsafe_allow_html=True)

    st.markdown("<p style='text-align: center'> Our goal is to raise awareness about the global transition towards renewable energy, by delivering valuable information through text & audio, and reach more people, in more ways.</p>", unsafe_allow_html=True)
    st.write("")

    # --- Image Carousel ---
    st.markdown("---")

    # Define your image paths
    # Make sure these images exist in the 'images' folder or update paths accordingly
    image_files = [
        'images/carousel_image1.jpg',  # Replace with your actual image file names
        'images/carousel_image2.jpg',
        'images/carousel_image3.jpg'
    ]
    # Create dummy image files if they don't exist for testing
    for img_path in image_files:
        if not os.path.exists(img_path):
            try:
                dummy_image = Image.new('RGB', (800, 400), color = 'skyblue')
                dummy_image.save(img_path)
                st.toast(f"Created dummy image: {img_path}", icon="ℹ️")
            except Exception as e:
                st.warning(f"Could not create dummy image {img_path}: {e}")

    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0

    # Display current image
    if image_files and os.path.exists(image_files[st.session_state.current_image_index]):
        st.image(image_files[st.session_state.current_image_index],use_container_width=True)
    elif image_files:
        st.warning(f"Image not found: {image_files[st.session_state.current_image_index]}")

    # Navigation buttons
    prev_col, _, next_col = st.columns([1,8,1])
    with prev_col:
        if st.button("⬅️ Previous", use_container_width=True) and image_files:
            st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(image_files)
            st.rerun()
    with next_col:
        if st.button("Next ➡️", use_container_width=True) and image_files:
            st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(image_files)
            st.rerun()
    st.write("") # Add some space after the carousel
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
