import streamlit as st
from PIL import Image
import os

def main():
    """Main function for the home page"""
    # Header
    st.markdown("<h1 style='text-align: center'>Welcome to ☀️Solar🔊Sound🍔Bytes</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center'>Mapping our global energy transition into tasty audio-bites.</h3>", unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

    st.markdown("""<p style='text-align: center'>
            SolarSoundBytes is a data-driven machine-learning project that explores the relationship between public opinion, media coverage,
            and renewable energy development worldwide.""", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: center'>
            Using Natural Language Processing (NLP) sentiment analysis of Twitter conversations and official news coverage,
            we analyze correlations with key renewable energy indicators including S&P 500 market performance and
            Ember's Monthly Wind and Solar Capacity Data, all during the same timeframe from 2022-01-02 to 2024-12-24.</p>""", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center'> Our goal is to raise awareness about the global transition towards renewable energy, by delivering valuable information through text & audio, and reach more people, in more ways.</p>", unsafe_allow_html=True)
    st.write("")

    # --- Image Carousel ---
    st.markdown("---")

    st.image('website/images/ren_en_Image.png', width=200, use_container_width=True)

    st.markdown("---")
    
    # Center the footer content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            # Create two columns for icon and text
            img_col, text_col = st.columns([1, 4])
            with img_col:
                st.image('website/images/LeWagonIcon.png', width=70)
            with text_col:
                st.markdown("""
                    <div style="margin-top: 0px;">
                        <div style="font-size: 14px; margin-bottom: 2px; white-space: nowrap;">🚁 Lift-off as final project of our <a href="https://www.lewagon.com/barcelona/data-science-course" target="_blank">🥾 Le Wagon  Data Science Bootcamp</a> batch #2012 in Barcelona 🏖️ </div>
                        <div style="font-size: 14px; margin-bottom: 2px; white-space: nowrap;">🫀 Created with love by the <a href="/about_us">☀️Solar🔊Sound🍔Bytes 🫂Team!</a> </div>
                        <div style="font-size: 14px; color: #666; white-space: nowrap;">💪 Please <a href="https://github.com/FadriPestalozzi/SolarSoundBytes/discussions/categories/ideas" target="_blank">tell us what you think 🧠</a> so we can reach for the stars together 🚀</div>
                    </div>
                """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("📷 Image not found")

# For backward compatibility when run directly
if __name__ == "__main__":
    st.set_page_config(page_title="SolarSoundBytes", page_icon="☀️🔊💻", layout="wide")
    main()
