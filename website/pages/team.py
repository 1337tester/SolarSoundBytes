import streamlit as st
from PIL import Image


st.header('The SolarSoundBytes Team: ')
# Create 3 columns for 3 teammembers
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Steffen Lauterbach")
    image = Image.open('images/SteffenLauterbach.png') # Pfad zu Ihrem Bild
    st.image(image)
    st.write('blablabla')
