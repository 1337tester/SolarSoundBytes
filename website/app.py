import streamlit as st

# Configure the app
st.set_page_config(
    page_title="SolarSoundBytes",
    page_icon="☀️🔊💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import page modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from website import home
from website.pages import dashboard, about_us, behind_scenes
from website.pages import challenges_&_further_work

# Define the navigation structure
pg = st.navigation({
    "Main": [
        st.Page(home.main, title="🏠 Home"),
        st.Page(about_us.main, title="👥 About Us"),
        st.Page(behind_scenes.main, title="🔧 Behind Scenes"),
        st.Page(challenges_&_further_work.main, title="🚧 Challenges & Further Work"),
    ],
    "Analytics": [
        st.Page(dashboard.main, title="📊 Dashboard"),
    ]
})

# Run the selected page
pg.run() 