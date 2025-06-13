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
from website.pages import cop28, event_russia, repowereu, solar_1tw, solar_oil, us_ira

# Define the navigation structure
pg = st.navigation({
    "Main": [
        st.Page(home.main, title="🏠 Home", icon="🏠"),
        st.Page(about_us.main, title="👥 About Us", icon="👥"),
        st.Page(behind_scenes.main, title="🔧 Behind Scenes", icon="🔧"),
    ],
    "Analytics": [
        st.Page(dashboard.main, title="📊 Dashboard", icon="📊"),
    ],
    "Event Analysis": [
        st.Page(event_russia.main, title="🇺🇦 Russia Invasion", icon="🇺🇦"),
        st.Page(cop28.main, title="🌍 COP28", icon="🌍"),
        st.Page(repowereu.main, title="⚡ REPowerEU", icon="⚡"),
        st.Page(solar_1tw.main, title="☀️ Solar 1TW", icon="☀️"),
        st.Page(solar_oil.main, title="🛢️ Solar Oil", icon="🛢️"),
        st.Page(us_ira.main, title="🇺🇸 US IRA", icon="🇺🇸"),
    ]
})

# Run the selected page
pg.run() 