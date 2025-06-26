import streamlit as st
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_components import get_emoji_title

# Page config
#st.set_page_config(page_title="About Us - SolarSoundBytes", layout="wide")

def render_about_us():
    # Custom CSS for better alignment and styling
    st.markdown("""
    <style>
        .team-title {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 2rem;
            color: #ffffff;
        }

        .team-member {
            text-align: center;
            padding: 20px;
            height: 100%;
        }

        .team-member h3 {
            color: #ffffff;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }

        .team-image {
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .social-links {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 1rem;
        }

        .social-link {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 8px 12px;
            background-color: #f0f2f6;
            border-radius: 20px;
            text-decoration: none;
            color: #333;
            transition: background-color 0.3s;
        }

        .social-link:hover {
            background-color: #e1e5e9;
        }

        .profile-description {
            text-align: justify;
            line-height: 1.6;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Add CSS to center images and style buttons
    st.markdown("""
    <style>
    .team-name {
        height: 48px;   /* Adjust as needed to fit the longest name */
        display: flex;
        align-items: flex-end; /* Align text to bottom of the area for nice look */
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #fff;
        margin-bottom: 1rem;
    }
    
    /* Custom styling for link buttons */
    .stLinkButton > a {
        padding: 4px 8px !important;
        font-size: 14px !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    /* Align footer section headers */
    .stats-section {
        text-align: left !important;
    }
    
    .stats-section h3 {
        text-align: left !important;
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stats-section .emoji {
        font-size: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # Centered title
    st.markdown(f'<h1 class="team-title">{get_emoji_title(include_team=True)}</h1>', unsafe_allow_html=True)

    # Create 3 equal columns with proper spacing
    col1, col2, col3 = st.columns(3, gap="medium")

    #FADRI PESTALOZZI
    with col1:
        st.markdown('<div class="team-name">Fadri Pestalozzi</div>', unsafe_allow_html=True)
        # st.subheader("Fadri Pestalozzi")
        image = Image.open('website/images/Fadri.jpeg').resize((250, 250))
        st.image(image)
        st.write("Mechanical engineer turned software developer proficient in Python, SQL, and Odoo. After building a strong backend foundation, he's currently diving into ML/AI through community‑driven bootcamps and open-source events. Motivated by collaborative impact and continuous upskilling.")
        # link buttons
        col_linkedin, col_github = st.columns(2)
        with col_linkedin:
            st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/fadri-pestalozzi/")
        with col_github:
            st.link_button("🐙 GitHub", "https://github.com/FadriPestalozzi")


    # STEFFEN LAUTERBACH
    with col2:
        st.markdown('<div class="team-name">Steffen Lauterbach</div>', unsafe_allow_html=True)
        # st.subheader("Steffen Lauterbach")
        image = Image.open('website/images/SteffenLauterbach.png').resize((250, 250))
        st.image(image)
        st.write("Renewable energy engineer and former research associate with deep experience in designing and optimizing clean energy systems. Passionate about bridging technical innovation with real-world impact. Committed to driving the next wave of green energy solutions.")
        # link buttons
        col_linkedin, col_github = st.columns(2)
        with col_linkedin:
            st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/92-steffen-lauterbach/")
        with col_github:
            st.link_button("🐙 GitHub", "https://github.com/SL14-SL14")

    # ENRIQUE FLORES ROLDÁN
    with col3:
        st.markdown('<div class="team-name">Enrique Flores Roldán</div>', unsafe_allow_html=True)
        # st.subheader("Enrique Flores Roldán")
        image = Image.open('website/images/Enrique.jpeg')
        width, height = image.size
        size = min(width, height)  # Use the smaller dimension
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image = image.crop((left, top, right, bottom))
        image = image.resize((250, 250), Image.Resampling.LANCZOS)
        st.image(image)
        st.write("Video producer with 12 years of experience crafting visual storytelling across TV, advertising, and corporate media. Now pursuing a career shift into ML and AI to fuse creativity with cutting‑edge technology. Eager to apply narrative expertise in building intelligent, engaging solutions.")
        # link buttons
        col_linkedin, col_github = st.columns(2)
        with col_linkedin:
            st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/enriqfr5/")
        with col_github:
            st.link_button("🐙 GitHub", "https://github.com/EFRdev")


    # Add some spacing and a footer section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # Team stats or additional info
    col_stats1, col_stats2, col_stats3 = st.columns(3, vertical_alignment='top')

    with col_stats1:
        st.markdown('<div class="stats-section">', unsafe_allow_html=True)
        st.markdown('<h3><span class="emoji">🎓</span> Education</h3>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: left;">Le Wagon Data Science Bootcamp</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats2:
        st.markdown('<div class="stats-section">', unsafe_allow_html=True)
        st.markdown('<h3><span class="emoji">🌍</span> Based in</h3>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: left;">Barcelona</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats3:
        st.markdown('<div class="stats-section">', unsafe_allow_html=True)
        st.markdown('<h3><span class="emoji">🚀</span> Mission</h3>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: left;">AI-powered insights tracking the global shift to renewable energy</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function for the about us page"""
    render_about_us()

if __name__ == "__main__":
    render_about_us()
