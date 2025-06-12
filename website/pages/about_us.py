import streamlit as st
from PIL import Image

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

    # Centered title
    st.markdown('<h1 class="team-title">The SolarSoundBytes Team</h1>', unsafe_allow_html=True)

    # Create 3 equal columns with proper spacing
    col1, col2, col3 = st.columns(3, gap="medium")

    #FADRI PESTALOZZI
    with col1:
        st.markdown('<div class="team-member">', unsafe_allow_html=True)
        st.subheader("Fadri Pestalozzi")
        try:
            image = Image.open('images/Fadri.jpeg')
            image = image.resize((250, 250), Image.Resampling.LANCZOS)
            st.image(image)
        except FileNotFoundError: st.info("📷 Image not found")
        st.markdown("""
        <div class="profile-description">
        Mechanical engineer turned software developer proficient in Python, SQL, and Odoo, now pivoting fully into tech. After building strong foundations in full-stack development, he's diving into ML/AI through community‑driven bootcamps and open-source events. Motivated by collaborative impact and continuous upskilling.
        </div>
        """, unsafe_allow_html=True)
    # Social links as buttons
        col_linkedin, col_github = st.columns(2)

        with col_linkedin:
            if st.button("🔗 LinkedIn", key="linkedin_fadri"):
                st.link_button("Go to LinkedIn", "https://www.linkedin.com/in/fadri-pestalozzi/")

        with col_github:
            if st.button("💻 GitHub", key="github_fadri"):
                st.link_button("Go to GitHub", "https://github.com/FadriPestalozzi")


    # STEFFEN LAUTERBACH
    with col2:
        st.markdown('<div class="team-member">', unsafe_allow_html=True)
        st.subheader("Steffen Lauterbach")
        try:
            image = Image.open('images/SteffenLauterbach.png')
            image = image.resize((250, 250), Image.Resampling.LANCZOS)
            st.image(image)
        except FileNotFoundError: st.info("📷 Image not found")
        # Description
        st.markdown("""
        <div class="profile-description">
        Renewable energy engineer and former research associate with deep experience in designing and optimizing clean energy systems. Passionate about bridging technical innovation with real-world impact. Committed to driving the next wave of green energy solutions.
        </div>
        """, unsafe_allow_html=True)

        # Social links with logos
        col_linkedin, col_github = st.columns(2)
        with col_linkedin:
            if st.button("🔗 LinkedIn", key="linkedin_Steffen"):
                st.link_button("Go to LinkedIn", "https://www.linkedin.com/in/92-steffen-lauterbach/")

        with col_github:
            if st.button("💻 GitHub", key="github_Steffen"):
                st.link_button("Go to GitHub", "https://github.com/SL14-SL14")




    # ENRIQUE FLORES ROLDÁN
    with col3:
        st.markdown('<div class="team-member">', unsafe_allow_html=True)
        st.subheader("Enrique Flores Roldán")
        try:
            image = Image.open('images/Enrique.jpeg')
            width, height = image.size
            size = min(width, height)  # Use the smaller dimension
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            image = image.crop((left, top, right, bottom))
            image = image.resize((250, 250), Image.Resampling.LANCZOS)
            st.image(image)
        except FileNotFoundError:
            st.info("📷 Image not found")
        # Description
        st.markdown("""
        <div class="profile-description">
        Video producer with 12 years of experience crafting visual storytelling across TV, advertising, and corporate media. Now pursuing a career shift into ML and AI to fuse creativity with cutting‑edge technology. Eager to apply narrative expertise in building intelligent, engaging solutions.
        </div>
        """, unsafe_allow_html=True)

        # Social links with logos
        col_linkedin, col_github = st.columns(2)
        with col_linkedin:
            if st.button("🔗 LinkedIn", key="linkedin_Enrique"):
                st.link_button("Go to LinkedIn", "https://www.linkedin.com/in/enriqfr5/")

        with col_github:
            if st.button("💻 GitHub", key="github_Enrique"):
                st.link_button("Go to GitHub", "https://github.com/EFRdev")


    # Add some spacing and a footer section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # Team stats or additional info
    col_stats1, col_stats2, col_stats3 = st.columns(3,vertical_alignment='center')

    with col_stats1:
        st.markdown("### 🎓 Education")
        st.write("Le Wagon Data Science Bootcamp")

    with col_stats2:
        st.markdown("### 🌍 Based in")
        st.write("Barcelona")

    with col_stats3:
        st.markdown("### 🚀 Mission")
        st.write("AI-powered insights tracking the global shift to renewable energy")

if __name__ == "__main__":
    render_about_us()
