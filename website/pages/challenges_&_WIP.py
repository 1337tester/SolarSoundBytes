import streamlit as st
from PIL import Image

# Page config
#st.set_page_config(page_title="About Us - SolarSoundBytes", layout="wide")

def render_limitations():
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
    st.markdown('<h1 class="team-title">Our Challenges and Furter Work</h1>', unsafe_allow_html=True)

    # Create 3 equal columns with proper spacing

    st.header("Challenges & Limitations")
    st.write("""
        - **Twitter API rate limits and restrictions** on historical data access, which lead us to scrape tweets and create our own dataset.
        - Scraped twits had **no location avilable**.
        - No high-quality news artcile dataset with manually labeled sentiment was available to train the model.
        - **Time constraints** limited the scope of development and testing.
        - During multiple rounds of fine-tuning, we observed that the **data was biased** toward positive sentiment,
        which led us to **scrape news articles** and create our own dataset.
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("Further Work")
    st.write("""
            - **Model Training and Fine-Tuning** specifically tailored for renewable energy discourse to improve the relevance and accuracy of input data.
            - **Live Data Integration:** Implement real-time data pipelines for continuous sentiment analysis and regular updates on renewable energy metrics.
            - **Audio Podcast Experience:** Enhance audio quality and develop a podcast with background music integration and multi-language support.
    """, unsafe_allow_html=True)

    # Add some spacing and a footer section
    st.write("")
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

def main():
    """Main function for the challenges & WIP page"""
    # Set page config (must be first Streamlit command)
    st.set_page_config(page_title="Challenges & WIP @ ☀️🔊🍔", page_icon="🚧", layout="wide")
    render_limitations()

if __name__ == "__main__":
    # Set page config (must be first Streamlit command)
    st.set_page_config(page_title="Challenges & WIP @ ☀️🔊🍔", page_icon="🚧", layout="wide")
    render_limitations()
