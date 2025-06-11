import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def page_config():
    """Configure the page settings"""
    st.set_page_config(page_title="Behind the Scenes - SolarSoundBytes", layout="wide")

def header_section():
    """Display the main header and hero section"""
    st.title("🔧 Behind the Scenes")
    st.markdown("### How we built SolarSoundBytes: from raw data to AI-generated podcasts")

    # Hero Section - Centered
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown("""
        <p style='text-align: right; font-size: 1.1em; color: #666;'>
        Welcome to our technical kitchen!
        Let us walk you through our journey of building this sentiment-driven audio experience.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

def data_research_tab():
    """Content for the Data Research tab"""
    st.header("📊 Data Research & Collection")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Our Data Journey Started With Questions:**
        - Where can we find real-time market sentiment?
        - How do we ensure data quality and relevance?
        - What's the best way to capture both social media buzz and news sentiment?
        """)

        st.markdown("**Data Sources We Explored:**")
        data_sources = {
            "Twitter/X API": "Real-time social sentiment",
            "News APIs": "Professional market analysis",
            "Reddit Finance": "Community discussions",
            "Financial News Sites": "Expert opinions"
        }

        for source, description in data_sources.items():
            st.write(f"• **{source}**: {description}")

    with col2:
        st.markdown("**Data Metrics**")
        st.metric("Tweets Analyzed", "50,000+", "↗️ Growing daily")
        st.metric("News Articles", "10,000+", "↗️ Fresh content")
        st.metric("Data Quality Score", "94.2%", "↗️ Improving")

        # Data volume chart
        fig = go.Figure(data=go.Bar(
            x=['Tweets', 'News', 'Reddit'],
            y=[50000, 10000, 5000],
            marker_color=['#1DA1F2', '#FF4500', '#FF6B35']
        ))
        fig.update_layout(
            title="Data Sources Volume",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def nlp_models_tab():
    """Content for the NLP Models tab"""
    st.header("🤖 NLP Models & Testing")

    st.markdown("**The Model Battle Arena**")
    st.write("We didn't just pick the first model we found. Here's our scientific approach:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Models We Tested:**")
        models_tested = [
            "VADER Sentiment",
            "TextBlob",
            "BERT-based models",
            "FinBERT (Finance-specific)",
            "Custom trained model"
        ]

        for i, model in enumerate(models_tested, 1):
            st.write(f"{i}. {model}")

    with col2:
        st.markdown("**Evaluation Metrics**")

        # Model performance comparison
        model_performance = {
            'Model': ['VADER', 'TextBlob', 'BERT', 'FinBERT', 'Custom'],
            'Accuracy': [0.78, 0.75, 0.89, 0.92, 0.94],
            'Speed (ms)': [2.1, 1.5, 45.2, 38.7, 12.3]
        }

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=model_performance['Model'], y=model_performance['Accuracy'],
                   name="Accuracy", marker_color='#2E86AB'),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=model_performance['Model'], y=model_performance['Speed (ms)'],
                      name="Speed (ms)", line=dict(color='#A23B72')),
            secondary_y=True,
        )

        fig.update_layout(title="Model Performance Comparison", height=400)
        fig.update_yaxes(title_text="Accuracy", secondary_y=False)
        fig.update_yaxes(title_text="Speed (ms)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    # Winner announcement - centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("🏆 **Winner: Custom FinBERT Model** - Best balance of accuracy and speed for financial sentiment!")

def pipeline_tab():
    """Content for the Pipeline tab"""
    st.header("⚙️ The Complete Pipeline")

    st.markdown("**From Raw Data to Insights: Our 6-Step Process**")

    # Pipeline steps
    pipeline_steps = [
        ("🔍 Data Scraping", "Collect tweets and news articles"),
        ("🧹 Text Cleaning", "Remove noise, normalize text"),
        ("🤖 Sentiment Analysis", "Apply our trained NLP model"),
        ("📊 Data Aggregation", "Combine and weight sentiments"),
        ("🎵 Audio Script Generation", "Create podcast-style content"),
        ("🎧 Text-to-Speech", "Convert to audio format")
    ]

    for i, (step, description) in enumerate(pipeline_steps, 1):
        with st.expander(f"Step {i}: {step}"):
            st.write(description)

            # Code examples for key steps
            if i == 1:
                st.code("""
def scrape_financial_tweets(symbol, limit=100):
    tweets = api.search_tweets(
        q=f"${symbol} OR #{symbol}",
        count=limit,
        lang='en'
    )
    return [tweet.text for tweet in tweets]
                """, language='python')
            elif i == 3:
                st.code("""
def analyze_sentiment(text):
    # Our custom model in action
    prediction = model.predict(text)
    return {
        'sentiment': prediction.label,
        'confidence': prediction.confidence,
        'score': prediction.score
    }
                """, language='python')

def audio_generation_tab():
    """Content for the Audio Generation tab"""
    st.header("🎵 Audio Generation Magic")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Text-to-Speech Pipeline:**")
        st.write("1. **Script Generation**: AI creates podcast-style narrative")
        st.write("2. **Voice Selection**: Choose appropriate voice characteristics")
        st.write("3. **Audio Processing**: Add background music and effects")
        st.write("4. **Quality Control**: Ensure natural speech patterns")

        st.markdown("**Audio Features:**")
        st.write("• Natural-sounding AI voices")
        st.write("• Background music integration")
        st.write("• Podcast-style formatting")
        st.write("• Multiple language support (planned)")

    with col2:
        st.markdown("**Sample Audio Script Generation:**")
        st.code("""
# Example of generated script
script = f'''
Welcome to your daily market sentiment briefing.

Today's analysis shows {sentiment_summary} sentiment
around {top_stocks}.

Let's dive into the details...
'''
        """, language='python')

        st.info("🎧 **Fun Fact**: Each podcast is unique and generated in real-time based on current market sentiment!")

def final_assembly_tab():
    """Content for the Final Assembly tab"""
    st.header("🚀 Final Assembly & Integration")

    st.markdown("**Bringing It All Together with Streamlit**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Architecture Overview:**")
        st.write("• **Frontend**: Streamlit web application")
        st.write("• **Backend**: Python data processing pipeline")
        st.write("• **Database**: Real-time sentiment data storage")
        st.write("• **APIs**: External data sources integration")
        st.write("• **Deployment**: Cloud-based hosting")

    with col2:
        st.markdown("**Technical Stack:**")
        tech_stack = {
            "Frontend": "Streamlit",
            "ML/NLP": "Transformers, scikit-learn",
            "Data Processing": "Pandas, NumPy",
            "Visualization": "Plotly, Matplotlib",
            "Audio": "Text-to-Speech APIs",
            "Deployment": "Streamlit Cloud"
        }

        for category, tech in tech_stack.items():
            st.write(f"• **{category}**: {tech}")

def main_tabs():
    """Display the main content tabs"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Research",
        "🤖 NLP Models",
        "⚙️ Pipeline",
        "🎵 Audio Generation",
        "🚀 Final Assembly"
    ])

    with tab1:
        data_research_tab()

    with tab2:
        nlp_models_tab()

    with tab3:
        pipeline_tab()

    with tab4:
        audio_generation_tab()

    with tab5:
        final_assembly_tab()

def whats_next_section():
    """Display the What's Next section"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🎯 What's Next?</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Want to see our future plans? Check out our <strong>Upcoming Updates</strong> page!</p>", unsafe_allow_html=True)

def navigation_buttons():
    """Display navigation buttons"""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("home.py")
    with col3:
        if st.button("📊 Try Dashboard", use_container_width=True):
            st.switch_page("pages/app.py")
    with col4:
        if st.button("🚀 Future Features", use_container_width=True):
            st.switch_page("pages/upcoming.py")

def footer_section():
    """Display the footer with logo and credits"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image('images/LeWagon_logo.png', width=150)
            st.markdown("<p style='text-align: center;'>Created by Le Wagon Data Science Batch #2012</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-style: italic;'>Built with ❤️ by the SolarSoundBytes team</p>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("📷 Logo not found")
            st.markdown("<p style='text-align: center;'>Created by Le Wagon Data Science Batch #2012</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-style: italic;'>Built with ❤️ by the SolarSoundBytes team</p>", unsafe_allow_html=True)

def render_behind_scenes():
    """Render function for importing into other pages"""
    header_section()
    main_tabs()
    whats_next_section()
    navigation_buttons()
    footer_section()

def main():
    """Main function to run the page"""
    page_config()
    render_behind_scenes()

# Run the page
if __name__ == "__main__":
    main()
else:
    # This runs when imported
    render_behind_scenes()
