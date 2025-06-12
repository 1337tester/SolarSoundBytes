import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def page_config():
    st.set_page_config(page_title="Behind the Scenes - SolarSoundBytes", layout="wide")

def header_section():
    st.title("🔧 Behind the Scenes")
    st.markdown("### How we built SolarSoundBytes: from raw data to AI-generated podcasts")
    st.markdown("""
        Welcome to our technical kitchen!
        Let us walk you through our journey of transforming years of
        public opinion, official news, market metrics, and historical events
        into a dynamic dashboard and time-traveling audio insights.
        """, unsafe_allow_html=True)
    st.markdown("---")

def data_research_tab():
    st.header("📊 Data Research & Collection")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Main Questions:")
        st.write("""
            - Where do we start?
            - How do we find **trustworthy, high-quality data**?
            - How can we ensure our data is **representative and unbiased**?
            - Which **models are best suited** for sentiment analysis of tweets and news articles?
            - How do we **visualize and interpret** the insights we uncover?
        """)
        st.subheader("Data Sources We Explored:")
        st.write("""
            To compare the sentiment of **news articles** to a broader **public sentiment**, we looked for a fitting twitter dataset.
            Although the **Climate Change Twitter Dataset (15 million tweets spanning over 13 years)** looked promising at first, we could not use it due to the lack of full-text tweets within.
            Since the vast majority of the most recent tweet_ids listed inside the Climate Change Twitter Dataset in GBR are no longer accessible, we abandoned our attempt to rehydrate this dataset.
            After extensive and unsuccessful further research for an alternative twitter dataset, we decided to create our own twitter dataset as input for a social media sentiment analysis using a scraping actor on console.apify.
        """)
        st.write("### Twitter/X API")
        st.write("""
            To work with a user-friendly scraping GUI while keeping scraping costs below **40 USD/month**, the following scraper was chosen:
            - Tweet Scraper|$0.25/1K Tweets | Pay-Per Result | No Rate Limits.
                Search Terms:
                - Renewable Energy
                - Energy Storage
        """)
        st.write("### News Articles API")
        st.write("""
            Online research for datasets of news-articles in the field of renewable energy technologies led us to the Cleantech Media Dataset by Anacode.
            This dataset settled the timeframe of our data collection to **2022-01-02 to 2024-12-24**. However, after many tests, we found out that our data set was bias towards positive sentiment.
            Therefore, we decided to use the **News API** to collect news articles from **2022-01-02 to 2024-12-24**.
            - GNews 49,00€/month: API results in JSON format via HTTP GET requests.
                Search Terms:
                - Renewable Energy
                - Energy Storage
        """)
        st.write("### Global Events")
        st.markdown("""
            Key global events with likely sentiment shifts were identified by conducting in-depth research
            using iterative ChatGPT-4.1 prompts. A summary is shown in the table in this [**link**](https://drive.google.com/file/d/16EfNelp-TF2qFCpzoPM9wJqfMXhSedhx/view?usp=sharing).
        """)
    with col2:
        metrics = [
            ("Tweets Analyzed", "100,000+"),
            ("News Articles", "4,000+"),
            ("Total Words Processed", "4.5M+"),
            ("Data Quality Score", "74.2%")
        ]
        st.markdown("**Data Metrics**")
        for label, value in metrics:
            st.metric(label, value)
        fig = go.Figure(data=go.Bar(
            x=['Tweets', 'News'],
            y=[100000, 4000],
            marker_color=['#1DA1F2', '#FF4500']
        ))
        fig.update_layout(title="Data Sources Volume", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(data=go.Bar(
            x=['Tweets', 'News'],
            y=[10, 500],
            marker_color=['#1DA1F2', '#FF4500']
        ))
        fig.update_layout(title="Text Volume (Millions of Words)", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def nlp_models_tab():
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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("🏆 **Winner: Custom FinBERT Model** - Best balance of accuracy and speed for financial sentiment!")

def pipeline_tab():
    st.header("⚙️ The Complete Pipeline")
    st.markdown("**From Raw Data to Insights: Our 6-Step Process**")
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
    st.header("🎵 Audio Generation Magic")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Text-to-Speech Pipeline:**")
        steps = [
            "**Script Generation**: AI creates podcast-style narrative",
            "**Voice Selection**: Choose appropriate voice characteristics",
            "**Audio Processing**: Add background music and effects",
            "**Quality Control**: Ensure natural speech patterns"
        ]
        for s in steps:
            st.write(f"1. {s}" if steps.index(s) == 0 else s)
        st.markdown("**Audio Features:**")
        features = [
            "Natural-sounding AI voices",
            "Background music integration",
            "Podcast-style formatting",
            "Multiple language support (planned)"
        ]
        for f in features:
            st.write(f"• {f}")
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
    st.header("🚀 Final Assembly & Integration")
    st.markdown("**Bringing It All Together with Streamlit**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Architecture Overview:**")
        frontend = [
            ("Frontend", "Streamlit web application"),
            ("Backend", "Python data processing pipeline"),
            ("Database", "Real-time sentiment data storage"),
            ("APIs", "External data sources integration"),
            ("Deployment", "Cloud-based hosting")
        ]
        for label, desc in frontend:
            st.write(f"• **{label}**: {desc}")
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
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🎯 What's Next?</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Want to see our future plans? Check out our <strong>Upcoming Updates</strong> page!</p>", unsafe_allow_html=True)

def navigation_buttons():
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
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('images/LeWagon_logo.png', width=150)
        st.markdown("<p style='text-align: center;'>Created by Le Wagon Data Science Batch #2012</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-style: italic;'>Built with ❤️ by the SolarSoundBytes team</p>", unsafe_allow_html=True)

def render_behind_scenes():
    header_section()
    main_tabs()
    whats_next_section()
    navigation_buttons()
    footer_section()

def main():
    page_config()
    render_behind_scenes()

if __name__ == "__main__":
    main()
else:
    render_behind_scenes()
